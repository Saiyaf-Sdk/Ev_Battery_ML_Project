"""Token-protected demonstration API; not certified for operational EV safety."""
from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .data import make_features
from .config import TARGET, IDENTIFIERS, SUSPECT_SCORES


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    features: dict[str, str | float | int | None] = Field(min_length=10, max_length=90)


class PredictResponse(BaseModel):
    failure_risk_score: float
    alarm_threshold: float
    alert: bool
    model_name: str
    disclaimer: str = "Synthetic data research model; not validated for real-world EV safety."


@asynccontextmanager
async def lifespan(app: FastAPI):
    configured = os.environ.get("EV_MODEL_PATH", "artifacts/model.joblib")
    path = Path(configured)
    if not path.is_file():
        raise RuntimeError(f"Trusted model artifact not found: {path}")
    # Never load joblib files from API input: pickle can run malicious code.
    bundle = joblib.load(path)
    if bundle['metadata'].get('fast_smoke_test_only'):
        raise RuntimeError('A smoke-test artifact cannot be served')
    if bundle['metadata'].get('include_suspect_scores'):
        raise RuntimeError('Composite-score sensitivity models cannot be served')
    app.state.bundle = bundle
    yield
    app.state.bundle = None


app = FastAPI(title='EV Battery Synthetic Research Demo', lifespan=lifespan,
              docs_url=None, redoc_url=None)


def require_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get('EV_API_KEY', '')
    if not expected:
        raise HTTPException(status_code=503, detail='API authentication not configured')
    if not x_api_key or not secrets.compare_digest(expected, x_api_key):
        raise HTTPException(status_code=401, detail='Invalid API key')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict', response_model=PredictResponse, dependencies=[Depends(require_key)])
def predict(body: PredictRequest):
    bundle = app.state.bundle
    meta = bundle['metadata']
    known = set(meta['raw_feature_columns'])
    provided = set(body.features)
    if provided - known:
        raise HTTPException(status_code=422,
                            detail=f"Unknown raw fields: {sorted(provided - known)[:5]}")
    if TARGET in provided or any(col in provided for col in IDENTIFIERS):
        raise HTTPException(status_code=422, detail='Identifiers and targets are disallowed')
    if provided.intersection(SUSPECT_SCORES):
        raise HTTPException(status_code=422, detail='Composite risk scores are not allowed')
    if sum(v is not None for v in body.features.values()) < 10:
        raise HTTPException(status_code=422, detail='At least 10 nonmissing telemetry values required')
    row = pd.DataFrame([body.features], columns=meta['raw_feature_columns'])
    # Explicit numeric coercion prevents hidden strings turning numeric input into objects.
    numeric_cols = bundle['model'].calibrated_classifiers_[0].estimator.named_steps['preprocess'].transformers_[0][2]
    for c in numeric_cols:
        if c in row.columns:
            value = row[c]
            if value.notna().any():
                try:
                    row[c] = pd.to_numeric(value, errors='raise')
                except (ValueError, TypeError, OverflowError):
                    raise HTTPException(status_code=422, detail=f'Invalid numeric field: {c}')
    x = make_features(row, include_suspect_scores=False)
    x = x.reindex(columns=meta['engineered_feature_columns'])
    try:
        p = float(bundle['model'].predict_proba(x)[0, 1])
    except (ValueError, TypeError) as exc:
        # Error messages may contain sensitive features; return only a generic client error.
        raise HTTPException(status_code=422, detail='Telemetry values cannot be processed') from exc
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise HTTPException(status_code=500, detail='Invalid model output')
    threshold = float(bundle['threshold'])
    return PredictResponse(failure_risk_score=round(p, 6), alarm_threshold=threshold,
                           alert=bool(p >= threshold), model_name=meta['winner'])
