from types import SimpleNamespace
import numpy as np
import pytest


try:
    from fastapi.testclient import TestClient
    from evbattery.api import app
except ImportError:
    TestClient = None
    app = None





class FakeModel:
    def __init__(self):
        self.calibrated_classifiers_ = [SimpleNamespace(
            estimator=SimpleNamespace(named_steps={
                'preprocess': SimpleNamespace(transformers_=[('num', None, ['battery_health_percent'])])
            })
        )]

    def predict_proba(self, x):
        return np.array([[0.8, 0.2]])


def test_api_auth_and_schemas(monkeypatch):
    if TestClient is None or app is None:
        pytest.skip("fastapi is not installed (Member D optional dependency)")
    monkeypatch.setenv('EV_API_KEY', 'example-secret')
    app.state.bundle = {

        'model': FakeModel(),
        'threshold': 0.25,
        'metadata': {
            'winner': 'test_dummy',
            'raw_feature_columns': ['battery_health_percent', 'cycle_count', 'battery_capacity_kwh',
                                    'state_of_charge', 'vehicle_brand', 'cell_temperature_max',
                                    'cell_temperature_avg', 'internal_resistance', 'previous_faults',
                                    'vehicle_age_years', 'terrain_type'],
            'engineered_feature_columns': ['battery_health_percent', 'cycle_count', 'battery_capacity_kwh',
                                          'state_of_charge', 'vehicle_brand', 'cell_temperature_max',
                                          'cell_temperature_avg', 'internal_resistance', 'previous_faults',
                                          'vehicle_age_years', 'terrain_type'],
        },
    }
    sample = {'features': {
        'battery_health_percent': 80, 'cycle_count': 500, 'battery_capacity_kwh': 65,
        'state_of_charge': 65, 'vehicle_brand': 'unknown', 'cell_temperature_max': 35,
        'cell_temperature_avg': 28, 'internal_resistance': 0.1, 'previous_faults': 0,
        'vehicle_age_years': 3,
    }}
    client = TestClient(app)  # not context-managed: no model-loading lifespan during unit test
    assert client.post('/predict', json=sample).status_code == 401
    headers = {'X-API-Key': 'example-secret'}
    response = client.post('/predict', json=sample, headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()['alert'] is False
    bad = {'features': {**sample['features'], 'not_a_real_feature': 5}}
    assert client.post('/predict', json=bad, headers=headers).status_code == 422
    suspect = {'features': {**sample['features'], 'thermal_runaway_risk': 30}}
    assert client.post('/predict', json=suspect, headers=headers).status_code == 422
