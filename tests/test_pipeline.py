import numpy as np
import pandas as pd
import pytest

from evbattery.config import TARGET, IDENTIFIERS, SUSPECT_SCORES

from evbattery.data import make_features, split_data, load_dataset
from evbattery.evaluation import choose_threshold, evaluate
from evbattery.modeling import pipe_for, candidate_models


def sample_frame(n=200):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        'vehicle_id': [f'v{i}' for i in range(n)],
        'battery_serial': [f'b{i}' for i in range(n)],
        'battery_failure': np.array([0] * (n * 9 // 10) + [1] * (n - n * 9 // 10)),
        'battery_health_percent': rng.uniform(60, 100, n),
        'thermal_runaway_risk': rng.uniform(0, 100, n),
        'battery_stress_index': rng.uniform(0, 100, n),
        'charging_quality_score': rng.uniform(0, 100, n),
        'vehicle_brand': rng.choice(['BrandA', 'BrandB', None], n),
        'cycle_count': rng.integers(0, 2000, n).astype(float),
        'vehicle_age_years': rng.uniform(1, 12, n),
        'cell_temperature_avg': rng.normal(30, 8, n),
        'cell_temperature_max': rng.normal(35, 8, n),
        'previous_faults': rng.integers(0, 3, n).astype(float),
        'BMS_warning_count': rng.integers(0, 4, n).astype(float),
        'sensor_fault_count': rng.integers(0, 2, n).astype(float),
    })


def test_no_identifiers_or_target_or_composite_scores():
    x = make_features(sample_frame())
    assert not {TARGET, *IDENTIFIERS, *SUSPECT_SCORES}.intersection(x.columns)
    assert 'cycles_per_year' in x.columns
    assert 'prior_fault_and_warning_count' in x.columns


def test_splits_are_disjoint_and_stratified():
    frame = sample_frame(500)
    s = split_data(frame)
    sets = [set(z.index) for z in (s.train, s.calibration, s.threshold, s.test)]
    assert len(set.union(*sets)) == len(frame)
    assert all(not sets[i] & sets[j] for i in range(4) for j in range(i + 1, 4))
    assert [len(z) for z in (s.train, s.calibration, s.threshold, s.test)] == [300, 50, 50, 100]
    assert all(z[TARGET].mean() == pytest.approx(0.1, abs=0.025)
               for z in (s.train, s.calibration, s.threshold, s.test))


def test_zero_cycles_and_years_safely_handled():
    data = sample_frame(100)
    data.loc[0, ['cycle_count', 'vehicle_age_years']] = 0
    x = make_features(data)
    assert np.isnan(x.loc[0, 'cycles_per_year'])
    assert not np.isinf(x.select_dtypes('number').to_numpy()).any()


def test_unknown_categories_and_missing_values_do_not_crash():
    raw = sample_frame()
    x = make_features(raw)
    estimator = pipe_for(x, candidate_models(fast=True)['logistic_regression'])
    estimator.fit(x, raw[TARGET])
    x_new = x.iloc[180:181].copy()
    x_new.loc[:, 'vehicle_brand'] = 'UnseenBrand'
    x_new.loc[:, 'battery_health_percent'] = np.nan
    assert estimator.predict_proba(x_new).shape == (1, 2)


def test_threshold_metrics_and_nonfinite_rejection():
    y = [0, 0, 1, 1]
    p = [0.1, 0.3, 0.7, 0.8]
    threshold, _ = choose_threshold(y, p)
    scores = evaluate(y, p, threshold)
    assert scores['tp'] + scores['tn'] + scores['fp'] + scores['fn'] == 4
    assert scores['average_precision'] == pytest.approx(1)
    with pytest.raises(ValueError):
        choose_threshold(y, [np.nan, 0.2, 0.7, 0.8])


def test_csv_rejects_duplicate_ids_and_bad_target(tmp_path):
    d = sample_frame()
    d.loc[0, 'battery_serial'] = d.loc[1, 'battery_serial']
    path = tmp_path / 'bad.csv'
    d.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        load_dataset(path)
