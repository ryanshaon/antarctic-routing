import json
from pathlib import Path

import numpy as np
import pytest

from antarctic_routing.config import load_config
from antarctic_routing.forecasting.calibration import (
    IsotonicCalibrator,
    climatological_exceedance,
    evaluate_probabilities,
    residual_ensemble,
)
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.synthetic import synthetic_history
from antarctic_routing.validation.metrics import brier_score

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
SEASON = CFG.project.season_months


def test_residual_ensemble_adds_whole_residual_fields_and_clips():
    pred = np.full((2, 3, 3), 0.5)
    bank = np.stack([np.full((2, 3, 3), 0.1), np.full((2, 3, 3), -0.7)])
    members = residual_ensemble(pred, bank, n_members=50, rng=np.random.default_rng(0))
    assert members.shape == (50, 2, 3, 3)
    values = set(np.round(np.unique(members), 6))
    assert values <= {0.6, 0.0}  # 0.5 + 0.1, or 0.5 - 0.7 clipped to 0
    # a member uses one residual field for every lead and cell (joint, coherent)
    assert all(len(np.unique(m)) == 1 for m in members)


def test_residual_ensemble_is_deterministic_for_seed():
    pred = np.random.default_rng(1).random((2, 4, 4))
    bank = np.random.default_rng(2).normal(0, 0.1, (10, 2, 4, 4))
    a = residual_ensemble(pred, bank, 8, np.random.default_rng(5))
    b = residual_ensemble(pred, bank, 8, np.random.default_rng(5))
    assert np.array_equal(a, b)


def _miscalibrated(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)
    outcome = (rng.uniform(0, 1, n) < p**2).astype(float)  # true probability is p^2
    return p, outcome


def test_isotonic_calibration_improves_brier_and_is_monotone():
    p_fit, o_fit = _miscalibrated(seed=0)
    p_test, o_test = _miscalibrated(seed=1)
    cal = IsotonicCalibrator.fit({1: (p_fit, o_fit)})
    calibrated = cal.transform(p_test, lead=1)
    assert brier_score(calibrated, o_test) < brier_score(p_test, o_test) - 0.01
    grid = np.linspace(0, 1, 101)
    assert np.all(np.diff(cal.transform(grid, lead=1)) >= -1e-12)
    assert cal.transform(np.array([0.5]), lead=1)[0] == pytest.approx(0.25, abs=0.05)


def test_calibrator_is_per_lead_and_json_roundtrips():
    p, o = _miscalibrated()
    cal = IsotonicCalibrator.fit({1: (p, o), 2: (p, (np.random.default_rng(3).uniform(size=p.size) < p).astype(float))})
    assert cal.transform(np.array([0.5]), 1)[0] != pytest.approx(cal.transform(np.array([0.5]), 2)[0], abs=0.05)
    again = IsotonicCalibrator.from_dict(json.loads(json.dumps(cal.to_dict())))
    x = np.linspace(0, 1, 37)
    assert np.allclose(again.transform(x, 2), cal.transform(x, 2))
    with pytest.raises(KeyError):
        cal.transform(x, 3)


@pytest.fixture(scope="module")
def history():
    grid = PolarGrid.from_domain(CFG.domain, resolution_km=50)
    return synthetic_history(CFG, grid, seasons=range(2010, 2016), seed=4)


def test_climatological_exceedance_uses_training_seasons_only(history):
    freq_a = climatological_exceedance(history, [2010, 2011], SEASON, tau=0.15)
    h2 = history.copy(deep=True)
    later = h2["time"].values >= np.datetime64("2012-11-01")
    h2["ice_concentration"].values[later] = 1.0
    freq_b = climatological_exceedance(h2, [2010, 2011], SEASON, tau=0.15)
    assert np.array_equal(freq_a, freq_b, equal_nan=True)
    finite = freq_a[np.isfinite(freq_a)]
    assert finite.min() >= 0 and finite.max() <= 1


def test_evaluate_probabilities_report(history):
    conc = np.nan_to_num(history["ice_concentration"].values, nan=0.0)

    def persistence_model(inputs, idx):
        return np.stack([np.repeat(conc[t][None], 3, axis=0) for t in idx])

    report = evaluate_probabilities(
        history, persistence_model, train_seasons=[2010, 2011, 2012], val_seasons=[2013, 2014],
        test_seasons=[2015], history_days=5, lead_days=3, season_months=SEASON, tau=0.15,
        n_members=10, seed=0, max_fit_points=50_000,
    )
    for key in ("raw", "calibrated", "climatology"):
        assert len(report["brier"][key]) == 3
        assert all(0 <= b <= 1 for b in report["brier"][key])
    assert report["calibrator"]["leads"] == [1, 2, 3]
    assert len(report["reliability"]["calibrated"]) == 10
    assert report["test_seasons"] == [2015]
    assert sum(report["brier"]["calibrated"]) <= sum(report["brier"]["raw"]) + 1e-3


def test_evaluate_probabilities_rejects_split_overlap(history):
    with pytest.raises(ValueError, match="overlap"):
        evaluate_probabilities(history, lambda i, t: None, [2010], [2010, 2011], [2015], 5, 3, SEASON)


def test_evaluate_probabilities_ignores_imputed_cells(history):
    """Corrupted values at imputed cells must not reach the bank, the isotonic fit or the scores."""
    truth = np.nan_to_num(history["ice_concentration"].values, nan=0.0)

    def oracle(inputs, idx):
        return np.stack([truth[t + 1: t + 4] for t in idx])

    h = history.copy(deep=True)
    cells = np.zeros(h["land_mask"].shape, bool)
    cells[4:12, 4:12] = True
    cells &= ~h["land_mask"].values
    test_days = np.where(h["time"].values >= np.datetime64("2015-11-01"))[0]
    for d in test_days:  # corrupt only test-season targets, so the bank cannot "learn" the corruption
        h["imputed_mask"].values[d, cells] = True
        h["ice_concentration"].values[d, cells] = 1.0 - h["ice_concentration"].values[d, cells]
    kwargs = dict(train_seasons=[2010, 2011, 2012], val_seasons=[2013, 2014], test_seasons=[2015],
                  history_days=5, lead_days=3, season_months=SEASON, tau=0.15, n_members=10, seed=0)
    assert max(evaluate_probabilities(h, oracle, **kwargs)["brier"]["raw"]) == pytest.approx(0.0)
    h["imputed_mask"].values[:] = False  # same corrupted values, now treated as observed
    assert max(evaluate_probabilities(h, oracle, **kwargs)["brier"]["raw"]) > 0.0
