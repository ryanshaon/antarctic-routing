"""Ensemble spread calibration of the drift ensemble (fixture data, not a real-data claim)."""

from datetime import date, timedelta

import numpy as np
import pytest

from antarctic_routing.config import DomainSection
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.validation.drift_calibration import (
    SELECTED,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    run_split,
    season_bounds,
    split_pairs,
)
from antarctic_routing.validation.drift_hindcast import DailyForcing, Observation, run_start_group
from antarctic_routing.validation.drift_uncertainty import (
    IDENTITY,
    SpreadModel,
    covered,
    energy_score,
    evaluate,
    fit,
    make_case,
    one_parameter_candidates,
    scale_spread,
    select,
    two_parameter_candidates,
)

DOMAIN = DomainSection(lat_min=-66.0, lat_max=-55.0, lon_min=-72.0, lon_max=-52.0)
D0 = date(2019, 1, 4)


@pytest.fixture(scope="module")
def grid():
    return PolarGrid.from_domain(DOMAIN, 50)


def forcing(grid, first, n, u_east=0.1):
    days = [first + timedelta(days=k) for k in range(n)]
    cx, cy = grid.rotate_en_to_xy(np.full(grid.shape, u_east), np.zeros(grid.shape), grid.lon2d)
    cur = (np.broadcast_to(cx, (n, *grid.shape)).copy(), np.broadcast_to(cy, (n, *grid.shape)).copy())
    wind = (np.zeros((n, *grid.shape)), np.zeros((n, *grid.shape)))
    return DailyForcing(grid, np.array(days, dtype="datetime64[D]"), cur, wind, "fixture")


def gaussian_cases(n_cases, sigma_member_m, sigma_obs_m, lead=7, seed=0, k=200):
    """Members ~ N(m, sigma_member); observation ~ N(m, sigma_obs) - a known spread mismatch."""
    rng = np.random.default_rng(seed)
    x0, y0 = PolarGrid.to_xy(np.array([-60.0]), np.array([-60.0]))
    m = np.array([float(x0[0]), float(y0[0])])
    out = []
    for i in range(n_cases):
        members = m + rng.normal(0, sigma_member_m, (k, 2))
        ox, oy = members.mean(0) + rng.normal(0, sigma_obs_m, 2)
        lat, lon = PolarGrid.to_latlon(np.array([ox]), np.array([oy]))
        out.append(make_case(members, float(lat[0]), float(lon[0]), lead, f"B{i % 5}", 2018, str(D0)))
    return out


def test_scaling_preserves_the_ensemble_mean_and_exited_members():
    rng = np.random.default_rng(1)
    xy = rng.normal(0, 1e4, (50, 2)) + [1e6, -2e6]
    xy[3] = np.nan
    for s in (0.0, 0.4, 1.0, 2.5):
        out = scale_spread(xy, s)
        ok = np.isfinite(xy).all(1)
        np.testing.assert_allclose(out[ok].mean(0), xy[ok].mean(0), rtol=0, atol=1e-6)
        assert np.isnan(out[3]).all()
        np.testing.assert_allclose(out[ok] - out[ok].mean(0), s * (xy[ok] - xy[ok].mean(0)), atol=1e-6)
    np.testing.assert_array_equal(scale_spread(xy, 1.0)[ok], xy[ok])


def test_spread_model_factor():
    assert IDENTITY.factor(7) == IDENTITY.factor(21) == 1.0
    m = SpreadModel(0.5, -0.5)
    assert m.factor(14) == pytest.approx(0.5) and m.factor(7) == pytest.approx(0.5 * 2 ** 0.5)


def test_consistent_ensemble_has_nominal_coverage():
    cases = gaussian_cases(400, 10_000, 10_000)
    cov = evaluate(cases, IDENTITY)["per_lead"][7]["coverage"]
    assert cov["0.50"] == pytest.approx(0.5, abs=0.07) and cov["0.95"] == pytest.approx(0.95, abs=0.04)
    assert evaluate(cases, IDENTITY)["per_lead"][7]["spread_error_ratio"] == pytest.approx(1.0, abs=0.1)


def test_overdispersed_ensemble_is_detected_and_fit_recovers_the_shrink_factor():
    cases = gaussian_cases(300, 20_000, 10_000)                # spread twice the real error
    before = evaluate(cases, IDENTITY)["per_lead"][7]
    assert before["coverage"]["0.50"] > 0.75 and before["spread_error_ratio"] > 1.7
    best = fit(cases, one_parameter_candidates())[0]["model"]
    assert best.c == pytest.approx(0.5, rel=0.15)
    after = evaluate(cases, best)["per_lead"][7]
    assert after["coverage"]["0.80"] == pytest.approx(0.8, abs=0.07)
    assert after["energy_score_km"] < before["energy_score_km"]


def test_coverage_is_monotone_in_the_scale_factor():
    c = gaussian_cases(1, 10_000, 15_000, seed=3)[0]
    hits = [covered(c, s, 0.8) for s in (0.2, 0.5, 1, 2, 5)]
    assert hits == sorted(hits)                                  # False ... True
    assert energy_score(c, 1.0) == pytest.approx(energy_score(c, 1.0))


def test_fit_is_reproducible_and_deterministic():
    a = gaussian_cases(60, 15_000, 10_000, seed=4)
    b = gaussian_cases(60, 15_000, 10_000, seed=4)
    cands = two_parameter_candidates((0.5, 0.7, 1.0), (-0.5, 0.0))
    ra, rb = fit(a, cands), fit(b, cands)
    assert [r["energy_score_km"] for r in ra] == [r["energy_score_km"] for r in rb]
    assert ra[0]["model"] == rb[0]["model"]


def test_selection_prefers_simpler_unless_clearly_better():
    v = {"none": {"mean_abs_coverage_error": 0.20}, "one_parameter": {"mean_abs_coverage_error": 0.05},
         "two_parameter": {"mean_abs_coverage_error": 0.04}}
    assert select(v) == "one_parameter"
    v["two_parameter"]["mean_abs_coverage_error"] = 0.02
    assert select(v) == "two_parameter"
    v["one_parameter"]["mean_abs_coverage_error"] = 0.25
    v["two_parameter"]["mean_abs_coverage_error"] = 0.21
    assert select(v) == "none"


def test_keep_members_does_not_change_scores(grid):
    f = forcing(grid, D0, 30)
    s = Observation("A1", D0, -60.0, -62.0)
    obs = [s, Observation("A1", D0 + timedelta(days=7), -60.0, -61.5)]
    pairs = [p for p in split_pairs(obs, obs)["train"] if p[0] == s]
    a = run_start_group([s], pairs, f, 5, 50, params=SELECTED.ensemble())
    b = run_start_group([s], pairs, f, 5, 50, params=SELECTED.ensemble(), keep_members=True)
    ra, rb = (next(r for r in x if r["lead"] == 7) for x in (a, b))
    assert ra["error_km"] == rb["error_km"] and rb["members_xy"].shape == (50, 2)
    assert "members_xy" not in ra


def test_train_and_validation_forcing_cannot_reach_the_test_seasons(grid):
    last = season_bounds(TRAIN_SEASONS + VALIDATION_SEASONS)[1]
    assert last == date(2022, 6, 30)
    f = forcing(grid, date(2018, 11, 1), 2000).subset(date(2018, 11, 1), last)
    s = Observation("A1", date(2022, 12, 2), -60.0, -62.0)        # test season 2022-23
    t = Observation("A1", date(2022, 12, 9), -60.0, -61.5)
    with pytest.raises(ValueError, match="outside"):
        run_split([(s, 7, t)], f, SELECTED, VALIDATION_SEASONS, keep_members=True)
    assert f.missing_days(s.day, t.day)
