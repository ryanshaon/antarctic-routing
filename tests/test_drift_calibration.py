"""Leakage-safe drift calibration split and fitting (fixture data, not a real-data claim)."""

from datetime import date, timedelta

import numpy as np
import pytest

from antarctic_routing.config import DomainSection
from antarctic_routing.iceberg.drift import drift_ensemble
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.validation.drift_calibration import (
    EXIT_PENALTY_KM,
    ORIGINAL,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    DriftParams,
    check_split,
    grid_search,
    objective,
    one_parameter_candidates,
    run_split,
    select,
    split_of,
    split_pairs,
    tracks,
)
from antarctic_routing.validation.drift_hindcast import DailyForcing, Observation, season_of_day

DOMAIN = DomainSection(lat_min=-66.0, lat_max=-55.0, lon_min=-72.0, lon_max=-52.0)
D0 = date(2019, 1, 4)                      # season 2018 (train)


@pytest.fixture(scope="module")
def grid():
    return PolarGrid.from_domain(DOMAIN, 50)


def forcing(grid, first, n, u_east=0.1):
    days = [first + timedelta(days=k) for k in range(n)]
    cx, cy = grid.rotate_en_to_xy(np.full(grid.shape, u_east), np.zeros(grid.shape), grid.lon2d)
    cur = (np.broadcast_to(cx, (n, *grid.shape)).copy(), np.broadcast_to(cy, (n, *grid.shape)).copy())
    wind = (np.zeros((n, *grid.shape)), np.zeros((n, *grid.shape)))
    return DailyForcing(grid, np.array(days, dtype="datetime64[D]"), cur, wind, "fixture")


def east_of(lat, lon, km):
    return lat, lon + np.rad2deg(km / (6371.0 * np.cos(np.deg2rad(lat))))


def weekly_track(bid, first, km_per_week, lat=-60.0, lon=-64.0, weeks=4):
    return [Observation(bid, first + timedelta(days=7 * k), *east_of(lat, lon, km_per_week * k)) for k in range(weeks)]


def test_seasons_are_chronological_and_disjoint():
    assert max(TRAIN_SEASONS) < min(VALIDATION_SEASONS) < min(TEST_SEASONS)
    assert 2023 in TEST_SEASONS and 2023 not in TRAIN_SEASONS + VALIDATION_SEASONS
    assert [split_of(s) for s in (2018, 2020, 2021, 2022, 2023, 2017)] == \
        ["train", "train", "validation", "test", "test", None]


def test_split_keeps_whole_tracks_and_drops_cross_season_targets():
    late = Observation("A1", date(2021, 2, 26), -60.0, -60.0)              # end of train season 2020
    next_season = Observation("A1", date(2021, 11, 19), -60.0, -59.0)      # validation season 2021
    obs = [late, Observation("A1", date(2021, 3, 5), -60.0, -59.9), next_season,
           Observation("A1", date(2021, 11, 26), -60.0, -58.9), Observation("B2", date(2023, 1, 6), -61.0, -60.0),
           Observation("B2", date(2023, 1, 13), -61.0, -59.8)]
    by = split_pairs(obs, obs)
    assert set(by) == {"train", "validation", "test"}
    assert tracks(by["train"]) == {("A1", 2020)} and tracks(by["validation"]) == {("A1", 2021)}
    assert tracks(by["test"]) == {("B2", 2022)}
    for sp, pairs in by.items():
        for _s, _, t in pairs:
            assert t is None or split_of(season_of_day(t.day)) == sp


def test_check_split_rejects_leakage():
    s = Observation("A1", D0, -60.0, -60.0)
    t = Observation("A1", D0 + timedelta(days=7), -60.0, -59.0)
    assert check_split({"train": [(s, 7, t)]}) == {("A1", 2018): "train"}
    with pytest.raises(ValueError, match="validation contains A1 2019-01-04 from season 2018"):
        check_split({"train": [(s, 7, t)], "validation": [(s, 14, None)]})
    test_obs = Observation("A1", date(2024, 1, 5), -60.0, -59.0)           # 2023-24 must never reach fitting
    with pytest.raises(ValueError, match="from season 2023"):
        check_split({"train": [(s, 7, test_obs)]})


def test_run_split_refuses_cases_and_cannot_see_forcing_outside_its_seasons(grid):
    f = forcing(grid, date(2018, 11, 1), 2000)                              # spans train .. test
    s = Observation("A1", date(2021, 12, 3), -60.0, -64.0)                 # validation season
    t = Observation("A1", date(2021, 12, 10), -60.0, -63.0)
    with pytest.raises(ValueError, match="outside"):
        run_split([(s, 7, t)], f, ORIGINAL, TRAIN_SEASONS)
    tr = f.subset(date(2018, 7, 1), date(2021, 6, 30))
    assert str(tr.days.max()) == "2021-06-30" and tr.missing_days(date(2021, 12, 3), date(2021, 12, 4))


def test_params_scale_currents_and_wind_coefficients(grid):
    assert ORIGINAL.ensemble() == {"beta": 1.0, "alpha_range": (0.01, 0.03)}
    lo, hi = DriftParams(0.3, 0.5).ensemble()["alpha_range"]
    assert (lo, hi) == pytest.approx((0.005, 0.015))
    cur, _ = forcing(grid, D0, 3).window(D0, D0 + timedelta(days=2))

    def run(**kw):
        return drift_ensemble([("A", -60.0, -64.0)], cur, None, 5, 24.0, np.random.default_rng(0),
                              sigma_pos_m=0.0, velocity_noise=0.0, **kw).xy[:, 0, -1]

    x0 = drift_ensemble([("A", -60.0, -64.0)], cur, None, 5, 24.0, np.random.default_rng(0),
                        sigma_pos_m=0.0, velocity_noise=0.0).xy[:, 0, 0]
    full, default, damped = run(), run(beta=1.0), run(beta=0.3)
    np.testing.assert_array_equal(full, default)                            # beta=1 is the unchanged model
    np.testing.assert_allclose(np.linalg.norm(damped - x0, axis=1), 0.3 * np.linalg.norm(full - x0, axis=1),
                               rtol=0.02)


def test_objective_penalises_grid_exit():
    rows = [{"lead": 7, "status": "evaluated", "error_km": 10.0},
            {"lead": 7, "status": "exited_grid"},
            {"lead": 14, "status": "evaluated", "error_km": 30.0},
            {"lead": 21, "status": "no_observation"}]
    o = objective(rows)
    assert o["per_lead"][7]["mean_error_km"] == pytest.approx((10 + EXIT_PENALTY_KM) / 2)
    assert o["per_lead"][7]["n_exited"] == 1 and o["per_lead"][21]["mean_error_km"] is None
    assert o["value"] == pytest.approx(((10 + EXIT_PENALTY_KM) / 2 + 30) / 2)


def test_grid_search_recovers_known_damping_and_is_reproducible(grid):
    f = forcing(grid, date(2018, 11, 1), 150, u_east=0.1)                  # 60.48 km/week if undamped
    obs = []
    for k, bid in enumerate(("A1", "B2", "C3")):
        obs += weekly_track(bid, D0 + timedelta(days=k), 0.4 * 60.48, lat=-60.0 + k)
    pairs = split_pairs(obs, obs)["train"]
    cands = one_parameter_candidates((0.2, 0.3, 0.4, 0.5, 1.0))
    a = grid_search(pairs, f, cands, TRAIN_SEASONS, n_members=40)
    b = grid_search(pairs, f, cands, TRAIN_SEASONS, n_members=40)
    worst = next(r for r in a if r["params"]["beta"] == 1.0)["objective"]["value"]
    assert a[0]["params"]["beta"] == 0.4 and a[0]["objective"]["value"] < 0.25 * worst
    assert [r["objective"]["value"] for r in a] == [r["objective"]["value"] for r in b]


def test_selection_prefers_the_simpler_model_unless_clearly_better():
    assert select(100.0, 96.0) == "one_parameter"
    assert select(100.0, 94.0) == "two_parameter"
