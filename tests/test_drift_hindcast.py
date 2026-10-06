"""Iceberg drift hindcast scoring against official positions (fixture data, not a real-data claim)."""

from datetime import date, timedelta

import numpy as np
import pytest

from antarctic_routing.config import DomainSection
from antarctic_routing.iceberg.drift import haversine_m
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.validation.drift_hindcast import (
    DailyForcing,
    Observation,
    covered_until,
    pair_observations,
    run_start_group,
    score_members,
    season_of_day,
    summarize,
)

DOMAIN = DomainSection(lat_min=-66.0, lat_max=-55.0, lon_min=-72.0, lon_max=-52.0)
D0 = date(2019, 1, 4)


@pytest.fixture(scope="module")
def grid():
    return PolarGrid.from_domain(DOMAIN, 50)


def forcing(grid, days, u_east=0.0, label="real", wind=False):
    """Uniform eastward current (m/s) on every date in ``days``."""
    ue = np.full(grid.shape, u_east)
    cx, cy = grid.rotate_en_to_xy(ue, np.zeros(grid.shape), grid.lon2d)
    t = len(days)
    cur = (np.broadcast_to(cx, (t, *grid.shape)).copy(), np.broadcast_to(cy, (t, *grid.shape)).copy())
    w = (np.zeros((t, *grid.shape)), np.zeros((t, *grid.shape))) if wind else None
    return DailyForcing(grid, np.array(days, dtype="datetime64[D]"), cur, w, label)


def days_from(start, n, skip=()):
    return [start + timedelta(days=k) for k in range(n) if k not in skip]


def east_of(lat, lon, km):
    return lat, lon + np.rad2deg(km / (6371.0 * np.cos(np.deg2rad(lat))))


def test_pairing_by_id_and_nearest_date_within_tolerance():
    a0 = Observation("A1", D0, -60, -60)
    obs = [a0, Observation("A1", D0 + timedelta(days=8), -60, -59),
           Observation("A1", D0 + timedelta(days=14), -60, -58), Observation("B2", D0 + timedelta(days=7), -61, -60)]
    pairs = {lead: t for _, lead, t in pair_observations(obs, [a0])}
    assert pairs[7].day == D0 + timedelta(days=8)          # 8 days is within +-1 of 7
    assert pairs[14].day == D0 + timedelta(days=14)
    assert pairs[21] is None                               # no A1 position near day 21; B2 never matches A1


def test_window_selects_by_exact_date_and_refuses_gaps(grid):
    f = forcing(grid, days_from(D0, 10, skip=(4,)), 0.1)
    assert f.missing_days(D0, D0 + timedelta(days=6)) == [str(D0 + timedelta(days=4))]
    assert covered_until(f, D0, D0 + timedelta(days=9)) == D0 + timedelta(days=3)
    with pytest.raises(ValueError, match="missing forcing"):
        f.window(D0, D0 + timedelta(days=6))
    cur, wind = f.window(D0, D0 + timedelta(days=3))
    assert cur.times_h.tolist() == [0, 24, 48, 72] and wind is None
    assert covered_until(f, D0 - timedelta(days=1), D0) is None


def test_uniform_current_drift_is_scored_against_the_observed_position(grid):
    lat0, lon0 = -60.0, -62.0
    f = forcing(grid, days_from(D0, 30), 0.1)                # 0.1 m/s east = 60.48 km per week
    start = Observation("A1", D0, lat0, lon0)
    right = Observation("A1", D0 + timedelta(days=7), *east_of(lat0, lon0, 60.48))
    wrong = Observation("A1", D0 + timedelta(days=14), lat0, lon0)          # did not move (e.g. grounded)
    rows = run_start_group([start], pair_observations([start, right, wrong], [start]), f, seed=1)
    by = {r["lead"]: r for r in rows}
    assert by[7]["status"] == "evaluated" and by[7]["error_km"] < 6.0 and by[7]["coverage"] == 1.0
    assert by[14]["status"] == "evaluated" and 110 < by[14]["error_km"] < 135
    assert by[14]["observed_displacement_km"] == pytest.approx(0.0)
    assert by[21]["status"] == "no_observation"
    assert by[7]["season"] == 2018


def test_missing_forcing_only_affects_leads_beyond_coverage(grid):
    f = forcing(grid, days_from(D0, 30, skip=(10,)), 0.05)
    start = Observation("A1", D0, -60.0, -62.0)
    obs = [start, Observation("A1", D0 + timedelta(days=7), -60.0, -61.5),
           Observation("A1", D0 + timedelta(days=14), -60.0, -61.0)]
    by = {r["lead"]: r for r in run_start_group([start], pair_observations(obs, [start]), f, seed=1)}
    assert by[7]["status"] == "evaluated"
    assert by[14]["status"] == "missing_forcing" and by[14]["missing_dates"] == [str(D0 + timedelta(days=10))]


def test_leaving_the_grid_is_a_failure_not_a_success(grid):
    f = forcing(grid, days_from(D0, 30), 2.0)                # 173 km/day eastward: leaves the grid
    lat, lon = -60.0, -50.0
    assert grid.cell_of(lat, lon) is not None
    start = Observation("A1", D0, lat, lon)
    gone = Observation("A1", D0 + timedelta(days=7), *east_of(lat, lon, 1200))
    (row,) = [r for r in run_start_group([start], pair_observations([start, gone], [start], leads=(7,)), f, seed=1)]
    assert row["status"] == "exited_grid" and row["coverage"] == 0.0 and "error_km" not in row


def test_low_coverage_is_reported_separately(grid):
    k = 10
    xy = np.full((k, 2), np.nan)
    x, y = PolarGrid.to_xy(np.array([-60.0]), np.array([-60.0]))
    xy[:3] = [x[0], y[0]]                                    # 3 of 10 members survive
    t = Observation("A1", D0, -60.0, -60.0)
    r = score_members(xy, t, min_coverage=0.5)
    assert r["status"] == "low_coverage" and r["coverage"] == pytest.approx(0.3) and r["n_surviving"] == 3
    assert r["error_km"] == pytest.approx(0.0, abs=1e-6)
    assert score_members(xy, t, min_coverage=0.3)["status"] == "evaluated"


def test_same_seed_gives_common_random_numbers_across_configs(grid):
    days = days_from(D0, 30)
    start = Observation("A1", D0, -60.0, -62.0)
    obs = [start, Observation("A1", D0 + timedelta(days=7), -60.0, -61.0)]
    pairs = pair_observations(obs, [start], leads=(7,))
    a = run_start_group([start], pairs, forcing(grid, days, 0.1, "a"), seed=3)[0]
    b = run_start_group([start], pairs, forcing(grid, days, 0.1, "b"), seed=3)[0]
    assert a["error_km"] == b["error_km"] and a["config"] == "a" and b["config"] == "b"


def test_summary_counts_every_status(grid):
    rows = [{"config": "real", "lead": 7, "status": "evaluated", "error_km": 10.0, "coverage": 1.0,
             "iceberg_id": "A1", "season": 2018},
            {"config": "real", "lead": 7, "status": "evaluated", "error_km": 30.0, "coverage": 0.9,
             "iceberg_id": "A1", "season": 2018},
            {"config": "real", "lead": 7, "status": "exited_grid", "coverage": 0.0, "iceberg_id": "B2", "season": 2018},
            {"config": "real", "lead": 14, "status": "no_observation", "iceberg_id": "A1", "season": 2018}]
    s = summarize(rows)["real"]
    assert s[7]["counts"] == {"evaluated": 2, "exited_grid": 1} and s[7]["n_evaluated"] == 2
    assert s[7]["error_km"]["mean"] == 20.0 and s[7]["n_tracks"] == 1
    assert s[14]["n_evaluated"] == 0 and s[21]["counts"] == {}


def test_season_label():
    assert season_of_day(date(2018, 11, 2)) == 2018 and season_of_day(date(2019, 2, 22)) == 2018


def test_haversine_matches_east_of_helper():
    lat, lon = east_of(-60.0, -62.0, 100.0)
    assert haversine_m(-60.0, -62.0, lat, lon) / 1e3 == pytest.approx(100.0, rel=1e-3)
