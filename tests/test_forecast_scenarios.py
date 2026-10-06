"""Forecast-driven joint scenarios (Stage 8.1-8.2)."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from antarctic_routing.config import load_config
from antarctic_routing.forecasting.scenarios import ForecastContext
from antarctic_routing.preprocessing.climatology import day_of_season
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.synthetic import synthetic_history

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
SEASON = CFG.project.season_months
L, H = 5, 3
TRAIN = [2010, 2011, 2012]
ISSUE = date(2014, 12, 10)


@pytest.fixture(scope="module")
def history():
    grid = PolarGrid.from_domain(CFG.domain, resolution_km=50)
    return synthetic_history(CFG, grid, seasons=range(2010, 2016), seed=5)


def _persistence(ds):
    conc = np.nan_to_num(ds["ice_concentration"].values, nan=0.0)

    def predict(inputs, idx):
        return np.repeat(inputs[:, L - 1: L], H, axis=1)  # uses only the input window

    return predict, conc


def _context(ds, seed=0):
    predictor, _ = _persistence(ds)
    return ForecastContext.build(ds, predictor, L, H, SEASON, TRAIN, bank_size=50, seed=seed)


def test_layers_are_observed_then_forecast_then_climatology(history):
    ctx = _context(history)
    world = ctx.scenarios(ISSUE, n_days=7, n_members=12, rng=np.random.default_rng(0))
    assert world.conc.shape == (12, 7, *history["land_mask"].shape)
    assert world.layer_source == ["observed", "forecast", "forecast", "forecast",
                                  "climatology", "climatology", "climatology"]
    t = ctx.index_of(ISSUE)
    obs = history["ice_concentration"].values[t]
    ocean = ~history["land_mask"].values
    assert np.allclose(world.conc[:, 0][:, ocean], obs[ocean][None])
    assert np.isnan(world.conc[:, :, ~ocean]).all()
    vals = world.conc[:, :, ocean]
    assert vals.min() >= 0 and vals.max() <= 1
    assert world.start.date() == ISSUE
    assert world.execution_mode == "controlled_synthetic"
    assert world.current_x.shape == (7, *ocean.shape) and world.wind_x is not None


def test_no_future_information_is_used(history):
    """Overwriting every observation after the issue date must not change the scenarios."""
    ctx_a = _context(history)
    a = ctx_a.scenarios(ISSUE, 7, 12, np.random.default_rng(1))
    tampered = history.copy(deep=True)
    future = tampered["time"].values > np.datetime64(ISSUE)
    tampered["ice_concentration"].values[future] = 1.0
    ctx_b = _context(tampered)
    b = ctx_b.scenarios(ISSUE, 7, 12, np.random.default_rng(1))
    assert np.array_equal(a.conc, b.conc, equal_nan=True)


def test_beyond_horizon_members_are_coherent_historical_anomaly_sequences(history):
    ctx = _context(history)
    world = ctx.scenarios(ISSUE, 8, 10, np.random.default_rng(2))
    ocean = ~history["land_mask"].values
    for k in range(10):
        season = world.meta["climatology_seasons"][k]
        assert season in TRAIN
        for h in range(H + 1, 8):
            day = ISSUE + timedelta(days=h)
            expected = np.clip(ctx.clim.mean_for(day) + ctx.anomaly(season, day), 0, 1)
            assert np.allclose(world.conc[k, h][ocean], expected[ocean], atol=1e-6)


def test_issue_without_enough_history_is_rejected(history):
    ctx = _context(history)
    with pytest.raises(ValueError, match="history"):
        ctx.scenarios(date(2014, 11, 2), 5, 4, np.random.default_rng(0))


def test_bank_and_climatology_come_from_training_seasons_only(history):
    ctx = _context(history)
    assert ctx.train_seasons == TRAIN
    assert set(ctx.bank_seasons) <= set(TRAIN)
    assert day_of_season(ISSUE, SEASON) >= 0


# ------------------------------------------------------------ daily wind forcing
def _daily_winds(shape, start: date, n: int, skip: tuple[int, ...] = ()):
    """Wind layers whose value encodes their calendar date (x = day number, y = -day number)."""
    days = [start + timedelta(days=i) for i in range(n) if i not in skip]
    wx = np.stack([np.full(shape, float(d.toordinal())) for d in days])
    return (wx, -wx), np.array(days, dtype="datetime64[D]")


def _daily_context(ds, winds, times, currents=None, sources=None):
    predictor, _ = _persistence(ds)
    return ForecastContext.build(ds, predictor, L, H, SEASON, TRAIN, bank_size=50, seed=0, currents=currents,
                                 winds=winds, wind_times=times, forcing_sources=sources)


def test_daily_winds_are_selected_by_calendar_date(history):
    shape = history["land_mask"].shape
    winds, times = _daily_winds(shape, ISSUE - timedelta(days=4), 15)   # issue is mid-way through the forcing
    world = _daily_context(history, winds, times).scenarios(ISSUE, 7, 6, np.random.default_rng(0))
    assert world.wind_x.shape == (7, *shape)
    for h in range(7):
        day = (ISSUE + timedelta(days=h)).toordinal()
        assert (world.wind_x[h] == day).all() and (world.wind_y[h] == -day).all()
    assert world.meta["forcing"]["wind_dates"] == [str(ISSUE), str(ISSUE + timedelta(days=6))]


@pytest.mark.parametrize("start_offset, n, skip, missing", [
    (1, 20, (), [str(ISSUE)]),                                                      # starts too late
    (-3, 8, (), [str(ISSUE + timedelta(days=5)), str(ISSUE + timedelta(days=6))]),  # ends too early
    (-2, 15, (5,), [str(ISSUE + timedelta(days=3))]),                               # gap inside the window
])
def test_missing_daily_wind_dates_raise_a_clear_error(history, start_offset, n, skip, missing):
    winds, times = _daily_winds(history["land_mask"].shape, ISSUE + timedelta(days=start_offset), n, skip)
    ctx = _daily_context(history, winds, times)
    with pytest.raises(ValueError, match="missing") as err:
        ctx.scenarios(ISSUE, 7, 4, np.random.default_rng(0))
    msg = str(err.value)
    assert f"covers {times[0]} .. {times[-1]}" in msg
    assert f"missing {len(missing)} date(s): {', '.join(missing)}" in msg


def test_wind_times_must_match_layers_and_be_one_per_day(history):
    shape = history["land_mask"].shape
    winds, times = _daily_winds(shape, ISSUE, 7)
    with pytest.raises(ValueError, match="6 timestamps but winds have 7 layers"):
        _daily_context(history, winds, times[:-1])
    with pytest.raises(ValueError, match="sorted"):
        _daily_context(history, winds, times[::-1])
    dup = times.copy()
    dup[3] = dup[2]
    with pytest.raises(ValueError, match="one UTC calendar day"):
        _daily_context(history, winds, dup)
    with pytest.raises(ValueError, match="without winds"):
        _daily_context(history, None, times)


def test_two_d_winds_behave_exactly_as_before(history):
    shape = history["land_mask"].shape
    rng = np.random.default_rng(3)
    wx, wy = rng.normal(size=shape), rng.normal(size=shape)
    ctx = _daily_context(history, (wx, wy), None)
    a = ctx.scenarios(ISSUE, 7, 6, np.random.default_rng(4))
    b = _context(history).scenarios(ISSUE, 7, 6, np.random.default_rng(4))
    assert ctx.wind_times is None
    assert np.array_equal(a.wind_x, np.broadcast_to(wx, (7, *shape)))
    assert np.array_equal(a.wind_y, np.broadcast_to(wy, (7, *shape)))
    assert np.array_equal(a.conc, b.conc, equal_nan=True)            # sea ice unaffected by the wind source
    assert np.array_equal(a.current_x, b.current_x)                  # still the schematic currents
    assert a.description == b.description and a.execution_mode == b.execution_mode


def test_era5_daily_winds_with_schematic_currents_are_reported_as_mixed(history):
    winds, times = _daily_winds(history["land_mask"].shape, ISSUE, 7)
    ctx = _daily_context(history, winds, times, sources={"winds": "ERA5 daily", "currents": "CMEMS time-mean"})
    fm = ctx.scenarios(ISSUE, 7, 4, np.random.default_rng(0)).meta["forcing"]
    assert fm["winds"] == "ERA5 daily" and fm["currents"] == "schematic"     # no CMEMS given -> schematic
    assert fm["status"] == "mixed" and fm["label"] == "ERA5 daily winds + schematic currents"
    cur = (np.zeros(history["land_mask"].shape),) * 2
    fm = _daily_context(history, winds, times, currents=cur,
                        sources={"winds": "ERA5 daily", "currents": "CMEMS time-mean"}
                        ).scenarios(ISSUE, 7, 4, np.random.default_rng(0)).meta["forcing"]
    assert fm["status"] == "real" and fm["label"] == "ERA5 daily winds + CMEMS time-mean currents"
    assert _context(history).scenarios(ISSUE, 7, 4, np.random.default_rng(0)).meta["forcing"]["status"] == "schematic"


# --------------------------------------------------------- daily current forcing
def _currents_ctx(ds, currents, current_times, winds=None, wind_times=None, sources=None, provenance=None):
    predictor, _ = _persistence(ds)
    return ForecastContext.build(ds, predictor, L, H, SEASON, TRAIN, bank_size=50, seed=0, currents=currents,
                                 winds=winds, wind_times=wind_times, current_times=current_times,
                                 forcing_sources=sources, forcing_provenance=provenance)


def test_daily_currents_are_selected_by_calendar_date(history):
    shape = history["land_mask"].shape
    cur, times = _daily_winds(shape, ISSUE - timedelta(days=3), 14)   # value encodes the date
    world = _currents_ctx(history, cur, times).scenarios(ISSUE, 7, 6, np.random.default_rng(0))
    assert world.current_x.shape == (7, *shape)
    for h in range(7):
        day = (ISSUE + timedelta(days=h)).toordinal()
        assert (world.current_x[h] == day).all() and (world.current_y[h] == -day).all()
    assert world.meta["forcing"]["current_dates"] == [str(ISSUE), str(ISSUE + timedelta(days=6))]
    assert "wind_dates" not in world.meta["forcing"]                  # winds stayed schematic 2-D


@pytest.mark.parametrize("start_offset, n, skip, missing", [
    (1, 20, (), [str(ISSUE)]),
    (-3, 8, (), [str(ISSUE + timedelta(days=5)), str(ISSUE + timedelta(days=6))]),
    (-2, 15, (5,), [str(ISSUE + timedelta(days=3))]),
])
def test_missing_daily_current_dates_raise_and_never_fall_back(history, start_offset, n, skip, missing):
    cur, times = _daily_winds(history["land_mask"].shape, ISSUE + timedelta(days=start_offset), n, skip)
    ctx = _currents_ctx(history, cur, times)
    with pytest.raises(ValueError, match="daily current forcing") as err:
        ctx.scenarios(ISSUE, 7, 4, np.random.default_rng(0))
    assert f"missing {len(missing)} date(s): {', '.join(missing)}" in str(err.value)


def test_current_times_are_validated(history):
    cur, times = _daily_winds(history["land_mask"].shape, ISSUE, 7)
    with pytest.raises(ValueError, match="6 timestamps but currents have 7 layers"):
        _currents_ctx(history, cur, times[:-1])
    with pytest.raises(ValueError, match="sorted"):
        _currents_ctx(history, cur, times[::-1])
    with pytest.raises(ValueError, match="current_times given without currents"):
        _currents_ctx(history, None, times)


def test_daily_winds_and_currents_together_are_real_with_provenance(history):
    shape = history["land_mask"].shape
    winds, wt = _daily_winds(shape, ISSUE - timedelta(days=1), 10)
    cur, ct = _daily_winds(shape, ISSUE - timedelta(days=2), 12)
    prov = {"winds": {"product": "ERA5", "forcing_sha256": "a"},
            "currents": {"product": "CMEMS", "forcing_sha256": "b"}}
    ctx = _currents_ctx(history, (cur[0] / 1e6, cur[1] / 1e6), ct, winds, wt,
                        sources={"winds": "ERA5 daily", "currents": "CMEMS daily"}, provenance=prov)
    world = ctx.scenarios(ISSUE, 7, 4, np.random.default_rng(0))
    fm = world.meta["forcing"]
    assert fm["status"] == "real" and fm["label"] == "ERA5 daily winds + CMEMS daily currents"
    assert fm["wind_dates"] == fm["current_dates"] == [str(ISSUE), str(ISSUE + timedelta(days=6))]
    assert fm["sources"] == prov
    for h in range(7):
        day = (ISSUE + timedelta(days=h)).toordinal()
        assert (world.wind_x[h] == day).all() and np.allclose(world.current_x[h], day / 1e6)


def test_schematic_fields_carry_no_real_provenance(history):
    winds, wt = _daily_winds(history["land_mask"].shape, ISSUE, 7)
    ctx = _currents_ctx(history, None, None, winds, wt, sources={"winds": "ERA5 daily"},
                        provenance={"winds": {"product": "ERA5"}, "currents": {"product": "CMEMS"}})
    fm = ctx.scenarios(ISSUE, 7, 4, np.random.default_rng(0)).meta["forcing"]
    assert fm["currents"] == "schematic" and fm["status"] == "mixed"
    assert fm["sources"] == {"winds": {"product": "ERA5"}}             # nothing claims real currents


def test_two_d_currents_behave_exactly_as_before(history):
    shape = history["land_mask"].shape
    rng = np.random.default_rng(5)
    cx, cy = rng.normal(size=shape), rng.normal(size=shape)
    ctx = _currents_ctx(history, (cx, cy), None)
    a = ctx.scenarios(ISSUE, 7, 6, np.random.default_rng(4))
    b = _context(history).scenarios(ISSUE, 7, 6, np.random.default_rng(4))
    assert ctx.current_times is None and ctx.current_layers(ISSUE, 7) is None
    assert np.array_equal(a.current_x, np.broadcast_to(cx, (7, *shape)))
    assert np.array_equal(a.conc, b.conc, equal_nan=True)
    assert np.array_equal(a.wind_x, b.wind_x)                          # still the schematic winds
    assert "current_dates" not in a.meta["forcing"] and "sources" not in a.meta["forcing"]


def test_replay_truth_world_uses_daily_currents_by_date(history):
    from antarctic_routing.replay import _truth_world

    cur, times = _daily_winds(history["land_mask"].shape, ISSUE - timedelta(days=1), 10)
    ctx = _currents_ctx(history, cur, times)
    world = _truth_world(ctx, ISSUE, 3)
    for h in range(world.current_x.shape[0]):
        assert (world.current_x[h] == (ISSUE + timedelta(days=h)).toordinal()).all()
    short = _currents_ctx(history, *_daily_winds(history["land_mask"].shape, ISSUE, 1))
    with pytest.raises(ValueError, match="missing"):
        _truth_world(short, ISSUE, 3)
