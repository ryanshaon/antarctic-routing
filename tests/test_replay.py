"""Historical replay: day-by-day decisions using only information available each day."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from antarctic_routing.config import load_config
from antarctic_routing.forecasting.scenarios import ForecastContext, grid_of
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.replay import observed_berg_exposure, run_replay
from antarctic_routing.routing.fuel import VesselModel
from antarctic_routing.routing.replan import ReplanPolicy
from antarctic_routing.synthetic import synthetic_history

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
L, H = 5, 3


def persistence(inputs, idx):
    return np.repeat(inputs[:, L - 1: L], H, axis=1)


@pytest.fixture(scope="module")
def context():
    ds = synthetic_history(CFG, PolarGrid.from_domain(CFG.domain, resolution_km=50), range(2010, 2016), seed=7)
    return ForecastContext.build(ds, persistence, L, H, CFG.project.season_months, [2010, 2011, 2012],
                                 bank_size=60, seed=0)


@pytest.fixture(scope="module")
def replay(tmp_path_factory, context):
    ctx, ds = context, context.ds
    grid = grid_of(ds)
    o = grid.cell_of(CFG.route.origin.lat, CFG.route.origin.lon)
    d = grid.cell_of(CFG.route.destination.lat, CFG.route.destination.lon)
    audit = tmp_path_factory.mktemp("replay") / "audit.jsonl"
    result = run_replay(ctx, VesselModel.from_config(CFG), o, d, start=date(2015, 11, 20), window_days=5,
                        max_wait_days=40, n_members=80, policy=ReplanPolicy(risk_budget=0.05),
                        risk_weights=[0, 100], voyage_days=3, rng=np.random.default_rng(0), audit_path=audit,
                        scenario_routes=0)
    return result, audit


def test_replay_waits_departs_and_arrives(replay):
    result, _ = replay
    phases = [d["phase"] for d in result["days"]]
    assert phases[0] == "port" and "at_sea" in phases
    assert result["arrived"] is True
    assert result["departure"] >= "2015-11-20"
    assert result["days_waited"] == phases.count("port") - 1


def test_every_decision_is_audited(replay):
    result, audit = replay
    lines = audit.read_text().splitlines()
    assert len(lines) == len(result["days"])
    assert all('"phase"' in line for line in lines)


def test_replay_scores_against_truth_and_naive_baseline(replay):
    result, _ = replay
    truth = result["truth"]
    assert truth["planner"]["observed_breach_cells"] >= 0
    assert {"observed_breach_cells", "hours", "fuel_index", "departure"} <= truth["naive"].keys()
    assert truth["naive"]["departure"] == "2015-11-20"
    assert result["execution_mode"] == "controlled_synthetic"


def test_port_decisions_only_use_issue_day_forecasts(replay):
    result, _ = replay
    for day in result["days"]:
        if day["phase"] == "port" and day["recommended_departure"]:
            assert day["recommended_departure"] >= day["day"]


def test_no_iceberg_keys_without_icebergs(replay):
    result, _ = replay
    assert "iceberg_hazard" not in result
    assert not any(k.startswith("berg_") for k in result["truth"]["planner"])


def test_replay_with_icebergs_adds_the_hazard_to_every_forecast_and_scores_reported_bergs(context):
    grid = grid_of(context.ds)
    o = grid.cell_of(CFG.route.origin.lat, CFG.route.origin.lon)
    d = grid.cell_of(CFG.route.destination.lat, CFG.route.destination.lon)
    issued = []

    def hazard(world, day):
        assert world.start.date() == day                   # only the forecast issued that day
        issued.append(day)
        return world

    berg = grid.cell_latlon(*d)
    result = run_replay(context, VesselModel.from_config(CFG), o, d, start=date(2015, 11, 20), window_days=5,
                        max_wait_days=40, n_members=80, policy=ReplanPolicy(risk_budget=0.05), risk_weights=[0, 100],
                        voyage_days=3, rng=np.random.default_rng(0), scenario_routes=0, hazard=hazard,
                        truth_bergs=lambda day: [("A99", *berg)], berg_radius_m=50_000.0)
    planned = [d_["day"] for d_ in result["days"] if d_["phase"] in ("port", "at_sea")]
    assert [x.isoformat() for x in issued] == planned
    assert result["iceberg_hazard"] is True
    truth = result["truth"]["planner"]
    assert truth["berg_nearest"] == "A99" and truth["berg_min_distance_km"] < 1.0
    assert truth["berg_footprint_cells"] >= 1 and truth["berg_radius_km"] == 50.0


def test_observed_berg_exposure_uses_the_planner_footprint():
    from _worlds import _grid

    grid = _grid()
    cells = [(5, c) for c in range(21)]
    hours = np.arange(21) * 2.0
    lat, lon = grid.cell_latlon(5, 10)
    near = observed_berg_exposure(cells, hours, date(2026, 12, 1), grid, lambda day: [("B1", lat, lon)], 10_000.0)
    assert near["berg_footprint_cells"] == 3 and near["berg_min_distance_km"] < 0.5 and near["berg_nearest"] == "B1"
    far = observed_berg_exposure(cells, hours, date(2026, 12, 1), grid, lambda day: [("B2", -70.0, 100.0)], 10_000.0)
    assert far["berg_footprint_cells"] == 0 and far["berg_min_distance_km"] > 1000
    seen = []
    observed_berg_exposure(cells, [0.0, 30.0] + [np.inf] * 19, date(2026, 12, 1), grid,
                           lambda day: seen.append(day) or [], 10_000.0)
    assert seen == [date(2026, 12, 1), date(2026, 12, 2)]    # the day each cell is reached; unreached skipped
