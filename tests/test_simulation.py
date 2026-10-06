"""Product voyage simulation (S4): POST /real/simulate frames, replans, summary and orchestration.

The archive here is a synthetic stand-in (the real archive and PyTorch are kept outside Git): the observed sea
ice is a synthetic history and the daily forecasts come from the synthetic generator. What is tested is the
orchestration around the existing engine: one frame per day, the replay's sailing step, the existing replan
rules, the summary, determinism and the absence of any fallback. tests/test_historical_real.py runs the real one.
"""

import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from antarctic_routing.api import historical as hist_api
from antarctic_routing.api.main import create_app
from antarctic_routing.common.provenance import sha256_file
from antarctic_routing.config import load_config
from antarctic_routing.forecasting.scenarios import ForecastContext, grid_of
from antarctic_routing.historical import (
    DailyForcing,
    HistoricalArchive,
    HistoricalUnavailable,
    IcebergSnapshot,
    UsnicArchive,
)
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.product import BANNERS
from antarctic_routing.replay import _sail, sail_day
from antarctic_routing.routing.optimizer import Route
from antarctic_routing.synthetic import generate_synthetic, synthetic_history

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "config.yaml"
CFG = load_config(CONFIG)
HEADER = "Iceberg,Length (NM),Width (NM),Latitude,Longitude,Remarks,Last Update\n"
L = 5


def persistence(inputs, idx):
    return np.repeat(inputs[:, L - 1: L], 21, axis=1)


class StandInPlanner:
    """The U-Net context's interface: ``ctx`` over the archive's observed ice, ``world`` for forecasts.

    ``block`` (optional) marks cells as ice-covered in every forecast issued after ``block_after``, so the
    existing replan rules meet a route ahead that exceeds the budget (a stand-in event, not real data).
    """

    def __init__(self, archive):
        self.archive, self.lead_days, self.calls = archive, 21, []
        self.ctx = ForecastContext.build(archive.ds, persistence, L, 21, CFG.project.season_months, [2021],
                                         bank_size=20, seed=0)
        self.block, self.block_after, self.clear = [], None, False

    def world(self, issue, n_days):
        self.calls.append((issue, n_days))
        # 120 members when clear: the Wilson upper bound of 0 breaches in 40 is above the 5% budget
        world = generate_synthetic(CFG, grid_of(self.archive.ds), issue, n_days, 120 if self.clear else 40, seed=2)
        if self.clear:      # open water everywhere: every route meets the budget
            world = replace(world, conc=np.where(np.isnan(world.conc), np.nan, 0.0).astype(world.conc.dtype))
        if self.block and self.block_after is not None and issue > self.block_after:
            conc = world.conc.copy()
            for r, c in self.block:
                conc[:, :, r, c] = 1.0
            world = replace(world, conc=conc)
        snap = IcebergSnapshot(date(2021, 11, 25), "AntarcticIcebergs_20211125.csv", "0" * 64, 6, [], [])
        return world, snap

    def drift_kwargs(self):
        return {}

    def observed_bergs(self, day):
        return [("A23A", -60.0, -60.0)]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sim")
    grid = PolarGrid.from_domain(CFG.domain, 25)
    ds = synthetic_history(CFG, grid, range(2021, 2023), seed=3)
    t = ds["time"].values.astype("datetime64[D]")
    zeros = np.zeros((t.size, *ds["land_mask"].shape))
    p = tmp / "AntarcticIcebergs_20211125.csv"
    p.write_text(HEADER + "A23A,20,5,-60.0,-60.0,,11/25/2021\n")
    archive = HistoricalArchive(json.loads((REPO / "config" / "real_historical.json").read_text()), tmp, ds,
                                DailyForcing("winds", "ERA5", t, zeros, zeros, ()),
                                DailyForcing("currents", "CMEMS", t, zeros, zeros, ()),
                                UsnicArchive([{"path": p.name, "sha256": sha256_file(p), "date": "2021-11-25"}], tmp),
                                CFG.project.season_months)
    planner = StandInPlanner(archive)
    mp = pytest.MonkeyPatch()
    mp.setenv("ANTROUTE_DATA_ROOT", str(tmp))
    mp.setattr(hist_api.HistoricalArchive, "load", classmethod(lambda cls, *a, **k: archive))
    mp.setattr(hist_api, "HistoricalPlanner", lambda a: planner)
    with TestClient(create_app(CONFIG)) as c:
        c.planner = planner
        yield c
    mp.undo()


BODY = {"origin": {"preset": "drake_passage"}, "destination": {"preset": "bransfield_strait"},
        "issue": "2021-12-01", "window_days": 3}


def simulate(c, body=BODY):
    r = c.post("/real/simulate?wait=true", json=body)
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["status"] == "done", job.get("error")
    return job["result"]


@pytest.fixture(scope="module")
def plain(client):
    client.planner.block = []
    plan = client.post("/real/plan", json=BODY).json()
    return plan, simulate(client, {**BODY, "departure": plan["route"]["departure_date"]})


# ------------------------------------------------------------------ frames
def test_simulation_starts_from_the_planned_route_and_arrives(plain):
    plan, s = plain
    assert s["status"] == "simulated"
    f0, last = s["frames"][0], s["frames"][-1]
    assert s["metadata"]["departure_date"] == plan["route"]["departure_date"]
    assert f0["phase"] == "departure" and f0["route"]["cells"] == plan["route"]["cells"]
    assert [f0["position"]["row"], f0["position"]["col"]] == plan["route"]["cells"][0]
    assert f0["forecast"]["risk"]["combined"]["p_breach_upper"] == plan["risk"]["combined"]["p_breach_upper"]
    assert f0["forecast"]["eta_utc"] == plan["route"]["eta_utc"]
    assert last["phase"] == "arrived" and s["summary"]["arrived"] is True
    assert [last["position"]["row"], last["position"]["col"]] == plan["route"]["cells"][-1]
    assert last["progress"]["fraction"] == 1.0 and last["progress"]["remaining_km"] == 0.0


def test_frames_are_ordered_daily_and_the_vessel_moves_forward(plain):
    _, s = plain
    frames = s["frames"]
    assert [f["index"] for f in frames] == list(range(len(frames)))
    dep = date.fromisoformat(s["metadata"]["departure_date"])
    for f in frames[:-1]:
        assert f["date"] == (dep + timedelta(days=f["day_of_voyage"])).isoformat()
        assert f["timestamp_utc"] == f"{f['date']}T00:00Z"
    sailed = [f["progress"]["sailed_km"] for f in frames]
    frac = [f["progress"]["fraction"] for f in frames]
    assert sailed == sorted(sailed) and frac == sorted(frac) and sailed[-1] > 0
    for a, b in zip(frames, frames[1:], strict=False):
        assert b["track"][: len(a["track"])] == a["track"]                  # the track only grows
        assert b["track"][-1] == b["position"]["xy_km"]
    for f in frames[1:-1]:
        assert f["route"]["cells"][0] == [f["position"]["row"], f["position"]["col"]]   # the route ahead
        assert f["decision"]["action"] in ("keep", "switch", "no_feasible_route")
        assert f["forecast"]["issued"] == f["date"]


def test_frames_carry_observations_not_invented_values(plain, client):
    _, s = plain
    a = client.planner.archive
    for f in s["frames"]:
        seg = f["observed"]["sea_ice_on_segment"]
        if f["index"] == 0:
            assert seg is None and f["segment"]["cells"] == []
            continue
        t = a.days.index(date.fromisoformat(seg["date"]))
        vals = [float(a.ds["ice_concentration"].values[t, r, c]) for r, c in f["segment"]["cells"]]
        assert seg["max_concentration"] == pytest.approx(max(vals))
        assert f["observed"]["icebergs"]["list_date"] == "2021-11-25"
        assert len(f["map"]["concentration_pct"]) == s["grid"]["nx"] * s["grid"]["ny"]


def test_no_replan_case_is_explicit(plain):
    _, s = plain
    u = s["summary"]
    assert u["replans"] == 0 and u["replan_note"] == "No replan was required during this voyage."
    assert not any(f["replanned"] for f in s["frames"])
    types = [e["type"] for e in s["events"]]
    assert types[0] == "departed" and types[-1] == "arrived" and "replan" not in types
    assert set(types[1:-1]) <= {"no_feasible_route"}          # alerts are reported, never turned into replans


def test_summary_uses_the_replay_scoring(plain):
    _, s = plain
    u = s["summary"]
    assert set(u["sailed"]) >= {"hours_through_observed_ice", "distance_km", "fuel_index", "observed_breach_cells",
                                "berg_footprint_cells", "berg_min_distance_km"}
    assert u["sailed"]["distance_km"] == pytest.approx(s["frames"][-1]["progress"]["sailed_km"], abs=0.1)
    assert u["simulated_hours"] > 0 and u["held_hours"] >= 0
    assert u["arrival_utc"] == s["frames"][-1]["timestamp_utc"]
    assert u["final_forecast_risk"]["combined"]["p_breach_upper"] is not None


def test_labels_and_provenance_are_real_historical(plain):
    _, s = plain
    m = s["metadata"]
    assert (m["mode"], m["execution_mode"], m["hindsight_forcing"], m["banners"]) == ("historical", "real", True,
                                                                                     BANNERS)
    assert "reanalysis" in m["hindsight_disclosure"] and m["provenance"]["issue"] == "2021-12-01"
    assert "observed sea ice" in m["simulation_note"]


def test_simulation_is_deterministic_and_reuses_the_plan(client, plain):
    _, s = plain
    calls = list(client.planner.calls)
    again = simulate(client, {**BODY, "departure": s["metadata"]["departure_date"]})
    assert json.dumps(again, sort_keys=True) == json.dumps(s, sort_keys=True)
    new = client.planner.calls[len(calls):]
    assert all(n == 1 + s["metadata"]["horizon_days"] for _, n in new), "the window was not rebuilt"
    assert len(new) == len(s["frames"]) - 2                               # one forecast per day at sea


# ------------------------------------------------------------------ replanning
def test_a_route_ahead_that_exceeds_the_budget_is_replanned_by_the_existing_rules(client):
    planner = client.planner
    body = {**BODY, "issue": "2021-12-02"}
    planner.clear = True
    plan = client.post("/real/plan", json=body).json()
    assert plan["status"] == "recommended"
    cells = [tuple(c) for c in plan["route"]["cells"]]
    planner.block = cells[-6:-3]               # ice on the route ahead (past the day-1 position) from day 1 on
    planner.block_after = date.fromisoformat(plan["route"]["departure_date"])
    try:
        s = simulate(client, {**body, "departure": plan["route"]["departure_date"]})
    finally:
        planner.block, planner.block_after, planner.clear = [], None, False
    rep = [e for e in s["events"] if e["type"] == "replan"]
    assert rep, [f["decision"] for f in s["frames"]]
    e = rep[0]
    f = s["frames"][e["frame"]]
    assert f["replanned"] and f["route_version"] == 1 and f["decision"]["action"] == "switch"
    assert "previous_route_exceeds_budget" in e["triggers"] and e["reasons"]
    assert e["old_route"]["p_breach_upper"] > 0.05 >= e["new_route"]["p_breach_upper"]
    assert f["route"]["cells"] == e["new_route"]["cells"] and e["new_route"]["cells"] != e["old_route"]["cells"]
    assert not set(map(tuple, e["new_route"]["cells"])) & set(planner.block)
    for k in ("expected_hours", "expected_fuel", "distance_km", "p_breach_upper"):
        assert e["change"][k] == pytest.approx(e["new_route"][k] - e["old_route"][k], abs=1e-3)
    later = s["frames"][e["frame"] + 1:]
    assert all(x["route_version"] >= 1 for x in later)
    assert s["summary"]["replans"] == len(rep) and s["summary"]["replan_note"] is None
    if s["summary"]["arrived"]:
        assert [s["frames"][-1]["position"]["row"], s["frames"][-1]["position"]["col"]] == list(cells[-1])


# ------------------------------------------------------------------ refusals, no fallback
def test_a_departure_that_differs_from_the_plan_is_refused(client, plain):
    plan, _ = plain
    other = (date.fromisoformat(plan["route"]["departure_date"]) + timedelta(days=1)).isoformat()
    s = simulate(client, {**BODY, "departure": other})
    assert s["status"] == "not_simulated" and "departs" in s["reason"] and "frames" not in s


def test_invalid_inputs_are_refused_before_any_work(client):
    calls = len(client.planner.calls)
    bad = client.post("/real/simulate?wait=true", json={**BODY, "destination": {"preset": "mcmurdo"}})
    assert bad.status_code == 422 and bad.json()["detail"]["status"] == "invalid_location"
    off = client.post("/real/simulate?wait=true", json={**BODY, "issue": "2022-07-01"})
    assert off.status_code == 422 and off.json()["detail"]["status"] == "out_of_coverage"
    assert len(client.planner.calls) == calls


def test_simulation_never_falls_back_to_synthetic(monkeypatch):
    monkeypatch.delenv("ANTROUTE_DATA_ROOT", raising=False)
    with TestClient(create_app(CONFIG)) as c:
        r = c.post("/real/simulate?wait=true", json=BODY)
        assert r.status_code == 503 and r.json()["detail"]["status"] == "unavailable"


def test_a_missing_model_fails_the_job_instead_of_substituting(client, monkeypatch, tmp_path):
    def broken(a):
        raise HistoricalUnavailable("blocked", "PyTorch is not installed")

    archive = client.planner.archive
    monkeypatch.setenv("ANTROUTE_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(hist_api.HistoricalArchive, "load", classmethod(lambda cls, *a, **k: archive))
    monkeypatch.setattr(hist_api, "HistoricalPlanner", broken)
    with TestClient(create_app(CONFIG)) as c:
        r = c.post("/real/simulate?wait=true", json=BODY)
        assert r.status_code == 200 and r.json()["status"] == "failed" and "PyTorch" in r.json()["error"]
        assert r.json()["result"] is None


def test_sail_day_is_the_replay_sailing_step(client):
    ctx = client.planner.ctx
    grid = grid_of(ctx.ds)
    o = grid.cell_of(CFG.route.origin.lat, CFG.route.origin.lon)
    cells = [o] + [(o[0], o[1] + k) for k in range(1, 12)]
    route = Route("test", cells, [0.0] * len(cells), 0.0, 0.0)
    day = date(2021, 12, 1)
    a, fin_a, hours = sail_day(route, ctx, day, _vessel())
    b, fin_b = _sail(route, ctx, day, _vessel())
    assert (a, fin_a) == (b, fin_b) and len(hours) == len(a) and all(0 < h <= 24 for h in hours)
    assert hours == sorted(hours)


def _vessel():
    from antarctic_routing.routing.fuel import VesselModel

    return VesselModel.from_config(CFG)
