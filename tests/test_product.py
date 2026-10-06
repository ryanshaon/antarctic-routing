"""One-call product plan: risk split, daily timeline, layers and the POST /real/plan orchestration."""

import json
import math
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from antarctic_routing.api import historical as hist_api
from antarctic_routing.api.main import create_app
from antarctic_routing.common.provenance import sha256_file
from antarctic_routing.config import load_config, wilson_upper_bound
from antarctic_routing.forecasting.scenarios import grid_of
from antarctic_routing.historical import (
    DailyForcing,
    HistoricalArchive,
    HistoricalUnavailable,
    IcebergSnapshot,
    UsnicArchive,
)
from antarctic_routing.iceberg.drift import add_iceberg_hazard
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.product import (
    BANNERS,
    daily_timeline,
    environment_layers,
    json_safe,
    risk_breakdown,
)
from antarctic_routing.routing.candidates import plan_candidates
from antarctic_routing.routing.fuel import VesselModel
from antarctic_routing.synthetic import generate_synthetic, synthetic_history

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "config.yaml"
CFG = load_config(CONFIG)
VESSEL = VesselModel.from_config(CFG)
R = CFG.routing


@pytest.fixture(scope="module")
def setup():
    grid = PolarGrid.from_domain(CFG.domain, 25)
    world = generate_synthetic(CFG, grid, date(2027, 1, 10), 7, 60, seed=4)
    o = grid.cell_of(CFG.route.origin.lat, CFG.route.origin.lon)
    d = grid.cell_of(CFG.route.destination.lat, CFG.route.destination.lon)
    plan = plan_candidates(world, VESSEL, o, d, R.risk_budget, R.risk_weights, R.risk_estimator, R.connectivity,
                           0, R.confidence, 1)
    cand = plan.recommended or plan.candidates[0]
    mid = cand.route.cells[len(cand.route.cells) // 2]
    lat, lon = grid.cell_latlon(*mid)
    bergy = add_iceberg_hazard(world, [("T1", lat, lon)], rng=np.random.default_rng(0), radius_m=60_000.0)
    return grid, world, bergy, cand


def _risk(world, cand, ev=None):
    return risk_breakdown(cand.route, world, VESSEL, 0.0, R.confidence, ev or cand.evaluation, R.risk_budget)


# ------------------------------------------------------------------ risk split
def test_combined_risk_is_the_engine_evaluation_and_dominates_its_parts(setup):
    _, _, bergy, cand = setup
    from antarctic_routing.routing.evaluate import evaluate_route

    ev = evaluate_route(cand.route, bergy, VESSEL, 0.0, R.confidence)
    risk = _risk(bergy, cand, ev)
    k = bergy.n_scenarios
    comb, ice, berg, inc = (risk[n] for n in ("combined", "sea_ice", "iceberg", "iceberg_only_increment"))
    assert risk["authoritative"] == "combined" and comb["breaches"] == ev.breaches
    assert comb["p_breach_upper"] == ev.p_breach_upper == wilson_upper_bound(ev.breaches, k, R.confidence)
    assert berg["breaches"] > 0, "the test berg sits on the route"
    assert comb["breaches"] >= max(ice["breaches"], berg["breaches"])
    assert comb["breaches"] == ice["breaches"] + inc["breaches"]
    assert inc["breaches"] <= berg["breaches"]
    for part in (comb, ice, berg, inc):
        assert part["n_scenarios"] == k and part["p_breach"] == part["breaches"] / k
        assert part["p_breach_upper"] == wilson_upper_bound(part["breaches"], k, R.confidence)
    assert comb["within_budget"] == (comb["p_breach_upper"] <= R.risk_budget)
    assert len(ice["segment_breach_prob"]) == len(berg["segment_breach_prob"]) == len(cand.route.cells)
    assert set(risk["definitions"]) >= {"combined", "sea_ice", "iceberg", "iceberg_only_increment"}


def test_without_icebergs_the_iceberg_risk_is_zero_and_sea_ice_equals_combined(setup):
    _, world, _, cand = setup
    risk = _risk(world, cand)
    assert risk["iceberg"]["breaches"] == 0 and "no in-grid USNIC icebergs" in risk["iceberg"]["note"]
    assert risk["sea_ice"]["breaches"] == risk["combined"]["breaches"]
    assert risk["iceberg_only_increment"]["breaches"] == 0


def test_risk_split_refuses_an_evaluation_it_cannot_reproduce(setup):
    _, _, bergy, cand = setup
    from antarctic_routing.routing.evaluate import evaluate_route

    other = evaluate_route(cand.route, bergy, VESSEL, 24.0, R.confidence)   # a different departure time
    wrong = replace(other, breaches=other.breaches + 1)
    with pytest.raises(RuntimeError, match="did not reproduce"):
        _risk(bergy, cand, wrong)


# ------------------------------------------------------------------ daily timeline and layers
def test_daily_timeline_covers_the_route_once_in_order(setup):
    grid, _, bergy, cand = setup
    from antarctic_routing.routing.evaluate import evaluate_route

    ev = evaluate_route(cand.route, bergy, VESSEL, 0.0, R.confidence)
    risk = _risk(bergy, cand, ev)
    days = daily_timeline(cand.route, bergy, VESSEL, date(2027, 1, 10), 0, ev.expected_hours, risk)
    assert [d["day_of_voyage"] for d in days] == list(range(1, len(days) + 1))
    assert days[0]["date"] == "2027-01-10" and len(days) == math.floor(ev.expected_hours / 24) + 1
    cells = [tuple(c) for d in days for c in d["cells"]]
    assert cells == [tuple(c) for c in cand.route.cells]
    assert sum(d["distance_km"] for d in days) == pytest.approx(ev.distance_km, abs=0.1 * len(days))
    assert days[-1]["position_end_of_day"] == [round(v, 5) for v in grid.cell_latlon(*cand.route.cells[-1])]
    seg = risk["combined"]["segment_breach_prob"]
    for d in days:
        i0, i1 = d["route_cell_index"]
        assert d["risk"]["combined"]["max_cell_breach_prob"] == round(max(seg[i0:i1 + 1]), 4)
        assert d["scenario_layer"] == min(d["day_of_voyage"] - 1, bergy.n_times - 1)
        assert set(d["sea_ice"]) == {"mean_concentration", "max_concentration", "max_p_ge_vessel_limit"}
        assert 0.0 <= d["iceberg"]["max_p_presence"] <= 1.0


def test_departure_later_in_the_window_shifts_the_timeline(setup):
    _, _, bergy, cand = setup
    from antarctic_routing.routing.evaluate import evaluate_route

    ev = evaluate_route(cand.route, bergy, VESSEL, 48.0, R.confidence)
    risk = risk_breakdown(cand.route, bergy, VESSEL, 48.0, R.confidence, ev, R.risk_budget)
    days = daily_timeline(cand.route, bergy, VESSEL, date(2027, 1, 10), 2, ev.expected_hours, risk)
    assert days[0]["date"] == "2027-01-12" and days[0]["scenario_layer"] == 2
    assert days[0]["nominal_hours_since_departure"][0] == 0.0


def test_environment_layers_are_percent_maps_with_land_as_none(setup):
    grid, _, bergy, _ = setup
    layers = environment_layers(bergy, VESSEL.tau, [0, 1, 1, 99])
    assert [x["scenario_layer"] for x in layers] == [0, 1, bergy.n_times - 1]
    land = bergy.land.ravel()
    for x in layers:
        assert len(x["p_ice_ge_limit_pct"]) == len(x["p_berg_pct"]) == grid.x.size * grid.y.size
        assert all(v is None for v, is_land in zip(x["p_ice_ge_limit_pct"], land, strict=True) if is_land)
        assert all(0 <= v <= 100 for v in x["p_ice_ge_limit_pct"] if v is not None)


def test_json_safe_drops_non_finite_numbers():
    assert json_safe({"a": [1.0, math.inf, np.float64("nan")], "b": np.int64(3), "c": (math.inf,)}) == \
        {"a": [1.0, None, None], "b": 3, "c": [None]}
    json.dumps(json_safe({"x": math.inf}), allow_nan=False)


# ------------------------------------------------------------------ POST /real/plan orchestration
HEADER = "Iceberg,Length (NM),Width (NM),Latitude,Longitude,Remarks,Last Update\n"


class StandInPlanner:
    """Stands in for the U-Net context (PyTorch and the real archive are kept outside Git)."""

    def __init__(self, archive):
        self.archive, self.lead_days, self.calls = archive, 21, 0

    def world(self, issue, n_days):
        self.calls += 1
        world = generate_synthetic(CFG, grid_of(self.archive.ds), issue, n_days, 40, seed=2)
        snap = IcebergSnapshot(date(2021, 11, 25), "AntarcticIcebergs_20211125.csv", "0" * 64, 6, [], [])
        return world, snap

    def drift_kwargs(self):
        return {}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("plan")
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


def test_plan_returns_one_coherent_result(client):
    r = client.post("/real/plan", json=BODY)
    assert r.status_code == 200, r.text
    j = r.json()
    assert set(j) >= {"status", "metadata", "locations", "horizon", "route", "risk", "departure", "daily", "layers",
                      "icebergs", "iceberg_tracks", "alternatives"}
    m = j["metadata"]
    assert (m["mode"], m["hindsight_forcing"], m["banners"]) == ("historical", True, BANNERS)
    assert "reanalysis" in m["hindsight_disclosure"] and "not a certified navigation route" in m["disclaimer"]
    assert m["issue_date"] == "2021-12-01" and m["scenario_days"] == 3 + j["horizon"]["horizon_days"]
    assert m["provenance"]["issue"] == "2021-12-01"
    assert j["locations"]["origin"]["id"] == "drake_passage"
    assert len(j["departure"]["options"]) == 3 and j["departure"]["depart_on_issue_date"]["lead_days"] == 0
    if j["status"] == "recommended":
        assert j["departure"]["recommended"] == j["route"]["departure_date"] and j["route"]["recommended"]
    rt = j["route"]
    assert rt["departure_utc"].endswith("T00:00:00Z") and rt["eta_utc"] and rt["distance_km"] > 0
    assert rt["fuel_index"]["expected"] > 0 and "relative" in rt["fuel_index"]["unit"]
    assert rt["latlon"][0] == [j["locations"]["origin"]["resolved"]["lat"], j["locations"]["origin"]["resolved"]["lon"]]
    assert j["risk"]["combined"]["p_breach_upper"] == next(
        o for o in j["departure"]["options"] if o["departure"] == rt["departure_date"])["p_breach_upper"]
    assert [d["date"] for d in j["daily"]][0] == rt["departure_date"]
    assert {x["scenario_layer"] for x in j["layers"]["layers"]} == {d["scenario_layer"] for d in j["daily"]}
    assert len(j["layers"]["land"]) == j["layers"]["grid"]["nx"] * j["layers"]["grid"]["ny"]


def test_plan_runs_one_scenario_build_per_request_and_is_deterministic(client):
    before = client.planner.calls
    a = client.post("/real/plan", json=BODY).content
    b = client.post("/real/plan", json=BODY).content
    assert a == b and client.planner.calls == before + 2


def test_plan_accepts_coordinates_and_reports_snapping(client):
    body = {**BODY, "origin": {"lat": -56.3, "lon": -66.0}, "destination": {"lat": -63.0, "lon": -59.0},
            "include_layers": False}
    j = client.post("/real/plan", json=body).json()
    assert j["locations"]["origin"]["id"] is None and j["layers"] is None
    ref = client.post("/real/plan", json=BODY).json()
    assert j["route"]["cells"] == ref["route"]["cells"] and j["risk"] == ref["risk"]


def test_plan_errors_are_explicit(client):
    bad = client.post("/real/plan", json={**BODY, "destination": {"preset": "mcmurdo"}})
    assert bad.status_code == 422 and bad.json()["detail"]["status"] == "invalid_location"
    out = client.post("/real/plan", json={**BODY, "origin": {"lat": -80.0, "lon": -60.0}})
    assert out.status_code == 422 and "outside the routing grid" in out.json()["detail"]["reason"]
    late = client.post("/real/plan", json={**BODY, "issue": "2022-02-26"})
    assert late.status_code == 422 and late.json()["detail"]["status"] == "out_of_coverage"
    off = client.post("/real/plan", json={**BODY, "issue": "2022-07-01"})
    assert off.status_code == 422 and "not in the sea-ice archive" in off.json()["detail"]["reason"]
    assert client.post("/real/plan", json={**BODY, "window_days": 15}).status_code == 422
    assert client.post("/real/plan", json={"origin": {"preset": "drake_passage"}, "issue": "2021-12-01"}) \
        .status_code == 422


def test_plan_never_falls_back_to_synthetic(monkeypatch):
    monkeypatch.delenv("ANTROUTE_DATA_ROOT", raising=False)
    with TestClient(create_app(CONFIG)) as c:
        r = c.post("/real/plan", json=BODY)
        assert r.status_code == 503 and r.json()["detail"]["status"] == "unavailable"


def test_plan_reports_a_missing_model_instead_of_substituting(client, monkeypatch, tmp_path):
    def broken(a):
        raise HistoricalUnavailable("blocked", "PyTorch is not installed")

    archive = client.planner.archive
    monkeypatch.setenv("ANTROUTE_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(hist_api.HistoricalArchive, "load", classmethod(lambda cls, *a, **k: archive))
    monkeypatch.setattr(hist_api, "HistoricalPlanner", broken)
    with TestClient(create_app(CONFIG)) as c:
        r = c.post("/real/plan", json=BODY)
        assert r.status_code == 503 and r.json()["detail"]["status"] == "blocked"
        assert "PyTorch" in r.json()["detail"]["reason"]
