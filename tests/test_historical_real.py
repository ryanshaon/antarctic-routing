"""Real Historical Data on the real archive: the API reproduces the frozen 2023-11-14 run.

Floats are compared to a relative 1e-9: the last digits depend on the platform and PyTorch build.

Runs only where the real-data archive (``ANTROUTE_DATA_ROOT``, default /mnt/project-files/real-data) and
PyTorch are present; the inputs are kept outside Git.
"""

import importlib.util
import json
import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from antarctic_routing.api.main import create_app

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "config.yaml"
REAL_DATA = Path(os.environ.get("ANTROUTE_DATA_ROOT", "/mnt/project-files/real-data"))
SPEC = json.loads((REPO / "config" / "real_historical.json").read_text())
FROZEN = json.loads((REPO / "docs" / "frozen_demo" / "frozen_demo_2023-11-14.json").read_text())
FROZEN_WINDOW = json.loads((REPO / "deploy" / "bundle" / "plan_window.json").read_text())

WINDOW_KEYS = ("departure", "feasible", "expected_hours", "expected_fuel", "p_breach", "p_breach_upper", "lead_days",
               "forecast_fraction", "support")


def assert_matches_frozen_window(options):
    assert len(options) == len(FROZEN_WINDOW["options"])
    for got, want in zip(options, FROZEN_WINDOW["options"], strict=True):
        for k in WINDOW_KEYS:
            assert got[k] == (pytest.approx(want[k], rel=1e-9) if isinstance(want[k], float) else want[k]), k


pytestmark = pytest.mark.skipif(
    not (REAL_DATA / SPEC["inputs"]["sea_ice"]["path"]).is_file() or importlib.util.find_spec("torch") is None,
    reason="the real-data archive and PyTorch are kept outside Git")


@pytest.fixture(scope="module")
def client():
    old = os.environ.get("ANTROUTE_DATA_ROOT")
    os.environ["ANTROUTE_DATA_ROOT"] = str(REAL_DATA)
    try:
        with TestClient(create_app(CONFIG)) as c:
            yield c
    finally:
        if old is None:
            os.environ.pop("ANTROUTE_DATA_ROOT", None)
        else:
            os.environ["ANTROUTE_DATA_ROOT"] = old


def test_status_and_dates_cover_seven_seasons(client):
    st = client.get("/real/historical/status").json()
    assert st["status"] == "available" and st["label"] == "Real Historical Data"
    assert "reanalysis" in st["hindsight_forcing"]
    d = client.get("/real/historical/dates").json()
    old = lambda ds: [x for x in ds if x < "2024-06-01"]                             # noqa: E731
    assert (len(old(d["route_dates"])), len(old(d["window_dates"]))) == (596, 521)   # the six frozen-file seasons
    assert (len(d["route_dates"]), len(d["window_dates"])) == (596 + 101, 521 + 88)
    seasons = d["seasons"]["window"]
    assert [s["season"] for s in seasons] == ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
                                              "2024-25"]
    assert [s["out_of_sample"] for s in seasons] == [False, False, False, False, True, True, True]
    assert [s["independent_evaluation"] for s in seasons] == [False] * 6 + [True]


def test_route_reproduces_the_frozen_selected_route(client):
    r = client.post("/real/historical/routes?wait=true", json={"issue": "2023-11-14"}).json()
    assert r["status"] == "done", r.get("error")
    res = r["result"]
    rec = res["candidates"][res["recommended_index"]]
    exp = FROZEN["expected"]
    assert (rec["expected_hours"], rec["expected_fuel"], rec["distance_km"], rec["p_breach_upper"]) == pytest.approx(
        (exp["expected_hours"], exp["expected_fuel"], exp["distance_km"], exp["p_breach_upper"]), rel=1e-9)
    assert rec["breaches"] == exp["breaches"] and len(rec["xy_km"]) == exp["route_cells"]
    assert (res["execution_mode"], res["data_status"], res["data_label"]) == ("real", "historical",
                                                                              "Real Historical Data")
    assert res["icebergs"]["list_date"] == "2023-11-09" and len(res["icebergs"]["drifted"]) == 7
    assert res["historical"]["season"]["out_of_sample"] is True
    assert res["historical"]["probability_calibration"]["applied_in_route_risk"] is False


def test_departure_window_matches_the_frozen_plan_window(client):
    r = client.post("/real/historical/departures?wait=true", json={"issue": "2023-11-14"}).json()
    assert r["status"] == "done", r.get("error")
    res = r["result"]
    assert_matches_frozen_window(res["options"])
    assert res["selected"] == FROZEN_WINDOW["selected"] and res["layer_source"] == FROZEN_WINDOW["layer_source"]


def test_out_of_coverage_dates_are_refused_with_the_reason(client):
    r = client.post("/real/historical/routes?wait=true", json={"issue": "2022-11-15"})
    assert r.status_code == 422 and "14 contiguous" in r.json()["detail"]["reason"]
    r = client.post("/real/historical/departures?wait=true", json={"issue": "2024-02-20"})
    assert r.status_code == 422 and "past the end of the season" in r.json()["detail"]["reason"]


def test_voyage_replan_and_replay_with_icebergs(client):
    v = client.post("/real/historical/voyages", json={"issue": "2023-11-14"})
    assert v.status_code == 201
    vid, route = v.json()["voyage_id"], v.json()["route"]["latlon"]
    wp = route[len(route) // 3]
    r = client.post(f"/real/historical/voyages/{vid}/replan", json={"lat": wp[0], "lon": wp[1], "issued": "2023-11-15"})
    assert r.status_code == 200 and r.json()["action"] in ("keep", "switch", "no_feasible_route")
    assert r.json()["usnic_list"] == "2023-11-09" and r.json()["data_label"] == "Real Historical Data"
    events = client.get(f"/voyages/{vid}/history").json()["events"]
    assert [e.get("event") for e in events] == ["planned", "replan"]
    assert client.post(f"/voyages/{vid}/replan", json={"lat": wp[0], "lon": wp[1],
                                                       "issued": "2023-11-15"}).status_code == 409
    props = client.get(f"/voyages/{vid}/export").json()["features"][0]["properties"]
    assert props["data_label"] == "Real Historical Data" and "reanalysis" in props["hindsight_forcing"]
    header = client.get(f"/voyages/{vid}/export?format=csv").text.splitlines()[0]
    assert header.endswith(",disclaimer,data_label,hindsight_forcing")
    rp = client.post("/real/historical/replay?wait=true", json={"start": "2023-11-14", "max_wait_days": 0}).json()
    assert rp["status"] == "done", rp.get("error")
    res = rp["result"]
    assert res["iceberg_hazard"] is True and res["departure"] == "2023-11-14" and res["arrived"] is True
    assert {"berg_footprint_cells", "berg_min_distance_km"} <= res["truth"]["planner"].keys()
    assert res["map"]["observed_date"] == "2023-11-14" and len(res["observed_bergs"]) == 7


def test_presets_resolve_on_the_real_routing_grid(client):
    body = client.get("/real/locations").json()
    rows = {p["id"]: p for p in body["presets"]}
    assert all(p["available"] for p in rows.values())
    assert [k for k, p in rows.items() if p["snapped"]] == ["rothera"]      # only Rothera is on a 25 km land cell
    assert (rows["drake_passage"]["resolved"]["row"], rows["drake_passage"]["resolved"]["col"]) == (29, 12)
    assert (rows["bransfield_strait"]["resolved"]["row"], rows["bransfield_strait"]["resolved"]["col"]) == (29, 47)


def test_a_chosen_route_plans_between_the_resolved_cells_with_its_own_horizon(client):
    body = {"issue": "2023-11-14", "origin": {"preset": "drake_passage"}, "destination": {"preset": "palmer_station"}}
    rs = client.post("/real/locations/resolve", json=body).json()
    assert rs["horizon"]["horizon_days"] == 7 and rs["coverage"]["route"]["ok"]
    r = client.post("/real/historical/routes?wait=true", json=body).json()
    assert r["status"] == "done", r.get("error")
    res = r["result"]
    assert res["route"] == {k: rs[k] for k in ("origin", "destination", "horizon")}
    rec = res["candidates"][res["recommended_index"]]
    for end, ll in (("origin", rec["latlon"][0]), ("destination", rec["latlon"][-1])):
        assert ll == [rs[end]["resolved"]["lat"], rs[end]["resolved"]["lon"]]
    assert res["data_label"] == "Real Historical Data"


PLAN = {"origin": {"preset": "drake_passage"}, "destination": {"preset": "bransfield_strait"}, "issue": "2023-11-14"}


def test_plan_reproduces_the_frozen_run_in_one_call_and_is_deterministic(client):
    r = client.post("/real/plan", json=PLAN)
    assert r.status_code == 200, r.text
    j, exp = r.json(), FROZEN["expected"]
    rt = j["route"]
    assert j["status"] == "recommended" and j["departure"]["recommended"] == "2023-11-14" == rt["departure_date"]
    assert (rt["expected_hours"], rt["fuel_index"]["expected"], rt["distance_km"]) == pytest.approx(
        (exp["expected_hours"], exp["expected_fuel"], exp["distance_km"]), rel=1e-9)
    assert j["risk"]["combined"]["p_breach_upper"] == pytest.approx(exp["p_breach_upper"], rel=1e-9)
    assert j["risk"]["combined"]["breaches"] == exp["breaches"]
    assert len(rt["latlon"]) == exp["route_cells"] and rt["eta_utc"] == "2023-11-15T13:30Z"
    assert_matches_frozen_window(j["departure"]["options"])
    risk = j["risk"]
    assert risk["combined"]["breaches"] == risk["sea_ice"]["breaches"] + risk["iceberg_only_increment"]["breaches"]
    m = j["metadata"]
    assert m["mode"] == "historical" and m["hindsight_forcing"] is True and "reanalysis" in m["hindsight_disclosure"]
    assert m["banners"] == ["HISTORICAL MODE", "ERA5/CMEMS hindsight forcing",
                            "Research estimate, not certified navigation"]
    assert m["provenance"]["forecast_model"]["sha256"] == SPEC["inputs"]["checkpoint"]["sha256"]
    assert j["icebergs"]["list_date"] == "2023-11-09" and len(j["iceberg_tracks"]) == 7
    assert [d["date"] for d in j["daily"]] == ["2023-11-14", "2023-11-15"]
    assert client.post("/real/plan", json=PLAN).content == r.content               # deterministic
    coords = client.post("/real/plan", json={**PLAN, "origin": {"lat": -56.3, "lon": -66.0},
                                             "destination": {"lat": -63.0, "lon": -59.0}}).json()
    assert coords["route"] == rt and coords["risk"] == risk


def test_plan_to_rothera_snaps_and_never_hides_an_infeasible_window(client):
    j = client.post("/real/plan", json={**PLAN, "destination": {"preset": "rothera"}}).json()
    assert j["locations"]["destination"]["snapped"] is True and j["horizon"]["horizon_days"] == 9
    assert j["metadata"]["scenario_days"] == 14 + 9
    assert j["status"] == "no_feasible_departure"          # heavy ice: the least risky option, flagged
    assert j["departure"]["recommended"] is None and j["route"]["recommended"] is False
    assert j["risk"]["combined"]["within_budget"] is False
    risk = j["risk"]
    assert risk["combined"]["breaches"] >= max(risk["sea_ice"]["breaches"], risk["iceberg"]["breaches"])
    assert j["daily"] and j["daily"][0]["date"] == j["route"]["departure_date"]


def test_plan_refuses_invalid_locations_and_uncovered_dates(client):
    r = client.post("/real/plan", json={**PLAN, "origin": {"preset": "mcmurdo"}})
    assert r.status_code == 422 and r.json()["detail"]["status"] == "invalid_location"
    r = client.post("/real/plan", json={**PLAN, "issue": "2024-02-20"})
    assert r.status_code == 422 and "past the end of the season" in r.json()["detail"]["reason"]


# ------------------------------------------------------------------ S4: voyage simulation
def _simulate(client, body):
    r = client.post("/real/simulate?wait=true", json=body).json()
    assert r["status"] == "done", r.get("error")
    return r["result"]


def test_simulation_sails_the_frozen_plan_and_needs_no_replan(client):
    body = {"origin": {"preset": "drake_passage"}, "destination": {"preset": "bransfield_strait"},
            "issue": "2023-11-14", "departure": "2023-11-14"}
    s = _simulate(client, body)
    assert s["status"] == "simulated" and s["metadata"]["execution_mode"] == "real"
    f0, last = s["frames"][0], s["frames"][-1]
    assert f0["forecast"]["route_ahead"]["expected_hours"] == pytest.approx(37.503171114273826, abs=1e-9)
    assert f0["forecast"]["risk"]["combined"]["p_breach_upper"] == pytest.approx(0.018845326377266575)
    assert [f["date"] for f in s["frames"]] == ["2023-11-14", "2023-11-15", "2023-11-15"]
    assert [f["phase"] for f in s["frames"]] == ["departure", "at_sea", "arrived"]
    assert s["frames"][1]["decision"]["action"] == "keep"
    u = s["summary"]
    assert u["arrived"] and u["replans"] == 0 and u["replan_note"] == "No replan was required during this voyage."
    assert u["sailed"]["distance_km"] == pytest.approx(837.75, abs=0.01)
    assert u["sailed"]["observed_breach_cells"] == 0 and u["sailed"]["berg_nearest"] == "A76B"
    assert [last["position"]["row"], last["position"]["col"]] == s["plan"]["route"]["cells"][-1]
    assert _simulate(client, body) == s                                                  # deterministic


def test_simulation_shows_a_real_replan_from_the_existing_rules(client):
    """Drake Passage -> Bransfield Strait, forecast issued 2019-01-30 (found by the S4 replan search)."""
    body = {"origin": {"preset": "drake_passage"}, "destination": {"preset": "bransfield_strait"},
            "issue": "2019-01-30"}
    s = _simulate(client, body)
    assert s["status"] == "simulated" and s["metadata"]["departure_date"] == "2019-02-12"
    assert s["plan"]["status"] == "no_feasible_departure"
    rep = [e for e in s["events"] if e["type"] == "replan"]
    assert len(rep) == 1 and s["summary"]["replans"] == 1 and s["summary"]["arrived"]
    e = rep[0]
    assert e["triggers"] == ["material_fuel_saving"] and e["issued"] == "2019-02-13"
    assert e["change"]["expected_fuel"] == pytest.approx(-40.274, abs=0.01)
    assert e["change"]["distance_km"] == pytest.approx(-41.602, abs=0.01)
    assert e["new_route"]["p_breach_upper"] <= 0.05 and e["deviation_km"] == 50.0
    f = s["frames"][e["frame"]]
    assert f["replanned"] and f["route"]["cells"] == e["new_route"]["cells"] != e["old_route"]["cells"]


# ------------------------------------------------------------------ 2024-25 (S5: registered additional season)
DB = {"origin": {"preset": "drake_passage"}, "destination": {"preset": "bransfield_strait"}, "window_days": 14}


def test_2024_25_is_discoverable_with_its_supported_range(client):
    d = client.get("/real/historical/dates?origin=drake_passage&destination=bransfield_strait").json()
    s = d["seasons"]["window"][-1]
    assert (s["season"], s["first"], s["last"], s["n_dates"]) == ("2024-25", "2024-11-14", "2025-02-09", 88)
    assert (s["forecast_model"], s["iceberg_drift"]) == ("independent_evaluation", "independent_evaluation")
    assert s["out_of_sample"] and s["in_sample_notes"] == [] and len(s["evaluation_notes"]) == 2
    new = [x for x in d["window_dates"] if x >= "2024-11-01"]
    assert new[0] == "2024-11-14" and new[-1] == "2025-02-09" and len(new) == 88


@pytest.mark.parametrize("issue", ["2024-11-14", "2024-12-27", "2025-02-09"])
def test_2024_25_plans_on_real_sea_ice_forcing_and_icebergs(client, issue):
    r = client.post("/real/plan", json={**DB, "issue": issue})
    assert r.status_code == 200, r.text
    p = r.json()
    m = p["metadata"]
    pv = m["provenance"]
    assert p["status"] == "recommended" and m["execution_mode"] == "real" and m["data_status"] == "historical"
    assert m["banners"][:3] == ["HISTORICAL MODE", "ERA5/CMEMS hindsight forcing",
                                "Research estimate, not certified navigation"]
    assert pv["sea_ice"]["file"] == "sea_ice_25km_2024_25.nc"
    assert pv["sea_ice"]["sha256"] == SPEC["inputs"]["sea_ice_additional"][0]["sha256"]
    assert "OSI-430-a v3.0" in pv["sea_ice"]["source"] and pv["sea_ice"]["observed_through"] == issue
    assert pv["forecast_model"]["sha256"] == SPEC["inputs"]["checkpoint"]["sha256"]
    assert pv["season"]["season"] == "2024-25" and pv["season"]["independent_evaluation"]
    files = {k: [f["file"] for f in v["files"]] for k, v in pv["forcing"].items()}
    assert files["winds"] and all(f.startswith("forcing_era5_daily_25km_202") for f in files["winds"])
    assert files["currents"] == ["forcing_cmems_daily_25km_20241101-20250228.nc"]
    assert pv["icebergs"]["file"].startswith("AntarcticIcebergs_202") and pv["icebergs"]["list_date"] <= issue
    assert pv["icebergs"]["age_days"] <= 14 and pv["icebergs"]["drifted"]
    assert p["route"]["distance_km"] == pytest.approx(837.75, abs=0.01)


def test_2024_25_forecasts_use_real_daily_forcing_never_schematic():
    from datetime import date

    from antarctic_routing.config import load_config
    from antarctic_routing.historical import HistoricalArchive
    a = HistoricalArchive.load(REAL_DATA, REPO / "config" / "real_historical.json",
                               load_config(CONFIG).project.season_months)
    assert (a.winds.product, a.currents.product) == ("ERA5", "CMEMS")
    assert a.winds.missing(date(2024, 11, 1), 120) == [] and a.currents.missing(date(2024, 11, 1), 120) == []
    assert a.winds.missing(date(2025, 2, 28), 2) == ["2025-03-01"]          # no forcing after the season: refused
    assert a.ds.attrs["execution_mode"] == "real"
    assert [f["file"] for f in a.sea_ice_files] == ["sea_ice_25km.nc", "sea_ice_25km_2024_25.nc"]
    assert str(a.ds["time"].values[-1])[:10] == "2025-02-28" and len(a.days) == 2403 + 120


def test_2024_25_refuses_unsupported_dates_with_the_reason(client):
    # dates up to the end of the archive (2025-02-28) are Real Historical Data only: uncovered ones are refused
    for issue, why in (("2024-11-13", "14 contiguous in-season days"),
                       ("2025-02-10", "daily ERA5 winds are missing"),
                       ("2024-06-15", "not in the sea-ice archive")):
        r = client.post("/real/plan", json={**DB, "issue": issue})
        assert r.status_code == 422 and r.json()["detail"]["status"] == "out_of_coverage", issue
        assert why in r.json()["detail"]["reason"], issue
        assert client.post("/real/simulate?wait=true", json={**DB, "issue": issue}).status_code == 422
    # a later date is an estimate (here a historical seasonal analogue), never a refusal for being off season
    r = client.post("/real/plan", json={**DB, "issue": "2025-03-05"})
    assert r.status_code == 200 and r.json()["metadata"]["mode"] == "forecast"
    f = r.json()["metadata"]["forecast"]
    assert (f["pathway"], f["analogue_date"]) == ("seasonal_analogue", "2024-03-05")
    assert "outside the Nov-Feb season" in f["pathway_reason"]


def test_2024_25_plan_and_simulation_are_deterministic(client):
    body = {**DB, "issue": "2024-11-14"}
    p1 = client.post("/real/plan", json=body).json()
    assert client.post("/real/plan", json=body).json() == p1
    s = _simulate(client, {**body, "departure": "2024-11-14"})
    assert s["status"] == "simulated" and s["metadata"]["provenance"]["sea_ice"]["file"] == "sea_ice_25km_2024_25.nc"
    assert [f["date"] for f in s["frames"]] == ["2024-11-14", "2024-11-15", "2024-11-15"]
    assert [f["phase"] for f in s["frames"]] == ["departure", "at_sea", "arrived"]
    u = s["summary"]
    assert u["arrived"] and u["replans"] == 0 and u["replan_note"] == "No replan was required during this voyage."
    assert u["sailed"]["distance_km"] == pytest.approx(837.75, abs=0.01) and u["sailed"]["observed_breach_cells"] == 0
    assert s["frames"][1]["observed"]["icebergs"]["file"].startswith("AntarcticIcebergs_2024")
    assert _simulate(client, {**body, "departure": "2024-11-14"}) == s


# ------------------------------------------------------------------ forecast mode (dates after the archive)
FC = {"origin": {"preset": "drake_passage"}, "destination": {"preset": "bransfield_strait"}, "issue": "2026-11-19"}
ARCHIVE_END = "2025-02-28"


def _no_observation_claims(obj, path=""):
    """No 'observed'/'observation' field carries a date after the archive, except the official iceberg list and
    values explicitly labelled as the analogue proxy."""
    if isinstance(obj, dict):
        if obj.get("source") == "proxy_analogue_observed":
            return
        for k, v in obj.items():
            _no_observation_claims(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _no_observation_claims(v, f"{path}/{i}")
    elif isinstance(obj, str):
        assert obj != "observed", path
        if "observ" in path.lower() and "/icebergs" not in path and "observations_for_requested_dates" not in path:
            assert all(d <= ARCHIVE_END for d in re.findall(r"\d{4}-\d{2}-\d{2}", obj)), (path, obj)


def test_2026_11_19_is_planned_in_forecast_mode_from_labelled_proxies(client):
    r = client.post("/real/plan", json=FC)
    assert r.status_code == 200, r.text
    j = r.json()
    m, f = j["metadata"], j["metadata"]["forecast"]
    assert (m["mode"], m["execution_mode"], m["data_status"]) == ("forecast", "modelled", "forecast_estimate")
    assert m["banners"][0] == "FORECAST / HACKATHON ESTIMATE" and m["hindsight_forcing"] is False
    assert m["requested_date"] == "2026-11-19" and "provenance" not in m
    assert m["forecast_disclosure"] == SPEC["forecast_mode"]["disclosure"]
    rt = j["route"]
    assert j["status"] == "recommended" and j["departure"]["recommended"] == "2026-11-19" == rt["departure_date"]
    assert rt["eta_utc"] == "2026-11-20T13:09Z" and rt["expected_hours"] == pytest.approx(37.1449, abs=1e-3)
    assert rt["distance_km"] == pytest.approx(837.75, abs=0.01)
    risk = j["risk"]
    assert risk["combined"]["breaches"] == 0 and risk["combined"]["n_scenarios"] == 200
    assert risk["combined"]["p_breach_upper"] == pytest.approx(0.018845326377266575)
    assert [o["departure"] for o in j["departure"]["options"]][:2] == ["2026-11-19", "2026-11-20"]
    assert len(j["departure"]["options"]) == 14 and [d["date"] for d in j["daily"]] == ["2026-11-19", "2026-11-20"]
    # the actual data and proxy dates
    dm = f["date_mapping"]
    assert (dm["analogue_start_date"], dm["analogue_season"], dm["offset_days"]) == ("2024-11-19", "2024-25", 730)
    assert dm["engine_days"] == ["2024-11-19", "2024-12-08"] and dm["shown_as"] == ["2026-11-19", "2026-12-08"]
    assert f["sea_ice"]["status"] == "proxy_analogue" and f["sea_ice"]["observed_window_used"] == \
        ["2024-11-06", "2024-11-19"]
    assert f["sea_ice"]["sha256"] == SPEC["inputs"]["sea_ice_additional"][0]["sha256"]
    assert f["sea_ice_forecast"]["model"]["sha256"] == SPEC["inputs"]["checkpoint"]["sha256"]
    assert f["forcing"]["status"] == "proxy_analogue_reanalysis"
    assert f["forcing"]["winds"]["dates_used"] == f["forcing"]["currents"]["dates_used"] == \
        ["2024-11-19", "2024-12-08"]
    b = f["icebergs"]
    assert (b["snapshot_date"], b["file"], b["age_days_at_requested_date"]) == \
        ("2026-10-01", "AntarcticIcebergs_20261001.csv", 49)
    assert b["sha256"] == SPEC["inputs"]["forecast_icebergs"][-1]["sha256"]
    assert (b["n_source_bergs"], b["n_in_grid"]) == (33, 7) and b["drift_model"]["members"] == 200
    assert (b["drift_model"]["beta"], b["drift_model"]["alpha_scale"], b["drift_model"]["spread_factor"]) == \
        (0.1, 0.1, 0.6053)
    assert j["icebergs"]["list_date"] == "2026-10-01"
    _no_observation_claims(j)
    # deterministic plan and departure window
    assert client.post("/real/plan", json=FC).content == r.content
    # the engine is the unchanged historical one: same sea-ice risk as the analogue day in historical mode
    hist = client.post("/real/plan", json={**FC, "issue": "2024-11-19"}).json()
    assert hist["metadata"]["mode"] == "historical" and hist["risk"]["sea_ice"] == risk["sea_ice"]


def test_forecast_dates_are_listed_and_every_later_date_is_estimated(client):
    d = client.get("/real/historical/dates?origin=drake_passage&destination=bransfield_strait").json()
    f = d["forecast"]
    assert (f["first"], f["last"], len(f["dates"])) == ("2026-11-14", "2027-01-29", 77)   # proxy-forecast range
    assert d["window_dates"][-1] == "2025-02-09"                                     # historical range unchanged
    e = d["estimate"]
    assert e["any_date_after"] == "2025-02-28" and e["analogue"]["available"] is True, e
    # dates the proxy forecast cannot serve fall back to a historical seasonal analogue, with the reason recorded
    for issue, analogue, why, kind in (
            ("2026-06-15", "2024-06-15", "outside the Nov-Feb season", "analogue_year"),
            ("2027-11-20", "2024-11-20", "no recent official iceberg", "analogue_year"),
            ("2026-11-13", "2024-11-13", "no archive season can start", "recent_official")):
        r = client.post("/real/plan", json={**FC, "issue": issue})
        assert r.status_code == 200, (issue, r.text)
        fm = r.json()["metadata"]["forecast"]
        assert (fm["pathway"], fm["analogue_date"], fm["icebergs"]["kind"]) == ("seasonal_analogue", analogue, kind)
        assert why in fm["pathway_reason"], issue
    # a past date the archive does not cover is still refused, never replaced by another date
    r = client.post("/real/plan", json={**FC, "issue": "2022-06-15"})
    assert r.status_code == 422 and r.json()["detail"]["status"] == "out_of_coverage"


def test_forecast_simulation_sails_the_proxy_and_is_deterministic(client):
    s = _simulate(client, FC)
    m = s["metadata"]
    assert s["status"] == "simulated" and m["mode"] == "forecast" and m["execution_mode"] == "modelled"
    assert m["departure_date"] == "2026-11-19" and "analogue" in m["simulation_note"]
    assert [f["date"] for f in s["frames"]] == ["2026-11-19", "2026-11-20", "2026-11-20"]
    assert [f["phase"] for f in s["frames"]] == ["departure", "at_sea", "arrived"]
    assert all(f["observed"]["sea_ice_source"] == "proxy_analogue_observed" for f in s["frames"])
    assert s["frames"][1]["observed"]["icebergs"]["list_date"] == "2026-10-01"
    seg = s["frames"][1]["observed"]["sea_ice_on_segment"]
    assert (seg["date"], seg["analogue_date"], seg["source"]) == ("2026-11-19", "2024-11-19", "proxy_analogue_observed")
    u = s["summary"]
    assert u["arrived"] and u["replans"] == 0 and u["arrival_utc"] == "2026-11-20T14:08Z"
    assert u["sailed"]["distance_km"] == pytest.approx(837.75, abs=0.01) and "forecast_mode" in u["notes"]
    _no_observation_claims(s)
    assert _simulate(client, FC) == s


# ------------------------------------------------------------------ any future date: historical seasonal analogue
EXAMPLES = (("2026-11-19", "proxy_forecast", "2024-11-19"), ("2026-12-15", "proxy_forecast", "2024-12-15"),
            ("2027-01-20", "proxy_forecast", "2025-01-20"), ("2027-02-15", "seasonal_analogue", "2024-02-15"),
            ("2027-08-14", "seasonal_analogue", "2024-08-14"), ("2028-03-03", "seasonal_analogue", "2024-03-03"),
            ("2030-08-14", "seasonal_analogue", "2024-08-14"))


@pytest.mark.parametrize("issue,pathway,analogue", EXAMPLES)
def test_every_example_future_date_is_accepted_and_records_its_analogue(client, issue, pathway, analogue):
    r = client.post("/real/plan", json={**FC, "issue": issue})
    assert r.status_code == 200, r.text
    j = r.json()
    m, f = j["metadata"], j["metadata"]["forecast"]
    assert (m["mode"], m["execution_mode"], m["data_status"]) == ("forecast", "modelled", "forecast_estimate")
    assert m["banners"][0] == "FORECAST / HACKATHON ESTIMATE" and m["requested_date"] == issue
    assert (f["pathway"], f["date_mapping"]["analogue_start_date"]) == (pathway, analogue)
    assert f["date_mapping"]["shown_as"][0] == issue                                 # never another date
    assert j["status"] in ("recommended", "no_feasible_departure") and j["route"] is not None
    _no_observation_claims(j)


def test_august_2027_is_a_labelled_seasonal_analogue_with_its_real_dates_and_caveats(client):
    r = client.post("/real/plan", json={**FC, "issue": "2027-08-14"})
    j = r.json()
    m, f = j["metadata"], j["metadata"]["forecast"]
    assert m["banners"] == ["FORECAST / HACKATHON ESTIMATE",
                            "Historical seasonal analogue: real sea ice, winds and currents of earlier years",
                            "Research estimate, not certified navigation"]
    assert m["forecast_disclosure"] == SPEC["seasonal_analogue"]["disclosure"]
    assert "not a meteorological or oceanographic forecast" in m["forecast_disclosure"]
    assert (f["pathway"], f["pathway_label"]) == ("seasonal_analogue", "Historical seasonal analogue")
    assert (f["requested_date"], f["analogue_date"], f["archive_last_date"]) == \
        ("2027-08-14", "2024-08-14", "2025-02-28")
    assert f["observations_for_requested_dates"].startswith("none")
    dm = f["date_mapping"]
    assert dm["engine_days"] == ["2024-08-14", "2024-09-02"] and dm["shown_as"] == ["2027-08-14", "2027-09-02"]
    assert "outside the Nov-Feb archive seasons" in dm["analogue_season"]
    sea = f["sea_ice"]
    assert sea["status"] == "historical_analogue_ensemble" and sea["member_years"] == list(range(2023, 2016, -1))
    assert sea["day_shifts"] == [-7, 7] and sea["members"] == 200 and 2024 not in sea["member_years"]
    pins = {e["path"].rsplit("/", 1)[-1]: e["sha256"] for e in SPEC["inputs"]["analogue_sea_ice"]}
    assert {x["file"]: x["sha256"] for x in sea["off_season_files"]}.items() <= pins.items()
    assert f["forcing"]["status"] == "historical_reanalysis_analogue"
    assert f["forcing"]["winds"]["dates_used"] == f["forcing"]["currents"]["dates_used"] == ["2024-08-14", "2024-09-02"]
    b = f["icebergs"]
    assert (b["kind"], b["snapshot_date"], b["file"]) == \
        ("analogue_year", "2024-08-08", "AntarcticIcebergs_20240808.csv")
    assert b["sha256"] in {e["sha256"] for e in SPEC["inputs"]["analogue_icebergs"]}
    assert (b["age_days_at_analogue_date"], b["max_age_days"]) == (6, 14)
    # winter ice: the estimate is honest about risk and confidence instead of refusing the date
    c = f["confidence"]
    assert c["level"] == "very low" and any("Not a forecast" in x for x in c["caveats"])
    assert c["sea_ice_spread"]["range"] >= 0.1
    assert j["status"] == "no_feasible_departure" and j["departure"]["recommended"] is None
    assert j["risk"]["combined"]["breaches"] > 100
    _no_observation_claims(j)
    assert client.post("/real/plan", json={**FC, "issue": "2027-08-14"}).content == r.content    # deterministic
    # 2030-08-14 uses the same analogue (the most recent August with real data) and is shown on its own dates
    far = client.post("/real/plan", json={**FC, "issue": "2030-08-14"}).json()
    ff = far["metadata"]["forecast"]
    assert ff["analogue_date"] == "2024-08-14" and ff["date_mapping"]["shown_as"][0] == "2030-08-14"
    assert far["risk"] == j["risk"] and far["route"]["departure_date"].startswith("2030-08-")
    assert any("5.5 years after the latest real observation" in x for x in ff["confidence"]["caveats"])


def test_analogue_simulation_sails_the_analogue_years_ice_and_never_fakes_a_gap(client):
    s = _simulate(client, {**FC, "issue": "2028-03-03"})
    m = s["metadata"]
    assert s["status"] == "simulated" and m["mode"] == "forecast" and m["execution_mode"] == "modelled"
    assert m["departure_date"] == "2028-03-03" and "Seasonal-analogue" in m["simulation_note"]
    assert [f["date"] for f in s["frames"]] == ["2028-03-03", "2028-03-04", "2028-03-04"]
    seg = s["frames"][1]["observed"]["sea_ice_on_segment"]
    assert (seg["date"], seg["analogue_date"], seg["source"]) == ("2028-03-03", "2024-03-03", "proxy_analogue_observed")
    assert s["frames"][1]["observed"]["icebergs"]["file"] == "AntarcticIcebergs_20240301.csv"
    u = s["summary"]
    assert u["arrived"] and u["replans"] == 0 and u["arrival_utc"] == "2028-03-04T14:09Z"
    assert u["sailed"]["distance_km"] == pytest.approx(837.75, abs=0.01)
    assert _simulate(client, {**FC, "issue": "2028-03-03"}) == s
    # the analogue year (2024) has no OSI SAF ice on 15-17 Sep upstream: the plan works, the voyage says why not
    assert client.post("/real/plan", json={**FC, "issue": "2027-09-05"}).status_code == 200
    gap = client.post("/real/simulate?wait=true", json={**FC, "issue": "2027-09-05"}).json()["result"]
    assert gap["status"] == "not_simulated"
    assert "2024-09-15, 2024-09-16, 2024-09-17 (missing upstream)" in gap["reason"]
