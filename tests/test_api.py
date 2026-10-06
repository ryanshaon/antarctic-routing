"""HTTP API contract (Stage 11)."""

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from antarctic_routing import DISCLAIMER
from antarctic_routing.api.main import create_app

CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
FAST = {"departure": "2026-12-20", "scenarios": 80, "resolution_km": 25, "scenario_routes": 0}
OPEN_WATER = {**FAST, "departure": "2027-01-10"}


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(CONFIG)) as c:
        yield c


def test_health_reports_version_and_disclaimer(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["disclaimer"] == DISCLAIMER and body["version"]


def test_config_endpoint_returns_validated_scenario(client):
    body = client.get("/config").json()
    assert body["config"]["routing"]["risk_budget"] == 0.05
    assert body["min_scenarios_for_budget"] == 73


def test_synchronous_route_plan_includes_map_and_candidates(client):
    r = client.post("/routes?wait=true", json=FAST)
    assert r.status_code == 200
    job = r.json()
    assert job["status"] == "done", job.get("error")
    res = job["result"]
    m = res["map"]
    assert len(m["p_ice"]) == m["nx"] * m["ny"] == len(m["land"])
    assert res["status"] in ("feasible", "infeasible")
    assert res["execution_mode"] == "controlled_synthetic"
    for cand in res["candidates"]:
        assert len(cand["xy_km"]) == len(cand["latlon"]) >= 2
        assert {"labels", "tags", "p_breach", "p_breach_upper", "expected_hours", "expected_fuel"} <= cand.keys()


def test_asynchronous_job_lifecycle(client):
    r = client.post("/routes", json=FAST)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(120):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.25)
    assert job["status"] == "done"


def test_unknown_job_is_404(client):
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_too_few_scenarios_for_the_budget_is_rejected(client):
    r = client.post("/routes?wait=true", json={**FAST, "scenarios": 20})
    assert r.status_code == 422
    assert "73" in r.text


def test_route_request_with_iceberg(client):
    r = client.post("/routes?wait=true", json={**FAST, "icebergs": [{"id": "A23A", "lat": -60.2, "lon": -62.6}]})
    res = r.json()["result"]
    assert res["icebergs"][0]["id"] == "A23A"
    assert res["map"]["p_berg"] is not None


def test_departure_window_job(client):
    r = client.post("/departures?wait=true", json={"start": "2026-12-08", "end": "2026-12-20", "step_days": 6,
                                                   "scenarios": 80, "resolution_km": 25})
    res = r.json()["result"]
    assert len(res["options"]) == 3 and "rule" in res


def test_voyage_lifecycle_replan_history_export(client):
    v = client.post("/voyages", json=OPEN_WATER)
    assert v.status_code == 201, v.text
    voyage = v.json()
    vid = voyage["voyage_id"]
    assert voyage["route"]["latlon"]
    lat, lon = voyage["route"]["latlon"][3]
    r = client.post(f"/voyages/{vid}/replan", json={"lat": lat, "lon": lon, "issued": "2026-12-21",
                                                     "scenarios": 80, "data_age_hours": 6})
    assert r.status_code == 200, r.text
    assert r.json()["action"] in ("keep", "switch", "no_feasible_route")
    hist = client.get(f"/voyages/{vid}/history").json()
    assert [h["event"] for h in hist["events"]] == ["planned", "replan"]
    gj = client.get(f"/voyages/{vid}/export?format=geojson").json()
    assert gj["type"] == "FeatureCollection"
    assert gj["features"][0]["properties"]["disclaimer"] == DISCLAIMER
    csv = client.get(f"/voyages/{vid}/export?format=csv")
    assert csv.headers["content-type"].startswith("text/csv")
    assert "planned_arrival_utc" in csv.text.splitlines()[0]
    # Only Real Historical Data voyages carry a data label; the synthetic export is unchanged.
    assert csv.text.splitlines()[0] == "waypoint_index,lat,lon,planned_arrival_utc,segment_breach_prob,disclaimer"
    assert "data_label" not in gj["features"][0]["properties"]


def test_replan_position_off_route_is_a_clear_400(client):
    vid = client.post("/voyages", json=OPEN_WATER).json()["voyage_id"]
    r = client.post(f"/voyages/{vid}/replan", json={"lat": -56.0, "lon": -70.5, "issued": "2026-12-21",
                                                     "scenarios": 80})
    assert r.status_code == 400 and "route" in r.json()["detail"]


def test_unknown_voyage_is_404(client):
    assert client.get("/voyages/nope/history").status_code == 404


def test_voyage_is_refused_when_no_route_meets_the_budget(client):
    r = client.post("/voyages", json={**FAST, "departure": "2026-11-20"})
    assert r.status_code == 409
    assert "risk budget" in r.json()["detail"]


def test_dashboard_page_and_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "Antarctic Ice-Risk Routing" in page.text
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_figures_endpoint_lists_validation_images(client):
    figs = client.get("/figures").json()["figures"]
    assert figs and all(f["url"].startswith("/figures/") for f in figs)
    assert client.get(figs[0]["url"]).status_code == 200
    assert any("backtest" in f["url"] for f in figs)


# ------------------------------------------------- data status and real-data bundle
def test_status_without_a_bundle_says_real_data_is_unavailable(client):
    st = client.get("/status").json()
    assert st["real_data"]["status"] == "unavailable" and "ANTROUTE_ARTIFACTS_DIR" in st["real_data"]["reason"]
    assert st["interactive_planner"]["status"] == "schematic"
    assert st["interactive_planner"]["execution_mode"] == "controlled_synthetic"
    assert client.get("/health").json()["data_modes"] == ["controlled_synthetic"]
    for path in ("/real/route", "/real/forecast-map", "/real/sea-ice", "/real/icebergs", "/provenance"):
        r = client.get(path)
        assert r.status_code == 503 and r.json()["detail"]["status"] == "unavailable", path


def test_synthetic_responses_are_labelled_schematic(client):
    plan = client.post("/routes?wait=true", json=FAST).json()["result"]
    assert plan["execution_mode"] == "controlled_synthetic" and plan["data_status"] == "schematic"
    assert None not in plan["map"]["p_ice"]          # the synthetic world has no missing ocean cells


def test_config_and_versions_expose_no_filesystem_paths(client):
    body = client.get("/config").json()
    assert body["config_file"] == "config.yaml" and len(body["config_sha256"]) == 64
    v = client.get("/versions").json()
    assert v["software"]["version"] and v["config_sha256"] == body["config_sha256"]
    assert "/" not in str(body["config_file"])


@pytest.mark.parametrize("path, payload", [
    ("/routes", {**FAST, "resolution_km": 1}),
    ("/routes", {**FAST, "seed": -1}),
    ("/routes", {**FAST, "icebergs": [{"id": f"B{i}", "lat": -60, "lon": -60} for i in range(21)]}),
    ("/routes", {**FAST, "berg_radius_km": 500}),
    ("/routes", {**FAST, "scenarios": 1000, "resolution_km": 5}),   # ~20 GB: refused before any work
    ("/voyages", {**FAST, "scenarios": 1000, "resolution_km": 10}),
    ("/departures", {"start": "2026-11-01", "end": "2027-02-28", "scenarios": 80, "resolution_km": 25}),
])
def test_expensive_or_invalid_requests_are_rejected(client, path, payload):
    r = client.post(path + ("?wait=true" if path != "/voyages" else ""), json=payload)
    assert r.status_code == 422
    if payload.get("scenarios") == 1000:
        assert "scenario-cells" in r.json()["detail"]


def test_replan_position_must_be_on_earth(client):
    vid = client.post("/voyages", json=OPEN_WATER).json()["voyage_id"]
    r = client.post(f"/voyages/{vid}/replan", json={"lat": -95, "lon": 10, "issued": "2027-01-11", "scenarios": 80})
    assert r.status_code == 422


@pytest.mark.parametrize("path", ["/figures/../config/config.yaml", "/static/../api/main.py",
                                  "/figures/%2e%2e/%2e%2e/config/config.yaml", "/static/%2e%2e/api/main.py"])
def test_static_mounts_do_not_traverse_paths(client, path):
    r = client.get(path)
    assert r.status_code == 404 or "risk_budget" not in r.text and "create_app" not in r.text


def test_requests_carry_an_id_and_unhandled_errors_are_json(monkeypatch):
    app = create_app(CONFIG)
    monkeypatch.setattr(app.state.service, "world", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.get("/health").headers["X-Request-ID"]
        r = c.post("/voyages", json=OPEN_WATER)
    assert r.status_code == 500 and r.json()["detail"] == "internal server error" and r.json()["error_id"]
    assert "boom" not in r.text


def test_cors_is_off_by_default_and_explicit_when_configured(client, monkeypatch):
    assert "access-control-allow-origin" not in client.get("/health", headers={"Origin": "https://x.example"}).headers
    monkeypatch.setenv("ANTROUTE_CORS_ORIGINS", "https://dash.example")
    with TestClient(create_app(CONFIG)) as c:
        ok = c.get("/health", headers={"Origin": "https://dash.example"})
        other = c.get("/health", headers={"Origin": "https://evil.example"})
    assert ok.headers["access-control-allow-origin"] == "https://dash.example"
    assert "access-control-allow-origin" not in other.headers
    monkeypatch.setenv("ANTROUTE_CORS_ORIGINS", "*")
    with pytest.raises(ValueError, match="not an exact origin"):
        create_app(CONFIG)


def test_job_and_voyage_stores_are_bounded(monkeypatch):
    monkeypatch.setenv("ANTROUTE_MAX_JOBS", "2")
    monkeypatch.setenv("ANTROUTE_MAX_VOYAGES", "1")
    with TestClient(create_app(CONFIG)) as c:
        ids = [c.post("/routes?wait=true", json=OPEN_WATER).json()["job_id"] for _ in range(3)]
        assert c.get("/health").json()["jobs"] == 2
        assert c.get(f"/jobs/{ids[0]}").status_code == 404 and c.get(f"/jobs/{ids[2]}").status_code == 200
        assert c.post("/voyages", json=OPEN_WATER).status_code == 201
        r = c.post("/voyages", json=OPEN_WATER)
        assert r.status_code == 503 and "voyage store is full" in r.json()["detail"]


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    from test_publish import make_bundle, small_world

    tmp = tmp_path_factory.mktemp("bundle")
    return make_bundle(tmp, small_world(), "real", "real")


def test_real_endpoints_serve_the_verified_bundle(bundle_dir):
    with TestClient(create_app(CONFIG, bundle_dir)) as c:
        h = c.get("/health").json()
        assert h["status"] == "ok" and h["data_modes"] == ["controlled_synthetic", "real"]
        st = c.get("/status").json()["real_data"]
        assert st["status"] == "available" and st["issue"] == "2026-12-20" and st["limitations"] == ["fixture"]
        dates = c.get("/real/forecast-dates").json()
        assert dates["layers"][0] == {"index": 0, "date": "2026-12-20", "source": "observed"}
        assert c.get("/real/sea-ice").json()["status"] == "historical"
        fm = c.get("/real/forecast-map", params={"layer": 1}).json()
        assert fm["status"] == "forecast" and fm["date"] == "2026-12-21" and len(fm["land"]) == fm["grid"]["nx"] * \
            fm["grid"]["ny"]
        assert c.get("/real/forecast-map", params={"layer": 99}).status_code == 422
        route = c.get("/real/route").json()
        g = fm["grid"]
        assert route["xy_km"] == [[g["x_km"][2], g["y_km"][1]], [g["x_km"][3], g["y_km"][2]]]
        assert c.get("/real/departure-window").json()["options"][0]["departure"] == "2026-12-20"
        bergs = c.get("/real/icebergs").json()
        assert bergs["status"] == "unavailable" and bergs["tracks"] is None     # fixture has no iceberg source
        assert c.get("/real/figures/departure_window.png").status_code == 404
        prov = c.get("/provenance").json()
        assert prov["bundle"]["bundle_id"] and "/data/" not in json.dumps(prov)


def test_a_damaged_bundle_is_reported_not_served(bundle_dir, tmp_path):
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(bundle_dir, broken)
    (broken / "forecast_maps.json").write_text("{}")
    with TestClient(create_app(CONFIG, broken)) as c:
        assert c.get("/health").json()["status"] == "degraded"
        r = c.get("/real/route")
        assert r.status_code == 503 and "checksum mismatch" in r.json()["detail"]["reason"]
        assert c.get("/status").json()["real_data"]["status"] == "failed"


def test_computations_share_one_slot_budget(monkeypatch):
    monkeypatch.setenv("ANTROUTE_MAX_COMPUTE", "1")
    app = create_app(CONFIG)
    svc = app.state.service
    with TestClient(app) as c:
        assert svc.compute_slots.acquire(blocking=False)     # another computation is running
        try:
            busy = c.post("/routes?wait=true", json=OPEN_WATER)
            assert busy.status_code == 503 and "retry" in busy.json()["detail"] and busy.headers["Retry-After"]
            assert c.post("/voyages", json=OPEN_WATER).status_code == 503
            queued = c.post("/routes", json=OPEN_WATER)
            assert queued.status_code == 202                         # background jobs queue for the slot
            time.sleep(0.3)
            assert c.get(f"/jobs/{queued.json()['job_id']}").json()["status"] == "queued"
        finally:
            svc.compute_slots.release()
        for _ in range(200):
            if c.get(f"/jobs/{queued.json()['job_id']}").json()["status"] == "done":
                break
            time.sleep(0.1)
        assert c.get(f"/jobs/{queued.json()['job_id']}").json()["status"] == "done"
        assert c.post("/routes?wait=true", json=OPEN_WATER).json()["status"] == "done"


@pytest.mark.parametrize("value", [
    "*", "null", "https://dash.example/", "https://dash.example/app", "http://dash.example", "https://Dash.example",
    "https://*.vercel.app", "https://user@dash.example", "dash.example", "https://dash.example:0",
])
def test_cors_origins_must_be_exact(value):
    from antarctic_routing.api.main import cors_origins

    with pytest.raises(ValueError, match="not an exact origin"):
        cors_origins(f"https://ok.example,{value}")


def test_cors_allows_the_dashboard_preflight_and_readable_errors(monkeypatch):
    from antarctic_routing.api.main import cors_origins

    assert cors_origins(" https://dash.example , http://localhost:5173,https://a.example:8443,") == [
        "https://dash.example", "http://localhost:5173", "https://a.example:8443"]
    monkeypatch.setenv("ANTROUTE_CORS_ORIGINS", "https://dash.example")
    monkeypatch.setenv("ANTROUTE_MAX_BODY_BYTES", "1024")
    with TestClient(create_app(CONFIG)) as c:
        pre = c.options("/routes", headers={"Origin": "https://dash.example", "Access-Control-Request-Method": "POST",
                                            "Access-Control-Request-Headers": "content-type"})
        assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] == "https://dash.example"
        assert "access-control-allow-credentials" not in pre.headers
        big = c.post("/routes", content=b"{" + b" " * 2048 + b"}",
                     headers={"Origin": "https://dash.example", "Content-Type": "application/json"})
        assert big.status_code == 413 and big.headers["access-control-allow-origin"] == "https://dash.example"


def test_request_bodies_are_size_limited_with_or_without_a_length(monkeypatch):
    monkeypatch.setenv("ANTROUTE_MAX_BODY_BYTES", "1024")
    body = json.dumps({**OPEN_WATER, "icebergs": [{"id": "X" * 40, "lat": -60, "lon": -60}] * 20}).encode()
    assert len(body) > 1024

    def chunks():
        yield body[:600]
        yield body[600:]

    with TestClient(create_app(CONFIG)) as c:
        r = c.post("/routes", content=body, headers={"Content-Type": "application/json"})
        assert r.status_code == 413 and "ANTROUTE_MAX_BODY_BYTES" in r.json()["detail"]
        r = c.post("/routes", content=chunks(), headers={"Content-Type": "application/json"})     # chunked upload
        assert r.status_code == 413
        assert c.post("/routes", json=OPEN_WATER).status_code == 202


def test_readiness_requires_a_verified_bundle(bundle_dir, tmp_path, client):
    import shutil

    r = client.get("/ready")
    assert r.status_code == 503 and r.json()["real_data"] == "unavailable"
    assert client.get("/health").status_code == 200                       # liveness is separate
    with TestClient(create_app(CONFIG, bundle_dir)) as c:
        ok = c.get("/ready")
        assert ok.status_code == 200 and ok.json()["ready"] is True and ok.json()["bundle_id"]
    broken = tmp_path / "broken"
    shutil.copytree(bundle_dir, broken)
    (broken / "manifest.json").write_text("{}")
    with TestClient(create_app(CONFIG, broken)) as c:
        r = c.get("/ready")
        assert r.status_code == 503 and r.json()["real_data"] == "failed" and "checksum mismatch" in r.json()["reason"]
    with TestClient(create_app(CONFIG, tmp_path / "missing")) as c:
        r = c.get("/ready")
        assert r.status_code == 503 and "no bundle.json" in r.json()["reason"]


def test_stored_voyages_and_grids_do_not_grow_memory(client):
    svc = client.app.state.service
    v = client.post("/voyages", json=OPEN_WATER).json()
    stored = svc.voyages[v["voyage_id"]]["world"]
    assert not hasattr(stored, "conc") and not hasattr(stored, "berg")       # no scenario arrays kept
    gj = client.get(f"/voyages/{v['voyage_id']}/export").json()
    assert gj["features"][0]["properties"]["execution_mode"] == "controlled_synthetic"
    from antarctic_routing.api.main import MAX_GRIDS

    for res in range(40, 60):
        svc.grid(float(res))
    assert len(svc._grids) <= MAX_GRIDS
