"""Origin/destination presets, snapping to the routing grid and the route-specific forecast horizon."""

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pyproj import Geod

from antarctic_routing.api import historical as hist_api
from antarctic_routing.api.main import create_app
from antarctic_routing.cli import _horizon_days
from antarctic_routing.config import load_config
from antarctic_routing.historical import DailyForcing, HistoricalArchive, UsnicArchive
from antarctic_routing.locations import (
    MAX_SNAP_KM,
    PRESETS,
    PRESETS_BY_ID,
    LocationError,
    LocationResolver,
    build_route,
    navigable_mask,
    route_horizon,
    route_horizon_days,
)
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.synthetic import schematic_land_mask, synthetic_history

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "config.yaml"
CFG = load_config(CONFIG)
GEOD = Geod(ellps="WGS84")
SPEED, STEP, LEADS = CFG.vessel.cruise_speed_kmh, CFG.grid.time_step_hours, CFG.forecast.lead_days

# The real archive's routing grid (OSI SAF 25 km, EPSG:3031): 65 rows x 67 columns.
ARCHIVE_GRID = PolarGrid(x=-3737500.0 + 25000.0 * np.arange(67), y=812500.0 + 25000.0 * np.arange(65),
                         resolution_m=25000.0)


@pytest.fixture(scope="module")
def resolver():
    return LocationResolver(ARCHIVE_GRID, schematic_land_mask(ARCHIVE_GRID))


# ------------------------------------------------------------------ presets
def test_presets_are_a_small_unique_list_inside_the_routing_grid():
    assert 5 <= len(PRESETS) <= 12
    assert len({p.id for p in PRESETS}) == len(PRESETS) and set(PRESETS_BY_ID) == {p.id for p in PRESETS}
    for p in PRESETS:
        assert p.name and p.region and p.id == p.id.lower()
        ARCHIVE_GRID.cell_of(p.lat, p.lon)          # raises if outside the grid
    o, d = CFG.route.origin, CFG.route.destination  # the configured route is offered as presets
    assert (PRESETS_BY_ID["drake_passage"].lat, PRESETS_BY_ID["drake_passage"].lon) == (o.lat, o.lon)
    assert (PRESETS_BY_ID["bransfield_strait"].lat, PRESETS_BY_ID["bransfield_strait"].lon) == (d.lat, d.lon)


def test_every_preset_resolves_to_a_navigable_cell(resolver):
    rows = resolver.presets()
    assert [r["id"] for r in rows] == [p.id for p in PRESETS]
    for r in rows:
        assert r["available"], r
        cell = (r["resolved"]["row"], r["resolved"]["col"])
        assert resolver.navigable[cell] and not resolver.land[cell]
        assert r["distance_km"] <= MAX_SNAP_KM


# ------------------------------------------------------------------ resolution and snapping
def _open_water_point(resolver):
    r, c = np.argwhere(resolver.navigable)[len(np.argwhere(resolver.navigable)) // 2]
    return ARCHIVE_GRID.cell_latlon(r, c), (int(r), int(c))


def test_a_point_in_a_navigable_cell_is_used_as_is(resolver):
    (lat, lon), cell = _open_water_point(resolver)
    res = resolver.resolve({"lat": lat + 0.01, "lon": lon})
    assert res.cell == cell and not res.snapped and res.distance_km < 25
    d = res.to_dict()
    assert d["requested"] == {"lat": lat + 0.01, "lon": lon} and (d["resolved"]["row"], d["resolved"]["col"]) == cell
    assert d["snapped"] is False and "navigable" in d["reason"]


def _coastal_land_cell(resolver):
    """A land cell with a navigable neighbour (so a point there is near the sea)."""
    land, nav = resolver.land, resolver.navigable
    for r, c in np.argwhere(land):
        if nav[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].any():
            return int(r), int(c)
    raise AssertionError("no coastal land cell")


def test_a_land_point_snaps_to_the_geodesically_nearest_navigable_cell(resolver):
    r, c = _coastal_land_cell(resolver)
    lat, lon = ARCHIVE_GRID.cell_latlon(r, c)
    res = resolver.resolve({"lat": lat, "lon": lon, "name": "On the coast"})
    assert res.snapped and res.cell != (r, c) and "land" in res.reason and res.name == "On the coast"
    assert not resolver.land[res.cell] and resolver.navigable[res.cell]
    rows, cols = np.nonzero(resolver.navigable)          # brute force: nothing navigable is closer
    d = GEOD.inv(np.full(rows.size, lon), np.full(rows.size, lat), ARCHIVE_GRID.lon2d[rows, cols],
                 ARCHIVE_GRID.lat2d[rows, cols])[2] / 1000.0
    assert math.isclose(res.distance_km, d.min(), rel_tol=1e-9)
    assert res.to_dict()["snapped"] is True


def test_snapping_never_lands_on_land_for_any_land_cell(resolver):
    for r, c in np.argwhere(resolver.land):
        lat, lon = ARCHIVE_GRID.cell_latlon(r, c)
        try:
            res = resolver.resolve({"lat": lat, "lon": lon})
        except LocationError as exc:            # inland: refused, never placed on land
            assert "nearest navigable" in str(exc)
            continue
        assert res.snapped and not resolver.land[res.cell] and resolver.navigable[res.cell]


def test_snapping_is_deterministic(resolver):
    r, c = _coastal_land_cell(resolver)
    lat, lon = ARCHIVE_GRID.cell_latlon(r, c)
    first = resolver.resolve({"lat": lat, "lon": lon})
    again = [LocationResolver(ARCHIVE_GRID, schematic_land_mask(ARCHIVE_GRID)).resolve({"lat": lat, "lon": lon})
             for _ in range(3)]
    assert all(a == first for a in again)
    by_id = resolver.resolve("rothera")
    assert by_id == resolver.resolve({"preset": "rothera"}) == resolver.resolve(PRESETS_BY_ID["rothera"])


def test_ocean_cut_off_from_the_open_sea_is_not_navigable():
    grid = PolarGrid(x=ARCHIVE_GRID.x[:7], y=ARCHIVE_GRID.y[:7], resolution_m=25000.0)
    land = np.zeros((7, 7), bool)
    land[2:5, 2:5] = True
    land[3, 3] = False                         # a one-cell "lake" enclosed by land
    nav = navigable_mask(land)
    assert not nav[3, 3] and nav.sum() == (~land).sum() - 1
    res = LocationResolver(grid, land).resolve({"lat": grid.lat2d[3, 3], "lon": grid.lon2d[3, 3]})
    assert res.snapped and res.cell != (3, 3) and nav[res.cell] and "cut off" in res.reason


def test_points_far_inland_are_refused_not_moved(resolver):
    grid = PolarGrid(x=ARCHIVE_GRID.x[:12], y=ARCHIVE_GRID.y[:12], resolution_m=25000.0)
    land = np.ones((12, 12), bool)
    land[:, :2] = False                        # sea only along one edge
    res = LocationResolver(grid, land)
    lat, lon = grid.cell_latlon(6, 11)         # nine cells (225 km) from the sea
    with pytest.raises(LocationError, match="nearest navigable routing cell"):
        res.resolve({"lat": lat, "lon": lon})
    assert res.resolve({"lat": grid.lat2d[6, 2], "lon": grid.lon2d[6, 2]}).snapped    # a coastal one is moved


@pytest.mark.parametrize("spec, message", [
    ({"lat": -75.0, "lon": -60.0}, "outside the routing grid"),
    ({"lat": -60.0, "lon": 10.0}, "outside the routing grid"),
    ({"lat": float("nan"), "lon": -60.0}, "not a valid latitude"),
    ({"lat": -60.0}, "both 'lat' and 'lon'"),
    ({"preset": "mcmurdo"}, "unknown preset"),
    ({"preset": "rothera", "lat": -60.0, "lon": -60.0}, "not both"),
    (42, "preset id or an object"),
])
def test_invalid_and_out_of_domain_locations_are_refused(resolver, spec, message):
    with pytest.raises(LocationError, match=message):
        resolver.resolve(spec)


# ------------------------------------------------------------------ route horizon
def test_configured_route_horizon_is_unchanged():
    o, d = CFG.route.origin, CFG.route.destination
    assert _horizon_days(CFG) == route_horizon_days(o.lat, o.lon, d.lat, d.lon, SPEED, STEP, LEADS) == 6


def test_route_horizon_rule_and_limits():
    hz = route_horizon(100.0, 20.0, 24.0, 21)
    assert hz.planning_hours == pytest.approx(15.0) and hz.required_days == 2 and hz.horizon_days == 2
    assert route_horizon(0.0, 20.0, 24.0, 21).horizon_days == 1
    long = route_horizon(10_000.0, 20.0, 24.0, 21)
    assert long.required_days == 64 and long.horizon_days == 21 and not long.supported
    assert route_horizon(1000.0, 20.0, 24.0, 21).horizon_days < route_horizon(2000.0, 20.0, 24.0, 21).horizon_days


def test_build_route_uses_the_resolved_cells_and_their_distance(resolver):
    spec = build_route(resolver, "drake_passage", "rothera", SPEED, STEP, LEADS)
    o, d = resolver.resolve("drake_passage"), resolver.resolve("rothera")
    assert spec.cells == (o.cell, d.cell)
    km = ARCHIVE_GRID.geodesic_distance_m(*o.cell, *d.cell) / 1000.0
    assert spec.horizon == route_horizon(km, SPEED, STEP, LEADS)
    assert spec.horizon_days > build_route(resolver, "drake_passage", "bransfield_strait", SPEED, STEP,
                                           LEADS).horizon_days
    assert spec.scenario_days() == 1 + spec.horizon_days and spec.scenario_days(14) == 14 + spec.horizon_days
    out = spec.to_dict()
    assert set(out) == {"origin", "destination", "horizon"} and out["horizon"]["supported"]


def test_build_route_refuses_routes_beyond_the_model_horizon_or_with_one_cell(resolver):
    with pytest.raises(LocationError, match="forecast horizon"):
        build_route(resolver, "drake_passage", "signy_island", SPEED, STEP, 3)
    with pytest.raises(LocationError, match="same routing cell"):
        build_route(resolver, "drake_passage", {"lat": -56.31, "lon": -66.0}, SPEED, STEP, LEADS)


# ------------------------------------------------------------------ API (synthetic stand-in archive)
HEADER = "Iceberg,Length (NM),Width (NM),Latitude,Longitude,Remarks,Last Update\n"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """The real endpoints over a small synthetic stand-in archive (the real one is kept outside Git)."""
    from antarctic_routing.common.provenance import sha256_file

    tmp = tmp_path_factory.mktemp("loc")
    grid = PolarGrid.from_domain(CFG.domain, 25)
    ds = synthetic_history(CFG, grid, range(2021, 2023), seed=3)
    t = ds["time"].values.astype("datetime64[D]")
    shape = ds["land_mask"].shape
    zeros = np.zeros((t.size, *shape))
    lists = []
    for d in (date(2021, 11, 15), date(2022, 2, 15)):
        p = tmp / f"AntarcticIcebergs_{d:%Y%m%d}.csv"
        p.write_text(HEADER + f"A23A,20,5,-60.0,-60.0,,{d:%m/%d/%Y}\n")
        lists.append({"path": p.name, "sha256": sha256_file(p), "date": d.isoformat()})
    archive = HistoricalArchive(json.loads((REPO / "config" / "real_historical.json").read_text()), tmp, ds,
                                DailyForcing("winds", "ERA5", t, zeros, zeros, ()),
                                DailyForcing("currents", "CMEMS", t, zeros, zeros, ()),
                                UsnicArchive(lists, tmp), CFG.project.season_months)
    mp = pytest.MonkeyPatch()
    mp.setenv("ANTROUTE_DATA_ROOT", str(tmp))
    mp.setattr(hist_api.HistoricalArchive, "load", classmethod(lambda cls, *a, **k: archive))
    with TestClient(create_app(CONFIG)) as c:
        c.grid, c.land = grid, ds["land_mask"].values.astype(bool)
        yield c
    mp.undo()


def _sea(c, k=0):
    r, col = np.argwhere(navigable_mask(c.land))[k]
    return {"lat": float(c.grid.lat2d[r, col]), "lon": float(c.grid.lon2d[r, col])}


def test_api_lists_presets_with_their_resolution(client):
    body = client.get("/real/locations").json()
    assert [p["id"] for p in body["presets"]] == [p.id for p in PRESETS]
    assert body["max_snap_km"] == MAX_SNAP_KM and "never placed on land" in body["snapping"]
    assert body["grid"]["resolution_km"] == 25.0 and body["data_label"] == "Real Historical Data"
    for p in body["presets"]:
        assert {"requested", "resolved", "snapped", "available", "reason"} <= set(p)


def test_api_resolves_a_pair_with_horizon_scenario_days_and_coverage(client):
    land_cell = np.argwhere(client.land & np.roll(navigable_mask(client.land), 1, axis=1))[0]
    on_land = {"lat": float(client.grid.lat2d[tuple(land_cell)]), "lon": float(client.grid.lon2d[tuple(land_cell)]),
               "name": "Coast"}
    body = client.post("/real/locations/resolve", json={"origin": _sea(client), "destination": on_land,
                                                        "issue": "2021-11-20", "window_days": 5}).json()
    assert body["origin"]["snapped"] is False and body["destination"]["snapped"] is True
    assert body["destination"]["requested"] == {"lat": on_land["lat"], "lon": on_land["lon"]}
    h = body["horizon"]["horizon_days"]
    assert body["scenario_days"] == {"route": 1 + h, "window": 5 + h}
    assert body["coverage"]["route"] == {"ok": True, "problems": []}
    late = client.post("/real/locations/resolve", json={"origin": _sea(client), "destination": on_land,
                                                        "issue": "2022-02-27"}).json()
    assert late["coverage"]["route"]["ok"] is False and "end of the season" in late["coverage"]["route"]["problems"][0]


def test_api_refuses_invalid_locations_with_a_reason(client):
    cases = [{"origin": {"preset": "nowhere"}, "destination": _sea(client)},
             {"origin": {"lat": -75.0, "lon": -60.0}, "destination": _sea(client)},
             {"origin": _sea(client), "destination": _sea(client)}]
    for body in cases:
        r = client.post("/real/locations/resolve", json=body)
        assert r.status_code == 422 and r.json()["detail"]["status"] == "invalid_location", body
    r = client.post("/real/historical/routes?wait=true", json={"issue": "2021-11-20", "origin": _sea(client)})
    assert r.status_code == 422 and "both origin and destination" in r.json()["detail"]["reason"]


def test_api_refuses_dates_the_route_horizon_cannot_cover(client):
    body = {"issue": "2022-02-25", "origin": _sea(client), "destination": _sea(client, 40)}
    for path in ("/real/historical/routes?wait=true", "/real/historical/departures?wait=true",
                 "/real/historical/voyages"):
        r = client.post(path, json=body)
        assert r.status_code == 422 and r.json()["detail"]["status"] == "out_of_coverage", path
    r = client.post("/real/historical/replay?wait=true", json={"start": "2022-02-20", "origin": _sea(client),
                                                               "destination": _sea(client, 40)})
    assert r.status_code == 422


def test_api_dates_follow_the_route_horizon(client):
    default = client.get("/real/historical/dates").json()
    assert "route" not in default and default["horizon_days"] == _horizon_days(CFG)
    far = client.get("/real/historical/dates?origin=drake_passage&destination=palmer_station").json()
    assert far["horizon_days"] == far["route"]["horizon"]["horizon_days"] > default["horizon_days"]
    assert len(far["route_dates"]) < len(default["route_dates"])
    assert set(far["route_dates"]) <= set(default["route_dates"])
    out = client.get("/real/historical/dates?origin=drake_passage&destination=mcmurdo")
    assert out.status_code == 422 and out.json()["detail"]["status"] == "invalid_location"


def test_api_serves_the_pick_map_grid_land_and_cell_positions(client):
    m = client.get("/real/locations").json()["map"]
    ny, nx = client.grid.shape
    assert (m["nx"], m["ny"], m["crs"]) == (nx, ny, "EPSG:3031")
    assert m["land"] == client.land.astype(int).ravel().tolist()
    assert m["navigable"] == navigable_mask(client.land).astype(int).ravel().tolist()
    k = 7 * nx + 11
    assert (m["lat"][k], m["lon"][k]) == (pytest.approx(client.grid.lat2d[7, 11], abs=1e-4),
                                          pytest.approx(client.grid.lon2d[7, 11], abs=1e-4))
    assert not any(n and l for n, l in zip(m["navigable"], m["land"], strict=True))        # land is never navigable


def test_api_dates_accept_points_resolved_with_the_same_snapping(client):
    land_cell = np.argwhere(client.land & np.roll(navigable_mask(client.land), 1, axis=1))[0]
    on_land = {"lat": float(client.grid.lat2d[tuple(land_cell)]), "lon": float(client.grid.lon2d[tuple(land_cell)])}
    sea = _sea(client)
    q = (f"origin_lat={sea['lat']}&origin_lon={sea['lon']}"
         f"&destination_lat={on_land['lat']}&destination_lon={on_land['lon']}")
    body = client.get(f"/real/historical/dates?{q}").json()
    resolved = client.post("/real/locations/resolve", json={"origin": sea, "destination": on_land}).json()
    assert body["route"]["origin"] == resolved["origin"] and body["route"]["destination"] == resolved["destination"]
    assert body["route"]["destination"]["snapped"] is True
    assert not client.land[tuple(body["route"]["destination"]["cell"])]                  # never a land endpoint
    assert body["horizon_days"] == resolved["horizon"]["horizon_days"]
    mixed = client.get(f"/real/historical/dates?origin=drake_passage&destination_lat={sea['lat']}"
                       f"&destination_lon={sea['lon']}").json()
    assert mixed["route"]["origin"]["id"] == "drake_passage" and mixed["route"]["destination"]["id"] is None
    for bad in ("origin_lat=-40&origin_lon=-60&destination=drake_passage",           # outside the grid
                "origin_lat=-62&destination=drake_passage"):                         # half a point
        r = client.get(f"/real/historical/dates?{bad}")
        assert r.status_code == 422 and r.json()["detail"]["status"] == "invalid_location", bad
    assert "outside the routing grid" in client.get(
        "/real/historical/dates?origin_lat=-40&origin_lon=-60&destination=drake_passage").json()["detail"]["reason"]
