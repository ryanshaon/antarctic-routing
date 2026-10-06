"""Real Historical Data mode: input catalogue, coverage rules and the API's no-fallback behaviour.

These tests need no real data; ``test_historical_real.py`` runs the real archive when it is present.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from antarctic_routing.api.main import create_app
from antarctic_routing.common.provenance import sha256_file
from antarctic_routing.config import load_config
from antarctic_routing.historical import (
    LABEL,
    DailyForcing,
    HistoricalArchive,
    HistoricalUnavailable,
    OutOfCoverage,
    UsnicArchive,
    season_label,
)
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.synthetic import synthetic_history

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "config.yaml"
SPEC = REPO / "config" / "real_historical.json"
FROZEN = REPO / "docs" / "frozen_demo" / "frozen_demo_2023-11-14.json"
CFG = load_config(CONFIG)
SHAPE = (3, 4)


def _forcing(path: Path, kind: str, start: str, n: int, value: float = 1.0) -> Path:
    t = np.datetime64(start) + np.arange(n)
    field = np.full((n, *SHAPE), value) + np.arange(n)[:, None, None]
    name = "wind" if kind == "winds" else "current"
    ds = xr.Dataset({f"{name}_x": (("time", "y", "x"), field), f"{name}_y": (("time", "y", "x"), -field)},
                    coords={"time": t, "y": np.arange(SHAPE[0]), "x": np.arange(SHAPE[1])},
                    attrs={f"{name}_time_mode": "daily"})
    ds.to_netcdf(path)
    return path


def _rec(path: Path, root: Path) -> dict:
    return {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}


# ------------------------------------------------------------------ forcing catalogue
def test_daily_forcing_joins_files_by_exact_date(tmp_path):
    a = _forcing(tmp_path / "b.nc", "winds", "2023-11-01", 3, 10.0)
    b = _forcing(tmp_path / "a.nc", "winds", "2023-11-04", 2, 20.0)
    f = DailyForcing.load([_rec(a, tmp_path), _rec(b, tmp_path)], tmp_path, "winds", SHAPE)
    assert [str(d) for d in f.times] == ["2023-11-01", "2023-11-02", "2023-11-03", "2023-11-04", "2023-11-05"]
    assert f.x[:, 0, 0].tolist() == [10.0, 11.0, 12.0, 20.0, 21.0] and (f.y == -f.x).all()
    assert f.missing(date(2023, 11, 4), 3) == ["2023-11-06"]
    assert f.missing(date(2023, 11, 1), 5) == []
    assert [x["file"] for x in f.files_for(date(2023, 11, 3), 2)] == ["b.nc", "a.nc"]
    assert [x["file"] for x in f.files_for(date(2023, 11, 4), 2)] == ["a.nc"]


def test_daily_forcing_refuses_overlapping_dates_and_missing_fields(tmp_path):
    a = _forcing(tmp_path / "a.nc", "winds", "2023-11-01", 3)
    b = _forcing(tmp_path / "b.nc", "winds", "2023-11-03", 2)
    with pytest.raises(ValueError, match="overlap"):
        DailyForcing.load([_rec(a, tmp_path), _rec(b, tmp_path)], tmp_path, "winds", SHAPE)
    with pytest.raises(ValueError, match="no daily currents"):
        DailyForcing.load([_rec(a, tmp_path)], tmp_path, "currents", SHAPE)
    with pytest.raises(ValueError, match="does not match"):
        DailyForcing.load([_rec(a, tmp_path)], tmp_path, "winds", (5, 5))


# ------------------------------------------------------------------ USNIC
HEADER = "Iceberg,Length (NM),Width (NM),Latitude,Longitude,Remarks,Last Update\n"


def _usnic(path: Path, rows: list[tuple]) -> Path:
    path.write_text(HEADER + "".join(f"{i},20,5,{lat},{lon},,{d}\n" for i, lat, lon, d in rows))
    return path


def _usnic_archive(tmp_path) -> UsnicArchive:
    a = _usnic(tmp_path / "AntarcticIcebergs_20231103.csv", [("A23A", -60.0, -60.0, "11/03/2023")])
    b = _usnic(tmp_path / "AntarcticIcebergs_20231110.csv", [("A23A", -61.0, -61.0, "11/10/2023"),
                                                            ("B09", -70.0, 100.0, "11/10/2023"),
                                                            ("C99", -62.0, -62.0, "11/20/2023")])
    return UsnicArchive([{**_rec(a, tmp_path), "date": "2023-11-03"}, {**_rec(b, tmp_path), "date": "2023-11-10"}],
                        tmp_path)


def test_usnic_uses_the_latest_list_on_or_before_the_day(tmp_path):
    usnic = _usnic_archive(tmp_path)
    assert usnic.list_for(date(2023, 11, 9), 14)[0] == date(2023, 11, 3)
    assert usnic.list_for(date(2023, 11, 10), 14)[0] == date(2023, 11, 10)
    with pytest.raises(OutOfCoverage, match="no USNIC iceberg list on or before"):
        usnic.list_for(date(2023, 11, 1), 14)
    with pytest.raises(OutOfCoverage, match="20 days old"):
        usnic.list_for(date(2023, 11, 30), 14)


def test_usnic_positions_never_use_reports_dated_after_the_day(tmp_path):
    usnic = _usnic_archive(tmp_path)
    _, _, _, bergs = usnic.positions(date(2023, 11, 12), 14)
    assert [b[0] for b in bergs] == ["A23A", "B09"]          # C99 is dated 2023-11-20
    grid = PolarGrid.from_domain(CFG.domain, 25)
    snap = usnic.snapshot(date(2023, 11, 12), grid, 14)
    assert snap.list_date == date(2023, 11, 10) and snap.age_days == 2
    assert [b[0] for b in snap.bergs] == ["A23A"] and snap.outside_grid == ["B09"]
    assert snap.summary()["drifted"] == ["A23A"]


# ------------------------------------------------------------------ coverage rules
@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hist")
    grid = PolarGrid.from_domain(CFG.domain, 100)
    ds = synthetic_history(CFG, grid, range(2020, 2023), seed=1)
    t = ds["time"].values.astype("datetime64[D]")                          # forcing has every day
    ds = ds.isel(time=np.flatnonzero(t != np.datetime64("2021-11-20")))        # the sea ice has a gap
    shape = ds["land_mask"].shape
    winds = DailyForcing("winds", "ERA5", t, np.zeros((t.size, *shape)), np.zeros((t.size, *shape)), ())
    cur_t = t[t < np.datetime64("2023-02-20")]                              # currents end early
    currents = DailyForcing("currents", "CMEMS", cur_t, np.zeros((cur_t.size, *shape)),
                            np.zeros((cur_t.size, *shape)), ())
    lists = [_usnic(tmp / f"AntarcticIcebergs_{d:%Y%m%d}.csv", [("A23A", -60.0, -60.0, f"{d:%m/%d/%Y}")])
             for d in (date(2020, 11, 1), date(2021, 11, 1), date(2021, 11, 30), date(2022, 11, 1),
                       date(2022, 12, 1), date(2023, 2, 10))]
    usnic = UsnicArchive([{**_rec(p, tmp), "date": f"{p.stem[-8:-4]}-{p.stem[-4:-2]}-{p.stem[-2:]}"} for p in lists],
                         tmp)
    spec = json.loads(SPEC.read_text())
    return HistoricalArchive(spec, tmp, ds, winds, currents, usnic, CFG.project.season_months)


def test_problems_explain_every_unsupported_issue_date(archive):
    assert archive.problems(date(2021, 11, 14), 7) == []
    assert "not in the sea-ice archive" in archive.problems(date(2021, 3, 1), 7)[0]
    assert "14 contiguous in-season days" in archive.problems(date(2021, 11, 13), 7)[0]
    assert "14 contiguous" in archive.problems(date(2021, 11, 25), 7)[0]       # history crosses the 11-20 gap
    assert archive.problems(date(2021, 12, 4), 7) == []
    assert any("past the end of the season" in p for p in archive.problems(date(2022, 2, 25), 7))
    assert any("CMEMS currents are missing" in p for p in archive.problems(date(2023, 2, 15), 7))
    assert any("days old" in p for p in archive.problems(date(2021, 12, 20), 7))   # USNIC list too old
    with pytest.raises(OutOfCoverage, match="no Real Historical Data run for 2021-11-13"):
        archive.check(date(2021, 11, 13), 7)


def test_available_dates_satisfy_every_rule(archive):
    dates = archive.available(7)
    assert dates and all(not archive.problems(d, 7) for d in dates)
    assert date(2021, 11, 14) in dates and date(2021, 11, 25) not in dates
    longer = archive.available(20)
    assert set(longer) <= set(dates) and len(longer) < len(dates)


def test_season_flags_name_the_components_fitted_on_a_season(archive):
    assert archive.season_info(2018)["in_sample_notes"] == [
        "the sea-ice U-Net used this season for validation (model selection)",
        "the iceberg drift and spread calibration used this season for training"]
    s21 = archive.season_info(2021)
    assert (s21["forecast_model"], s21["iceberg_drift"], s21["out_of_sample"]) == ("test", "validation", False)
    s23 = archive.season_info(2023)
    assert s23["out_of_sample"] and s23["in_sample_notes"] == [] and s23["season"] == "2023-24"
    assert season_label(2009) == "2009-10"
    cov = archive.coverage(7)
    assert [c["season"] for c in cov] == ["2020-21", "2021-22", "2022-23"]
    assert sum(c["n_dates"] for c in cov) == len(archive.available(7))


# ------------------------------------------------------------------ load-time verification
def _mini_spec(tmp_path, *, beta=0.1) -> tuple[Path, Path]:
    root = tmp_path / "data"
    root.mkdir()
    spec = json.loads(SPEC.read_text())
    files = {"sea_ice": "ice.nc", "checkpoint": "best.pt", "drift_selection": "sel.json",
             "spread_calibration": "spread.json"}
    (root / "ice.nc").write_bytes(b"ice")
    (root / "best.pt").write_bytes(b"model")
    (root / "sel.json").write_text(json.dumps({"selected_params": {"beta": beta, "alpha_scale": 0.1}}))
    (root / "spread.json").write_text(json.dumps({"selected_model": {"c": 0.6053, "q": 0.0}}))
    for name, f in files.items():
        spec["inputs"][name] = _rec(root / f, root)
    for name in ("sea_ice_additional", "wind_forcing", "current_forcing", "icebergs"):
        spec["inputs"][name] = []
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return root, path


def test_load_blocks_on_a_missing_input_and_fails_on_a_changed_one(tmp_path):
    root, spec = _mini_spec(tmp_path)
    (root / "best.pt").unlink()
    with pytest.raises(HistoricalUnavailable, match="checkpoint not found") as e:
        HistoricalArchive.load(root, spec, [11, 12, 1, 2])
    assert e.value.status == "blocked"
    (root / "best.pt").write_bytes(b"other model")
    with pytest.raises(HistoricalUnavailable, match="checkpoint .* has SHA-256") as e:
        HistoricalArchive.load(root, spec, [11, 12, 1, 2])
    assert e.value.status == "failed"
    with pytest.raises(HistoricalUnavailable, match="does not exist"):
        HistoricalArchive.load(tmp_path / "nowhere", spec, [11, 12, 1, 2])


def test_load_fails_when_drift_parameters_differ_from_the_calibration(tmp_path):
    root, spec = _mini_spec(tmp_path, beta=0.2)
    with pytest.raises(HistoricalUnavailable, match="drift selection") as e:
        HistoricalArchive.load(root, spec, [11, 12, 1, 2])
    assert e.value.status == "failed"


def test_pinned_spec_keeps_the_frozen_parameters_and_discloses_hindsight_forcing():
    spec, frozen = json.loads(SPEC.read_text()), json.loads(FROZEN.read_text())
    assert spec["label"] == LABEL == "Real Historical Data"
    assert {k: spec["parameters"][k] for k in frozen["parameters"]} == frozen["parameters"]
    assert spec["parameters"]["seed"] == 42 and spec["parameters"]["window_days"] == 14
    for name in ("sea_ice", "checkpoint", "drift_selection", "spread_calibration"):
        assert spec["inputs"][name] == {k: frozen["inputs"][name][k] for k in ("path", "sha256")}
    assert "reanalysis" in spec["hindsight_forcing"] and "only known afterwards" in spec["hindsight_forcing"]
    assert spec["splits"]["forecast_model"]["train"] == list(range(2004, 2018))
    assert spec["splits"]["iceberg_drift"] == {**spec["splits"]["iceberg_drift"], "train": [2018, 2019, 2020],
                                               "validation": [2021], "test": [2022, 2023]}
    wind = spec["inputs"]["wind_forcing"]
    assert all(a["last"] < b["first"] for a, b in zip(wind, wind[1:], strict=False))       # no overlapping files


# ------------------------------------------------------------------ API without an archive
@pytest.fixture()
def no_archive(monkeypatch):
    monkeypatch.delenv("ANTROUTE_DATA_ROOT", raising=False)
    with TestClient(create_app(CONFIG)) as c:
        yield c


def test_api_reports_unavailable_and_never_substitutes(no_archive):
    c = no_archive
    st = c.get("/real/historical/status").json()
    assert st["status"] == "unavailable" and "ANTROUTE_DATA_ROOT" in st["reason"] and st["label"] == LABEL
    assert c.get("/status").json()["historical"]["status"] == "unavailable"
    calls = [("get", "/real/historical/dates", None),
             ("post", "/real/historical/routes?wait=true", {"issue": "2023-11-14"}),
             ("post", "/real/historical/departures?wait=true", {"issue": "2023-11-14"}),
             ("post", "/real/historical/voyages", {"issue": "2023-11-14"}),
             ("post", "/real/historical/replay?wait=true", {"start": "2023-11-14"})]
    for method, path, body in calls:
        r = getattr(c, method)(path, **({"json": body} if body else {}))
        assert r.status_code == 503, path
        assert r.json()["detail"]["status"] == "unavailable"


def test_api_blocks_when_the_archive_files_are_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTROUTE_DATA_ROOT", str(tmp_path))
    with TestClient(create_app(CONFIG)) as c:
        st = c.get("/real/historical/status").json()
        assert st["status"] == "blocked" and "not found" in st["reason"]
        assert c.post("/real/historical/routes?wait=true", json={"issue": "2023-11-14"}).status_code == 503


def test_window_length_is_capped_at_the_frozen_14_days(no_archive):
    r = no_archive.post("/real/historical/departures?wait=true", json={"issue": "2023-11-14", "window_days": 15})
    assert r.status_code == 422


def test_real_and_synthetic_voyages_are_never_replanned_on_the_other_data(no_archive):
    c = no_archive
    svc = c.app.state.service
    svc.voyages["real1"] = {"mode": "historical", "events": []}
    r = c.post("/voyages/real1/replan", json={"lat": -60, "lon": -60, "issued": "2023-11-15"})
    assert r.status_code == 409 and "Real Historical Data" in r.json()["detail"]
    v = c.post("/voyages", json={"departure": "2027-01-10", "scenarios": 80, "resolution_km": 25,
                                 "scenario_routes": 0})
    assert v.status_code == 201
    r = c.post(f"/real/historical/voyages/{v.json()['voyage_id']}/replan",
               json={"lat": -60, "lon": -60, "issued": "2023-11-15"})
    assert r.status_code == 409 and "controlled-synthetic" in r.json()["detail"]
