"""CMEMS subset box: must cover the whole EPSG:3031 routing grid, not just the config lat/lon box."""

import json
import sys
import types
from datetime import date
from pathlib import Path

import pytest

from antarctic_routing.cli import main
from antarctic_routing.config import load_config
from antarctic_routing.ingestion.cmems import (
    AREA_MARGIN_DEG,
    REANALYSIS,
    cmems_bbox,
    cmems_subset_kwargs,
    make_fetcher,
    request_for,
)
from antarctic_routing.ingestion.era5 import grid_latlon_extent
from antarctic_routing.preprocessing.grid import PolarGrid
from test_cli import CONFIG
from test_era5_area import corner_latlon

DOMAIN = load_config(CONFIG).domain


@pytest.mark.parametrize("res_km", [10, 25, 50])
def test_bbox_contains_every_cell_centre_and_corner_with_margin(res_km):
    grid = PolarGrid.from_domain(DOMAIN, res_km)
    lat_min, lat_max, lon_min, lon_max = cmems_bbox(grid)
    for lat, lon in ((grid.lat2d, grid.lon2d), corner_latlon(grid)):
        assert lat.min() - lat_min >= AREA_MARGIN_DEG - 1e-9
        assert lat_max - lat.max() >= AREA_MARGIN_DEG - 1e-9
        assert lon.min() - lon_min >= AREA_MARGIN_DEG - 1e-9
        assert lon_max - lon.max() >= AREA_MARGIN_DEG - 1e-9


def test_bbox_is_wider_than_the_configured_box_and_snapped():
    grid = PolarGrid.from_domain(DOMAIN, 25)
    lat_min, lat_max, lon_min, lon_max = cmems_bbox(grid)
    assert lat_min < DOMAIN.lat_min and lat_max > DOMAIN.lat_max
    assert lon_min < DOMAIN.lon_min and lon_max > DOMAIN.lon_max
    for v in (lat_min, lat_max, lon_min, lon_max):
        assert v / 0.25 == pytest.approx(round(v / 0.25))
    g = grid_latlon_extent(grid)
    assert g[0] < DOMAIN.lat_min and g[3] > DOMAIN.lon_max   # why the config box alone is not enough
    with pytest.raises(ValueError, match="margin"):
        cmems_bbox(grid, margin_deg=-1)


def test_subset_kwargs_use_the_bbox_and_only_surface_uo_vo():
    grid = PolarGrid.from_domain(DOMAIN, 25)
    bbox = cmems_bbox(grid)
    kw = cmems_subset_kwargs(DOMAIN, date(2023, 11, 1), date(2023, 12, 31), "c.nc", bbox=bbox)
    assert (kw["minimum_latitude"], kw["maximum_latitude"], kw["minimum_longitude"], kw["maximum_longitude"]) == bbox
    assert kw["variables"] == ["uo", "vo"] and kw["dataset_id"] == REANALYSIS
    assert kw["minimum_depth"] == 0.0 and kw["maximum_depth"] == 1.0
    assert kw["start_datetime"] == "2023-11-01T00:00:00" and kw["end_datetime"] == "2023-12-31T23:59:59"
    req = request_for(DOMAIN, date(2023, 11, 1), date(2023, 12, 31), grid=grid)
    assert req.bbox == bbox and req.variables == ("uo", "vo")
    assert req.key() != request_for(DOMAIN, date(2023, 11, 1), date(2023, 12, 31)).key()


class _FakeToolbox:
    calls: list = []

    @classmethod
    def subset(cls, **kw):
        cls.calls.append(kw)
        Path(kw["output_directory"], kw["output_filename"]).write_bytes(b"not-real-netcdf")
        return types.SimpleNamespace(model_dump=lambda **_: {"status": "000", "file_size": 0.0})


@pytest.fixture
def fake_toolbox(monkeypatch):
    _FakeToolbox.calls = []
    monkeypatch.setitem(sys.modules, "copernicusmarine", _FakeToolbox)
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "test-only")
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "test-only")
    return _FakeToolbox.calls


def test_fetcher_sends_the_grid_bbox_and_moves_the_file_into_place(fake_toolbox, tmp_path):
    grid = PolarGrid.from_domain(DOMAIN, 25)
    dest = tmp_path / "c.nc.part"
    meta = make_fetcher(DOMAIN, grid=grid, margin_deg=1.0)(
        request_for(DOMAIN, date(2023, 11, 1), date(2023, 11, 2), grid=grid, margin_deg=1.0), dest)
    (kw,) = fake_toolbox
    assert kw["output_filename"].endswith(".nc")
    assert (kw["minimum_latitude"], kw["maximum_latitude"],
            kw["minimum_longitude"], kw["maximum_longitude"]) == cmems_bbox(grid, 1.0)
    assert dest.is_file() and not list(tmp_path.glob("*.part.nc"))
    assert meta["product_kind"] == "reanalysis" and meta["subset_request"]["variables"] == ["uo", "vo"]
    assert meta["subset_response"]["status"] == "000"
    assert "password" not in json.dumps(meta).lower() and "test-only" not in json.dumps(meta)


def test_fetch_forcing_cli_requests_full_grid_coverage(fake_toolbox, tmp_path):
    rc = main(["fetch-forcing", "--config", CONFIG, "--start", "2023-11-01", "--end", "2023-11-02",
               "--root", str(tmp_path / "raw"), "--skip-era5", "--resolution-km", "25",
               "--cmems-margin-deg", "0.75"])
    assert rc == 0
    (kw,) = fake_toolbox
    grid = PolarGrid.from_domain(DOMAIN, 25)
    south, north, west, east = cmems_bbox(grid, 0.75)
    assert (kw["minimum_latitude"], kw["maximum_latitude"], kw["minimum_longitude"],
            kw["maximum_longitude"]) == (south, north, west, east)
    lat, lon = corner_latlon(grid)
    assert south <= lat.min() and lat.max() <= north and west <= lon.min() and lon.max() <= east
    manifest = json.loads(next((tmp_path / "raw").rglob("*.manifest.json")).read_text())
    assert manifest["request"]["bbox"] == [south, north, west, east]


def test_fetcher_without_credentials_is_blocked(fake_toolbox, monkeypatch, tmp_path):
    from antarctic_routing.ingestion.base import MissingCredentials

    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_USERNAME")
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(MissingCredentials):
        make_fetcher(DOMAIN)(request_for(DOMAIN, date(2023, 11, 1), date(2023, 11, 2)), tmp_path / "c.nc")
    assert fake_toolbox == []


def test_build_forcing_cli_daily_currents(tmp_path):
    import xarray as xr

    from test_forcing import write_daily_cmems

    cm = write_daily_cmems(tmp_path / "c.nc", [0.1, 0.2], [0.0, 0.1])
    out = tmp_path / "f.nc"
    assert main(["build-forcing", "--config", CONFIG, "--cmems", str(cm), "--current-time-mode", "daily",
                 "--out", str(out)]) == 0
    stage = json.loads(out.with_suffix(".stage-result.json").read_text())
    assert stage["parameters"]["current_time_mode"] == "daily" and "wind_time_mode" not in stage["parameters"]
    with xr.open_dataset(out) as ds:
        assert ds["current_x"].dims == ("time", "y", "x") and ds.attrs["current_time_mode"] == "daily"
