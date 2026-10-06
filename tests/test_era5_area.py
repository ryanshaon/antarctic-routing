"""ERA5 request area: must cover the whole EPSG:3031 routing grid, not just the config lat/lon box."""

import json
import sys
import types
from datetime import date

import numpy as np
import pytest

from antarctic_routing.cli import main
from antarctic_routing.config import DomainSection, load_config
from antarctic_routing.ingestion.era5 import (
    AREA_MARGIN_DEG,
    era5_area,
    era5_request,
    grid_latlon_extent,
    make_fetcher,
    request_for,
)
from antarctic_routing.preprocessing.grid import PolarGrid
from test_cli import CONFIG

DOMAIN = load_config(CONFIG).domain


def corner_latlon(grid: PolarGrid):
    """Lat/lon of every cell corner (an independent lattice of grid-edge points)."""
    half = grid.resolution_m / 2
    corners = PolarGrid(x=np.r_[grid.x - half, grid.x[-1] + half],
                        y=np.r_[grid.y - half, grid.y[-1] + half], resolution_m=grid.resolution_m)
    return corners.lat2d, corners.lon2d


@pytest.mark.parametrize("res_km", [10, 25, 50])
def test_area_contains_every_cell_centre_and_corner_with_margin(res_km):
    grid = PolarGrid.from_domain(DOMAIN, res_km)
    north, west, south, east = era5_area(grid)
    for lat, lon in ((grid.lat2d, grid.lon2d), corner_latlon(grid)):
        assert lat.min() - south >= AREA_MARGIN_DEG - 1e-9
        assert north - lat.max() >= AREA_MARGIN_DEG - 1e-9
        assert lon.min() - west >= AREA_MARGIN_DEG - 1e-9
        assert east - lon.max() >= AREA_MARGIN_DEG - 1e-9


def test_area_is_wider_than_the_configured_box_and_on_the_era5_lattice():
    grid = PolarGrid.from_domain(DOMAIN, 25)
    north, west, south, east = era5_area(grid)
    assert south < DOMAIN.lat_min and north > DOMAIN.lat_max
    assert west < DOMAIN.lon_min and east > DOMAIN.lon_max
    for v in (north, west, south, east):
        assert v / 0.25 == pytest.approx(round(v / 0.25))
    # the grid's own extent reaches outside the box, which is why the box alone is not enough
    lat_min, lat_max, lon_min, lon_max = grid_latlon_extent(grid)
    assert lat_min < DOMAIN.lat_min and lon_max > DOMAIN.lon_max


def test_margin_is_configurable():
    grid = PolarGrid.from_domain(DOMAIN, 25)
    tight, wide = era5_area(grid, margin_deg=0.0), era5_area(grid, margin_deg=2.0)
    assert wide[0] >= tight[0] + 2.0 - 0.25 and wide[2] <= tight[2] - 2.0 + 0.25
    assert wide[1] <= tight[1] - 2.0 + 0.25 and wide[3] >= tight[3] + 2.0 - 0.25
    with pytest.raises(ValueError, match="margin"):
        era5_area(grid, margin_deg=-0.1)


def test_grid_containing_the_pole_is_rejected():
    grid = PolarGrid(x=np.array([-50e3, 50e3]), y=np.array([-50e3, 50e3]), resolution_m=100e3)
    with pytest.raises(ValueError, match="Pole"):
        era5_area(grid)


def test_request_body_and_cache_key_use_the_grid_area():
    grid = PolarGrid.from_domain(DOMAIN, 25)
    area = era5_area(grid)
    body = era5_request(DOMAIN, date(2023, 11, 3), date(2023, 11, 24), area=area)
    assert body["area"] == area
    req = request_for(DOMAIN, date(2023, 11, 3), date(2023, 11, 24), grid)
    assert req.bbox == (area[2], area[0], area[1], area[3])   # lat_min, lat_max, lon_min, lon_max
    assert req.key() != request_for(DOMAIN, date(2023, 11, 3), date(2023, 11, 24)).key()
    # configured domain itself is untouched and still the default for a bare request
    assert era5_request(DOMAIN, date(2023, 11, 3), date(2023, 11, 4))["area"] == [-55.0, -72.0, -66.0, -52.0]
    assert DOMAIN == DomainSection(lat_min=-66.0, lat_max=-55.0, lon_min=-72.0, lon_max=-52.0)


class _RecordingClient:
    calls: list = []

    def retrieve(self, dataset, body, target):
        self.calls.append((dataset, body))
        with open(target, "wb") as f:
            f.write(b"not-real-netcdf")   # placeholder bytes; only the request body is under test


@pytest.fixture
def fake_cdsapi(monkeypatch, tmp_path):
    _RecordingClient.calls = []
    monkeypatch.setitem(sys.modules, "cdsapi", types.SimpleNamespace(Client=_RecordingClient))
    monkeypatch.setenv("CDSAPI_KEY", "test-only")
    return _RecordingClient.calls


def test_fetcher_sends_the_grid_area(fake_cdsapi, tmp_path):
    grid = PolarGrid.from_domain(DOMAIN, 10)
    fetch = make_fetcher(DOMAIN, grid, margin_deg=1.0)
    fetch(request_for(DOMAIN, date(2023, 11, 3), date(2023, 11, 4), grid, 1.0), tmp_path / "e.nc")
    (_, body), = fake_cdsapi
    assert body["area"] == era5_area(grid, 1.0)


def test_fetch_forcing_cli_requests_full_grid_coverage(fake_cdsapi, tmp_path):
    rc = main(["fetch-forcing", "--config", CONFIG, "--start", "2023-11-03", "--end", "2023-11-04",
               "--root", str(tmp_path / "raw"), "--skip-cmems", "--resolution-km", "25",
               "--era5-margin-deg", "0.75"])
    assert rc == 0
    (_, body), = fake_cdsapi
    grid = PolarGrid.from_domain(DOMAIN, 25)
    assert body["area"] == era5_area(grid, 0.75)
    north, west, south, east = body["area"]
    lat, lon = corner_latlon(grid)
    assert south <= lat.min() and lat.max() <= north and west <= lon.min() and lon.max() <= east
    manifest = json.loads(next((tmp_path / "raw").rglob("*.manifest.json")).read_text())
    assert manifest["request"]["bbox"] == [south, north, west, east]
