"""Atmospheric forcing: ERA5 single levels from the Copernicus Climate Data Store.

Requires the ``cdsapi`` client and a CDS API key (``~/.cdsapirc`` or the
``CDSAPI_URL``/``CDSAPI_KEY`` environment variables). ERA5 is hourly; daily
summaries are derived downstream.

The request area is derived from the polar-stereographic routing grid, not from
the configured lat/lon box: the grid is the smallest EPSG:3031 rectangle that
encloses that box, so its corners reach well outside it in lat/lon. A box-only
request would leave those cells without wind (zero-filled by
``ingestion.forcing``).
"""

from __future__ import annotations

import math
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from pyproj import Transformer

from antarctic_routing.config import DomainSection
from antarctic_routing.ingestion.base import DownloadRequest, MissingCredentials, MissingDependency
from antarctic_routing.preprocessing.grid import CRS, PolarGrid

DATASET = "reanalysis-era5-single-levels"
VARIABLES = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
)

AREA_MARGIN_DEG = 0.5   # two ERA5 0.25 deg cells, so bilinear interpolation has neighbours at the grid edge
ERA5_STEP_DEG = 0.25
_TO_LL = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)


def grid_latlon_extent(grid: PolarGrid, samples: int = 200) -> tuple[float, float, float, float]:
    """``(lat_min, lat_max, lon_min, lon_max)`` covered by the grid's cells, edges included.

    The outer cell edges are densely sampled because straight EPSG:3031 edges
    are curved in lat/lon. Cell centres are included as well.
    """
    half = grid.resolution_m / 2
    x0, x1 = grid.x[0] - half, grid.x[-1] + half
    y0, y1 = grid.y[0] - half, grid.y[-1] + half
    if x0 <= 0 <= x1 and y0 <= 0 <= y1:
        raise ValueError("grid contains the South Pole; a lat/lon box cannot describe it")
    ex, ey = np.linspace(x0, x1, samples), np.linspace(y0, y1, samples)
    bx = np.r_[ex, ex, np.full(samples, x0), np.full(samples, x1)]
    by = np.r_[np.full(samples, y0), np.full(samples, y1), ey, ey]
    xx, yy = np.meshgrid(grid.x, grid.y)
    lon, lat = _TO_LL.transform(np.r_[bx, xx.ravel()], np.r_[by, yy.ravel()])
    if lon.max() - lon.min() > 180:
        raise ValueError("grid crosses the antimeridian; unsupported (as for config.domain)")
    return float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())


def era5_area(grid: PolarGrid, margin_deg: float = AREA_MARGIN_DEG) -> list[float]:
    """CDS ``area`` (N, W, S, E) covering the whole grid plus ``margin_deg``, snapped outward to 0.25 deg."""
    if margin_deg < 0:
        raise ValueError("margin_deg must be >= 0")
    lat_min, lat_max, lon_min, lon_max = grid_latlon_extent(grid)

    def down(v):
        return math.floor(v / ERA5_STEP_DEG) * ERA5_STEP_DEG

    def up(v):
        return math.ceil(v / ERA5_STEP_DEG) * ERA5_STEP_DEG

    return [min(up(lat_max + margin_deg), 90.0), max(down(lon_min - margin_deg), -180.0),
            max(down(lat_min - margin_deg), -90.0), min(up(lon_max + margin_deg), 180.0)]


def era5_request(domain: DomainSection, start: date, end: date, hours: tuple[int, ...] = (0, 6, 12, 18),
                 area: list[float] | None = None) -> dict:
    """CDS request body. ``area`` (N, W, S, E) defaults to the configured domain box;
    real downloads pass :func:`era5_area` so the whole routing grid is covered."""
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    if area is None:
        area = [domain.lat_max, domain.lon_min, domain.lat_min, domain.lon_max]
    return {
        "product_type": ["reanalysis"],
        "variable": list(VARIABLES),
        "year": sorted({f"{d:%Y}" for d in days}),
        "month": sorted({f"{d:%m}" for d in days}),
        "day": sorted({f"{d:%d}" for d in days}),
        "time": [f"{h:02d}:00" for h in hours],
        "area": list(area),  # N, W, S, E
        "data_format": "netcdf",
    }


def _has_credentials() -> bool:
    return bool(os.environ.get("CDSAPI_KEY")) or Path.home().joinpath(".cdsapirc").is_file()


def make_fetcher(domain: DomainSection, grid: PolarGrid | None = None, margin_deg: float = AREA_MARGIN_DEG):
    area = era5_area(grid, margin_deg) if grid is not None else None

    def fetch(request: DownloadRequest, dest: Path) -> dict:
        try:
            import cdsapi
        except ImportError as exc:
            raise MissingDependency("cdsapi is not installed (pip install '.[ingest]')") from exc
        if not _has_credentials():
            raise MissingCredentials("CDS API key not configured (~/.cdsapirc or CDSAPI_KEY)")
        body = era5_request(domain, request.start, request.end, area=area)
        cdsapi.Client().retrieve(DATASET, body, str(dest))
        return {"source_url": f"https://cds.climate.copernicus.eu/datasets/{DATASET}", "product_version": DATASET}

    return fetch


def request_for(domain: DomainSection, start: date, end: date, grid: PolarGrid | None = None,
                margin_deg: float = AREA_MARGIN_DEG) -> DownloadRequest:
    if grid is None:
        bbox = (domain.lat_min, domain.lat_max, domain.lon_min, domain.lon_max)
    else:
        north, west, south, east = era5_area(grid, margin_deg)
        bbox = (south, north, west, east)
    return DownloadRequest("era5", DATASET, VARIABLES, start, end, bbox)
