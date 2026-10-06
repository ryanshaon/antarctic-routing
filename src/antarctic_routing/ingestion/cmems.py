"""Ocean currents and temperature: Copernicus Marine Service (CMEMS).

Uses the ``copernicusmarine`` toolbox. Credentials come from
``COPERNICUSMARINE_SERVICE_USERNAME``/``COPERNICUSMARINE_SERVICE_PASSWORD`` or a
prior ``copernicusmarine login``. Record whether the chosen dataset is a
reanalysis (``_my_``) or analysis/forecast (``_anfc_``) product.

As for ERA5 (:func:`ingestion.era5.era5_area`), real downloads request the
lat/lon extent of the whole EPSG:3031 routing grid plus a margin, not the
configured lat/lon box: the grid corners reach well outside that box.
"""

from __future__ import annotations

import math
import os
from datetime import date
from pathlib import Path

from antarctic_routing.config import DomainSection
from antarctic_routing.ingestion.base import DownloadRequest, MissingCredentials, MissingDependency
from antarctic_routing.ingestion.era5 import grid_latlon_extent
from antarctic_routing.preprocessing.grid import PolarGrid

REANALYSIS = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
ANALYSIS_FORECAST = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
VARIABLES = ("uo", "vo")   # eastward/northward surface currents; the only CMEMS fields build-forcing uses

AREA_MARGIN_DEG = 0.5   # six 1/12 deg CMEMS cells, so bilinear interpolation has neighbours at the grid edge
AREA_STEP_DEG = 0.25    # bounds are snapped outward to a tidy lattice (the subset service clips to its own cells)


def cmems_bbox(grid: PolarGrid, margin_deg: float = AREA_MARGIN_DEG) -> tuple[float, float, float, float]:
    """``(lat_min, lat_max, lon_min, lon_max)`` covering the whole grid plus ``margin_deg``, snapped outward."""
    if margin_deg < 0:
        raise ValueError("margin_deg must be >= 0")
    lat_min, lat_max, lon_min, lon_max = grid_latlon_extent(grid)

    def down(v):
        return math.floor(v / AREA_STEP_DEG) * AREA_STEP_DEG

    def up(v):
        return math.ceil(v / AREA_STEP_DEG) * AREA_STEP_DEG

    return (max(down(lat_min - margin_deg), -90.0), min(up(lat_max + margin_deg), 90.0),
            max(down(lon_min - margin_deg), -180.0), min(up(lon_max + margin_deg), 180.0))


def cmems_subset_kwargs(domain: DomainSection, start: date, end: date, output_filename: str,
                        dataset_id: str = REANALYSIS,
                        bbox: tuple[float, float, float, float] | None = None) -> dict:
    """``copernicusmarine.subset`` arguments. ``bbox`` (lat_min, lat_max, lon_min, lon_max)
    defaults to the configured domain box; real downloads pass :func:`cmems_bbox`."""
    lat_min, lat_max, lon_min, lon_max = bbox or (domain.lat_min, domain.lat_max, domain.lon_min, domain.lon_max)
    return {
        "dataset_id": dataset_id,
        "variables": list(VARIABLES),
        "minimum_latitude": lat_min,
        "maximum_latitude": lat_max,
        "minimum_longitude": lon_min,
        "maximum_longitude": lon_max,
        "minimum_depth": 0.0,
        "maximum_depth": 1.0,
        "start_datetime": f"{start.isoformat()}T00:00:00",
        "end_datetime": f"{end.isoformat()}T23:59:59",
        "output_filename": output_filename,
    }


def make_fetcher(domain: DomainSection, dataset_id: str = REANALYSIS, grid: PolarGrid | None = None,
                 margin_deg: float = AREA_MARGIN_DEG):
    bbox = cmems_bbox(grid, margin_deg) if grid is not None else None

    def fetch(request: DownloadRequest, dest: Path) -> dict:
        try:
            import copernicusmarine
        except ImportError as exc:
            raise MissingDependency("copernicusmarine is not installed (pip install '.[ingest]')") from exc
        has_env = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME") and os.environ.get(
            "COPERNICUSMARINE_SERVICE_PASSWORD")
        has_file = Path.home().joinpath(".copernicusmarine").exists()
        if not (has_env or has_file):
            raise MissingCredentials("Copernicus Marine credentials not configured")
        # the toolbox insists on a .nc suffix, so write beside ``dest`` and move it into place
        tmp = dest.with_name(dest.name + ".nc")
        kwargs = cmems_subset_kwargs(domain, request.start, request.end, tmp.name, dataset_id, bbox)
        response = copernicusmarine.subset(**kwargs, output_directory=str(dest.parent), overwrite=True,
                                           disable_progress_bar=True)
        tmp.replace(dest)
        meta = {"source_url": f"https://data.marine.copernicus.eu/product/{dataset_id}",
                "product_version": dataset_id,
                "product_kind": "reanalysis" if "_my_" in dataset_id else "analysis_forecast",
                "subset_request": {k: v for k, v in kwargs.items() if k != "output_filename"}}
        if hasattr(response, "model_dump"):
            meta["subset_response"] = response.model_dump(mode="json", exclude={"filename", "output_directory",
                                                                                 "file_path"})
        return meta

    return fetch


def request_for(domain: DomainSection, start: date, end: date, dataset_id: str = REANALYSIS,
                grid: PolarGrid | None = None, margin_deg: float = AREA_MARGIN_DEG) -> DownloadRequest:
    bbox = cmems_bbox(grid, margin_deg) if grid is not None else (
        domain.lat_min, domain.lat_max, domain.lon_min, domain.lon_max)
    return DownloadRequest("cmems", dataset_id, VARIABLES, start, end, bbox)
