import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from antarctic_routing.config import DomainSection
from antarctic_routing.ingestion.base import (
    DownloadRequest,
    MissingCredentials,
    MissingDependency,
    run_download,
)
from antarctic_routing.ingestion.cmems import cmems_subset_kwargs
from antarctic_routing.ingestion.era5 import era5_request
from antarctic_routing.ingestion.icebergs import (
    iceberg_provenance,
    iceberg_quality_flags,
    icebergs_in_domain,
    parse_coordinate,
    read_iceberg_positions,
)
from antarctic_routing.ingestion.sea_ice import osisaf_url

DOMAIN = DomainSection(lat_min=-66.0, lat_max=-55.0, lon_min=-72.0, lon_max=-52.0)


def _req(**kw):
    base = dict(source="era5", product="reanalysis-era5-single-levels", variables=("u10", "v10"),
                start=date(2024, 12, 1), end=date(2024, 12, 3), bbox=(-66.0, -55.0, -72.0, -52.0))
    base.update(kw)
    return DownloadRequest(**base)


def test_request_key_is_deterministic_and_parameter_sensitive():
    assert _req().key() == _req().key()
    assert _req().key() != _req(end=date(2024, 12, 4)).key()
    assert _req().key() != _req(variables=("u10",)).key()


def test_successful_download_writes_file_manifest_and_checksum(tmp_path):
    def fetch(request, dest):
        dest.write_bytes(b"netcdf-bytes")
        return {"source_url": "https://example.invalid/x.nc", "product_version": "v1"}

    result = run_download(_req(), tmp_path, fetch, suffix=".nc")
    assert result.status == "passed" and result.execution_mode == "real"
    out = result.outputs[0]
    assert out["sha256"] and Path(out["path"]).is_file()
    manifest = json.loads(next(tmp_path.rglob("*.manifest.json")).read_text())
    assert manifest["request"]["variables"] == ["u10", "v10"]
    assert manifest["product_version"] == "v1"
    assert manifest["sha256"] == out["sha256"]


def test_second_run_uses_cache_without_fetching(tmp_path):
    calls = []

    def fetch(request, dest):
        calls.append(1)
        dest.write_bytes(b"abc")
        return {}

    run_download(_req(), tmp_path, fetch)
    second = run_download(_req(), tmp_path, fetch)
    assert len(calls) == 1
    assert any("cached" in w for w in second.warnings)


def test_missing_credentials_is_blocked_not_faked(tmp_path):
    def fetch(request, dest):
        raise MissingCredentials("CDS API key not configured")

    result = run_download(_req(), tmp_path, fetch)
    assert result.status == "blocked"
    assert not list(tmp_path.rglob("*.nc"))


def test_missing_dependency_is_blocked(tmp_path):
    def fetch(request, dest):
        raise MissingDependency("cdsapi is not installed")

    assert run_download(_req(), tmp_path, fetch).status == "blocked"


def test_failed_download_leaves_no_partial_file(tmp_path):
    def fetch(request, dest):
        dest.write_bytes(b"half")
        raise ConnectionError("reset")

    result = run_download(_req(), tmp_path, fetch)
    assert result.status == "failed"
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_osisaf_url_pattern():
    url = osisaf_url(date(2024, 12, 5))
    assert url.startswith("https://thredds.met.no/thredds/fileServer/osisaf/met.no/ice/conc/2024/12/")
    assert url.endswith("ice_conc_sh_polstere-100_multi_202412051200.nc")


def test_era5_request_uses_north_west_south_east_area():
    req = era5_request(DOMAIN, date(2024, 12, 30), date(2025, 1, 2))
    assert req["area"] == [-55.0, -72.0, -66.0, -52.0]
    assert "10m_u_component_of_wind" in req["variable"]
    assert set(req["year"]) == {"2024", "2025"}
    assert req["data_format"] == "netcdf"


def test_cmems_subset_kwargs():
    kw = cmems_subset_kwargs(DOMAIN, date(2024, 12, 1), date(2024, 12, 3), "out.nc")
    assert kw["minimum_latitude"] == -66.0 and kw["maximum_longitude"] == -52.0
    assert set(kw["variables"]) == {"uo", "vo"}
    assert kw["start_datetime"].startswith("2024-12-01")


@pytest.mark.parametrize(
    "text,expected",
    [("-65.5", -65.5), ("65.5S", -65.5), ("65 30S", -65.5), ("65 30'S", -65.5), ("58 15W", -58.25), ("12.0E", 12.0)],
)
def test_parse_coordinate_formats(text, expected):
    assert parse_coordinate(text) == pytest.approx(expected)


def test_read_iceberg_positions_normalises_columns(tmp_path):
    path = tmp_path / "usnic.csv"
    path.write_text(
        "Iceberg,Length (NM),Width (NM),Latitude,Longitude,Updated\n"
        "A23A,38,32,61 30S,46 0W,12/01/2024\n"
        "D30B,10,5,-63.25,-57.5,12/02/2024\n"
    )
    rows = read_iceberg_positions(path)
    assert rows[0]["iceberg_id"] == "A23A"
    assert rows[0]["lat"] == pytest.approx(-61.5) and rows[0]["lon"] == pytest.approx(-46.0)
    assert rows[0]["length_km"] == pytest.approx(38 * 1.852)
    assert rows[1]["date"] == date(2024, 12, 2)


# Official USNIC header (AntarcticIcebergs_20261001.csv from usicecenter.gov, BOM + CRLF) with three of its real rows.
USNIC_OFFICIAL = (
    "\ufeffIceberg,Length (NM),Width (NM),Latitude,Longitude,Area (sqMI),Area (sqNM),Area (sqKM),Last Update\r\n"
    "A76C,16,7,-55.41,-25.27,107.53,81.19,278.49,10/01/2026\r\n"
    "A83,12,7,-60.77,-52.12,73.29,55.34,189.82,10/01/2026\r\n"
    "B51,15,3,-74.25,-131.7,30.50,23.03,79.00,10/01/2026\r\n"
)
# Official files are kept outside Git; tests that need them skip when they are absent.
REAL_DATA = Path(os.environ.get("ANTROUTE_DATA_ROOT", "/mnt/project-files/real-data"))
USNIC_REAL_FILE = REAL_DATA / "iceberg/raw/AntarcticIcebergs_20261001.csv"


def test_read_iceberg_positions_accepts_official_usnic_last_update_column(tmp_path):
    path = tmp_path / "AntarcticIcebergs_20261001.csv"
    path.write_bytes(USNIC_OFFICIAL.encode("utf-8"))
    rows = read_iceberg_positions(path)
    assert [r["iceberg_id"] for r in rows] == ["A76C", "A83", "B51"]
    assert [(r["lat"], r["lon"]) for r in rows] == [(-55.41, -25.27), (-60.77, -52.12), (-74.25, -131.7)]
    assert {r["date"] for r in rows} == {date(2026, 10, 1)}
    assert rows[1]["length_km"] == pytest.approx(12 * 1.852) and rows[1]["width_km"] == pytest.approx(7 * 1.852)
    if USNIC_REAL_FILE.exists():  # the downloaded official file is kept outside Git
        import csv

        real = read_iceberg_positions(USNIC_REAL_FILE)
        with USNIC_REAL_FILE.open(newline="", encoding="utf-8-sig") as fh:
            src = list(csv.DictReader(fh))
        assert len(real) == len(src) == 33
        assert [r["iceberg_id"] for r in real] == [s["Iceberg"] for s in src]
        assert len({r["iceberg_id"] for r in real}) == 33
        assert [(r["lat"], r["lon"]) for r in real] == [(float(s["Latitude"]), float(s["Longitude"])) for s in src]
        assert {r["date"] for r in real} == {date(2026, 10, 1)}


USNIC_C39 = "C39,8,3,-65.73,56.18,17.37,13.12,44.99,10/01/2026\r\n"  # real row below the tracking criterion


def test_usnic_provenance_records_source_counts_bounds_and_domain(tmp_path):
    path = tmp_path / "AntarcticIcebergs_20261001.csv"
    path.write_bytes((USNIC_OFFICIAL + USNIC_C39).encode("utf-8"))
    rows = read_iceberg_positions(path)
    rec = iceberg_provenance(path, rows, DOMAIN, source_url="https://usicecenter.gov/File/DownloadCurrent?pId=134")
    assert rec["source"].startswith("U.S. National Ice Center") and rec["product"] == "Antarctic Iceberg Data, CSV"
    assert rec["filename"] == path.name and rec["bytes"] == path.stat().st_size and len(rec["sha256"]) == 64
    assert rec["execution_mode"] == "real" and rec["update_dates"] == ["2026-10-01"]
    assert rec["n_rows"] == rec["n_unique_icebergs"] == 4
    assert rec["lat_bounds"] == [-74.25, -55.41] and rec["lon_bounds"] == [-131.7, 56.18]
    assert [r["iceberg_id"] for r in rec["in_domain"]] == ["A83"]
    assert rec["quality_flags"]["below_usnic_tracking_criterion"] == ["C39"]
    assert all(not v for k, v in rec["quality_flags"].items() if k != "below_usnic_tracking_criterion")


def test_usnic_quality_flags_report_but_never_drop_records(tmp_path):
    path = tmp_path / "usnic.csv"
    # corrupted copy of real rows: a duplicated record and a sign-flipped latitude
    text = USNIC_OFFICIAL + "A83,12,7,-60.77,-52.12,73.29,55.34,189.82,10/01/2026\r\n"
    path.write_bytes(text.replace("-74.25", "74.25").encode("utf-8"))
    rows = read_iceberg_positions(path)
    flags = iceberg_quality_flags(rows)
    assert len(rows) == 4
    assert flags["duplicate_iceberg_ids"] == ["A83"] and flags["duplicate_positions"] == ["A83", "A83"]
    assert flags["not_southern_hemisphere"] == ["B51"]


@pytest.mark.skipif(not USNIC_REAL_FILE.exists(), reason="official USNIC file is kept outside Git")
def test_real_usnic_snapshot_passes_ingestion_and_provenance():
    rows = read_iceberg_positions(USNIC_REAL_FILE)
    rec = iceberg_provenance(USNIC_REAL_FILE, rows, DOMAIN)
    assert rec["sha256"] == "f512ce937a1a78f37acd5bbf513a35d6104c477728664a2fdf264110555e7507"
    assert rec["n_rows"] == rec["n_unique_icebergs"] == 33 and rec["update_dates"] == ["2026-10-01"]
    assert rec["lat_bounds"] == [-74.25, -55.41] and rec["lon_bounds"] == [-174.86, 164.02]
    assert [r["iceberg_id"] for r in icebergs_in_domain(rows, DOMAIN)] == ["A83", "D33A", "D33C", "D33D"]
    assert rec["quality_flags"]["below_usnic_tracking_criterion"] == ["C39", "D15D"]
    assert all(not v for k, v in rec["quality_flags"].items() if k != "below_usnic_tracking_criterion")


def test_read_iceberg_positions_ignores_completely_empty_usnic_rows(tmp_path):
    """Some official USNIC archive files (e.g. AntarcticIcebergs_20211112.csv) end with ',,,,,,' rows."""
    header = "Iceberg,Length (NM),Width (NM),Latitude,Longitude,Remarks,Last Update\r\n"
    valid = ("D30A,37,11,-73.69,-26.38,weddw,11/12/2021\r\n"
             "D30B,10,5,-71.55,-14.95,weddw,11/12/2021\r\n")
    path = tmp_path / "AntarcticIcebergs_20211112.csv"
    path.write_bytes((header + valid + ",,,,,,\r\n" * 5 + " , ,,,\t,,\r\n").encode("utf-8"))
    rows = read_iceberg_positions(path)
    assert [r["iceberg_id"] for r in rows] == ["D30A", "D30B"]
    assert [(r["lat"], r["lon"]) for r in rows] == [(-73.69, -26.38), (-71.55, -14.95)]
    assert {r["date"] for r in rows} == {date(2021, 11, 12)}
    partial = tmp_path / "partial.csv"  # a partially populated row is still an error, not skipped
    partial.write_bytes((header + valid + "D31,,,,,,\r\n").encode("utf-8"))
    with pytest.raises(ValueError, match="unrecognised date"):
        read_iceberg_positions(partial)


# ------------------------------------------------------------- routing-grid filter
USNIC_20231109 = REAL_DATA / "iceberg/raw/archive/AntarcticIcebergs_20231109.csv"
IN_GRID_20231109 = ["A23A", "A76B", "A76C", "A80A", "A80D", "D20A", "D30B"]


def _grid25():
    from antarctic_routing.preprocessing.grid import PolarGrid

    return PolarGrid.from_domain(DOMAIN, resolution_km=25)


def test_in_grid_mask_uses_the_polar_grid_not_the_latlon_box():
    from antarctic_routing.ingestion.icebergs import in_grid_mask

    grid = _grid25()
    # real 2023-11-09 positions: A23A/D20A/D30B are outside the config box but on the polar grid,
    # A74A (71.5S) is south of it and B22A (Amundsen Sea) is far away
    lat = [-63.13, -62.7, -60.88, -65.31, -71.51, -74.0]
    lon = [-51.67, -50.5, -44.99, -58.29, -55.01, -108.0]
    assert in_grid_mask(lat, lon, grid).tolist() == [True, True, True, True, False, False]


def test_in_grid_mask_boundaries_match_the_grid_cell_convention():
    from antarctic_routing.ingestion.icebergs import in_grid_mask

    grid = _grid25()
    res, half = grid.resolution_m, grid.resolution_m / 2
    x_lo, x_hi = grid.x[0] - half, grid.x[-1] + half
    y_lo, y_hi = grid.y[0] - half, grid.y[-1] + half
    ym, xm = grid.y[grid.y.size // 2], grid.x[grid.x.size // 2]
    xs = np.array([x_lo - 1, x_lo + 1, x_hi - 1, x_hi + 1, xm, xm, xm, xm])
    ys = np.array([ym, ym, ym, ym, y_lo - 1, y_lo + 1, y_hi - 1, y_hi + 1])
    lat, lon = grid.to_latlon(xs, ys)
    mask = in_grid_mask(lat, lon, grid)
    assert mask.tolist() == [False, True, True, False, False, True, True, False]
    # identical to PolarGrid.cell_of (used for route endpoints) for random points around the edges
    rng = np.random.default_rng(0)
    px = rng.uniform(x_lo - 3 * res, x_hi + 3 * res, 400)
    py = rng.uniform(y_lo - 3 * res, y_hi + 3 * res, 400)
    lat, lon = grid.to_latlon(px, py)

    def has_cell(a, b):
        try:
            grid.cell_of(a, b)
            return True
        except ValueError:
            return False

    assert in_grid_mask(lat, lon, grid).tolist() == [has_cell(a, b) for a, b in zip(lat, lon, strict=True)]
    assert in_grid_mask([], [], grid).shape == (0,)


@pytest.mark.skipif(not USNIC_20231109.exists(), reason="official USNIC archive file is kept outside Git")
def test_real_20231109_snapshot_keeps_the_seven_bergs_on_the_25km_grid():
    from antarctic_routing.ingestion.icebergs import in_grid_mask

    rows = read_iceberg_positions(USNIC_20231109)
    rec = iceberg_provenance(USNIC_20231109, rows)
    assert len(rows) == rec["n_rows"] == 49 and rec["update_dates"] == ["2023-11-09"]
    mask = in_grid_mask([r["lat"] for r in rows], [r["lon"] for r in rows], _grid25())
    assert [r["iceberg_id"] for r, m in zip(rows, mask, strict=True) if m] == IN_GRID_20231109
    assert int((~mask).sum()) == 42
    assert "A80D" in rec["quality_flags"]["below_usnic_tracking_criterion"]   # flags still cover every row
