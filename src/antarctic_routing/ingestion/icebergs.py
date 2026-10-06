"""Iceberg positions from U.S. National Ice Center (USNIC) Antarctic iceberg lists.

USNIC tracks only *large* icebergs (roughly > 10 nautical miles on the long
axis, e.g. A23A). Smaller bergs, bergy bits and growlers - the ones most
relevant to ship strikes - are NOT included, so any layer built from this data
is a "tracked large-iceberg hazard", not a complete encounter risk.

The USNIC download location changes over time, so this module imports a file
the operator has already downloaded; it never fetches a URL itself.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import numpy as np

from antarctic_routing.common.provenance import sha256_file

NM_TO_KM = 1.852
USNIC_SOURCE = "U.S. National Ice Center (USNIC), official"
USNIC_PRODUCT = "Antarctic Iceberg Data, CSV"
USNIC_MIN_LENGTH_NM, USNIC_MIN_AREA_SQNM = 10.0, 20.0  # USNIC tracking criterion (either suffices)
_DMS = re.compile(r"^\s*(-?\d+(?:\.\d+)?)(?:\s+(\d+(?:\.\d+)?)'?)?\s*([NSEW])?\s*$", re.IGNORECASE)


def parse_coordinate(text: str) -> float:
    """Parse ``-65.5``, ``65.5S``, ``65 30S`` or ``65 30'S`` to signed decimal degrees."""
    m = _DMS.match(str(text))
    if not m:
        raise ValueError(f"unrecognised coordinate: {text!r}")
    deg, minutes, hemi = float(m.group(1)), m.group(2), (m.group(3) or "").upper()
    value = abs(deg) + (float(minutes) / 60.0 if minutes else 0.0)
    negative = deg < 0 or hemi in ("S", "W")
    return -value if negative else value


def _parse_date(text: str) -> date:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d-%b-%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date: {text!r}")


def read_iceberg_positions(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        for raw in csv.DictReader(fh):
            if all(not (" ".join(v) if isinstance(v, list) else (v or "")).strip() for v in raw.values()):
                continue  # separators only (e.g. ",,,,,," at the end of some USNIC archive files)
            r = {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}
            rows.append({
                "iceberg_id": r["iceberg"].upper(),
                "date": _parse_date(r.get("updated") or r.get("last update") or r.get("date", "")),
                "lat": parse_coordinate(r["latitude"]),
                "lon": parse_coordinate(r["longitude"]),
                "length_km": float(r["length (nm)"]) * NM_TO_KM if r.get("length (nm)") else None,
                "width_km": float(r["width (nm)"]) * NM_TO_KM if r.get("width (nm)") else None,
                "area_sqnm": float(r["area (sqnm)"]) if r.get("area (sqnm)") else None,
            })
    return rows


def iceberg_quality_flags(rows: list[dict]) -> dict[str, list]:
    """Suspicious records, reported by iceberg id; nothing is dropped."""
    ids = Counter(r["iceberg_id"] for r in rows)
    positions = Counter((r["lat"], r["lon"]) for r in rows)
    flags: dict[str, list] = {
        "invalid_coordinates": [r["iceberg_id"] for r in rows
                                if not (-90 <= r["lat"] <= 90 and -180 <= r["lon"] <= 180)],
        "not_southern_hemisphere": [r["iceberg_id"] for r in rows if r["lat"] >= 0],
        "duplicate_iceberg_ids": sorted(i for i, n in ids.items() if n > 1),
        "duplicate_positions": sorted(r["iceberg_id"] for r in rows if positions[(r["lat"], r["lon"])] > 1),
        "below_usnic_tracking_criterion": [],
    }
    for r in rows:
        length_nm = r["length_km"] / NM_TO_KM if r["length_km"] is not None else None
        if length_nm is None and r.get("area_sqnm") is None:
            continue
        if not ((length_nm or 0) >= USNIC_MIN_LENGTH_NM or (r.get("area_sqnm") or 0) >= USNIC_MIN_AREA_SQNM):
            flags["below_usnic_tracking_criterion"].append(r["iceberg_id"])
    return flags


def icebergs_in_domain(rows: list[dict], domain) -> list[dict]:
    """Rows inside a lat/lon box (e.g. the configured ``DomainSection``)."""
    return [r for r in rows
            if domain.lat_min <= r["lat"] <= domain.lat_max and domain.lon_min <= r["lon"] <= domain.lon_max]


def in_grid_mask(lats, lons, grid) -> np.ndarray:
    """True where a point falls in a cell of the EPSG:3031 ``grid`` (its real extent, not a lat/lon box).

    Same convention as the drift presence layers: a point belongs to the cell whose
    half-open interval ``[edge, edge + res)`` contains it, so the outer low edges are
    inside and the outer high edges are outside.
    """
    px, py = grid.to_xy(np.asarray(lats, float), np.asarray(lons, float))
    res = grid.resolution_m
    cols = np.floor((np.asarray(px) - (grid.x[0] - res / 2)) / res)
    rows = np.floor((np.asarray(py) - (grid.y[0] - res / 2)) / res)
    return (cols >= 0) & (cols < grid.x.size) & (rows >= 0) & (rows < grid.y.size)


def iceberg_provenance(path: str | Path, rows: list[dict], domain=None, source_url: str | None = None) -> dict:
    """Provenance and data-quality summary for an operator-supplied USNIC iceberg list."""
    path = Path(path)
    dates = sorted({r["date"] for r in rows})
    record = {
        "source": USNIC_SOURCE,
        "product": USNIC_PRODUCT,
        "source_url": source_url,
        "filename": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "execution_mode": "real",
        "update_dates": [d.isoformat() for d in dates],
        "n_rows": len(rows),
        "n_unique_icebergs": len({r["iceberg_id"] for r in rows}),
        "lat_bounds": [min(r["lat"] for r in rows), max(r["lat"] for r in rows)] if rows else None,
        "lon_bounds": [min(r["lon"] for r in rows), max(r["lon"] for r in rows)] if rows else None,
        "quality_flags": iceberg_quality_flags(rows),
    }
    if domain is not None:
        record["in_domain"] = [{"iceberg_id": r["iceberg_id"], "lat": r["lat"], "lon": r["lon"]}
                               for r in icebergs_in_domain(rows, domain)]
    return record
