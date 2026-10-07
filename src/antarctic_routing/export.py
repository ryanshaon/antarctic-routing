"""Route exports: GeoJSON (RFC 7946, [lon, lat] order) and CSV.

Every export carries the research-only disclaimer, the execution mode of the
underlying data, the forecast issue time and the configuration checksum.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

from antarctic_routing import DISCLAIMER, __version__
from antarctic_routing.routing.candidates import Candidate
from antarctic_routing.synthetic import ScenarioSet


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat() + "Z"


def _waypoints(candidate: Candidate, world: ScenarioSet) -> list[dict]:
    grid = world.grid
    seg = candidate.evaluation.segment_breach_prob
    fuel = candidate.evaluation.segment_fuel or [None] * len(seg)
    rows = []
    for i, ((r, c), hours) in enumerate(zip(candidate.route.cells, candidate.route.arrival_hours, strict=True)):
        rows.append({
            "waypoint_index": i,
            "lat": round(float(grid.lat2d[r, c]), 6),
            "lon": round(float(grid.lon2d[r, c]), 6),
            "planned_arrival_utc": _iso(world.start + timedelta(hours=hours)),
            "hours_from_departure": round(float(hours), 3),
            "segment_breach_prob": round(float(seg[i]), 6),
            "segment_fuel_index": None if fuel[i] is None else round(float(fuel[i]), 3),
        })
    return rows


def route_to_geojson(
    candidate: Candidate,
    world: ScenarioSet,
    issued: str | None = None,
    config_sha256: str | None = None,
) -> dict:
    wps = _waypoints(candidate, world)
    ev = candidate.evaluation
    props = {
        "labels": candidate.labels,
        "tags": candidate.tags,
        "feasible": candidate.feasible,
        **ev.summary(),
        "departure_utc": _iso(world.start),
        "forecast_issued_utc": issued,
        "execution_mode": world.execution_mode,
        "data_description": world.description,
        "software_version": __version__,
        "config_sha256": config_sha256,
        "disclaimer": DISCLAIMER,
    }
    features = [{
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[w["lon"], w["lat"]] for w in wps]},
        "properties": props,
    }]
    for w in wps:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [w["lon"], w["lat"]]},
            "properties": {k: v for k, v in w.items() if k not in ("lat", "lon")},
        })
    return {"type": "FeatureCollection", "features": features}


def route_to_csv(candidate: Candidate, world: ScenarioSet, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wps = [{**w, "execution_mode": world.execution_mode, "disclaimer": DISCLAIMER}
           for w in _waypoints(candidate, world)]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(wps[0].keys()))
        writer.writeheader()
        writer.writerows(wps)
    return out
