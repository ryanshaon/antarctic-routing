"""Read-only publication of precomputed real-data results (API / dashboard).

Heavy work (NetCDF ingestion, U-Net inference, ensembles, routing) runs on a
backend or worker. Its outputs are packed into a *bundle*: a directory with a
``bundle.json`` index that records the SHA-256 of every file. The API only
reads bundles; it never computes science on request and never falls back to
synthetic data when a bundle is missing or damaged.

Bundle files
------------
``plan_window.json``      plan-window output, absolute paths reduced to file names
``forecast_maps.json``    per-layer observed/forecast maps from :func:`map_layers`
``manifest.json``         run provenance (inputs, checksums, parameters), paths reduced
``departure_window.png``  plan-window figure
``bundle.json``           index: kind, execution mode, limitations, file checksums
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from antarctic_routing.common.provenance import sha256_file, utc_now, write_json_artifact
from antarctic_routing.routing.hazard import exceedance_probability

BUNDLE_INDEX = "bundle.json"
BUNDLE_SCHEMA = "antroute-bundle/1"
REQUIRED_FILES = ("plan_window.json", "forecast_maps.json", "manifest.json")
OPTIONAL_FILES = ("departure_window.png",)

# Data-status labels shown by the API and dashboard.
REAL, HISTORICAL, FORECAST, SCHEMATIC, UNAVAILABLE = "real", "historical", "forecast", "schematic", "unavailable"
DATA_STATUSES = (REAL, HISTORICAL, FORECAST, SCHEMATIC, UNAVAILABLE)


class BundleError(RuntimeError):
    """A bundle is missing, incomplete or does not match its recorded checksums."""


def to_percent(a: np.ndarray) -> list:
    """Fractions -> integer percent per cell (row-major); non-finite cells become ``None``."""
    flat = np.asarray(a, float).ravel()
    out = np.round(np.where(np.isfinite(flat), flat, 0.0) * 100).astype(int).tolist()
    return [v if ok else None for v, ok in zip(out, np.isfinite(flat).tolist(), strict=True)]


def grid_geometry(grid) -> dict:
    """Grid description used by the dashboard to draw cells (km, EPSG:3031)."""
    res_km = grid.resolution_m / 1000.0
    return {
        "crs": "EPSG:3031", "nx": int(grid.x.size), "ny": int(grid.y.size), "res_km": res_km,
        "x0_km": float(grid.x[0] / 1000.0 - res_km / 2), "y0_km": float(grid.y[0] / 1000.0 - res_km / 2),
        "rotation_rad": float(np.deg2rad(np.median(grid.lon2d))),
        "x_km": [float(v / 1000.0) for v in grid.x], "y_km": [float(v / 1000.0) for v in grid.y],
    }


def map_layers(world, tau: float) -> dict:
    """Per-layer maps of a scenario set, for display only (no new science).

    ``observed`` is layer 0 concentration where the layer is an observation. Every
    layer carries the member mean concentration, the fraction of members with
    concentration >= ``tau`` and, if icebergs were drifted, the fraction of
    members with iceberg presence. Land cells are ``None``.
    """
    sources = list(world.layer_source or ["unknown"] * world.n_times)
    layers = []
    for t in range(world.n_times):
        conc = world.conc[:, t]
        n = np.isfinite(conc).sum(axis=0)
        mean = np.where(world.land | (n == 0), np.nan, np.nansum(conc, axis=0) / np.maximum(n, 1))
        p_ice = np.where(world.land, np.nan, exceedance_probability(conc, tau))
        layer = {
            "index": t, "date": (world.start + timedelta(hours=t * world.time_step_hours)).date().isoformat(),
            "source": sources[t], "mean_concentration_pct": to_percent(mean), "p_ice_ge_limit_pct": to_percent(p_ice),
            "p_berg_pct": None,
        }
        if world.berg is not None:
            layer["p_berg_pct"] = to_percent(np.where(world.land, np.nan, world.berg[:, t].mean(axis=0)))
        layers.append(layer)
    observed = None
    if sources and sources[0] == "observed":
        obs = np.where(world.land, np.nan, world.conc[0, 0])
        observed = {"date": layers[0]["date"], "concentration_pct": to_percent(obs)}
    return {
        "kind": "forecast_maps", "execution_mode": world.execution_mode, "tau": float(tau),
        "n_members": int(world.n_scenarios), "time_step_hours": float(world.time_step_hours),
        "grid": grid_geometry(world.grid), "land": world.land.astype(np.uint8).ravel().tolist(),
        "observed": observed, "layers": layers,
    }


def strip_paths(obj: Any) -> Any:
    """Copy of ``obj`` with absolute filesystem paths reduced to their file names."""
    if isinstance(obj, dict):
        return {k: strip_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_paths(v) for v in obj]
    if isinstance(obj, str) and os.path.isabs(obj) and "\n" not in obj:
        return Path(obj).name
    return obj


def canonical_sha256(payload: Any) -> str:
    """SHA-256 of ``payload`` with paths stripped and keys sorted (machine-independent)."""
    encoded = json.dumps(strip_paths(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_bundle(dest: Path, plan_window: Path, forecast_maps: Path, manifest: dict, *,
                 figure: Path | None = None, kind: str = "frozen_demo", limitations: list[str] = (),
                 bundle_id: str | None = None) -> Path:
    """Write a bundle to ``dest`` (created; existing bundle files are replaced)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(plan_window).read_text())
    execution_mode = plan.get("execution_mode")
    write_json_artifact(dest / "plan_window.json", strip_paths(plan))
    shutil.copyfile(forecast_maps, dest / "forecast_maps.json")
    write_json_artifact(dest / "manifest.json", strip_paths(manifest))
    names = list(REQUIRED_FILES)
    if figure is not None:
        shutil.copyfile(figure, dest / "departure_window.png")
        names.append("departure_window.png")
    files = {n: {"sha256": sha256_file(dest / n), "bytes": (dest / n).stat().st_size} for n in names}
    index = {
        "schema": BUNDLE_SCHEMA, "kind": kind, "created_utc": utc_now(), "execution_mode": execution_mode,
        "bundle_id": bundle_id or f"{kind}-{plan.get('issue', 'unknown')}-{canonical_sha256(plan)[:12]}",
        "issue": plan.get("issue"), "plan_window_canonical_sha256": canonical_sha256(plan),
        "source_plan_window_sha256": sha256_file(plan_window), "limitations": list(limitations), "files": files,
    }
    write_json_artifact(dest / BUNDLE_INDEX, index)
    return dest / BUNDLE_INDEX


@dataclass
class Bundle:
    """A verified, read-only bundle held in memory."""

    root: Path
    index: dict
    plan_window: dict
    forecast_maps: dict
    manifest: dict
    files: dict = field(default_factory=dict)

    @property
    def figure(self) -> Path | None:
        p = self.root / "departure_window.png"
        return p if "departure_window.png" in self.index.get("files", {}) else None


def load_bundle(root: str | Path) -> Bundle:
    """Load and verify a bundle; raises :class:`BundleError` with the reason on any problem."""
    root = Path(root)
    idx_path = root / BUNDLE_INDEX
    if not idx_path.is_file():
        raise BundleError(f"no {BUNDLE_INDEX} in the configured artifacts directory")
    try:
        index = json.loads(idx_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"{BUNDLE_INDEX} is not readable JSON ({type(exc).__name__})") from None
    if index.get("schema") != BUNDLE_SCHEMA:
        raise BundleError(f"unsupported bundle schema {index.get('schema')!r}")
    files = index.get("files") or {}
    for name in REQUIRED_FILES:
        if name not in files:
            raise BundleError(f"bundle index does not list required file {name}")
    for name, rec in files.items():
        if name not in REQUIRED_FILES + OPTIONAL_FILES:
            raise BundleError(f"unexpected file {name!r} in bundle index")
        path = root / name
        if not path.is_file():
            raise BundleError(f"bundle file {name} is missing")
        if sha256_file(path) != rec.get("sha256"):
            raise BundleError(f"checksum mismatch for {name}")
    data = {n: json.loads((root / n).read_text()) for n in REQUIRED_FILES}
    plan = data["plan_window.json"]
    if canonical_sha256(plan) != index.get("plan_window_canonical_sha256"):
        raise BundleError("plan_window.json does not match the canonical checksum recorded in the bundle")
    if plan.get("execution_mode") != index.get("execution_mode"):
        raise BundleError("execution mode in plan_window.json differs from the bundle index")
    return Bundle(root=root, index=index, plan_window=plan, forecast_maps=data["forecast_maps.json"],
                  manifest=data["manifest.json"], files=files)


def source_statuses(bundle: Bundle) -> list[dict]:
    """Data status of each input/output in a bundle, derived from its own records."""
    plan, idx, man = bundle.plan_window, bundle.index, bundle.manifest
    issue = plan.get("issue")
    real = plan.get("execution_mode") == "real"
    past = HISTORICAL if real else SCHEMATIC
    fp = plan.get("forcing_provenance") or {}
    product = ((man.get("sea_ice") or {}).get("dataset") or {}).get("source_product") or "sea-ice dataset"

    def forcing(kind: str) -> dict:
        src = fp.get(kind)
        if not src:
            return {"name": kind, "status": UNAVAILABLE, "label": f"no {kind} record in the bundle"}
        if src == "schematic":
            return {"name": kind, "status": SCHEMATIC, "label": f"schematic {kind} (no real forcing supplied)"}
        dates = fp.get("wind_dates" if kind == "winds" else "current_dates")
        span = f" {dates[0]}..{dates[1]}" if dates else ""
        return {"name": kind, "status": past, "label": f"{src}{span}"}

    n_forecast = sum(1 for s in plan.get("layer_source") or [] if s == "forecast")
    n_clim = sum(1 for s in plan.get("layer_source") or [] if s == "climatology")
    berg_src = plan.get("iceberg_source") or {}
    drift = plan.get("iceberg_drift") or {}
    out = [
        {"name": "sea_ice_observed", "status": past if (plan.get("layer_source") or [""])[0] == "observed"
         else UNAVAILABLE, "label": f"{product} observed {issue} (EPSG:3031 25 km)"},
        {"name": "sea_ice_forecast", "status": FORECAST if real else SCHEMATIC,
         "label": f"U-Net ensemble issued {issue}: {n_forecast} forecast day(s), {n_clim} climatology-anomaly "
                  f"day(s), {plan.get('n_members')} members"},
        forcing("winds"),
        forcing("currents"),
    ]
    if berg_src:
        dates = ",".join(berg_src.get("update_dates") or [])
        out.append({"name": "icebergs", "status": past if berg_src.get("execution_mode") == "real" else SCHEMATIC,
                    "label": f"{berg_src.get('source', 'iceberg list')} {dates}: {len(plan.get('icebergs') or [])} "
                             f"in grid drifted (beta {drift.get('beta')}, spread {drift.get('spread_factor')})"})
    else:
        out.append({"name": "icebergs", "status": UNAVAILABLE, "label": "no iceberg source in this run"})
    sel = plan.get("selected")
    out.append({"name": "route", "status": FORECAST if sel else UNAVAILABLE,
                "label": f"selected departure {sel} from the forecast issued {issue}" if sel
                else "no departure met the risk budget"})
    out.append({"name": "land_mask", "status": REAL if real else SCHEMATIC,
                "label": "sea-ice product land mask on the 25 km grid; not a navigational coastline" if real
                else "schematic land"})
    out.insert(0, {"name": "bundle", "status": past, "label": f"{idx.get('kind')} {idx.get('bundle_id')}"})
    return out


def write_compact_json(path: Path, payload: Any) -> Path:
    """Atomic compact JSON (large map arrays; no indentation)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(destination)
    return destination


def iceberg_tracks(world, bergs, seed: int, radius_m: float, *, beta: float, alpha_range: tuple[float, float],
                   spread_factor: float) -> list[dict]:
    """Daily ensemble-mean track of each drifted berg, for display.

    The drift ensemble is recomputed as :func:`add_iceberg_hazard` built it (same seed and parameters)
    and its presence layers must equal ``world.berg`` cell for cell; otherwise this raises instead of
    showing tracks that the planner did not use. The check works at grid-cell resolution.
    """
    from antarctic_routing.iceberg.drift import ForcingField, drift_ensemble, presence_layers, scale_ensemble_spread
    from antarctic_routing.preprocessing.grid import PolarGrid

    ens = drift_ensemble(bergs, ForcingField.from_scenarios_current(world), ForcingField.from_scenarios_wind(world),
                         world.n_scenarios, world.horizon_hours, np.random.default_rng(seed),
                         alpha_range=alpha_range, beta=beta)
    ens = scale_ensemble_spread(ens, spread_factor)
    pres = presence_layers(ens, world.grid, world.n_times, world.time_step_hours, radius_m) & ~world.land[None, None]
    if world.berg is None or not np.array_equal(pres, world.berg):
        raise ValueError("recomputed iceberg ensemble does not match the planner's hazard; tracks not exported")
    out = []
    step = int(round(world.time_step_hours))
    for b, name in enumerate(ens.ids):
        days = []
        for t in range(world.n_times):
            i = int(np.argmin(np.abs(ens.times_h - t * step)))
            pts = ens.xy[:, b, i]                                      # (K, 2) metres, NaN once off the grid
            ok = np.isfinite(pts).all(axis=1)
            if not ok.any():
                days.append({"layer": t, "on_grid_fraction": 0.0})
                continue
            m = pts[ok].mean(axis=0)
            lat, lon = PolarGrid.to_latlon(np.array([m[0]]), np.array([m[1]]))
            dist = np.hypot(*(pts[ok] - m).T) / 1000.0
            days.append({"layer": t, "hours": float(ens.times_h[i]), "x_km": float(m[0] / 1000.0),
                         "y_km": float(m[1] / 1000.0), "lat": round(float(lat[0]), 4), "lon": round(float(lon[0]), 4),
                         "on_grid_fraction": float(ok.mean()),
                         "spread_p90_km": round(float(np.quantile(dist, 0.9)), 2)})
        out.append({"id": name, "start_lat": bergs[b][1], "start_lon": bergs[b][2], "daily_mean": days})
    return out
