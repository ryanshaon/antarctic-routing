"""Stage 11 - FastAPI backend.

Endpoints
---------
GET  /health                       liveness: service status, version, disclaimer (200 while the process runs)
GET  /ready                        readiness: 200 only when the real-data bundle is loaded and verified, else 503
GET  /status                      data status of every source (real / historical / forecast / schematic / unavailable)
GET  /versions                     software, model and artifact versions
GET  /provenance                   provenance of the published real-data bundle
GET  /config                       validated scenario configuration
GET  /real/...                     read-only real-data results from a verified bundle (see below)
POST /routes[?wait=true]           plan risk-budgeted routes (job)
POST /departures[?wait=true]       sweep a departure window (job)
GET  /jobs/{id}                    job status and result
POST /voyages                      create a voyage from a route plan
POST /voyages/{id}/replan          replan from the vessel position
GET  /voyages/{id}/history         planned route and every replan decision
GET  /voyages/{id}/export          GeoJSON or CSV of the current route
GET  /real/historical/...          Real Historical Data: the real-data planner on past seasons
                                   (see :mod:`antarctic_routing.api.historical`)
GET  /                             dashboard (static)

Long computations run as background jobs so HTTP requests are never held open
for a whole ensemble; ``?wait=true`` runs them inline (scripts and tests).
Responses carry geometry and summaries, never full scientific arrays.

Three data paths, never mixed:

* the interactive planner (``/routes``, ``/departures``, ``/voyages``) runs on
  the **controlled-synthetic** schematic world and says so in every response;
* ``/real/*`` serves precomputed results of a real-data run from the bundle in
  ``ANTROUTE_ARTIFACTS_DIR`` (see :mod:`antarctic_routing.publish`), verified by
  checksum at start-up. With no bundle, or a damaged one, these endpoints return
  503 with the reason; nothing is substituted;
* ``/real/historical/*`` runs the real-data planner on request for past issue
  dates from the verified archive in ``ANTROUTE_DATA_ROOT`` ("Real Historical
  Data", hindsight forcing disclosed). Without it they return 503 with the reason.

Environment: ``ANTROUTE_CONFIG``, ``ANTROUTE_ARTIFACTS_DIR``, ``ANTROUTE_FIGURES``,
``ANTROUTE_FIGURES_MODE``, ``ANTROUTE_CORS_ORIGINS`` (comma-separated https origins),
``ANTROUTE_MAX_JOBS``, ``ANTROUTE_MAX_VOYAGES``, ``ANTROUTE_MAX_SCENARIO_CELLS``, ``ANTROUTE_MAX_COMPUTE``,
``ANTROUTE_MAX_BODY_BYTES``, ``ANTROUTE_GIT_COMMIT``, ``ANTROUTE_DATA_ROOT``, ``ANTROUTE_HISTORICAL_SPEC``.
The API reads no data-service credentials.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import platform
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from importlib import metadata
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from antarctic_routing import DISCLAIMER, __version__
from antarctic_routing.api.historical import register as register_historical
from antarctic_routing.common.provenance import sha256_file, utc_now
from antarctic_routing.config import ProjectConfig, load_config, min_scenarios_for_budget
from antarctic_routing.export import route_to_geojson
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.publish import (
    SCHEMATIC,
    UNAVAILABLE,
    Bundle,
    BundleError,
    load_bundle,
    source_statuses,
    to_percent,
)
from antarctic_routing.routing.candidates import Candidate, plan_candidates
from antarctic_routing.routing.departure import sweep_departures
from antarctic_routing.routing.fuel import VesselModel
from antarctic_routing.routing.hazard import exceedance_probability
from antarctic_routing.routing.replan import ReplanPolicy, replan
from antarctic_routing.synthetic import ScenarioSet, generate_synthetic

log = logging.getLogger("antarctic_routing.api")

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
MAX_DEPARTURE_DATES = 62
MAX_GRIDS = 8
# The repository's docs/images figures come from controlled-synthetic runs.
FIGURES_MODE = os.environ.get("ANTROUTE_FIGURES_MODE", "controlled_synthetic")
# The dashboard needs nothing from other origins when the API serves it.
DASHBOARD_CSP = ("default-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; "
                 "frame-ancestors 'none'; form-action 'self'")
SYNTHETIC_LABEL = "controlled-synthetic scenarios on a schematic Drake Passage world (not real conditions)"
FIGURE_CAPTIONS = {
    "backtest": "Replay backtest vs naive and ice-edge-buffer baselines (held-out seasons)",
    "replay": "Day-by-day voyage replay: departure advice and sailed track vs observed ice",
    "departure_window": "Departure window from one forecast issue",
    "trust_horizon": "Forecast trust horizon (season-blocked bootstrap)",
    "forecast_skill": "U-Net vs baselines: MAE and ice-edge error by lead time",
    "reliability": "Probability calibration: reliability and Brier score",
    "sensitivity": "Fuel/speed assumption sensitivity",
    "route_map_iceberg": "Iceberg drift ensemble forcing a detour",
    "route_map": "Recommended route under the risk budget",
    "route_map_infeasible": "Infeasible departure: destination iced in",
    "departure_chart": "Departure sweep with fresh scenarios per date",
}


SEED = Field(default=42, ge=0, le=2**32 - 1)
RESOLUTION = Field(default=None, ge=5, le=100)   # finer grids make a request arbitrarily expensive


class Berg(BaseModel):
    id: str = Field(min_length=1, max_length=40)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class RouteRequest(BaseModel):
    departure: date
    scenarios: int | None = Field(default=None, ge=1, le=1000)
    resolution_km: float | None = RESOLUTION
    seed: int = SEED
    scenario_routes: int = Field(default=2, ge=0, le=10)
    icebergs: list[Berg] = Field(default=[], max_length=20)
    berg_radius_km: float = Field(default=10.0, ge=0, le=100)


class DepartureRequest(BaseModel):
    start: date
    end: date
    step_days: int = Field(default=1, ge=1, le=60)
    scenarios: int | None = Field(default=None, ge=1, le=1000)
    resolution_km: float | None = RESOLUTION
    seed: int = SEED


class ReplanRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    issued: date
    scenarios: int | None = Field(default=None, ge=1, le=1000)
    data_age_hours: float = Field(default=0.0, ge=0, le=24 * 365)
    seed: int = SEED


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def cors_origins(raw: str) -> list[str]:
    """Browser origins allowed to call the API: each exactly ``https://host[:port]`` (``http`` only for localhost).

    A wildcard, a path, a trailing slash or upper case would either open the API to every site or never match the
    ``Origin`` header a browser sends, so they stop the server at start-up instead of failing silently.
    """
    out = []
    for item in (o.strip() for o in raw.split(",")):
        if not item:
            continue
        try:
            u = urlsplit(item)
            port_ok = u.port is None or u.port > 0
        except ValueError:
            port_ok = False
        local = u.hostname in ("localhost", "127.0.0.1") if port_ok else False
        if not (port_ok and re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", u.hostname or "")
                and (u.scheme == "https" or (u.scheme == "http" and local))
                and item == f"{u.scheme}://{u.netloc}" and item == item.lower() and "@" not in u.netloc):
            raise ValueError(f"ANTROUTE_CORS_ORIGINS: {item!r} is not an exact origin; use https://host[:port] "
                             "(http only for localhost) with no path, trailing slash or wildcard")
        out.append(item)
    return out


@dataclass(frozen=True)
class RouteContext:
    """What a stored voyage needs to export its route. Keeping the whole scenario world instead would hold
    hundreds of MB per voyage (about 215 MB for 200 scenarios on the 10 km grid)."""

    grid: PolarGrid
    start: datetime
    execution_mode: str
    description: str

    @classmethod
    def of(cls, world: ScenarioSet) -> RouteContext:
        return cls(world.grid, world.start, world.execution_mode, world.description)


class BodySizeLimit:
    """Refuse request bodies over ``max_bytes`` with 413, whether or not they declare a Content-Length."""

    def __init__(self, app, max_bytes: int) -> None:
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.max_bytes):
            code = 413 if declared.isdigit() else 400
            return await JSONResponse({"detail": self._message() if code == 413 else "invalid Content-Length"},
                                      status_code=code)(scope, receive, send)
        seen = 0

        async def limited():
            nonlocal seen
            message = await receive()
            seen += len(message.get("body", b""))
            if seen > self.max_bytes:     # FastAPI passes HTTPException through its body parsing
                raise HTTPException(413, self._message())
            return message

        return await self.app(scope, limited, send)

    def _message(self) -> str:
        return f"request body larger than {self.max_bytes} bytes (ANTROUTE_MAX_BODY_BYTES)"


def _package_versions() -> dict:
    out = {}
    for name in ("numpy", "scipy", "xarray", "pyproj", "torch", "fastapi", "pydantic"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _horizon_days(cfg: ProjectConfig) -> int:
    from antarctic_routing.cli import _horizon_days as h

    return h(cfg)


class Service:
    def __init__(self, config_path: str | Path, artifacts_dir: str | Path | None = None) -> None:
        self.config_path = Path(config_path)
        self.cfg = load_config(self.config_path)
        self.config_sha256 = sha256_file(self.config_path)
        self.vessel = VesselModel.from_config(self.cfg)
        self.horizon_days = _horizon_days(self.cfg)
        self._grids: dict[float, PolarGrid] = {}
        self._grid_lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.voyages: dict[str, dict] = {}
        self.max_jobs = _env_int("ANTROUTE_MAX_JOBS", 200)
        self.max_voyages = _env_int("ANTROUTE_MAX_VOYAGES", 100)
        # scenarios x grid cells per world; 200 scenarios on the 10 km grid (5.3 M) peak at about 1.3 GB RSS
        self.max_scenario_cells = _env_int("ANTROUTE_MAX_SCENARIO_CELLS", 6_000_000)
        self.lock = threading.Lock()
        # one budget for every computation, in a request (?wait=true, voyages, replans) or a background job,
        # so peak memory is about idle + max_compute x the largest allowed request
        self.max_compute = _env_int("ANTROUTE_MAX_COMPUTE", 2)
        self.compute_slots = threading.BoundedSemaphore(self.max_compute)
        self.pool = ThreadPoolExecutor(max_workers=self.max_compute)
        self.bundle: Bundle | None = None
        self.bundle_status, self.bundle_reason = UNAVAILABLE, "no real-data bundle configured (ANTROUTE_ARTIFACTS_DIR)"
        if artifacts_dir:
            try:
                self.bundle = load_bundle(artifacts_dir)
                self.bundle_status, self.bundle_reason = "available", None
                log.info("real-data bundle %s loaded and verified", self.bundle.index.get("bundle_id"))
            except BundleError as exc:
                self.bundle_status, self.bundle_reason = "failed", f"real-data bundle rejected: {exc}"
                log.error("%s", self.bundle_reason)

    @contextmanager
    def inline_slot(self):
        if not self.compute_slots.acquire(blocking=False):
            raise HTTPException(503, "the server is busy with other computations (ANTROUTE_MAX_COMPUTE); retry "
                                     "shortly, or submit a job without ?wait=true", headers={"Retry-After": "10"})
        try:
            yield
        finally:
            self.compute_slots.release()

    def require_bundle(self) -> Bundle:
        if self.bundle is None:
            raise HTTPException(503, {"status": self.bundle_status, "reason": self.bundle_reason})
        return self.bundle

    # -------------------------------------------------------------- helpers
    def grid(self, resolution_km: float | None) -> PolarGrid:
        res = float(resolution_km or self.cfg.grid.resolution_km)
        with self._grid_lock:
            if res not in self._grids:
                if len(self._grids) >= MAX_GRIDS:     # resolution_km is any number: keep the cache bounded
                    self._grids.pop(next(iter(self._grids)))
                self._grids[res] = PolarGrid.from_domain(self.cfg.domain, res)
            return self._grids[res]

    def check_scenarios(self, n: int | None, resolution_km: float | None = None) -> int:
        n = n or self.cfg.forecast.route_scenarios
        r = self.cfg.routing
        if r.risk_estimator == "wilson_upper":
            need = min_scenarios_for_budget(r.risk_budget, r.confidence)
            if n < need:
                raise HTTPException(422, f"{n} scenarios cannot certify the {r.risk_budget:.0%} risk budget; "
                                         f"at least {need} are required")
        g = self.grid(resolution_km)
        size = n * g.shape[0] * g.shape[1]
        if size > self.max_scenario_cells:
            raise HTTPException(422, f"{n} scenarios on a {g.shape[0]}x{g.shape[1]} grid exceed this server's limit "
                                     f"of {self.max_scenario_cells:,} scenario-cells; use fewer scenarios or a "
                                     f"coarser grid")
        return n

    def world(self, start: date, n: int, resolution_km: float | None, seed: int,
              bergs: list[Berg] = (), berg_radius_km: float = 10.0) -> ScenarioSet:
        world = generate_synthetic(self.cfg, self.grid(resolution_km), start, _horizon_days(self.cfg), n, seed=seed)
        if bergs:
            from antarctic_routing.iceberg.drift import add_iceberg_hazard

            world = add_iceberg_hazard(world, [(b.id, b.lat, b.lon) for b in bergs],
                                       rng=np.random.default_rng(seed), radius_m=berg_radius_km * 1e3)
        return world

    def endpoints(self, grid: PolarGrid):
        o, d = self.cfg.route.origin, self.cfg.route.destination
        return grid.cell_of(o.lat, o.lon), grid.cell_of(d.lat, d.lon)

    # --------------------------------------------------------- serialisers
    @staticmethod
    def map_payload(world: ScenarioSet, tau: float, hours: float) -> dict:
        g = world.grid
        t = world.time_index(hours / 2 if np.isfinite(hours) else 0.0)
        p_ice = np.where(world.land, 0.0, exceedance_probability(world.conc[:, t], tau))
        p_berg = world.berg[:, t].mean(axis=0) if world.berg is not None else None
        res_km = g.resolution_m / 1000.0
        return {
            "nx": int(g.x.size), "ny": int(g.y.size), "res_km": res_km,
            "x0_km": float(g.x[0] / 1000.0 - res_km / 2), "y0_km": float(g.y[0] / 1000.0 - res_km / 2),
            "rotation_rad": float(np.deg2rad(np.median(g.lon2d))),
            "layer_day": t,
            "land": world.land.astype(np.uint8).ravel().tolist(),
            "p_ice": to_percent(p_ice),   # None = ocean cell without data, never shown as open water
            "p_berg": np.round(p_berg * 100).astype(int).ravel().tolist() if p_berg is not None else None,
        }

    @staticmethod
    def candidate_payload(c: Candidate, world: ScenarioSet) -> dict:
        g = world.grid
        return {
            "labels": c.labels, "tags": c.tags, "feasible": c.feasible,
            **c.evaluation.summary(),
            "xy_km": [[float(g.x[col] / 1000.0), float(g.y[row] / 1000.0)] for row, col in c.route.cells],
            "latlon": [[round(float(g.lat2d[row, col]), 5), round(float(g.lon2d[row, col]), 5)]
                       for row, col in c.route.cells],
            "segment_breach_prob": c.evaluation.segment_breach_prob,
        }

    def plan_payload(self, world: ScenarioSet, plan, bergs=()) -> dict:
        g = world.grid
        o, d = self.endpoints(g)
        rec_hours = plan.recommended.evaluation.expected_hours if plan.recommended else 24.0
        return {
            "status": plan.status, "explanation": plan.explanation, "risk_budget": plan.risk_budget,
            "estimator": plan.estimator, "departure_utc": world.start.isoformat() + "Z",
            "n_scenarios": world.n_scenarios, "execution_mode": world.execution_mode,
            "data_description": world.description, "data_status": SCHEMATIC, "data_label": SYNTHETIC_LABEL,
            "recommended_index": next((i for i, c in enumerate(plan.candidates) if c is plan.recommended), None),
            "candidates": [self.candidate_payload(c, world) for c in plan.candidates],
            "origin_xy_km": [float(g.x[o[1]] / 1000), float(g.y[o[0]] / 1000)],
            "destination_xy_km": [float(g.x[d[1]] / 1000), float(g.y[d[0]] / 1000)],
            "icebergs": [b.model_dump() for b in bergs],
            "map": self.map_payload(world, self.vessel.tau, rec_hours),
            "disclaimer": DISCLAIMER,
        }

    # ---------------------------------------------------------------- jobs
    def submit(self, kind: str, fn, wait: bool):
        job_id = uuid.uuid4().hex[:12]
        job = {"job_id": job_id, "kind": kind, "status": "queued", "created_at": utc_now(), "result": None,
               "error": None}
        with self.lock:
            if len(self.jobs) >= self.max_jobs:      # forget the oldest finished jobs first
                for old in [k for k, j in self.jobs.items() if j["status"] in ("done", "failed")]:
                    del self.jobs[old]
                    if len(self.jobs) < self.max_jobs:
                        break
            if len(self.jobs) >= self.max_jobs:
                raise HTTPException(503, "job store is full of unfinished jobs; retry later")
            self.jobs[job_id] = job

        def run():
            job["status"] = "running"
            try:
                job["result"] = fn()
                job["status"] = "done"
            except Exception as exc:  # noqa: BLE001 - surfaced to the client as a failed job
                log.exception("job %s (%s) failed", job_id, kind)
                job["status"], job["error"] = "failed", f"{type(exc).__name__}: {exc}"
            job["finished_at"] = utc_now()

        def run_queued():     # background jobs wait for a compute slot; requests never do
            with self.compute_slots:
                run()

        if wait:
            try:
                with self.inline_slot():
                    run()
            except HTTPException:
                with self.lock:
                    self.jobs.pop(job_id, None)
                raise
            return JSONResponse(job, status_code=200)
        self.pool.submit(run_queued)
        return JSONResponse({"job_id": job_id, "status": "queued"}, status_code=202)


def create_app(config_path: str | Path | None = None, artifacts_dir: str | Path | None = None) -> FastAPI:
    svc = Service(config_path or os.environ.get("ANTROUTE_CONFIG", "config/config.yaml"),
                  artifacts_dir if artifacts_dir is not None else os.environ.get("ANTROUTE_ARTIFACTS_DIR") or None)
    app = FastAPI(title="Antarctic Vessel Routing & Ice-Risk API", version=__version__,
                  description=DISCLAIMER)
    app.state.service = svc
    hs = register_historical(app, svc)

    # innermost, so a 413 still carries the CORS headers and the browser can read it
    app.add_middleware(BodySizeLimit, max_bytes=_env_int("ANTROUTE_MAX_BODY_BYTES", 64 * 1024))
    origins = cors_origins(os.environ.get("ANTROUTE_CORS_ORIGINS", ""))
    if origins:   # only for a separately hosted dashboard; same-origin needs none
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type"], allow_credentials=False,
                           expose_headers=["X-Request-ID"], max_age=600)

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        rid = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        log.info("%s %s -> %d %.0f ms [%s]", request.method, request.url.path, response.status_code,
                 1000 * (time.perf_counter() - t0), rid)
        return response

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        rid = uuid.uuid4().hex[:12]
        log.error("unhandled error on %s %s [%s]", request.method, request.url.path, rid,
                  exc_info=(type(exc), exc, exc.__traceback__))
        return JSONResponse({"detail": "internal server error", "error_id": rid}, status_code=500)

    @app.get("/health")
    def health():
        modes = ["controlled_synthetic"] + (["real"] if svc.bundle is not None else [])
        return {"status": "degraded" if svc.bundle_status == "failed" else "ok", "version": __version__,
                "time": utc_now(), "disclaimer": DISCLAIMER, "data_modes": modes, "real_data": svc.bundle_status,
                "jobs": len(svc.jobs), "voyages": len(svc.voyages)}

    @app.get("/ready")
    def ready():
        # Deployment health check: a release whose bundle is missing or fails verification is never "ready",
        # so the platform keeps the previous release instead of serving one without its real data.
        if svc.bundle is None:
            return JSONResponse({"ready": False, "real_data": svc.bundle_status, "reason": svc.bundle_reason},
                                status_code=503)
        return {"ready": True, "real_data": svc.bundle_status, "bundle_id": svc.bundle.index.get("bundle_id")}

    @app.get("/status")
    def status():
        b = svc.bundle
        real = {"status": svc.bundle_status, "reason": svc.bundle_reason, "sources": []}
        if b is not None:
            real.update(bundle_id=b.index.get("bundle_id"), kind=b.index.get("kind"), issue=b.index.get("issue"),
                        execution_mode=b.index.get("execution_mode"), sources=source_statuses(b),
                        limitations=b.index.get("limitations", []))
        return {
            "version": __version__, "time": utc_now(), "disclaimer": DISCLAIMER,
            "historical": hs.status(),
            "interactive_planner": {"status": SCHEMATIC, "execution_mode": "controlled_synthetic",
                                    "label": SYNTHETIC_LABEL, "endpoints": ["/routes", "/departures", "/voyages"]},
            "real_data": real,
            "figures": {"status": SCHEMATIC if FIGURES_MODE == "controlled_synthetic" else FIGURES_MODE,
                        "execution_mode": FIGURES_MODE},
        }

    @app.get("/versions")
    def versions():
        b = svc.bundle
        model = None
        if b is not None:
            fm = (b.manifest.get("sea_ice") or {}).get("forecast_model") or {}
            model = {"id": fm.get("id"), "sha256": fm.get("sha256"), "code_commit": fm.get("code_commit")}
        return {"software": {"name": "antarctic-routing", "version": __version__,
                             "git_commit": os.environ.get("ANTROUTE_GIT_COMMIT")
                             or os.environ.get("RENDER_GIT_COMMIT") or None},
                "python": platform.python_version(), "packages": _package_versions(),
                "config_sha256": svc.config_sha256, "forecast_model": model,
                "bundle": None if b is None else {"bundle_id": b.index.get("bundle_id"),
                                                  "created_utc": b.index.get("created_utc"),
                                                  "schema": b.index.get("schema")}}

    @app.get("/provenance")
    def provenance():
        b = svc.require_bundle()
        return {"bundle": b.index, "manifest": b.manifest}

    @app.get("/config")
    def config():
        r = svc.cfg.routing
        return {"config": svc.cfg.model_dump(mode="json"),
                "min_scenarios_for_budget": min_scenarios_for_budget(r.risk_budget, r.confidence),
                "config_file": svc.config_path.name, "config_sha256": svc.config_sha256}

    # ------------------------------------------------- real-data (read-only)
    def _layers(b: Bundle) -> list[dict]:
        return b.forecast_maps.get("layers") or []

    def _grid(b: Bundle) -> dict:
        return {"grid": b.forecast_maps.get("grid"), "land": b.forecast_maps.get("land")}

    @app.get("/real/plan-window")
    def real_plan_window():
        b = svc.require_bundle()
        return {**b.plan_window, "data_status": source_statuses(b)}

    @app.get("/real/forecast-dates")
    def real_forecast_dates():
        b = svc.require_bundle()
        fp = b.plan_window.get("forcing_provenance") or {}
        return {"issue": b.plan_window.get("issue"),
                "layers": [{"index": la["index"], "date": la["date"], "source": la["source"]} for la in _layers(b)],
                "departure_dates": [o["departure"] for o in b.plan_window.get("options", [])],
                "wind_dates": fp.get("wind_dates"), "current_dates": fp.get("current_dates")}

    @app.get("/real/sea-ice")
    def real_sea_ice():
        b = svc.require_bundle()
        obs = b.forecast_maps.get("observed")
        if obs is None:
            raise HTTPException(404, "this bundle has no observed sea-ice layer")
        return {"status": "historical", "label": "observed sea-ice concentration (%) on the issue date",
                **obs, **_grid(b)}

    @app.get("/real/forecast-map")
    def real_forecast_map(layer: int = Query(0, ge=0)):
        b = svc.require_bundle()
        layers = _layers(b)
        if layer >= len(layers):
            raise HTTPException(422, f"layer must be between 0 and {len(layers) - 1}")
        la = layers[layer]
        status = {"observed": "historical", "forecast": "forecast", "climatology": "forecast"}.get(la["source"],
                                                                                                    UNAVAILABLE)
        return {"status": status, "tau": b.forecast_maps.get("tau"), "n_members": b.forecast_maps.get("n_members"),
                **la, **_grid(b)}

    @app.get("/real/route")
    def real_route():
        b = svc.require_bundle()
        sel = b.plan_window.get("selected_route")
        if sel is None:
            return {"status": UNAVAILABLE, "selected": None, "explanation": b.plan_window.get("explanation")}
        g = b.forecast_maps.get("grid") or {}
        xs, ys = g.get("x_km") or [], g.get("y_km") or []
        xy = [[xs[c], ys[r]] for r, c in sel.get("cells_row_col", [])] if xs and ys else []
        return {"status": "forecast", "selected": b.plan_window.get("selected"),
                "explanation": b.plan_window.get("explanation"), **sel, "xy_km": xy}

    @app.get("/real/departure-window")
    def real_departure_window():
        b = svc.require_bundle()
        p = b.plan_window
        routing = b.manifest.get("routing") or {}
        return {"status": "forecast", "issue": p.get("issue"), "selected": p.get("selected"),
                "explanation": p.get("explanation"), "rule": p.get("rule"), "options": p.get("options", []),
                "risk_budget": routing.get("risk_budget"), "trust_horizon_days": p.get("trust_horizon_days"),
                "n_members": p.get("n_members")}

    @app.get("/real/icebergs")
    def real_icebergs():
        b = svc.require_bundle()
        p = b.plan_window
        src = p.get("iceberg_source")
        return {"status": "historical" if src and src.get("execution_mode") == "real" else UNAVAILABLE,
                "source": src, "drifted": p.get("icebergs", []), "outside_grid": p.get("icebergs_outside_grid", []),
                "drift": p.get("iceberg_drift"), "hazard": p.get("iceberg_hazard"),
                "tracks": b.forecast_maps.get("iceberg_tracks"),      # daily ensemble-mean drift, forecast
                "presence_on_selected_route": (p.get("selected_route") or {}).get("iceberg_presence_on_route")}

    @app.get("/real/figures/departure_window.png")
    def real_figure():
        b = svc.require_bundle()
        if b.figure is None:
            raise HTTPException(404, "this bundle has no departure-window figure")
        return FileResponse(b.figure, media_type="image/png")

    @app.post("/routes")
    def routes(req: RouteRequest, wait: bool = Query(False)):
        n = svc.check_scenarios(req.scenarios, req.resolution_km)

        def work():
            world = svc.world(req.departure, n, req.resolution_km, req.seed, req.icebergs, req.berg_radius_km)
            o, d = svc.endpoints(world.grid)
            r = svc.cfg.routing
            plan = plan_candidates(world, svc.vessel, o, d, r.risk_budget, r.risk_weights, r.risk_estimator,
                                   r.connectivity, req.scenario_routes, r.confidence, req.seed)
            return svc.plan_payload(world, plan, req.icebergs)

        return svc.submit("routes", work, wait)

    @app.post("/departures")
    def departures(req: DepartureRequest, wait: bool = Query(False)):
        n = svc.check_scenarios(req.scenarios, req.resolution_km)
        if req.end < req.start:
            raise HTTPException(422, "end must not be before start")
        if (req.end - req.start).days // req.step_days + 1 > MAX_DEPARTURE_DATES:
            raise HTTPException(422, f"at most {MAX_DEPARTURE_DATES} departure dates per sweep")

        def work():
            dates, cur = [], req.start
            while cur <= req.end:
                dates.append(cur)
                cur += timedelta(days=req.step_days)
            grid = svc.grid(req.resolution_km)
            o, d = svc.endpoints(grid)
            r = svc.cfg.routing
            sweep = sweep_departures(dates, lambda day: svc.world(day, n, req.resolution_km, req.seed), svc.vessel,
                                     o, d, r.risk_budget, r.risk_weights, r.risk_estimator, r.connectivity, 0,
                                     r.confidence, req.seed)
            return {**sweep.to_dict(), "execution_mode": "controlled_synthetic", "data_status": SCHEMATIC,
                    "data_label": SYNTHETIC_LABEL, "disclaimer": DISCLAIMER}

        return svc.submit("departures", work, wait)

    @app.get("/jobs/{job_id}")
    def job(job_id: str):
        if job_id not in svc.jobs:
            raise HTTPException(404, "unknown job")
        return svc.jobs[job_id]

    @app.post("/voyages", status_code=201)
    def create_voyage(req: RouteRequest):
        n = svc.check_scenarios(req.scenarios, req.resolution_km)
        if len(svc.voyages) >= svc.max_voyages:
            raise HTTPException(503, "voyage store is full (in-memory, ANTROUTE_MAX_VOYAGES); restart to clear it")
        with svc.inline_slot():
            world = svc.world(req.departure, n, req.resolution_km, req.seed, req.icebergs, req.berg_radius_km)
            o, d = svc.endpoints(world.grid)
            r = svc.cfg.routing
            plan = plan_candidates(world, svc.vessel, o, d, r.risk_budget, r.risk_weights, r.risk_estimator,
                                   r.connectivity, req.scenario_routes, r.confidence, req.seed)
        if plan.recommended is None:
            raise HTTPException(409, plan.explanation)
        vid = uuid.uuid4().hex[:12]
        cand = plan.recommended
        svc.voyages[vid] = {"request": req, "candidate": cand, "world": RouteContext.of(world), "events": [{
            "event": "planned", "at": utc_now(), "departure": req.departure.isoformat(),
            "explanation": plan.explanation, **cand.evaluation.summary()}]}
        return {"voyage_id": vid, "route": svc.candidate_payload(cand, world), "explanation": plan.explanation,
                "execution_mode": world.execution_mode, "data_status": SCHEMATIC, "disclaimer": DISCLAIMER}

    def _voyage(vid: str) -> dict:
        if vid not in svc.voyages:
            raise HTTPException(404, "unknown voyage")
        return svc.voyages[vid]

    @app.post("/voyages/{vid}/replan")
    def replan_voyage(vid: str, req: ReplanRequest):
        v = _voyage(vid)
        if v.get("mode") == "historical":     # never replan a real voyage on a synthetic world
            raise HTTPException(409, "this voyage uses Real Historical Data; replan it with "
                                     "/real/historical/voyages/{id}/replan")
        n = svc.check_scenarios(req.scenarios, v["request"].resolution_km)
        with svc.inline_slot():
            world = svc.world(req.issued, n, v["request"].resolution_km, req.seed)
            try:
                position = world.grid.cell_of(req.lat, req.lon)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from None
            r = svc.cfg.routing
            policy = ReplanPolicy(r.risk_budget, r.risk_estimator, r.confidence)
            try:
                decision = replan(world, position, v["candidate"].route, svc.vessel, policy, r.risk_weights,
                                  r.connectivity, 0, data_age_hours=req.data_age_hours, seed=req.seed)
            except ValueError as exc:
                raise HTTPException(400, f"{exc}; send a position on the current route") from None
        if decision.action == "switch":
            v["candidate"] = decision.new_candidate
        else:
            v["candidate"] = Candidate(v["candidate"].labels, decision.previous_route, decision.previous_evaluation,
                                       decision.action == "keep", decision.previous_evaluation.p_breach_upper)
        v["world"] = RouteContext.of(world)
        record = decision.record(issued=req.issued.isoformat(), event="replan")
        record.pop("previous", None)
        record.pop("new", None)
        v["events"].append(record)
        return {**record, "route": svc.candidate_payload(v["candidate"], world)}

    @app.get("/voyages/{vid}/history")
    def history(vid: str):
        return {"voyage_id": vid, "events": _voyage(vid)["events"]}

    @app.get("/voyages/{vid}/export")
    def export(vid: str, format: str = Query("geojson", pattern="^(geojson|csv)$")):  # noqa: A002
        v = _voyage(vid)
        gj = route_to_geojson(v["candidate"], v["world"], issued=v["world"].start.isoformat() + "Z")
        labels = v.get("data_labels")  # Real Historical Data voyages carry their label and hindsight disclosure
        if labels:
            gj["features"][0]["properties"].update(labels)
        if format == "geojson":
            return gj
        buf = io.StringIO()
        rows = [f["properties"] for f in gj["features"][1:]]
        coords = [f["geometry"]["coordinates"] for f in gj["features"][1:]]
        extra = ["data_label", "hindsight_forcing"] if labels else []
        w = csv.writer(buf)
        w.writerow(["waypoint_index", "lat", "lon", "planned_arrival_utc", "segment_breach_prob", "disclaimer",
                    *extra])
        for p, (lon, lat) in zip(rows, coords, strict=True):
            w.writerow([p["waypoint_index"], lat, lon, p["planned_arrival_utc"], p["segment_breach_prob"],
                        DISCLAIMER, *(labels[k] for k in extra)])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=voyage_{vid}.csv"})

    figures_dir = Path(os.environ.get("ANTROUTE_FIGURES", svc.config_path.resolve().parent.parent / "docs" / "images"))

    @app.get("/figures")
    def figures():
        if not figures_dir.is_dir():
            return {"figures": [], "execution_mode": FIGURES_MODE}
        order = list(FIGURE_CAPTIONS)
        rank = {name: i for i, name in enumerate(order)}
        files = sorted(figures_dir.glob("*.png"), key=lambda p: (rank.get(p.stem, 99), p.stem))
        return {"execution_mode": FIGURES_MODE,
                "figures": [{"url": f"/figures/{p.name}", "caption": FIGURE_CAPTIONS.get(p.stem, p.stem)}
                            for p in files]}

    if figures_dir.is_dir():
        app.mount("/figures", StaticFiles(directory=figures_dir), name="figures")

    if DASHBOARD_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def dashboard():
            return FileResponse(DASHBOARD_DIR / "index.html", headers={"Content-Security-Policy": DASHBOARD_CSP})

    return app


app = create_app() if os.environ.get("ANTROUTE_CONFIG") else None
