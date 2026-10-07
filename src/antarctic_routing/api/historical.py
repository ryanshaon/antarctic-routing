"""Real Historical Data endpoints - the real-data planner on past seasons, computed on request.

GET  /real/historical/status                 availability, coverage, hindsight-forcing disclosure, limitations
GET  /real/historical/dates                  issue dates the archive supports, with in-sample flags per season
POST /real/historical/routes[?wait=true]     risk-budgeted route for a departure on the issue date (job)
POST /real/historical/departures[?wait=true] departure window from one forecast issue (job)
POST /real/historical/voyages                create a voyage from a real route plan
POST /real/historical/voyages/{id}/replan    replan from a position on the route with a later forecast
POST /real/historical/replay[?wait=true]     day-by-day replay with icebergs, scored against observations (job)
GET  /real/locations                         origin/destination presets, resolved on the routing grid
POST /real/locations/resolve                 resolve an origin/destination pair: snapped cells, route horizon,
                                             scenario days and (with an issue date) coverage
POST /real/plan                              one call: resolved ends, recommended departure and route, ETA,
                                             distance, fuel index, sea-ice / iceberg / combined risk, daily
                                             timeline and map layers (see :mod:`antarctic_routing.product`)
GET  /real/plans                             the most recent saved plans (every /real/plan result is saved; see
                                             :mod:`antarctic_routing.store`)
GET  /real/plans/{id}                        a saved plan and the request that produced it
GET  /real/plans/{id}/brief                  its PDF brief (see :mod:`antarctic_routing.brief`)
POST /real/simulate[?wait=true]              the voyage /real/plan chose, sailed day by day through the observed
                                             ice with the existing daily replanning, as playback frames (job; see
                                             :mod:`antarctic_routing.simulation`)

``/real/plan`` and ``/real/simulate`` also accept any issue date after the archive's last day
(``metadata.mode == "forecast"``, a labelled estimate). When the forecast pathway can serve it, it is planned there:
the same engine from a labelled analogue start with proxy sea ice and forcing and the latest official iceberg list
(see :mod:`antarctic_routing.forecast_mode`). Otherwise - another month, another year - it is a **historical seasonal
analogue** estimate from real observations of the same calendar period in earlier years
(``metadata.forecast.pathway == "seasonal_analogue"``; see :mod:`antarctic_routing.seasonal_analogue`).

Routes, departure windows, voyages and replays take an optional ``origin`` and ``destination`` (a preset id
``{"preset": "palmer_station"}`` or a point ``{"lat": .., "lon": ..}``; both or neither). Points are snapped to
the nearest navigable 25 km cell (never onto land) and the forecast horizon is computed for that route (see
:mod:`antarctic_routing.locations`). Without them the configured route is used exactly as before.

The voyage history and export endpoints (``/voyages/{id}/history``, ``/export``) serve these voyages too.

Inputs are the verified files of ``config/real_historical.json`` under ``ANTROUTE_DATA_ROOT``; the model
and the scenario context load once, on the first computation. Parameters are the frozen ones (200 joint
scenarios, seed 42, calibrated drift), so nothing here is tunable per request. A missing input or
dependency answers 503 with the reason and a date the archive cannot support answers 422; nothing
synthetic is ever substituted. Every response is labelled "Real Historical Data" and carries the
hindsight-forcing disclosure.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field

from antarctic_routing import DISCLAIMER
from antarctic_routing.common.provenance import utc_now
from antarctic_routing.forecast_mode import (
    BANNERS as FORECAST_BANNERS,
)
from antarctic_routing.forecast_mode import (
    DATA_STATUS as FORECAST_DATA_STATUS,
)
from antarctic_routing.forecast_mode import (
    EXECUTION_MODE as FORECAST_EXECUTION_MODE,
)
from antarctic_routing.forecast_mode import (
    ForecastPlanner,
    ForecastUnavailable,
    forecast_metadata,
    forecast_usnic,
    resolve,
    shift_dates,
)
from antarctic_routing.forecasting.scenarios import grid_of
from antarctic_routing.historical import (
    DATA_STATUS,
    LABEL,
    HistoricalArchive,
    HistoricalPlanner,
    HistoricalUnavailable,
    OutOfCoverage,
    UsnicArchive,
    drift_kwargs,
    provenance,
)
from antarctic_routing.ingestion.icebergs import in_grid_mask
from antarctic_routing.locations import MAX_SNAP_KM, LocationError, LocationResolver, RouteSpec, build_route
from antarctic_routing.product import (
    BANNERS,
    TIMELINE_NOTE,
    chosen_candidate,
    daily_timeline,
    environment_layers,
    json_safe,
    risk_breakdown,
    route_summary,
)
from antarctic_routing.publish import UNAVAILABLE, grid_geometry, iceberg_tracks, to_percent
from antarctic_routing.routing.candidates import Candidate, plan_candidates
from antarctic_routing.routing.departure import plan_from_issue
from antarctic_routing.routing.replan import ReplanPolicy, replan
from antarctic_routing.seasonal_analogue import (
    AnalogueData,
    AnaloguePlanner,
    AnalogueUnavailable,
    analogue_metadata,
)
from antarctic_routing.seasonal_analogue import (
    resolve as resolve_analogue,
)
from antarctic_routing.store import PlanStore

log = logging.getLogger("antarctic_routing.api")

LAND_LABEL = "Land (sea-ice product mask; not a navigational coastline)"
DATA_AGE_BASIS = ("Historical mode: a replan uses the sea-ice analysis of its issue date, assumed in hand that day "
                  "(operational product latency is not modelled); the USNIC list age is reported separately.")


class LocationIn(BaseModel):
    """A preset id, or a WGS84 point (optionally named)."""

    preset: str | None = Field(default=None, min_length=1, max_length=40)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    name: str | None = Field(default=None, max_length=80)


class RouteEnds(BaseModel):
    origin: LocationIn | None = None
    destination: LocationIn | None = None


class HistoricalRouteRequest(RouteEnds):
    issue: date


class HistoricalWindowRequest(RouteEnds):
    issue: date
    window_days: int = Field(default=14, ge=1, le=14)


class ResolveRequest(BaseModel):
    origin: LocationIn
    destination: LocationIn
    issue: date | None = None
    window_days: int = Field(default=14, ge=1, le=14)


class PlanRequest(BaseModel):
    origin: LocationIn
    destination: LocationIn
    issue: date
    window_days: int = Field(default=14, ge=1, le=14)
    include_layers: bool = True


class SimulateRequest(BaseModel):
    """The inputs of a /real/plan request; ``departure`` (optional) must match the plan's chosen departure."""
    origin: LocationIn
    destination: LocationIn
    issue: date
    window_days: int = Field(default=14, ge=1, le=14)
    departure: date | None = None


class HistoricalReplanRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    issued: date


class HistoricalReplayRequest(RouteEnds):
    start: date
    max_wait_days: int = Field(default=7, ge=0, le=21)


class HistoricalService:
    """Loads the verified archive (cheap) and the U-Net scenario context (heavy) once, on demand."""

    def __init__(self, svc, data_root: str | None) -> None:
        self.svc = svc
        self.data_root = data_root
        self.spec_path = Path(os.environ.get("ANTROUTE_HISTORICAL_SPEC")
                              or svc.config_path.resolve().parent / "real_historical.json")
        self._archive: HistoricalArchive | None = None
        self._planner: HistoricalPlanner | None = None
        self._resolver: LocationResolver | None = None
        self._error: HistoricalUnavailable | None = None
        self._analogue: AnalogueData | None = None
        self._analogue_error: HistoricalUnavailable | None = None
        self._lock = threading.Lock()

    def _unavailable(self) -> HistoricalUnavailable | None:
        if not self.data_root:
            return HistoricalUnavailable(UNAVAILABLE, "no real-data archive configured (set ANTROUTE_DATA_ROOT)")
        if not self.spec_path.is_file():
            return HistoricalUnavailable("blocked", f"input list {self.spec_path.name} not found")
        return self._error

    def archive(self) -> HistoricalArchive:
        with self._lock:
            if self._archive is None and self._unavailable() is None:
                try:
                    self._archive = HistoricalArchive.load(self.data_root, self.spec_path,
                                                           self.svc.cfg.project.season_months)
                    log.info("Real Historical Data archive verified under %s", self.data_root)
                except HistoricalUnavailable as exc:
                    self._error = exc
                    log.error("Real Historical Data %s: %s", exc.status, exc.reason)
            err = self._unavailable()
        if err is not None:
            raise HTTPException(503, {"status": err.status, "reason": err.reason, "label": LABEL})
        return self._archive

    def planner(self) -> HistoricalPlanner:
        archive = self.archive()
        with self._lock:
            if self._planner is None and self._error is None:
                try:
                    self._planner = HistoricalPlanner(archive)
                    log.info("Real Historical Data forecast context built")
                except HistoricalUnavailable as exc:
                    self._error = exc
                    log.error("Real Historical Data %s: %s", exc.status, exc.reason)
            err = self._error
        if err is not None:
            raise HTTPException(503, {"status": err.status, "reason": err.reason, "label": LABEL})
        return self._planner

    def analogue(self) -> AnalogueData:
        """The year-round observations for seasonal-analogue estimates (raises HistoricalUnavailable)."""
        archive = self.archive()
        with self._lock:
            if self._analogue is None and self._analogue_error is None:
                try:
                    self._analogue = AnalogueData.load(archive)
                    log.info("Seasonal-analogue inputs verified under %s", self.data_root)
                except HistoricalUnavailable as exc:
                    self._analogue_error = exc
                    log.error("Seasonal analogue %s: %s", exc.status, exc.reason)
            if self._analogue_error is not None:
                raise self._analogue_error
            return self._analogue

    def resolver(self) -> LocationResolver:
        """Snaps locations on the archive's routing grid and land mask (the cells routes actually use)."""
        a = self.archive()
        with self._lock:
            if self._resolver is None:
                self._resolver = LocationResolver(grid_of(a.ds), a.ds["land_mask"].values.astype(bool))
        return self._resolver

    def route(self, origin: LocationIn | None, destination: LocationIn | None) -> RouteSpec:
        """Resolved ends + route horizon; the configured route when neither end is given (422 when invalid)."""
        cfg = self.svc.cfg
        if (origin is None) != (destination is None):
            raise HTTPException(422, {"status": "invalid_location", "label": LABEL, "reason": "give both origin and "
                                      "destination, or neither for the configured route"})
        if origin is None:
            o, d = cfg.route.origin, cfg.route.destination
            origin = LocationIn(lat=o.lat, lon=o.lon, name="Configured origin")
            destination = LocationIn(lat=d.lat, lon=d.lon, name="Configured destination")
        try:
            spec = build_route(self.resolver(), _loc(origin), _loc(destination), cfg.vessel.cruise_speed_kmh,
                               cfg.grid.time_step_hours, cfg.forecast.lead_days)
        except LocationError as exc:
            raise HTTPException(422, {"status": "invalid_location", "reason": str(exc), "label": LABEL}) from None
        return spec

    def status(self) -> dict:
        out = {"label": LABEL, "execution_mode": "real", "data_status": DATA_STATUS}
        try:
            a = self.archive()
        except HTTPException as exc:
            return {**out, **exc.detail, "status": exc.detail["status"]}
        h = self.svc.horizon_days
        return {**out, "status": "available", "reason": None, "model_loaded": self._planner is not None,
                "hindsight_forcing": a.spec["hindsight_forcing"], "limitations": a.spec["limitations"],
                "parameters": a.params, "horizon_days": h, "window_days_max": a.params["window_days"],
                "coverage": {"route": a.coverage(1 + h), "window": a.coverage(a.params["window_days"] + h)}}


def _loc(x: LocationIn) -> dict:
    return x.model_dump(exclude_none=True)


def _labels(a: HistoricalArchive) -> dict:
    return {"execution_mode": "real", "data_status": DATA_STATUS, "data_label": LABEL,
            "hindsight_forcing": a.spec["hindsight_forcing"], "disclaimer": DISCLAIMER}


def register(app: FastAPI, svc) -> HistoricalService:
    hs = HistoricalService(svc, os.environ.get("ANTROUTE_DATA_ROOT") or None)
    cfg = svc.cfg
    H = svc.horizon_days
    store = app.state.plan_store = PlanStore.from_env()

    def ends(origin, destination) -> tuple[tuple[int, int], tuple[int, int], int, RouteSpec | None]:
        """Route cells and horizon: the exact configured path when no ends are given, else the resolved route."""
        if origin is None and destination is None:
            return (*svc.endpoints(grid_of(hs.archive().ds)), H, None)
        spec = hs.route(origin, destination)
        return (*spec.cells, spec.horizon_days, spec)

    def xy(grid, cell) -> list[float]:
        return [float(grid.x[cell[1]] / 1000), float(grid.y[cell[0]] / 1000)]

    def with_route(out: dict, grid, o, d, spec: RouteSpec | None) -> dict:
        out["origin_xy_km"], out["destination_xy_km"] = xy(grid, o), xy(grid, d)
        if spec is not None:
            out["route"] = spec.to_dict()
        return out

    def coverage_or_422(issue: date, n_days: int) -> HistoricalArchive:
        a = hs.archive()
        try:
            a.check(issue, n_days)
        except OutOfCoverage as exc:
            raise HTTPException(422, {"status": "out_of_coverage", "reason": str(exc), "label": LABEL}) from None
        return a

    def tracks(planner: HistoricalPlanner, world, snap) -> list[dict]:
        if not snap.bergs:
            return []
        p = planner.archive.params
        return iceberg_tracks(world, snap.bergs, p["seed"], p["berg_radius_km"] * 1e3, **planner.drift_kwargs())

    def plan_at_issue(issue: date, o, d, h: int):
        planner = hs.planner()
        p = planner.archive.params
        world, snap = planner.world(issue, 1 + h)
        r = cfg.routing
        plan = plan_candidates(world, svc.vessel, o, d, r.risk_budget, r.risk_weights, r.risk_estimator,
                               r.connectivity, p["scenario_routes"], r.confidence, p["seed"])
        return planner, world, snap, plan

    def route_payload(planner, world, snap, plan, issue: date, n_days: int) -> dict:
        a = planner.archive
        out = svc.plan_payload(world, plan)
        out.update(_labels(a), land_label=LAND_LABEL, icebergs=snap.summary(),
                   iceberg_tracks=tracks(planner, world, snap),
                   historical=provenance(a, cfg, issue, n_days, snap, planner))
        return out

    @app.get("/real/historical/status")
    def historical_status():
        return hs.status()

    @app.get("/real/historical/dates")
    def historical_dates(origin: str | None = Query(None, max_length=40),
                         destination: str | None = Query(None, max_length=40),
                         origin_lat: float | None = Query(None, ge=-90, le=90),
                         origin_lon: float | None = Query(None, ge=-180, le=180),
                         destination_lat: float | None = Query(None, ge=-90, le=90),
                         destination_lon: float | None = Query(None, ge=-180, le=180)):
        """Supported dates; each end is a preset id, or a point given by ``*_lat`` and ``*_lon``."""
        a = hs.archive()

        def to_loc(pid, lat, lon):
            if pid is not None or (lat is None and lon is None):
                return None if pid is None else LocationIn(preset=pid)
            if lat is None or lon is None:
                raise HTTPException(422, {"status": "invalid_location", "label": LABEL,
                                          "reason": "a point needs both latitude and longitude"})
            return LocationIn(lat=lat, lon=lon)

        _, _, h, spec = ends(to_loc(origin, origin_lat, origin_lon),
                             to_loc(destination, destination_lat, destination_lon))
        n_route, n_window = 1 + h, a.params["window_days"] + h
        out = {**_labels(a), "horizon_days": h, "window_days_max": a.params["window_days"],
               "route_dates": [d.isoformat() for d in a.available(n_route)],
               "window_dates": [d.isoformat() for d in a.available(n_window)],
               "seasons": {"route": a.coverage(n_route), "window": a.coverage(n_window)}}
        if spec is not None:
            out["route"] = spec.to_dict()
        out["forecast"] = forecast_dates(a, n_window)
        out["estimate"] = estimate_info(a)
        return out

    def estimate_info(a: HistoricalArchive) -> dict:
        """Any date after the archive is accepted: the forecast pathway on its dates, else a seasonal analogue."""
        info = {"any_date_after": a.days[-1].isoformat(), "label": "Forecast / hackathon estimate",
                "pathways": {"proxy_forecast": "the dates listed under 'forecast'",
                             "seasonal_analogue": "every other date after the archive"},
                "analogue": {"available": True, "reason": None, "label": "Historical seasonal analogue"}}
        try:
            hs.analogue()
        except HistoricalUnavailable as exc:
            info["analogue"].update(available=False, reason=f"{exc.status}: {exc.reason}")
        return info

    forecast_dates_cache: dict[int, dict] = {}

    def forecast_dates(a: HistoricalArchive, n_days: int) -> dict:
        """Requested dates after the archive that forecast mode can plan (an analogue start and a recent
        official iceberg list exist), for the date picker."""
        if n_days in forecast_dates_cache:
            return forecast_dates_cache[n_days]
        fm = a.spec.get("forecast_mode", {})
        info = {"mode": "forecast", "label": fm.get("label", "Forecast / hackathon estimate"),
                "disclosure": fm.get("disclosure"), "banners": FORECAST_BANNERS, "dates": [],
                "first": None, "last": None, "available": False, "reason": None}
        try:
            usnic = forecast_usnic(a)
        except ForecastUnavailable as exc:
            info["reason"] = str(exc)
            return info
        latest = max(d for d, _, _ in usnic.lists)
        day, stop = a.days[-1] + timedelta(days=1), latest + timedelta(days=int(fm.get("usnic_max_age_days", 120)))
        dates = []
        while day <= stop:
            try:
                resolve(a, day, n_days, usnic)
                dates.append(day.isoformat())
            except ForecastUnavailable:
                pass
            day += timedelta(days=1)
        info.update(dates=dates, first=dates[0] if dates else None, last=dates[-1] if dates else None,
                    available=bool(dates), reason=None if dates else "no date after the archive has an analogue "
                    "start and a recent official iceberg list")
        forecast_dates_cache[n_days] = info
        return info

    @app.get("/real/locations")
    def locations():
        res = hs.resolver()
        g = res.grid
        return {**_labels(hs.archive()), "presets": res.presets(), "max_snap_km": MAX_SNAP_KM,
                "grid": {"crs": "EPSG:3031", "resolution_km": g.resolution_m / 1000.0, "shape": list(g.shape),
                         "navigable_cells": int(res.navigable.sum())},
                "snapping": "A location on land, or in ocean cut off from the open sea, moves to the nearest "
                            f"navigable 25 km cell within {MAX_SNAP_KM:.0f} km; it is never placed on land.",
                "note": "Presets are research waypoints for decision support, not approved anchorages.",
                # For the dashboard's "Pick on map": the routing grid, its land mask and each cell centre's
                # WGS84 position (display and click-to-coordinate only; snapping stays on the server).
                "map": {**grid_geometry(g), "land": res.land.astype(np.uint8).ravel().tolist(),
                        "navigable": res.navigable.astype(np.uint8).ravel().tolist(),
                        "lat": np.round(g.lat2d, 4).ravel().tolist(), "lon": np.round(g.lon2d, 4).ravel().tolist()}}

    @app.post("/real/locations/resolve")
    def resolve_locations(req: ResolveRequest):
        spec = hs.route(req.origin, req.destination)
        a = hs.archive()
        n_route, n_window = spec.scenario_days(), spec.scenario_days(req.window_days)
        out = {**_labels(a), **spec.to_dict(), "window_days": req.window_days,
               "scenario_days": {"route": n_route, "window": n_window}}
        if req.issue is not None:
            out["issue"] = req.issue.isoformat()
            out["coverage"] = {k: {"ok": not (pr := a.problems(req.issue, n)), "problems": pr}
                               for k, n in (("route", n_route), ("window", n_window))}
        return out

    # The last few plans' chosen route (small objects), so "Simulate Voyage" after "Plan Route" does not
    # rebuild the same 14-day window. A miss recomputes it with the same deterministic calls.
    plan_cache: dict[tuple, dict] = {}
    plan_cache_lock = threading.Lock()

    def select_plan(planner: HistoricalPlanner, issue: date, window_days: int, o, d, n_days: int,
                    key: tuple | None = None):
        """The departure window from one issue and the route it stands for (what /real/plan shows)."""
        p, r, vessel = planner.archive.params, cfg.routing, svc.vessel
        world, snap = planner.world(issue, n_days)
        sweep = plan_from_issue(world, list(range(window_days)), vessel, o, d, r.risk_budget,
                                r.risk_weights, r.risk_estimator, r.connectivity, p["scenario_routes"],
                                r.confidence, p["seed"])
        option = sweep.selected
        if option is None:     # show the least risky option, clearly not recommended
            usable = [x for x in sweep.options if chosen_candidate(x.plan) is not None]
            option = min(usable, key=lambda x: (x.p_breach_upper, x.departure)) if usable else None
        cand = chosen_candidate(option.plan) if option is not None else None
        route = risk = None
        if cand is not None:
            step = world.time_step_hours
            route = route_summary(cand, world, option, step)
            risk = risk_breakdown(cand.route, world, vessel, option.lead_days * step, r.confidence,
                                  cand.evaluation, r.risk_budget)
        with plan_cache_lock:
            plan_cache[key or (tuple(o), tuple(d), issue, window_days)] = {
                "recommended": sweep.selected is not None, "explanation": sweep.explanation,
                "option": option, "cand": cand, "route": route, "risk": risk}
            while len(plan_cache) > 16:
                plan_cache.pop(next(iter(plan_cache)))
        return world, snap, sweep, option, cand, route, risk

    def plan_result(planner, issue: date, window_days: int, include_layers: bool, o, d, n_days: int,
                    key: tuple | None = None) -> tuple[dict, object, object]:
        """Window, route, risk, daily timeline, layers and berg tracks (shared by both modes)."""
        vessel = svc.vessel
        world, snap, sweep, option, cand, route, risk = select_plan(planner, issue, window_days, o, d, n_days, key)
        daily, layers = [], None
        if cand is not None:
            daily = daily_timeline(cand.route, world, vessel, issue, option.lead_days,
                                   cand.evaluation.expected_hours, risk)
            if include_layers:
                layers = {"grid": grid_geometry(world.grid), "land": world.land.astype(np.uint8).ravel().tolist(),
                          "vessel_limit": vessel.tau,
                          "layers": environment_layers(world, vessel.tau, [x["scenario_layer"] for x in daily])}
        status = ("recommended" if sweep.selected is not None
                  else "no_feasible_departure" if cand is not None else "no_route")
        body = {
            "status": status, "explanation": sweep.explanation,
            "route": route, "risk": risk,
            "departure": {"recommended": sweep.to_dict()["selected"], "rule": sweep.rule,
                          "explanation": sweep.explanation, "options": [x.to_dict() for x in sweep.options],
                          "depart_on_issue_date": sweep.options[0].to_dict()},
            "alternatives": ([svc.candidate_payload(c, world) for c in option.plan.candidates]
                             if option is not None else []),
            "daily": daily, "daily_note": TIMELINE_NOTE, "layers": layers,
            "icebergs": snap.summary(), "iceberg_tracks": tracks(planner, world, snap),
        }
        return body, world, snap

    def estimate_setup(issue: date, n_days: int):
        """("forecast", setup) when the forecast pathway can serve ``issue``, else ("analogue", setup) from a
        historical seasonal analogue; 422 only when neither can (both reasons are given)."""
        a = hs.archive()
        try:
            recent = forecast_usnic(a)
        except ForecastUnavailable as exc:
            recent, reason = None, str(exc)
        else:
            try:
                return "forecast", resolve(a, issue, n_days, recent)
            except ForecastUnavailable as exc:
                reason = str(exc)
        try:
            data = hs.analogue()
            setup = resolve_analogue(data, issue, n_days,
                                     recent or UsnicArchive(a.spec["inputs"]["icebergs"], a.root), reason)
        except (HistoricalUnavailable, AnalogueUnavailable) as exc:
            why = f"{exc.status}: {exc.reason}" if isinstance(exc, HistoricalUnavailable) else str(exc)
            raise HTTPException(422, {"status": "forecast_unavailable", "label": "Forecast / hackathon estimate",
                                      "reason": f"{reason}; and no historical seasonal analogue: {why}"}) from None
        return "analogue", setup

    def estimate_planner(kind: str, setup):
        if kind == "forecast":
            return ForecastPlanner(hs.planner(), setup)
        a = hs.archive()
        return AnaloguePlanner(hs.analogue(), setup, drift_kwargs(a.params))

    def forecast_meta(a: HistoricalArchive, setup, snap, n_days: int, members: int, planner, **extra) -> dict:
        if isinstance(planner, AnaloguePlanner):
            fm = analogue_metadata(planner.data, setup, snap, members, planner.data.days[-1])
            banners = [FORECAST_BANNERS[0], "Historical seasonal analogue: real sea ice, winds and currents of "
                       "earlier years", FORECAST_BANNERS[2]]
        else:
            fm = forecast_metadata(a, setup, snap, provenance(a, cfg, setup.analogue, n_days, snap, planner), members)
            fm.update(pathway="proxy_forecast", pathway_label="Proxy forecast (analogue start)")
            banners = FORECAST_BANNERS
        return {"mode": "forecast", "label": fm["label"], "execution_mode": FORECAST_EXECUTION_MODE,
                "data_status": FORECAST_DATA_STATUS, "issue_date": setup.requested.isoformat(),
                "requested_date": setup.requested.isoformat(), "hindsight_forcing": False,
                "proxy_forcing": True, "forecast_disclosure": fm["disclosure"], "banners": banners,
                "disclaimer": DISCLAIMER, **extra, "forecast": fm}

    FORECAST_SIM_NOTE = (
        "Forecast-estimate voyage simulation: the vessel sails the planned route one day at a time through the "
        "analogue season's observed sea ice (a proxy, not an observation of the requested dates); each following "
        "day a new forecast is issued from the same proxy inputs (frozen U-Net, analogue ERA5/CMEMS reanalysis, "
        "the latest official USNIC list held at its last positions, calibrated drift) and the existing replanning "
        "rules decide whether to keep or change the route. Dates are shown on the requested timeline; the analogue "
        "dates are listed under Data & Confidence. Positions are the last route cell reached each day (25 km cells).")

    ANALOGUE_SIM_NOTE = (
        "Seasonal-analogue voyage simulation: the vessel sails the planned route one day at a time through the real "
        "observed sea ice of the analogue date (an earlier year; a proxy, not an observation of the requested dates). "
        "Each following day a new estimate is issued from the same analogue inputs (sea ice of the same calendar days "
        "in other years, analogue ERA5/CMEMS reanalysis, the stated official USNIC list, calibrated drift) and the "
        "existing replanning rules decide whether to keep or change the route. Dates are shown on the requested "
        "timeline; the analogue dates are listed under Data & Confidence.")

    # Engine dates are analogue dates; these subtrees hold real-world dates and are never shifted.
    REAL_DATES = frozenset({"icebergs", "provenance", "analogue_date"})

    def relabel_proxy(body: dict) -> dict:
        """In forecast mode the scenario-layer-0 'observed' ice is the analogue season's: say so."""
        for x in body.get("daily") or []:
            if x.get("layer_source") == "observed":
                x["layer_source"] = "proxy_analogue_observed"
        for x in (body.get("layers") or {}).get("layers", []):
            if x.get("source") == "observed":
                x["source"] = "proxy_analogue_observed"
        return body

    @app.post("/real/plan")
    def real_plan(req: PlanRequest):
        """Locations -> joint real scenarios -> departure window -> route -> risk split -> daily timeline.

        Dates after the archive are planned in forecast mode (labelled analogue start; see forecast_mode)."""
        spec = hs.route(req.origin, req.destination)
        o, d, h = *spec.cells, spec.horizon_days
        n_days = req.window_days + h
        ends_out = {"locations": {"origin": spec.origin.to_dict(), "destination": spec.destination.to_dict()},
                    "horizon": spec.horizon.to_dict()}
        if req.issue > hs.archive().days[-1]:
            kind, setup = estimate_setup(req.issue, n_days)
            with svc.inline_slot():
                planner = estimate_planner(kind, setup)
                body, world, snap = plan_result(planner, setup.analogue, req.window_days, req.include_layers, o, d,
                                                n_days, key=(kind, tuple(o), tuple(d), req.issue, req.window_days))
            a = hs.archive()
            meta = forecast_meta(a, setup, snap, n_days, world.n_scenarios, planner,
                                 window_days=req.window_days, horizon_days=h, scenario_days=n_days,
                                 n_scenarios=world.n_scenarios,
                                 layer_source=["proxy_analogue_observed" if x == "observed" else x
                                               for x in world.layer_source])
            body = shift_dates(relabel_proxy(body), setup.offset_days, REAL_DATES)
            out = {"status": body["status"], "explanation": body["explanation"], "metadata": meta, **ends_out}
            out.update({k: v for k, v in body.items() if k not in out})
            return keep(req, json_safe(out))
        a = coverage_or_422(req.issue, n_days)
        with svc.inline_slot():
            planner = hs.planner()
            body, world, snap = plan_result(planner, req.issue, req.window_days, req.include_layers, o, d, n_days)
        meta = {
            "mode": "historical", "label": LABEL, "execution_mode": "real", "data_status": DATA_STATUS,
            "issue_date": req.issue.isoformat(), "hindsight_forcing": True,
            "hindsight_disclosure": a.spec["hindsight_forcing"], "banners": BANNERS, "disclaimer": DISCLAIMER,
            "window_days": req.window_days, "horizon_days": h, "scenario_days": n_days,
            "n_scenarios": world.n_scenarios, "layer_source": world.layer_source,
            "provenance": provenance(a, cfg, req.issue, n_days, snap, planner),
        }
        out = {"status": body["status"], "explanation": body["explanation"], "metadata": meta, **ends_out}
        out.update({k: v for k, v in body.items() if k not in out})
        return keep(req, json_safe(out))

    def keep(req: PlanRequest, out: dict) -> dict:
        """Save the result (see :mod:`antarctic_routing.store`) and return it with its ``plan_id``."""
        return {**out, "plan_id": store.save(req.model_dump(mode="json", exclude_none=True), out)}

    def saved_or_404(plan_id: uuid.UUID) -> dict:
        hit = store.get(str(plan_id))
        if hit is None:
            raise HTTPException(404, {"status": "unknown_plan", "label": LABEL,
                                      "reason": "no saved plan has this id (plans kept only in memory are lost "
                                                "when the server restarts)"})
        return hit

    @app.get("/real/plans")
    def saved_plans(limit: int = Query(20, ge=1, le=100)):
        """The most recent saved plans, newest first (summaries only)."""
        return {"storage": store.storage, "plans": store.list(limit)}

    @app.get("/real/plans/{plan_id}")
    def saved_plan(plan_id: uuid.UUID):
        """A saved plan exactly as ``POST /real/plan`` returned it, with the request that produced it."""
        hit = saved_or_404(plan_id)
        return {"plan_id": str(plan_id), "request": hit["request"], "plan": hit["plan"]}

    @app.get("/real/plans/{plan_id}/brief")
    def saved_plan_brief(plan_id: uuid.UUID):
        """PDF brief of a saved plan: decision, route map, departure window and candidate routes."""
        from antarctic_routing.brief import plan_brief

        return Response(plan_brief(saved_or_404(plan_id)["plan"]), media_type="application/pdf",
                        headers={"Content-Disposition": f"attachment; filename=voyage_brief_{plan_id}.pdf"})

    @app.post("/real/simulate")
    def real_simulate(req: SimulateRequest, wait: bool = Query(False)):
        """Simulate the voyage /real/plan chose, day by day, with the existing daily replanning (job).

        In forecast mode the same simulation runs on the analogue days with the forecast-mode planner; the
        'observed' ice it sails through is the analogue season's (labelled proxy) and dates are shown on the
        requested timeline."""
        spec = hs.route(req.origin, req.destination)
        o, d, h = *spec.cells, spec.horizon_days
        n_days = req.window_days + h
        forecast = req.issue > hs.archive().days[-1]
        kind, setup = estimate_setup(req.issue, n_days) if forecast else (None, None)
        a = hs.archive() if forecast else coverage_or_422(req.issue, n_days)
        engine_issue = setup.analogue if forecast else req.issue
        key = ((kind, tuple(o), tuple(d), req.issue, req.window_days) if forecast
               else (tuple(o), tuple(d), req.issue, req.window_days))
        to_engine = (lambda x: x - timedelta(days=setup.offset_days)) if forecast else (lambda x: x)  # noqa: E731

        def work():
            from antarctic_routing.simulation import VoyageSimulator

            planner = estimate_planner(kind, setup) if forecast else hs.planner()
            r, p = cfg.routing, a.params
            with plan_cache_lock:
                hit = plan_cache.get(key)
            if hit is None:
                select_plan(planner, engine_issue, req.window_days, o, d, n_days, key)
                with plan_cache_lock:
                    hit = plan_cache[key]
            option, cand = hit["option"], hit["cand"]
            if forecast:
                snap = planner.snapshot(engine_issue, grid_of(a.ds))
                meta = forecast_meta(a, setup, snap, n_days, p["members"], planner, window_days=req.window_days,
                                     horizon_days=h, n_scenarios=p["members"])
                base = {"execution_mode": FORECAST_EXECUTION_MODE, "data_status": FORECAST_DATA_STATUS,
                        "data_label": meta["label"], "hindsight_forcing": False, "disclaimer": DISCLAIMER,
                        "metadata": meta}
            else:
                base = {**_labels(a), "metadata": {
                    "mode": "historical", "label": LABEL, "execution_mode": "real", "data_status": DATA_STATUS,
                    "issue_date": req.issue.isoformat(), "hindsight_forcing": True,
                    "hindsight_disclosure": a.spec["hindsight_forcing"], "banners": BANNERS,
                    "disclaimer": DISCLAIMER, "window_days": req.window_days, "horizon_days": h,
                    "n_scenarios": p["members"]}}
            base["locations"] = {"origin": spec.origin.to_dict(), "destination": spec.destination.to_dict()}
            shown = (lambda x: setup.requested_day(x)) if forecast else (lambda x: x)  # noqa: E731
            if cand is None:
                return {**base, "status": "not_simulated", "reason": "the plan found no route to sail"}
            if req.departure is not None and to_engine(req.departure) != option.departure:
                return {**base, "status": "not_simulated",
                        "reason": f"the plan for these inputs departs {shown(option.departure)}, "
                                  f"not {req.departure}"}
            dep = option.departure
            sea_days = max(1, math.ceil(cand.evaluation.expected_hours / 24)) \
                if math.isfinite(cand.evaluation.expected_hours) else 1
            check = planner.problems if forecast else a.problems
            gaps = [f"{shown(dep + timedelta(days=k))}: {'; '.join(pr)}" for k in range(1, sea_days + 1)
                    if (pr := check(dep + timedelta(days=k), 1 + h))]
            if gaps:
                return {**base, "status": "not_simulated",
                        "reason": "the archive cannot issue the daily forecasts this voyage needs (" +
                                  " | ".join(gaps) + ")"}
            if kind == "analogue" and (miss := planner.truth_missing(dep, sea_days + 1)):
                return {**base, "status": "not_simulated",
                        "reason": f"the analogue year has no observed sea ice on {', '.join(map(str, miss))} (missing "
                                  "upstream), so the voyage cannot be sailed through it; the plan itself is unaffected"}
            policy = ReplanPolicy(r.risk_budget, r.risk_estimator, r.confidence)
            sim = VoyageSimulator(planner, svc.vessel, policy, r.risk_weights, r.connectivity, p["scenario_routes"],
                                  p["seed"], h).run(dep, cand.route, cand.evaluation, hit["risk"])
            grid = grid_of(a.ds)
            note = sim.pop("note")
            out = {"status": "simulated",
                   "plan": {"status": "recommended" if hit["recommended"] else "no_feasible_departure",
                            "explanation": hit["explanation"], "route": hit["route"],
                            "risk": {k: hit["risk"][k] for k in ("combined", "sea_ice", "iceberg", "risk_budget")}},
                   "grid": {**grid_geometry(grid), "land": a.ds["land_mask"].values.astype(np.uint8).ravel().tolist(),
                            "vessel_limit": svc.vessel.tau},
                   **sim}
            if forecast:
                for f in out["frames"]:
                    if f.get("map") and f["map"].get("source") == "observed":
                        f["map"]["source"] = "proxy_analogue_observed"
                    f["observed"]["sea_ice_source"] = "proxy_analogue_observed"
                    seg = f["observed"].get("sea_ice_on_segment")
                    if seg:                 # the analogue day's real observation, shown on the requested day
                        seg.update(source="proxy_analogue_observed", analogue_date=seg["date"])
                out["summary"]["notes"]["forecast_mode"] = (
                    "Forecast mode: the 'observed' sea ice sailed through and scored against is the analogue "
                    "season's real observation (a proxy, not the requested year), and icebergs are held at their "
                    "last official report." if kind == "forecast" else
                    "Seasonal analogue: the 'observed' sea ice sailed through and scored against is the real "
                    f"observation of the analogue date's year ({setup.analogue.year}, a proxy, not the requested "
                    "year); the plan's sea-ice members came from other years.")
                out = shift_dates(out, setup.offset_days, REAL_DATES)
                base["metadata"].update(departure_date=shown(dep).isoformat(),
                                        simulation_note=FORECAST_SIM_NOTE if kind == "forecast"
                                        else ANALOGUE_SIM_NOTE)
            else:
                base["metadata"].update(departure_date=dep.isoformat(), simulation_note=note,
                                        provenance=provenance(a, cfg, req.issue, n_days, None, planner))
            return json_safe({**base, **out})

        return svc.submit("real_simulate", work, wait)

    @app.post("/real/historical/routes")
    def historical_routes(req: HistoricalRouteRequest, wait: bool = Query(False)):
        o, d, h, spec = ends(req.origin, req.destination)
        coverage_or_422(req.issue, 1 + h)

        def work():
            planner, world, snap, plan = plan_at_issue(req.issue, o, d, h)
            return with_route(route_payload(planner, world, snap, plan, req.issue, 1 + h), world.grid, o, d, spec)

        return svc.submit("historical_routes", work, wait)

    @app.post("/real/historical/departures")
    def historical_departures(req: HistoricalWindowRequest, wait: bool = Query(False)):
        o, d, h, spec = ends(req.origin, req.destination)
        n_days = req.window_days + h
        coverage_or_422(req.issue, n_days)

        def work():
            planner = hs.planner()
            a, p, r = planner.archive, planner.archive.params, cfg.routing
            world, snap = planner.world(req.issue, n_days)
            sweep = plan_from_issue(world, list(range(req.window_days)), svc.vessel, o, d, r.risk_budget,
                                    r.risk_weights, r.risk_estimator, r.connectivity, p["scenario_routes"],
                                    r.confidence, p["seed"])
            sel = sweep.selected
            out = {**sweep.to_dict(), **_labels(a), "issue": req.issue.isoformat(), "window_days": req.window_days,
                   "horizon_days": h, "risk_budget": r.risk_budget, "n_scenarios": world.n_scenarios,
                   "layer_source": world.layer_source, "land_label": LAND_LABEL, "icebergs": snap.summary(),
                   "iceberg_tracks": tracks(planner, world, snap),
                   "historical": provenance(a, cfg, req.issue, n_days, snap, planner),
                   "selected_route": None, "map": None}
            with_route(out, world.grid, o, d, spec)
            if sel is not None and sel.plan.recommended is not None:
                mid = sel.lead_days * world.time_step_hours + sel.expected_hours / 2
                out["selected_route"] = svc.candidate_payload(sel.plan.recommended, world)
                out["map"] = svc.map_payload(world, svc.vessel.tau, 2 * mid)   # map_payload shows hours / 2
            else:
                out["map"] = svc.map_payload(world, svc.vessel.tau, 0.0)
            return out

        return svc.submit("historical_departures", work, wait)

    @app.post("/real/historical/voyages", status_code=201)
    def historical_create_voyage(req: HistoricalRouteRequest):
        o, d, h, spec = ends(req.origin, req.destination)
        coverage_or_422(req.issue, 1 + h)
        if len(svc.voyages) >= svc.max_voyages:
            raise HTTPException(503, "voyage store is full (in-memory, ANTROUTE_MAX_VOYAGES); restart to clear it")
        with svc.inline_slot():
            planner, world, snap, plan = plan_at_issue(req.issue, o, d, h)
            payload = with_route(route_payload(planner, world, snap, plan, req.issue, 1 + h), world.grid, o, d, spec)
        if plan.recommended is None:
            raise HTTPException(409, plan.explanation)
        from antarctic_routing.api.main import RouteContext

        vid = uuid.uuid4().hex[:12]
        cand = plan.recommended
        svc.voyages[vid] = {"mode": "historical", "issue": req.issue, "last_issue": req.issue, "candidate": cand,
                            "horizon_days": h,
                            "world": RouteContext.of(world), "data_labels": _labels(planner.archive), "events": [{
                                "event": "planned", "at": utc_now(), "departure": req.issue.isoformat(),
                                "issued": req.issue.isoformat(), "explanation": plan.explanation,
                                "usnic_list": snap.list_date.isoformat(), **cand.evaluation.summary()}]}
        return {"voyage_id": vid, "route": svc.candidate_payload(cand, world), "explanation": plan.explanation,
                **{k: payload[k] for k in ("map", "origin_xy_km", "destination_xy_km", "icebergs", "iceberg_tracks",
                                           "historical", "land_label", "route") if k in payload},
                **_labels(planner.archive)}

    @app.post("/real/historical/voyages/{vid}/replan")
    def historical_replan(vid: str, req: HistoricalReplanRequest):
        if vid not in svc.voyages:
            raise HTTPException(404, "unknown voyage")
        v = svc.voyages[vid]
        if v.get("mode") != "historical":
            raise HTTPException(409, "this voyage was planned on controlled-synthetic data; replan it with "
                                     "/voyages/{id}/replan")
        if req.issued < v["last_issue"]:
            raise HTTPException(422, f"issued must not be before the last forecast used ({v['last_issue']})")
        h = v.get("horizon_days", H)
        coverage_or_422(req.issued, 1 + h)
        with svc.inline_slot():
            planner = hs.planner()
            p, r = planner.archive.params, cfg.routing
            world, snap = planner.world(req.issued, 1 + h)
            try:
                position = world.grid.cell_of(req.lat, req.lon)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from None
            policy = ReplanPolicy(r.risk_budget, r.risk_estimator, r.confidence)
            try:
                decision = replan(world, position, v["candidate"].route, svc.vessel, policy, r.risk_weights,
                                  r.connectivity, p["scenario_routes"], data_age_hours=0.0, seed=p["seed"])
            except ValueError as exc:
                raise HTTPException(400, f"{exc}; send a position on the current route") from None
        if decision.action == "switch":
            v["candidate"] = decision.new_candidate
        else:
            v["candidate"] = Candidate(v["candidate"].labels, decision.previous_route, decision.previous_evaluation,
                                       decision.action == "keep", decision.previous_evaluation.p_breach_upper)
        from antarctic_routing.api.main import RouteContext

        v["world"], v["last_issue"] = RouteContext.of(world), req.issued
        record = decision.record(issued=req.issued.isoformat(), event="replan", data_age_basis=DATA_AGE_BASIS,
                                 usnic_list=snap.list_date.isoformat(), usnic_age_days=snap.age_days)
        record.pop("previous", None)
        record.pop("new", None)
        v["events"].append(record)
        return {**record, "route": svc.candidate_payload(v["candidate"], world),
                "map": svc.map_payload(world, svc.vessel.tau, v["candidate"].evaluation.expected_hours),
                "icebergs": snap.summary(), "iceberg_tracks": tracks(planner, world, snap), "land_label": LAND_LABEL,
                "historical": provenance(planner.archive, cfg, req.issued, 1 + h, snap, planner),
                **_labels(planner.archive)}

    @app.post("/real/historical/replay")
    def historical_replay(req: HistoricalReplayRequest, wait: bool = Query(False)):
        a = hs.archive()
        o, d, h, spec = ends(req.origin, req.destination)
        window = a.params["window_days"]
        coverage_or_422(req.start, window + h)
        ok = 0
        while ok < req.max_wait_days and not a.problems(req.start + timedelta(days=ok + 1), window + h):
            ok += 1
        if ok < req.max_wait_days:
            raise HTTPException(422, {"status": "out_of_coverage", "label": LABEL,
                                      "reason": f"waiting {req.max_wait_days} days from {req.start} needs forecasts "
                                                f"past the archive's coverage; at most {ok} wait day(s) are possible"})

        def work():
            from antarctic_routing.replay import run_replay

            planner = hs.planner()
            p, r = planner.archive.params, cfg.routing
            grid = grid_of(planner.archive.ds)
            policy = ReplanPolicy(r.risk_budget, r.risk_estimator, r.confidence)
            result = run_replay(planner.ctx, svc.vessel, o, d, req.start, window, req.max_wait_days, p["members"],
                                policy, r.risk_weights, h, np.random.default_rng(p["seed"]),
                                connectivity=r.connectivity, scenario_routes=p["scenario_routes"], seed=p["seed"],
                                hazard=planner.berg_hazard, truth_bergs=planner.observed_bergs,
                                berg_radius_m=p["berg_radius_km"] * 1e3)
            return with_route(replay_payload(planner, grid, result, req.start, window + h), grid, o, d, spec)

        return svc.submit("historical_replay", work, wait)

    def replay_payload(planner: HistoricalPlanner, grid, result: dict, start: date, n_days: int) -> dict:
        a = planner.archive
        xy = lambda cells: [[float(grid.x[c] / 1000), float(grid.y[rw] / 1000)] for rw, c in cells]   # noqa: E731
        ll = lambda cells: [[round(float(grid.lat2d[rw, c]), 5), round(float(grid.lon2d[rw, c]), 5)]  # noqa: E731
                            for rw, c in cells]
        out = {**result, **_labels(a), "land_label": LAND_LABEL,
               "historical": provenance(a, cfg, start, n_days, None, planner)}
        o, d = svc.endpoints(grid)
        out["origin_xy_km"], out["destination_xy_km"] = xy([o])[0], xy([d])[0]
        for key in ("sailed", "naive"):
            cells = result.get(f"{key}_cells")
            if cells:
                out[f"{key}_xy_km"], out[f"{key}_latlon"] = xy(cells), ll(cells)
        day = date.fromisoformat(result["departure"]) if result.get("departure") else start
        out["map"] = observed_map(a, grid, day)
        bergs = planner.observed_bergs(day)
        inside = in_grid_mask([b[1] for b in bergs], [b[2] for b in bergs], grid) if bergs else []
        out["observed_bergs"] = [{"id": b[0], "lat": b[1], "lon": b[2],
                                  "xy_km": [float(v) / 1000 for v in grid.to_xy(b[1], b[2])]}
                                 for b, m in zip(bergs, inside, strict=True) if m]
        return out

    def observed_map(a: HistoricalArchive, grid, day: date) -> dict:
        """Observed sea-ice concentration (percent) on ``day``, in the dashboard's map layout."""
        t = a.days.index(day)
        land = a.ds["land_mask"].values.astype(bool)
        conc = np.where(land, 0.0, a.ds["ice_concentration"].values[t])
        geo = grid_geometry(grid)
        return {**{k: geo[k] for k in ("nx", "ny", "res_km", "x0_km", "y0_km", "rotation_rad")}, "layer_day": 0,
                "observed_date": day.isoformat(), "land": land.astype(np.uint8).ravel().tolist(),
                "p_ice": to_percent(conc), "p_berg": None}

    return hs
