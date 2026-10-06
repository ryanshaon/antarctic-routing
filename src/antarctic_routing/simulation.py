"""Product voyage simulation (hackathon S4): one planned historical voyage as daily frames.

This is the existing historical replay loop (:mod:`antarctic_routing.replay`) applied to the route a
``POST /real/plan`` selected, recorded frame by frame for playback. Nothing here is new science:

* **Departure** - the voyage starts on the plan's departure date, on the plan's route.
* **Each day at sea** - the vessel advances 24 h along its active route through the *observed* sea ice
  (:func:`replay.sail_day`, exactly as the replay sails). At the next 00:00 UTC a new real forecast is issued
  (:meth:`HistoricalPlanner.world`, the same call as ``/real/historical/voyages/{id}/replan``: frozen U-Net,
  ERA5/CMEMS hindsight forcing, the USNIC list of that day with the calibrated drift) and
  :func:`routing.replan.replan` decides, by its own rules, whether to keep or switch the route.
* **Arrival** - when the destination is reached the sailed track is scored through the observed ice with
  :func:`replay.score_track` (time, distance, fuel index, observed exposure, reported-berg exposure).

A frame holds only values these functions return, plus display geometry (cell coordinates, geodesic
distance along cells) and observations read from the archive (observed concentration on the sailed cells,
the USNIC list in force). The vessel position is the last route cell reached within the day; the frame's
time is the day boundary (00:00 UTC), when the next forecast is issued.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
from pyproj import Geod

from antarctic_routing.iceberg.drift import haversine_m
from antarctic_routing.product import risk_breakdown
from antarctic_routing.publish import to_percent
from antarctic_routing.replay import sail_day, score_track
from antarctic_routing.routing.graph import RoutingGrid
from antarctic_routing.routing.optimizer import Route
from antarctic_routing.routing.replan import ReplanPolicy, replan

SIMULATION_NOTE = (
    "Historical voyage simulation: the vessel sails the planned route through the observed sea ice, one day at a "
    "time; each following day a new forecast is issued (frozen U-Net, ERA5/CMEMS hindsight forcing, that day's "
    "USNIC list with calibrated drift) and the existing replanning rules decide whether to keep or change the "
    "route. Positions are the last route cell reached each day (25 km cells); frame times are day boundaries."
)

REPLAN_RULES = {
    "previous_route_exceeds_budget": "The route ahead no longer met the risk budget under the new forecast.",
    "material_risk_reduction": "A new route lowered the risk upper bound by at least 2 percentage points for at "
                               "most 5% extra fuel.",
    "material_fuel_saving": "A new route saved at least 5% fuel while meeting the risk budget.",
    "stale_input": "Inputs were older than the staleness limit.",
}


class SimulationError(ValueError):
    """The voyage cannot be simulated (e.g. the archive does not cover the days it needs)."""


def _iso(t: datetime) -> str:
    """UTC time to the nearest minute (as /real/plan's ETA)."""
    t = (t + timedelta(seconds=30)).replace(second=0, microsecond=0)
    return t.strftime("%Y-%m-%dT%H:%MZ")


def _finite(v):
    return float(v) if v is not None and math.isfinite(v) else None


def _cells_payload(grid, cells) -> dict:
    return {"cells": [list(map(int, c)) for c in cells],
            "xy_km": [[float(grid.x[c] / 1000), float(grid.y[r] / 1000)] for r, c in cells],
            "latlon": [[round(float(grid.lat2d[r, c]), 5), round(float(grid.lon2d[r, c]), 5)] for r, c in cells]}


def _risk_short(risk: dict) -> dict:
    """The per-frame risk numbers (the full breakdown carries per-segment arrays)."""
    keys = ("breaches", "n_scenarios", "p_breach", "p_breach_upper")
    out = {k: {q: risk[k].get(q) for q in keys} for k in ("combined", "sea_ice", "iceberg")}
    out["combined"]["within_budget"] = risk["combined"]["within_budget"]
    out["risk_budget"] = risk["risk_budget"]
    if "note" in risk["iceberg"]:
        out["iceberg"]["note"] = risk["iceberg"]["note"]
    return out


def _route_metrics(ev) -> dict:
    return {"p_breach": ev.p_breach, "p_breach_upper": ev.p_breach_upper, "breaches": ev.breaches,
            "n_scenarios": ev.n_scenarios, "expected_hours": _finite(ev.expected_hours),
            "expected_fuel": _finite(ev.expected_fuel), "distance_km": _finite(ev.distance_km)}


class VoyageSimulator:
    """Runs one simulation; ``archive``/``planner`` are the verified historical ones."""

    def __init__(self, planner, vessel, policy: ReplanPolicy, risk_weights, connectivity: int, scenario_routes: int,
                 seed: int, horizon_days: int, max_sea_days: int = 10) -> None:
        self.planner, self.archive = planner, planner.archive
        self.vessel, self.policy = vessel, policy
        self.risk_weights, self.connectivity, self.scenario_routes = risk_weights, connectivity, scenario_routes
        self.seed, self.h, self.max_sea_days = seed, horizon_days, max_sea_days
        self.ctx = planner.ctx
        self.ds = self.archive.ds
        from antarctic_routing.forecasting.scenarios import grid_of

        self.grid = grid_of(self.ds)
        self.land = self.ds["land_mask"].values.astype(bool)
        # forecast mode's planner brings its own coverage rule and iceberg snapshot (forecast_mode.py)
        self.problems = getattr(planner, "problems", None) or self.archive.problems
        self._usnic = getattr(planner, "usnic_positions", None) or self._archive_usnic

    def _archive_usnic(self, day: date):
        list_date, path, _, bergs = self.archive.usnic.positions(day, self.archive.params["usnic_max_age_days"])
        return list_date, path, (day - list_date).days, bergs

    # ------------------------------------------------------------------ observations (display)
    def observed_layer(self, day: date) -> dict | None:
        if day not in self.archive.days:
            return None
        t = self.archive.days.index(day)
        conc = np.where(self.land, np.nan, self.ds["ice_concentration"].values[t])
        return {"date": day.isoformat(), "source": "observed", "concentration_pct": to_percent(conc)}

    def observed_on_cells(self, day: date, cells) -> dict:
        if not cells or day not in self.archive.days:
            return {"date": day.isoformat(), "mean_concentration": None, "max_concentration": None}
        t = self.archive.days.index(day)
        v = np.array([self.ds["ice_concentration"].values[t, r, c] for r, c in cells], dtype=float)
        v = v[np.isfinite(v)]
        return {"date": day.isoformat(), "mean_concentration": float(v.mean()) if v.size else None,
                "max_concentration": float(v.max()) if v.size else None}

    def usnic(self, day: date, position) -> dict:
        list_date, path, age, bergs = self._usnic(day)
        lat, lon = self.grid.cell_latlon(*position)
        inside = [b for b in bergs if self._in_grid(b[1], b[2])]
        near = min(((float(haversine_m(lat, lon, b[1], b[2])) / 1000.0, b[0]) for b in inside), default=None)
        return {"list_date": list_date.isoformat(), "file": path.name, "age_days": age,
                "in_grid": [{"id": b[0], "lat": b[1], "lon": b[2],
                             "xy_km": [float(v) / 1000 for v in self.grid.to_xy(b[1], b[2])]} for b in inside],
                "nearest_km": round(near[0], 1) if near else None, "nearest_id": near[1] if near else None}

    def _in_grid(self, lat, lon) -> bool:
        try:
            self.grid.cell_of(lat, lon)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------ the voyage
    def run(self, departure: date, route: Route, plan_eval, plan_risk: dict) -> dict:
        grid, a = self.grid, self.archive
        rgrid = RoutingGrid.build(grid, ~self.land, self.connectivity)
        t0 = datetime(departure.year, departure.month, departure.day)
        sailed = [tuple(route.cells[0])]
        sailed_hours = [0.0]
        version = 0
        frames, events = [], []
        dist_planned = _cum_km_grid(grid, route.cells)[-1]

        def frame(k: int, phase: str, position, active: Route, ev, risk: dict | None, decision: dict | None,
                  segment, seg_day: date | None, issued: date | None, at_hours: float | None = None) -> dict:
            now = t0 + timedelta(hours=24 * k if at_hours is None else at_hours)
            day = now.date()
            cum = _cum_km_grid(grid, sailed)
            ahead = _cum_km_grid(grid, active.cells)[-1] if len(active.cells) > 1 else 0.0
            rem_h = _finite(ev.expected_hours) if ev is not None else 0.0
            lat, lon = grid.cell_latlon(*position)
            return {
                "index": len(frames), "day_of_voyage": k, "phase": phase,
                "timestamp_utc": _iso(now), "date": day.isoformat(),
                "position": {"lat": round(float(lat), 5), "lon": round(float(lon), 5), "row": int(position[0]),
                             "col": int(position[1]), "xy_km": [float(grid.x[position[1]] / 1000),
                                                                float(grid.y[position[0]] / 1000)]},
                "progress": {"sailed_km": round(float(cum[-1]), 1), "remaining_km": round(float(ahead), 1),
                             "fraction": round(float(cum[-1] / (cum[-1] + ahead)), 4) if cum[-1] + ahead > 0 else 1.0,
                             "sailed_hours": round(sailed_hours[-1], 2), "cells_sailed": len(sailed) - 1},
                "segment": {**_cells_payload(grid, segment), "sailed_on": seg_day.isoformat() if seg_day else None},
                "track": _cells_payload(grid, sailed)["xy_km"],
                "route": {**_cells_payload(grid, active.cells), "version": version},
                "forecast": None if ev is None else {
                    "issued": issued.isoformat() if issued else None,
                    "route_ahead": _route_metrics(ev),
                    "risk": _risk_short(risk) if risk else None,
                    "eta_utc": _iso(now + timedelta(hours=rem_h)) if rem_h is not None else None,
                },
                "observed": {"sea_ice_on_segment": self.observed_on_cells(seg_day, segment) if seg_day else None,
                             "icebergs": self.usnic(day, position)},
                "map": self.observed_layer(day),
                "decision": decision,
                "replanned": bool(decision and decision["action"] == "switch"),
                "route_version": version,
            }

        position = tuple(route.cells[0])
        frames.append(frame(0, "departure", position, route, plan_eval, plan_risk, None, [], None, None))
        events.append({"type": "departed", "frame": 0, "timestamp_utc": frames[0]["timestamp_utc"],
                       "explanation": f"Departed on the planned route ({len(route.cells)} cells)."})
        day, status, reason, held = departure, "incomplete", None, 0.0
        for k in range(1, self.max_sea_days + 1):
            path, finished, hours = sail_day(route, self.ctx, day, self.vessel)
            seg_day = day
            base = 24.0 * (k - 1)
            sailed.extend(tuple(c) for c in path)
            sailed_hours.extend(base + h for h in hours)
            if not finished:     # the replay holds the vessel at the last cell reached until the next forecast
                held += 24.0 - (hours[-1] if hours else 0.0)
            position = sailed[-1]
            day = day + timedelta(days=1)
            if finished:
                status = "arrived"
                done = Route("arrived", [position], [0.0], 0.0, 0.0)
                frames.append(frame(k, "arrived", position, done, None, None, None, list(path), seg_day, None,
                                    at_hours=sailed_hours[-1]))
                events.append({"type": "arrived", "frame": len(frames) - 1,
                               "timestamp_utc": _iso(t0 + timedelta(hours=sailed_hours[-1])),
                               "explanation": "Destination reached."})
                break
            problems = self.problems(day, 1 + self.h)
            if problems:
                reason = f"no Real Historical Data forecast for {day}: " + "; ".join(problems)
                stop = Route("stopped", list(route.cells[route.cells.index(position):]), [0.0], 0.0, 0.0)
                frames.append(frame(k, "stopped", position, stop, None, None, None, list(path), seg_day, None))
                events.append({"type": "stopped", "frame": len(frames) - 1,
                               "timestamp_utc": frames[-1]["timestamp_utc"],
                               "explanation": f"Simulation stopped: {reason}."})
                break
            world, _snap = self.planner.world(day, 1 + self.h)
            decision = replan(world, position, route, self.vessel, self.policy, self.risk_weights, self.connectivity,
                              self.scenario_routes, seed=self.seed, rgrid=rgrid)
            old_rest, old_ev = decision.previous_route, decision.previous_evaluation
            if decision.action == "switch":
                version += 1
                active, ev = decision.new_candidate.route, decision.new_candidate.evaluation
            else:
                active, ev = old_rest, old_ev
            risk = risk_breakdown(active, world, self.vessel, 0.0, self.policy.confidence, ev, self.policy.risk_budget)
            rec = {"action": decision.action, "triggers": decision.triggers, "explanation": decision.explanation,
                   "alert": decision.alert, "deviation_km": round(float(decision.deviation_km), 1),
                   "reasons": [REPLAN_RULES[t] for t in decision.triggers if t in REPLAN_RULES]}
            frames.append(frame(k, "at_sea", position, active, ev, risk, rec, list(path), seg_day, day))
            if decision.action == "switch":
                new = decision.new_candidate.evaluation
                d = lambda x, y: None if x is None or y is None else round(y - x, 3)   # noqa: E731
                om, nm = _route_metrics(old_ev), _route_metrics(new)
                events.append({
                    "type": "replan", "frame": len(frames) - 1, "timestamp_utc": frames[-1]["timestamp_utc"],
                    **rec, "issued": day.isoformat(),
                    "old_route": {**_cells_payload(grid, old_rest.cells), **om},
                    "new_route": {**_cells_payload(grid, decision.new_candidate.route.cells), **nm},
                    "change": {"expected_hours": d(om["expected_hours"], nm["expected_hours"]),
                               "expected_fuel": d(om["expected_fuel"], nm["expected_fuel"]),
                               "distance_km": d(om["distance_km"], nm["distance_km"]),
                               "p_breach_upper": d(om["p_breach_upper"], nm["p_breach_upper"])},
                })
            elif decision.action == "no_feasible_route":
                events.append({"type": "no_feasible_route", "frame": len(frames) - 1,
                               "timestamp_utc": frames[-1]["timestamp_utc"], **rec, "issued": day.isoformat()})
            route = decision.route
        else:
            reason = f"the destination was not reached within {self.max_sea_days} days at sea"
            events.append({"type": "stopped", "frame": len(frames) - 1, "timestamp_utc": frames[-1]["timestamp_utc"],
                           "explanation": f"Simulation stopped: {reason}."})

        replans = [e for e in events if e["type"] == "replan"]
        p = a.params
        truth = score_track(sailed, self.ctx, departure, self.vessel, self.planner.observed_bergs,
                            p["berg_radius_km"] * 1e3)
        last_fc = next((f["forecast"] for f in reversed(frames) if f["forecast"]), None)
        summary = {
            "status": status, "reason": reason, "arrived": status == "arrived",
            "departure_utc": _iso(t0),
            "arrival_utc": _iso(t0 + timedelta(hours=sailed_hours[-1])) if status == "arrived" else None,
            "days_at_sea": len(frames) - 1,
            "simulated_hours": round(sailed_hours[-1], 2),
            "held_hours": round(held, 2),
            "replans": len(replans),
            "replan_note": None if replans else "No replan was required during this voyage.",
            "no_feasible_route_alerts": sum(e["type"] == "no_feasible_route" for e in events),
            "sailed": {"hours_through_observed_ice": _finite(truth["hours"]),
                       "distance_km": _finite(truth["distance_km"]), "fuel_index": _finite(truth["fuel_index"]),
                       "observed_breach_cells": truth["observed_breach_cells"],
                       "hazard_hours": truth["hazard_hours"],
                       **{k: truth.get(k) for k in ("berg_footprint_cells", "berg_min_distance_km", "berg_nearest",
                                                    "berg_radius_km")},
                       "cells": len(sailed)},
            "planned": {**_route_metrics(plan_eval), "geodesic_km": round(float(dist_planned), 1)},
            "final_forecast_risk": last_fc["risk"] if last_fc else None,
            "final_forecast_issued": last_fc["issued"] if last_fc else None,
            "notes": {
                "simulated_hours": "Departure to arrival in the daily simulation. Each day the vessel stops at the "
                                   "last route cell it reaches and waits there for the next day's forecast "
                                   "(held_hours in total), as the historical replay does.",
                "sailed": "Sailed values re-sail the whole track from departure through the observed sea ice "
                          "(replay scoring): time, distance, fuel index, observed cells with ice at or above the "
                          "vessel limit, and cells inside a USNIC-reported berg's footprint.",
                "final_forecast_risk": "Forecast risk of the route ahead at the last daily forecast before "
                                       "arrival (Wilson 95% upper bound, as planned).",
            },
        }
        return {"frames": frames, "events": events, "summary": summary, "note": SIMULATION_NOTE,
                "sailed_track": _cells_payload(grid, sailed)}


def _cum_km_grid(grid, cells) -> np.ndarray:
    """Cumulative WGS84 geodesic distance along cell centres (as ``evaluate_route`` measures distance)."""
    cells = list(cells)
    if len(cells) < 2:
        return np.zeros(len(cells))
    rows = np.array([c[0] for c in cells])
    cols = np.array([c[1] for c in cells])
    lat, lon = grid.lat2d[rows, cols], grid.lon2d[rows, cols]
    seg = np.atleast_1d(Geod(ellps="WGS84").inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[2]) / 1000.0
    return np.concatenate([[0.0], np.cumsum(seg)])
