"""Result assembly for the one-call product plan (``POST /real/plan``): no new science.

Everything here re-reads what the existing engine already produced, i.e. the joint scenarios and the
route evaluations of :func:`antarctic_routing.routing.evaluate.evaluate_route`.

Risk decomposition (:func:`risk_breakdown`)
-------------------------------------------
The authoritative risk is the existing joint one. A route breaches in scenario k when, at the vessel's
arrival time in k, it meets sea ice >= the vessel limit, an iceberg footprint, or an impassable leg; the
risk is the share of the K scenarios that breach, with the Wilson 95 % upper bound.

The same route is re-sailed through the same scenarios twice more. Arrival times depend only on ice and
currents, never on icebergs or on the vessel limit, so the timing is identical in all three passes:

* sea-ice risk: the icebergs are removed (``berg=None``), so only ice >= limit or impassable counts;
* iceberg risk: the vessel limit is set to infinity, so only an iceberg footprint or impassable counts;
* iceberg-only increment: breaches where the iceberg pass breaches and the ice pass does not.

Each component has its own count, share and Wilson bound. The two components overlap: impassable
scenarios count in both, and so does a scenario that meets both hazards. So they need not add up to the
combined risk, and combined >= max(sea ice, iceberg) always holds.

Daily timeline (:func:`daily_timeline`)
---------------------------------------
Per-cell breach probabilities come from the evaluations and are exact, because each scenario is scored at
its own arrival time. Assigning cells to calendar days is a *nominal* display timeline: the time to a cell is
taken as proportional to the distance sailed, scaled to the expected voyage time. The engine does not keep
per-scenario arrival times for large ensembles, so this is labelled as nominal and is not used for any risk.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta

import numpy as np

from antarctic_routing.config import wilson_upper_bound
from antarctic_routing.publish import to_percent
from antarctic_routing.routing.evaluate import RouteEvaluation, evaluate_route
from antarctic_routing.routing.fuel import VesselModel
from antarctic_routing.routing.hazard import exceedance_probability
from antarctic_routing.routing.optimizer import Route
from antarctic_routing.synthetic import ScenarioSet

RISK_DEFINITIONS = {
    "combined": "Authoritative. Share of the joint scenarios in which the route meets sea ice at or above the "
                "vessel limit, an iceberg footprint, or an impassable leg, with the Wilson 95% upper bound "
                "(the value compared with the risk budget).",
    "sea_ice": "The same route and scenarios with the icebergs removed: sea ice at or above the vessel limit, or "
               "an impassable leg.",
    "iceberg": "The same route and scenarios with the sea-ice limit switched off: an iceberg footprint (USNIC "
               "positions + calibrated drift ensemble) or an impassable leg.",
    "iceberg_only_increment": "Scenarios that breach only because of icebergs (the iceberg pass breaches, the "
                              "sea-ice pass does not); combined = sea-ice breaches + this increment.",
    "overlap_note": "Sea-ice and iceberg risk overlap (impassable legs, and scenarios meeting both hazards count in "
                    "both), so they need not add up to the combined risk.",
}
TIMELINE_NOTE = ("Nominal display timeline: cells are assigned to days by the share of distance sailed, scaled to "
                 "the expected voyage time. Per-cell probabilities are exact (each scenario is scored at its own "
                 "arrival time); the day assignment is approximate and is not used for any risk.")


def _component(n: int, k: int, confidence: float, ev: RouteEvaluation | None = None) -> dict:
    out = {"breaches": int(n), "n_scenarios": int(k), "p_breach": n / k,
           "p_breach_upper": wilson_upper_bound(n, k, confidence)}
    if ev is not None:
        out["segment_breach_prob"] = ev.segment_breach_prob
        out["impassable_scenarios"] = ev.impassable_scenarios
    return out


def risk_breakdown(route: Route, world: ScenarioSet, vessel: VesselModel, depart_hours: float, confidence: float,
                   combined: RouteEvaluation, risk_budget: float) -> dict:
    """Sea-ice, iceberg and combined risk of one route over the same scenarios (see the module docstring)."""
    k = world.n_scenarios
    ice = evaluate_route(route, replace(world, berg=None), vessel, depart_hours, confidence)
    berg = (evaluate_route(route, world, replace(vessel, tau=math.inf), depart_hours, confidence)
            if world.berg is not None else None)
    both = evaluate_route(route, world, vessel, depart_hours, confidence)
    if (both.breaches, both.expected_hours) != (combined.breaches, combined.expected_hours):
        raise RuntimeError("re-evaluating the selected route did not reproduce the planner's evaluation")
    out = {
        "authoritative": "combined",
        "risk_budget": risk_budget,
        "confidence": confidence,
        "estimator": "wilson_upper",
        "combined": {**_component(combined.breaches, k, confidence, combined),
                     "within_budget": combined.p_breach_upper <= risk_budget},
        "sea_ice": _component(ice.breaches, k, confidence, ice),
        "iceberg": (_component(berg.breaches, k, confidence, berg) if berg is not None
                    else {**_component(0, k, confidence), "segment_breach_prob": [0.0] * len(route.cells),
                          "impassable_scenarios": None, "note": "no in-grid USNIC icebergs for this issue date"}),
        "iceberg_only_increment": _component(combined.breaches - ice.breaches, k, confidence),
        "definitions": RISK_DEFINITIONS,
    }
    return out


def _cum_km(world: ScenarioSet, cells) -> np.ndarray:
    from pyproj import Geod

    g = world.grid
    rows = np.array([c[0] for c in cells])
    cols = np.array([c[1] for c in cells])
    lat, lon = g.lat2d[rows, cols], g.lon2d[rows, cols]
    if len(cells) < 2:
        return np.zeros(len(cells))
    seg = np.atleast_1d(Geod(ellps="WGS84").inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[2]) / 1000.0
    return np.concatenate([[0.0], np.cumsum(seg)])


def daily_timeline(route: Route, world: ScenarioSet, vessel: VesselModel, issue: date, lead_days: int,
                   expected_hours: float, risk: dict) -> list[dict]:
    """One entry per calendar day of the selected voyage (nominal day assignment, exact per-cell values)."""
    g, step = world.grid, world.time_step_hours
    cells = [tuple(c) for c in route.cells]
    cum = _cum_km(world, cells)
    total = float(cum[-1]) if cum.size else 0.0
    hours = cum / total * expected_hours if total > 0 and math.isfinite(expected_hours) else np.zeros(len(cells))
    start_h = lead_days * step
    t_abs = start_h + hours                                   # hours after the issue (00:00 UTC of issue day)
    day_of = np.floor(t_abs / step).astype(int)
    seg = {name: np.asarray(risk[name]["segment_breach_prob"], float) for name in ("combined", "sea_ice", "iceberg")}
    rows = np.array([c[0] for c in cells])
    cols = np.array([c[1] for c in cells])
    out = []
    for d in range(int(day_of[0]), int(day_of[-1]) + 1):
        sel = np.flatnonzero(day_of == d)
        layer = int(min(max(d, 0), world.n_times - 1))
        r, c = rows[sel], cols[sel]
        conc = world.conc[:, layer][:, r, c] if sel.size else np.empty((world.n_scenarios, 0))
        with np.errstate(invalid="ignore"):
            n_valid = np.isfinite(conc).sum(axis=0)
            mean_c = np.where(n_valid > 0, np.nansum(conc, axis=0) / np.maximum(n_valid, 1), np.nan)
        p_ice = exceedance_probability(conc, vessel.tau) if sel.size else np.empty(0)
        p_berg = (world.berg[:, layer][:, r, c].mean(axis=0) if world.berg is not None and sel.size
                  else np.zeros(sel.size))
        h0, h1 = max(d * step, start_h), min((d + 1) * step, float(t_abs[-1]))
        entry = {
            "date": (issue + timedelta(days=d)).isoformat(),
            "day_of_voyage": d - int(day_of[0]) + 1,
            "scenario_layer": layer,
            "layer_source": world.layer_source[layer] if world.layer_source else None,
            "nominal_hours_since_departure": [round(h0 - start_h, 2), round(h1 - start_h, 2)],
            "route_cell_index": [int(sel[0]), int(sel[-1])] if sel.size else None,
            "cells": [[int(a), int(b)] for a, b in zip(r, c, strict=True)],
            "latlon": [[round(float(g.lat2d[a, b]), 5), round(float(g.lon2d[a, b]), 5)]
                       for a, b in zip(r, c, strict=True)],
            "position_end_of_day": ([round(float(g.lat2d[r[-1], c[-1]]), 5), round(float(g.lon2d[r[-1], c[-1]]), 5)]
                                    if sel.size else None),
            "distance_km": round(float(cum[sel[-1]] - cum[max(sel[0] - 1, 0)]), 1) if sel.size else 0.0,
            "sea_ice": {
                "mean_concentration": _stat(mean_c, np.nanmean),
                "max_concentration": _stat(mean_c, np.nanmax),
                "max_p_ge_vessel_limit": _stat(p_ice, np.nanmax),
            },
            "iceberg": {"max_p_presence": _stat(p_berg, np.max)},
            "risk": {name: {"max_cell_breach_prob": _stat(seg[name][sel], np.max)} for name in seg},
        }
        out.append(entry)
    return out


def _stat(a: np.ndarray, fn) -> float | None:
    a = np.asarray(a, float)
    if a.size == 0 or not np.isfinite(a).any():
        return None
    return round(float(fn(a)), 4)


def environment_layers(world: ScenarioSet, tau: float, days: list[int]) -> list[dict]:
    """Display maps for the given scenario layers: P(ice >= limit) and P(iceberg) in percent (``None`` on land)."""
    out = []
    for t in sorted(set(int(min(max(d, 0), world.n_times - 1)) for d in days)):
        p_ice = np.where(world.land, np.nan, exceedance_probability(world.conc[:, t], tau))
        layer = {"scenario_layer": t,
                 "date": (world.start + timedelta(hours=t * world.time_step_hours)).date().isoformat(),
                 "source": world.layer_source[t] if world.layer_source else None,
                 "p_ice_ge_limit_pct": to_percent(p_ice), "p_berg_pct": None}
        if world.berg is not None:
            layer["p_berg_pct"] = to_percent(np.where(world.land, np.nan, world.berg[:, t].mean(axis=0)))
        out.append(layer)
    return out


# --------------------------------------------------------------------------- the one-call result
BANNERS = ["HISTORICAL MODE", "ERA5/CMEMS hindsight forcing", "Research estimate, not certified navigation"]


def json_safe(obj):
    """Non-finite floats -> None (the API's JSON encoder refuses inf/NaN); numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def chosen_candidate(plan):
    """The route a departure option stands for (as :mod:`routing.departure` picks it)."""
    return plan.recommended or next((c for c in plan.candidates if "lowest_risk" in c.tags), None)


def route_summary(cand, world: ScenarioSet, option, step_hours: float) -> dict:
    """The selected route in product terms (values straight from the engine's evaluation)."""
    g, ev = world.grid, cand.evaluation
    dep = option.departure
    eta = None
    if math.isfinite(ev.expected_hours):
        eta = (np.datetime64(dep.isoformat()) + np.timedelta64(int(round(ev.expected_hours * 60)), "m"))
        eta = str(eta) + "Z"
    return {
        "recommended": option.feasible,
        "departure_date": dep.isoformat(),
        "departure_utc": f"{dep.isoformat()}T00:00:00Z",
        "time_resolution": f"{step_hours:g} h forecast layers: departures are whole days at 00:00 UTC",
        "lead_days": option.lead_days,
        "expected_hours": ev.expected_hours, "hours_p10": ev.hours_p10, "hours_p90": ev.hours_p90,
        "eta_utc": eta,
        "distance_km": ev.distance_km,
        "fuel_index": {"expected": ev.expected_fuel, "p10": ev.fuel_p10, "p90": ev.fuel_p90,
                       "unit": "relative fuel index (distance-km weighted by an ice penalty), not litres or tonnes"},
        "labels": list(cand.labels), "tags": list(cand.tags),
        "forecast_fraction": option.forecast_fraction, "support": option.support,
        "beyond_horizon_fraction": ev.beyond_horizon_fraction,
        "impassable_scenarios": ev.impassable_scenarios,
        "cells": [[int(r), int(c)] for r, c in cand.route.cells],
        "latlon": [[round(float(g.lat2d[r, c]), 5), round(float(g.lon2d[r, c]), 5)] for r, c in cand.route.cells],
        "xy_km": [[float(g.x[c] / 1000.0), float(g.y[r] / 1000.0)] for r, c in cand.route.cells],
    }
