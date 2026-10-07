"""Joint-scenario route evaluation.

Instead of multiplying per-cell probabilities (which assumes independence that
does not exist between neighbouring ice cells), a fixed route is *sailed* through
every joint scenario k:

    B_R^(k) = 1[route meets C >= tau_v (or an iceberg) at its arrival time in k]
    P_hat(B_R) = (1/K) sum_k B_R^(k)

Travel time and fuel are simulated per scenario too, so their spread reflects
the same uncertainty as the risk estimate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from pyproj import Geod

from antarctic_routing.config import wilson_upper_bound
from antarctic_routing.routing.fuel import VesselModel, ice_penalty
from antarctic_routing.routing.optimizer import Route
from antarctic_routing.synthetic import ScenarioSet

_GEOD = Geod(ellps="WGS84")


@dataclass
class RouteEvaluation:
    n_scenarios: int
    breaches: int
    p_breach: float
    p_breach_upper: float
    expected_hours: float
    hours_p10: float
    hours_p90: float
    expected_fuel: float
    fuel_p10: float
    fuel_p90: float
    distance_km: float
    beyond_horizon_fraction: float
    impassable_scenarios: int
    segment_breach_prob: list[float]
    destination_breach_prob: float = 0.0
    scenario_arrival_hours: list[list[float]] | None = None
    # Per waypoint, for exports. Kept out of summary() so existing payloads are unchanged.
    segment_fuel: list[float] | None = None        # expected fuel index of the leg ending here (0 at the start)
    waypoint_hours: list[float | None] | None = None   # expected hours from departure; None if no scenario gets there

    def summary(self) -> dict:
        out = asdict(self)
        out.pop("segment_breach_prob")
        out.pop("scenario_arrival_hours")
        out.pop("segment_fuel")
        out.pop("waypoint_hours")
        return out


def evaluate_route(
    route: Route,
    world: ScenarioSet,
    vessel: VesselModel,
    depart_hours: float = 0.0,
    confidence: float = 0.95,
) -> RouteEvaluation:
    K = world.n_scenarios
    ks = np.arange(K)
    step, last = world.time_step_hours, world.n_times - 1
    conc = world.conc
    rows = np.array([c[0] for c in route.cells])
    cols = np.array([c[1] for c in route.cells])
    grid = world.grid
    lat, lon = grid.lat2d[rows, cols], grid.lon2d[rows, cols]
    _, _, seg_m = _GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    seg_km = np.atleast_1d(np.asarray(seg_m) / 1000.0)
    dx = np.diff(grid.x[cols])
    dy = np.diff(grid.y[rows])
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    ux, uy = dx / norm, dy / norm

    def idx(t: np.ndarray) -> np.ndarray:
        safe = np.where(np.isfinite(t), t, world.horizon_hours)
        return np.clip((safe // step).astype(int), 0, last)

    def conc_at(i: np.ndarray, r: int, c: int) -> np.ndarray:
        v = conc[ks, i, r, c].astype(float)
        return np.where(np.isfinite(v), v, 1.0)  # missing data treated as hazardous

    def berg_at(i: np.ndarray, r: int, c: int) -> np.ndarray:
        return world.berg[ks, i, r, c] if world.berg is not None else np.zeros(K, bool)

    t = np.full(K, float(depart_hours))
    fuel = np.zeros(K)
    i0 = idx(t)
    hit0 = (conc_at(i0, rows[0], cols[0]) >= vessel.tau) | berg_at(i0, rows[0], cols[0])
    breach = hit0.copy()
    impassable = np.zeros(K, bool)
    seg_prob = [float(hit0.mean())]
    seg_fuel = [0.0]
    arrivals = [t.copy()]

    for e in range(len(seg_km)):
        r0, c0, r1, c1 = rows[e], cols[e], rows[e + 1], cols[e + 1]
        i = idx(t)
        c_mid = 0.5 * (conc_at(i, r0, c0) + conc_at(i, r1, c1))
        vs = np.maximum(vessel.cruise_kmh * (1 - vessel.ice_k * c_mid), vessel.min_kmh)
        upar = (world.current_x[i, r0, c0] * ux[e] + world.current_y[i, r0, c0] * uy[e]) * 3.6
        vg = vs + upar
        stuck = vg <= 0
        impassable |= stuck
        t = t + np.where(stuck, np.inf, seg_km[e] / np.where(stuck, 1.0, vg))
        j = idx(t)
        cj = conc_at(j, r1, c1)
        hit = (cj >= vessel.tau) | berg_at(j, r1, c1) | stuck
        breach |= hit
        leg_fuel = seg_km[e] * (1 + vessel.lam * np.asarray(ice_penalty(cj, vessel.penalty, vessel.tau)))
        fuel += leg_fuel
        seg_fuel.append(float(leg_fuel.mean()))
        seg_prob.append(float(hit.mean()))
        arrivals.append(t.copy())

    hours = t - depart_hours
    finite = hours[np.isfinite(hours)]
    n_breach = int(breach.sum())

    def pct(a: np.ndarray, q: float) -> float:
        return float(np.percentile(a, q)) if a.size else float("inf")

    def mean_hours(a: np.ndarray) -> float | None:
        reached = a[np.isfinite(a)]                   # scenarios in which the vessel gets this far
        return float(reached.mean() - depart_hours) if reached.size else None

    return RouteEvaluation(
        n_scenarios=K,
        breaches=n_breach,
        p_breach=n_breach / K,
        p_breach_upper=wilson_upper_bound(n_breach, K, confidence),
        expected_hours=float(hours.mean()) if finite.size == K else float("inf"),
        hours_p10=pct(finite, 10),
        hours_p90=pct(finite, 90),
        expected_fuel=float(fuel.mean()),
        fuel_p10=pct(fuel, 10),
        fuel_p90=pct(fuel, 90),
        distance_km=float(seg_km.sum()),
        beyond_horizon_fraction=float(np.mean(t > world.horizon_hours)),
        impassable_scenarios=int(impassable.sum()),
        segment_breach_prob=seg_prob,
        destination_breach_prob=seg_prob[-1],
        scenario_arrival_hours=np.stack(arrivals, axis=1).tolist() if K <= 50 else None,
        segment_fuel=seg_fuel,
        waypoint_hours=[mean_hours(a) for a in arrivals],
    )
