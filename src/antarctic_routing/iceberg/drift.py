"""Stage 5 - iceberg drift: physics baseline, ensembles, hazard layers, learned correction.

Physics (transparent baseline):

    dx/dt = beta * u_o(x, t) + alpha * u_a(x, t)

integrated with Heun's RK2 in EPSG:3031. Forcing velocities are true ground
speeds (m/s) in grid-aligned components; because the stereographic projection
is conformal but stretches distances by the point scale factor k, the map
position advances at k * velocity. alpha (~0.01-0.03) is an *effective* wind
coefficient that depends on iceberg size and shape - treat it as uncertain.
beta is the ocean-current coupling coefficient: 1 (the default, used by the
planner) applies the surface current as given; beta < 1 represents a large
berg responding to a weaker keel-depth/sea-ice-damped current than the
0.5 m model level. It is only changed by an explicit calibration.

Ensembles perturb the initial position, alpha (per member and berg) and a
member-wide velocity error shared by all bergs (forcing errors are spatially
correlated). Presence layers mark, per member and day, every cell within a
safety radius of any berg; they become ``ScenarioSet.berg`` so route risk is
evaluated jointly with sea ice.

A learned correction (gradient boosting on physics residuals) is validated
with whole icebergs held out, so no iceberg appears in both train and test.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from pyproj import Proj
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation

from antarctic_routing.preprocessing.grid import PolarGrid

EARTH_RADIUS_M = 6371e3
_PROJ = Proj("EPSG:3031")


def haversine_m(lat1, lon1, lat2, lon2, radius: float = EARTH_RADIUS_M):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi, dlam = p2 - p1, np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


@dataclass
class ForcingField:
    """Gridded, time-varying vector field (grid-aligned x/y components, m/s)."""

    times_h: np.ndarray
    grid: PolarGrid
    u: np.ndarray  # (T, ny, nx)
    v: np.ndarray

    def __post_init__(self) -> None:
        t = np.asarray(self.times_h, float)
        u, v = np.asarray(self.u, float), np.asarray(self.v, float)
        if t.size == 1:
            t, u, v = np.r_[t, t + 1.0], np.concatenate([u, u]), np.concatenate([v, v])
        self.times_h, self.u, self.v = t, u, v
        axes = (t, self.grid.y, self.grid.x)
        self._iu = RegularGridInterpolator(axes, u, bounds_error=False, fill_value=np.nan)
        self._iv = RegularGridInterpolator(axes, v, bounds_error=False, fill_value=np.nan)

    @classmethod
    def constant(cls, grid: PolarGrid, u: float, v: float, hours: float) -> ForcingField:
        shape = (2, *grid.shape)
        return cls(np.array([0.0, float(hours)]), grid, np.full(shape, float(u)), np.full(shape, float(v)))

    @classmethod
    def from_scenarios_current(cls, world) -> ForcingField:
        times = np.arange(world.n_times) * world.time_step_hours
        return cls(times, world.grid, world.current_x, world.current_y)

    @classmethod
    def from_scenarios_wind(cls, world) -> ForcingField | None:
        if world.wind_x is None:
            return None
        times = np.arange(world.n_times) * world.time_step_hours
        return cls(times, world.grid, world.wind_x, world.wind_y)

    def sample(self, t_h, px, py) -> tuple[np.ndarray, np.ndarray]:
        t = np.clip(np.broadcast_to(np.asarray(t_h, float), np.shape(px)), self.times_h[0], self.times_h[-1])
        pts = np.column_stack([np.ravel(t), np.ravel(py), np.ravel(px)])
        return self._iu(pts).reshape(np.shape(px)), self._iv(pts).reshape(np.shape(px))


def _scale(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    lon, lat = _PROJ(px, py, inverse=True)
    return np.asarray(_PROJ.get_factors(lon, lat).meridional_scale)


def _integrate(px, py, alpha, noise, current, wind, hours, dt_hours, beta: float = 1.0):
    n = int(round(hours / dt_hours))
    dt = dt_hours * 3600.0
    out = np.full((n + 1, px.size, 2), np.nan)
    out[0, :, 0], out[0, :, 1] = px, py
    x, y = px.astype(float).copy(), py.astype(float).copy()

    def vel(t, x, y):
        uo, vo = current.sample(t, x, y)
        uo, vo = beta * uo, beta * vo
        if wind is not None:
            ua, va = wind.sample(t, x, y)
            uo, vo = uo + alpha * ua, vo + alpha * va
        return uo + noise[:, 0], vo + noise[:, 1]

    for i in range(n):
        t = i * dt_hours
        u1, v1 = vel(t, x, y)
        k1 = _scale(x, y)
        xp, yp = x + dt * k1 * u1, y + dt * k1 * v1
        u2, v2 = vel(t + dt_hours, xp, yp)
        k2 = _scale(np.nan_to_num(xp), np.nan_to_num(yp))
        x = x + 0.5 * dt * (k1 * u1 + k2 * u2)
        y = y + 0.5 * dt * (k1 * v1 + k2 * v2)
        bad = ~(np.isfinite(x) & np.isfinite(y))
        x[bad], y[bad] = np.nan, np.nan
        out[i + 1, :, 0], out[i + 1, :, 1] = x, y
    times = np.arange(n + 1) * dt_hours
    return times, out


@dataclass
class Trajectory:
    times_h: np.ndarray
    xy: np.ndarray  # (n+1, 2)

    @property
    def exited(self) -> bool:
        return bool(np.isnan(self.xy[-1]).any())


def drift_trajectory(x0, y0, current: ForcingField, wind: ForcingField | None, alpha: float,
                     hours: float, dt_hours: float = 1.0) -> Trajectory:
    times, out = _integrate(np.array([x0], float), np.array([y0], float), np.array([alpha], float),
                            np.zeros((1, 2)), current, wind, hours, dt_hours)
    return Trajectory(times, out[:, 0])


@dataclass
class DriftEnsemble:
    ids: list[str]
    times_h: np.ndarray
    xy: np.ndarray      # (K, B, n+1, 2)
    alpha: np.ndarray   # (K, B)


def drift_ensemble(
    bergs: Sequence[tuple[str, float, float]],
    current: ForcingField,
    wind: ForcingField | None,
    n_members: int,
    hours: float,
    rng: np.random.Generator,
    sigma_pos_m: float = 2_000.0,
    alpha_range: tuple[float, float] = (0.01, 0.03),
    velocity_noise: float = 0.03,
    dt_hours: float = 1.0,
    beta: float = 1.0,
) -> DriftEnsemble:
    K, B = n_members, len(bergs)
    x0, y0 = PolarGrid.to_xy(np.array([b[1] for b in bergs]), np.array([b[2] for b in bergs]))
    px = np.repeat(np.asarray(x0)[None], K, 0) + rng.normal(0, sigma_pos_m, (K, B)) if sigma_pos_m else \
        np.repeat(np.asarray(x0)[None], K, 0)
    py = np.repeat(np.asarray(y0)[None], K, 0) + rng.normal(0, sigma_pos_m, (K, B)) if sigma_pos_m else \
        np.repeat(np.asarray(y0)[None], K, 0)
    alpha = rng.uniform(alpha_range[0], alpha_range[1], (K, B))
    member_noise = rng.normal(0, velocity_noise, (K, 1, 2)) if velocity_noise else np.zeros((K, 1, 2))
    noise = np.broadcast_to(member_noise, (K, B, 2)).reshape(K * B, 2)
    times, out = _integrate(px.ravel(), py.ravel(), alpha.ravel(), noise, current, wind, hours, dt_hours, beta)
    xy = out.reshape(times.size, K, B, 2).transpose(1, 2, 0, 3)
    return DriftEnsemble([b[0] for b in bergs], times, xy, alpha)


def scale_ensemble_spread(ens: DriftEnsemble, factor: float) -> DriftEnsemble:
    """Members scaled about the ensemble mean of each berg at every time: ``m + factor * (x_k - m)``.

    A post-hoc spread calibration: the mean of the surviving members (and so the
    ensemble-mean trajectory) is unchanged, and members that left the grid stay NaN.
    """
    if factor == 1.0:
        return ens
    xy = ens.xy.copy()
    ok = np.isfinite(xy).all(-1, keepdims=True)                      # (K, B, T, 1)
    n = ok.sum(0, keepdims=True)
    mean = np.where(ok, xy, 0.0).sum(0, keepdims=True) / np.maximum(n, 1)
    xy = np.where(ok, mean + factor * (xy - mean), np.nan)
    return replace(ens, xy=xy)


def presence_layers(ens: DriftEnsemble, grid: PolarGrid, n_layers: int, layer_hours: float,
                    radius_m: float = 5_000.0) -> np.ndarray:
    """(K, T, ny, nx) True where any berg is within ``radius_m`` during layer t."""
    K = ens.xy.shape[0]
    pres = np.zeros((K, n_layers, *grid.shape), bool)
    res = grid.resolution_m
    for t in range(n_layers):
        sel = (ens.times_h >= t * layer_hours) & (ens.times_h <= (t + 1) * layer_hours)
        pts = ens.xy[:, :, sel]  # (K, B, s, 2)
        cols = np.floor((pts[..., 0] - (grid.x[0] - res / 2)) / res)
        rows = np.floor((pts[..., 1] - (grid.y[0] - res / 2)) / res)
        ok = np.isfinite(cols) & np.isfinite(rows)
        ok &= (cols >= 0) & (cols < grid.x.size) & (rows >= 0) & (rows < grid.y.size)
        k_idx = np.broadcast_to(np.arange(K)[:, None, None], ok.shape)
        pres[k_idx[ok], t, rows[ok].astype(int), cols[ok].astype(int)] = True
    r = int(np.ceil(radius_m / res))
    if r > 0:
        yy, xx = np.mgrid[-r: r + 1, -r: r + 1]
        disk = (xx**2 + yy**2) <= r**2
        pres = binary_dilation(pres, structure=disk[None, None])
    return pres


def density(presence: np.ndarray) -> np.ndarray:
    """P_i(t): fraction of members with at least one berg in (or near) cell i."""
    return presence.mean(axis=0)


def group_split(ids: np.ndarray, test_fraction: float = 0.3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Boolean train/test masks that keep every iceberg entirely on one side."""
    ids = np.asarray(ids)
    unique = np.unique(ids)
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(test_fraction * unique.size)))
    test_ids = set(rng.choice(unique, n_test, replace=False).tolist())
    test = np.array([i in test_ids for i in ids])
    return ~test, test


class ResidualCorrector:
    """Gradient-boosted correction of 24 h physics drift residuals."""

    def __init__(self, seed: int = 0) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        self.models = [GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3)
                       for _ in range(2)]

    @staticmethod
    def features(x, y, uw, vw, current: ForcingField) -> np.ndarray:
        uo, vo = current.sample(0.0, np.asarray(x, float), np.asarray(y, float))
        lat, _ = PolarGrid.to_latlon(x, y)
        return np.column_stack([uw, vw, uo, vo, lat])

    def fit(self, feats: np.ndarray, residual_xy: np.ndarray) -> ResidualCorrector:
        for i, m in enumerate(self.models):
            m.fit(feats, residual_xy[:, i])
        return self

    def predict(self, feats: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict(feats) for m in self.models])

    @classmethod
    def fit_evaluate(cls, rows, grid: PolarGrid, current: ForcingField, alpha_physics: float,
                     seed: int = 0, test_fraction: float = 0.3):
        """rows: (iceberg_id, day, x, y, wind_u, wind_v) daily observations."""
        by_berg: dict[str, list] = {}
        for r in rows:
            by_berg.setdefault(r[0], []).append(r)
        ids, start, obs, phys, feats = [], [], [], [], []
        for bid, seq in by_berg.items():
            seq.sort(key=lambda r: r[1])
            for a, b in zip(seq, seq[1:], strict=False):
                if b[1] - a[1] != 1 or not np.isfinite([a[2], a[3], b[2], b[3]]).all():
                    continue
                wind = ForcingField.constant(grid, a[4], a[5], 24.0)
                end = drift_trajectory(a[2], a[3], current, wind, alpha_physics, 24.0).xy[-1]
                if not np.isfinite(end).all():
                    continue
                ids.append(bid)
                start.append((a[2], a[3]))
                obs.append((b[2], b[3]))
                phys.append(end)
                feats.append(cls.features(a[2], a[3], a[4], a[5], current)[0])
        ids, start, obs, phys, feats = map(np.asarray, (ids, start, obs, phys, feats))
        train, test = group_split(ids, test_fraction, seed)
        model = cls(seed).fit(feats[train], (obs - phys)[train])
        corrected = phys[test] + model.predict(feats[test])

        def km(pred):
            lat_p, lon_p = PolarGrid.to_latlon(pred[:, 0], pred[:, 1])
            lat_o, lon_o = PolarGrid.to_latlon(obs[test][:, 0], obs[test][:, 1])
            return float(np.mean(haversine_m(lat_o, lon_o, lat_p, lon_p)) / 1000.0)

        report = {
            "train_icebergs": sorted(set(ids[train].tolist())),
            "test_icebergs": sorted(set(ids[test].tolist())),
            "n_test_steps": int(test.sum()),
            "mean_error_km": {"no_move": km(start[test]), "physics": km(phys[test]), "corrected": km(corrected)},
        }
        return model, report


def add_iceberg_hazard(
    world,
    bergs: Sequence[tuple[str, float, float]],
    rng: np.random.Generator,
    sigma_pos_m: float = 2_000.0,
    alpha_range: tuple[float, float] = (0.01, 0.03),
    velocity_noise: float = 0.03,
    radius_m: float = 5_000.0,
    dt_hours: float = 1.0,
    beta: float = 1.0,
    spread_factor: float = 1.0,
):
    """Return ``world`` with joint iceberg presence: scenario k gets drift member k.

    Drift uses the scenario set's own currents and winds, so the iceberg and
    sea-ice hazards are evaluated together in each joint scenario. ``beta`` (current
    coupling) and ``spread_factor`` (:func:`scale_ensemble_spread`) default to the
    uncalibrated model; calibrated values are passed explicitly.
    """
    current = ForcingField.from_scenarios_current(world)
    wind = ForcingField.from_scenarios_wind(world)
    ens = drift_ensemble(bergs, current, wind, world.n_scenarios, world.horizon_hours, rng,
                         sigma_pos_m=sigma_pos_m, alpha_range=alpha_range, velocity_noise=velocity_noise,
                         dt_hours=dt_hours, beta=beta)
    ens = scale_ensemble_spread(ens, spread_factor)
    pres = presence_layers(ens, world.grid, world.n_times, world.time_step_hours, radius_m)
    pres &= ~world.land[None, None]
    if world.berg is not None:
        pres |= world.berg
    names = ", ".join(b[0] for b in bergs)
    return replace(world, berg=pres,
                   description=f"{world.description} + iceberg drift ensemble ({names}, radius {radius_m / 1e3:g} km)")
