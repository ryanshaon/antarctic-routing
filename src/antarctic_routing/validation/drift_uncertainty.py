"""Spread (uncertainty) calibration of the iceberg drift ensemble against official positions.

The drift ensemble (:func:`antarctic_routing.iceberg.drift.drift_ensemble`) perturbs the initial
position (N(0, sigma_pos) per member and berg), the wind coefficient alpha (uniform per member and
berg) and adds one constant velocity error per member (N(0, velocity_noise), shared by all bergs).
A constant velocity error makes the spread grow linearly with lead time.

Adjustment: members are scaled about the ensemble mean at each lead,

    x_k' = m + s(L) * (x_k - m),    s(L) = c * (L / REF_LEAD_DAYS) ** q,

so the ensemble mean (and the deterministic physics) is unchanged by construction. ``c`` is an
overall spread inflation (c < 1 shrinks); ``q`` changes how spread grows with lead (q = -0.5 turns
the linear growth of a constant velocity error into the sqrt(t) growth of a random walk).

Prediction regions are ensemble-based: the level-p region is every point whose Mahalanobis
distance from the ensemble mean (ensemble covariance) is within the p-quantile of the members'
own distances. Scaling the members by s scales the covariance by s**2 and leaves the members'
distances unchanged, so the observation's distance is simply divided by s**2.

Distances are true kilometres: projected EPSG:3031 offsets are divided by the local map scale factor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from antarctic_routing.iceberg.drift import _scale
from antarctic_routing.preprocessing.grid import PolarGrid

LEVELS = (0.5, 0.8, 0.95)
REF_LEAD_DAYS = 14.0
SIMPLER_MARGIN = 0.02     # the 2-parameter model must cut validation coverage error by 2 points more


@dataclass(frozen=True)
class SpreadModel:
    c: float = 1.0
    q: float = 0.0

    def factor(self, lead_days: float) -> float:
        return float(self.c * (lead_days / REF_LEAD_DAYS) ** self.q)

    def as_dict(self) -> dict:
        return {"c": self.c, "q": self.q, "factor_7_14_21": [self.factor(d) for d in (7, 14, 21)]}


IDENTITY = SpreadModel()


def scale_spread(members_xy: np.ndarray, factor: float) -> np.ndarray:
    """Members (K, 2) scaled about their mean; members that left the grid stay NaN."""
    ok = np.isfinite(members_xy).all(1)
    out = members_xy.astype(float).copy()
    if ok.any():
        m = members_xy[ok].mean(0)
        out[ok] = m + factor * (members_xy[ok] - m)
    return out


@dataclass
class Case:
    """One forecast at one lead: member deviations from the ensemble mean and the observed error (km)."""

    iceberg_id: str
    season: int
    start: str
    lead: int
    dev: np.ndarray          # (K, 2) member - mean, km
    err: np.ndarray          # (2,) observation - mean, km
    pair_mean_km: float = 0.0  # mean |d_k - d_l| (for the energy score; independent of s)
    d2_members: np.ndarray | None = None
    d2_obs: float = 0.0

    @property
    def track(self) -> str:
        return f"{self.iceberg_id}|{self.season}"

    @property
    def error_km(self) -> float:
        return float(np.hypot(*self.err))

    @property
    def spread_km(self) -> float:
        """RMS distance of members from the ensemble mean."""
        return float(np.sqrt((self.dev ** 2).sum(1).mean()))


def make_case(members_xy: np.ndarray, obs_lat: float, obs_lon: float, lead: int, iceberg_id: str = "",
              season: int = 0, start: str = "") -> Case:
    ok = np.isfinite(members_xy).all(1)
    m = members_xy[ok].mean(0)
    k = float(_scale(np.array([m[0]]), np.array([m[1]]))[0])
    ox, oy = PolarGrid.to_xy(np.array([obs_lat]), np.array([obs_lon]))
    dev = (members_xy[ok] - m) / k / 1e3
    err = (np.array([float(ox[0]), float(oy[0])]) - m) / k / 1e3
    c = Case(iceberg_id, season, start, lead, dev, err)
    diff = dev[:, None, :] - dev[None, :, :]
    n = dev.shape[0]
    c.pair_mean_km = float(np.hypot(diff[..., 0], diff[..., 1]).sum() / (n * n))
    cov = np.cov(dev.T) + 1e-9 * np.eye(2)
    inv = np.linalg.inv(cov)
    c.d2_members = np.einsum("ki,ij,kj->k", dev, inv, dev)
    c.d2_obs = float(err @ inv @ err)
    return c


def covered(case: Case, factor: float, level: float) -> bool:
    return bool(case.d2_obs / factor ** 2 <= np.quantile(case.d2_members, level))


def energy_score(case: Case, factor: float) -> float:
    """Multivariate CRPS of the scaled ensemble (km); lower is better, proper."""
    return float(np.hypot(*(factor * case.dev - case.err).T).mean() - 0.5 * factor * case.pair_mean_km)


def evaluate(cases: Sequence[Case], model: SpreadModel, levels=LEVELS) -> dict:
    """Per lead: coverage of nominal regions, spread vs error, energy score; plus overall summaries."""
    out: dict = {"model": model.as_dict(), "per_lead": {}}
    errs = []
    for lead in sorted({c.lead for c in cases}):
        cs = [c for c in cases if c.lead == lead]
        s = model.factor(lead)
        spread = np.array([s * c.spread_km for c in cs])
        err = np.array([c.error_km for c in cs])
        cov = {f"{lv:.2f}": float(np.mean([covered(c, s, lv) for c in cs])) for lv in levels}
        errs += [abs(cov[f"{lv:.2f}"] - lv) for lv in levels]
        out["per_lead"][lead] = {
            "n": len(cs), "tracks": len({c.track for c in cs}), "factor": s, "coverage": cov,
            "mean_spread_km": float(spread.mean()), "rms_spread_km": float(np.sqrt((spread ** 2).mean())),
            "rmse_km": float(np.sqrt((err ** 2).mean())), "mean_error_km": float(err.mean()),
            "median_error_km": float(np.median(err)),
            "spread_error_ratio": float(np.sqrt((spread ** 2).mean()) / np.sqrt((err ** 2).mean())),
            "energy_score_km": float(np.mean([energy_score(c, s) for c in cs]))}
    pl = out["per_lead"].values()
    out["mean_abs_coverage_error"] = float(np.mean(errs))
    out["mean_energy_score_km"] = float(np.mean([v["energy_score_km"] for v in pl]))
    return out


def fit(cases: Sequence[Case], candidates: Sequence[SpreadModel]) -> list[dict]:
    """Mean (over leads) energy score for every candidate, best first; ties keep candidate order."""
    leads = sorted({c.lead for c in cases})
    by = {lead: [c for c in cases if c.lead == lead] for lead in leads}
    res = []
    for i, m in enumerate(candidates):
        per = {lead: float(np.mean([energy_score(c, m.factor(lead)) for c in by[lead]])) for lead in leads}
        res.append({"model": m, "params": m.as_dict(), "energy_score_km": float(np.mean(list(per.values()))),
                    "per_lead": per, "_i": i})
    res.sort(key=lambda r: (r["energy_score_km"], r["_i"]))
    return res


def select(val: dict[str, dict], margin: float = SIMPLER_MARGIN) -> str:
    """Lowest validation coverage error among 'none', 'one_parameter', 'two_parameter', preferring simpler.

    ``one_parameter`` must beat ``none``; ``two_parameter`` must beat ``one_parameter`` by ``margin``.
    """
    mace = {k: v["mean_abs_coverage_error"] for k, v in val.items()}
    pick = "one_parameter" if mace["one_parameter"] < mace["none"] else "none"
    if mace["two_parameter"] < mace[pick] - margin:
        pick = "two_parameter"
    return pick


def spread_error_relation(cases: Sequence[Case], n_bins: int = 4) -> dict:
    """Per lead: Spearman rank correlation of spread and error, and error by spread quantile bin."""
    out = {}
    for lead in sorted({c.lead for c in cases}):
        cs = [c for c in cases if c.lead == lead]
        sp = np.array([c.spread_km for c in cs])
        er = np.array([c.error_km for c in cs])
        rs, re = np.argsort(np.argsort(sp)), np.argsort(np.argsort(er))
        rho = float(np.corrcoef(rs, re)[0, 1]) if len(cs) > 2 else None
        edges = np.quantile(sp, np.linspace(0, 1, n_bins + 1))
        idx = np.clip(np.searchsorted(edges, sp, side="right") - 1, 0, n_bins - 1)
        bins = [{"spread_km": [float(edges[b]), float(edges[b + 1])], "n": int((idx == b).sum()),
                 "rms_spread_km": float(np.sqrt((sp[idx == b] ** 2).mean())) if (idx == b).any() else None,
                 "rmse_km": float(np.sqrt((er[idx == b] ** 2).mean())) if (idx == b).any() else None}
                for b in range(n_bins)]
        out[lead] = {"n": len(cs), "spearman_spread_error": rho, "bins": bins}
    return out


C_GRID = tuple(float(round(v, 4)) for v in np.exp(np.linspace(np.log(0.1), np.log(3.0), 35)))
Q_GRID = tuple(float(round(v, 2)) for v in np.arange(-1.5, 0.51, 0.1))


def one_parameter_candidates(cs=C_GRID) -> list[SpreadModel]:
    return [SpreadModel(c, 0.0) for c in cs]


def two_parameter_candidates(cs=C_GRID, qs=Q_GRID) -> list[SpreadModel]:
    return [SpreadModel(c, q) for c in cs for q in qs]
