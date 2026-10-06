"""Hindcast of the iceberg drift ensemble against official USNIC weekly positions.

For every observed iceberg position on the routing grid (the *start*), the
unchanged drift ensemble (:func:`iceberg.drift.drift_ensemble`, the same
parameters as :func:`iceberg.drift.add_iceberg_hazard`) is run from that
position with daily forcing layers selected by exact UTC date, and compared
with the *same iceberg ID* observed again about 7, 14 or 21 days later. The
forecast is read at the exact observed elapsed time, not at the nominal lead.

Scoring rules (nothing is quietly dropped; every start/lead gets a status):

* ``evaluated`` - at least ``min_coverage`` of the members are still on the
  grid at the observation time; error is from the mean of those members.
* ``low_coverage`` - some members survive but fewer than ``min_coverage``;
  reported with its error and coverage, never pooled with ``evaluated``.
* ``exited_grid`` - every member left the grid. A failure, not a success,
  even if the observed berg also left it.
* ``missing_forcing`` - a forcing date between start and observation is absent.
* ``no_observation`` - the berg has no official position within +-``tolerance_days``
  of the nominal lead.

All errors are great-circle distances in km.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from antarctic_routing.iceberg.drift import ForcingField, drift_ensemble, haversine_m
from antarctic_routing.preprocessing.grid import PolarGrid

LEADS = (7, 14, 21)
# add_iceberg_hazard defaults: the configuration the planner uses
ENSEMBLE_PARAMS = {"sigma_pos_m": 2_000.0, "alpha_range": (0.01, 0.03), "velocity_noise": 0.03, "dt_hours": 1.0}


@dataclass(frozen=True)
class Observation:
    iceberg_id: str
    day: date
    lat: float
    lon: float
    source_file: str = ""


@dataclass
class DailyForcing:
    """Daily (00 UTC) grid-aligned currents and winds, one layer per calendar date."""

    grid: PolarGrid
    days: np.ndarray                     # datetime64[D], sorted, unique
    current: tuple[np.ndarray, np.ndarray]   # (T, ny, nx) each, or (ny, nx) for a constant field
    wind: tuple[np.ndarray, np.ndarray] | None
    label: str = ""
    meta: dict = field(default_factory=dict)

    def subset(self, first: date, last: date) -> DailyForcing:
        """Only the dates ``first .. last``: data outside a split is then unreachable, not merely unused."""
        keep = (self.days >= np.datetime64(first, "D")) & (self.days <= np.datetime64(last, "D"))
        cur = self.current if np.ndim(self.current[0]) == 2 else (self.current[0][keep], self.current[1][keep])
        wind = None if self.wind is None else (self.wind[0][keep], self.wind[1][keep])
        return DailyForcing(self.grid, self.days[keep], cur, wind, self.label,
                            {**self.meta, "subset": [str(first), str(last)]})

    def missing_days(self, start: date, end: date) -> list[str]:
        need = np.datetime64(start, "D") + np.arange((end - start).days + 1)
        have = set(self.days.tolist())
        return [str(d) for d in need if d.astype(object) not in have]

    def window(self, start: date, end: date) -> tuple[ForcingField, ForcingField | None]:
        """Forcing fields with time 0 h = ``start`` 00 UTC, covering ``start .. end`` by exact date."""
        missing = self.missing_days(start, end)
        if missing:
            raise ValueError(f"{self.label}: missing forcing date(s) {', '.join(missing)}")
        idx = np.searchsorted(self.days, np.datetime64(start, "D") + np.arange((end - start).days + 1))
        hours = np.arange(idx.size) * 24.0

        def layers(f):
            u, v = f
            if np.ndim(u) == 2:
                return np.broadcast_to(u, (idx.size, *u.shape)), np.broadcast_to(v, (idx.size, *v.shape))
            return u[idx], v[idx]

        cur = ForcingField(hours, self.grid, *layers(self.current))
        wind = None if self.wind is None else ForcingField(hours, self.grid, *layers(self.wind))
        return cur, wind


def pair_observations(obs: Sequence[Observation], starts: Sequence[Observation], leads=LEADS,
                      tolerance_days: int = 1) -> list[tuple[Observation, int, Observation | None]]:
    """``(start, lead, target)`` for every start and lead: the same iceberg's observation
    closest to ``start.day + lead`` within ``tolerance_days`` (None if there is none)."""
    by_id: dict[str, dict[date, Observation]] = {}
    for o in obs:
        by_id.setdefault(o.iceberg_id, {})[o.day] = o
    out = []
    for s in starts:
        track = by_id.get(s.iceberg_id, {})
        for lead in leads:
            best = None
            for k in sorted(range(-tolerance_days, tolerance_days + 1), key=abs):
                t = date.fromordinal(s.day.toordinal() + lead + k)
                if t in track:
                    best = track[t]
                    break
            out.append((s, lead, best))
    return out


def _latlon(xy):
    lat, lon = PolarGrid.to_latlon(xy[..., 0], xy[..., 1])
    return np.asarray(lat), np.asarray(lon)


def score_members(members_xy: np.ndarray, target: Observation, min_coverage: float) -> dict:
    """Score one ensemble (K, 2) at the observation time against ``target``."""
    ok = np.isfinite(members_xy).all(1)
    cov = float(ok.mean())
    rec = {"coverage": cov, "n_members": int(members_xy.shape[0]), "n_surviving": int(ok.sum())}
    if not ok.any():
        return {**rec, "status": "exited_grid"}
    mean = members_xy[ok].mean(0)
    lat_m, lon_m = _latlon(mean)
    lat_k, lon_k = _latlon(members_xy[ok])
    err = haversine_m(target.lat, target.lon, lat_m, lon_m) / 1e3
    member_err = haversine_m(target.lat, target.lon, lat_k, lon_k) / 1e3
    spread = haversine_m(lat_m, lon_m, lat_k, lon_k) / 1e3
    rec.update(status="evaluated" if cov >= min_coverage else "low_coverage",
               error_km=float(err), median_member_error_km=float(np.median(member_err)),
               min_member_error_km=float(member_err.min()),
               spread_p90_km=float(np.percentile(spread, 90)),
               obs_within_p90_spread=bool(err <= np.percentile(spread, 90)),
               forecast_lat=float(lat_m), forecast_lon=float(lon_m))
    return rec


def season_of_day(d: date) -> int:
    """Austral season label: Nov-Dec of year Y and Jan-Feb of Y+1 are season Y."""
    return d.year if d.month >= 7 else d.year - 1


def covered_until(forcing: DailyForcing, start: date, end: date) -> date | None:
    """Last date ``d <= end`` such that every date ``start .. d`` has forcing (None if ``start`` is missing)."""
    have = set(forcing.days.tolist())
    last = None
    for k in range((end - start).days + 1):
        d = np.datetime64(start, "D") + k
        if d.astype(object) not in have:
            break
        last = d.astype(object)
    return last


def run_start_group(starts: Sequence[Observation], pairs, forcing: DailyForcing, seed: int,
                    n_members: int = 200, min_coverage: float = 0.5, params: dict | None = None,
                    keep_members: bool = False) -> list[dict]:
    """Drift every berg starting on one date together (as the planner does) and score all its leads.

    ``pairs`` are this group's ``(start, lead, target)`` from :func:`pair_observations`.
    ``params`` overrides :data:`ENSEMBLE_PARAMS` (e.g. a calibrated ``beta``/``alpha_range``).
    ``keep_members`` adds the raw member positions (K, 2) at the observation time as ``members_xy``.
    """
    ens_params = {**ENSEMBLE_PARAMS, **(params or {})}
    day0 = starts[0].day
    if any(s.day != day0 for s in starts):
        raise ValueError("one start date per group")
    results, wanted = [], []
    for s, lead, t in pairs:
        if t is None:
            results.append({"iceberg_id": s.iceberg_id, "start": str(s.day), "lead": lead, "config": forcing.label,
                            "status": "no_observation"})
        else:
            wanted.append((s, lead, t))
    if not wanted:
        return results
    end_day = covered_until(forcing, day0, max(t.day for _, _, t in wanted))
    ids = list(dict.fromkeys(s.iceberg_id for s in starts))
    ens = None
    if end_day is not None and end_day > day0:
        cur, wind = forcing.window(day0, end_day)
        first = {s.iceberg_id: s for s in starts}
        bergs = [(i, first[i].lat, first[i].lon) for i in ids]
        ens = drift_ensemble(bergs, cur, wind, n_members, (end_day - day0).days * 24.0,
                             np.random.default_rng(seed), **ens_params)
    for s, lead, t in wanted:
        base = {"iceberg_id": s.iceberg_id, "start": str(s.day), "lead": lead, "config": forcing.label,
                "obs_day": str(t.day), "elapsed_days": (t.day - s.day).days, "season": season_of_day(s.day),
                "start_lat": s.lat, "start_lon": s.lon, "obs_lat": t.lat, "obs_lon": t.lon,
                "observed_displacement_km": float(haversine_m(s.lat, s.lon, t.lat, t.lon) / 1e3)}
        if ens is None or t.day > end_day:
            results.append({**base, "status": "missing_forcing", "missing_dates": forcing.missing_days(day0, t.day)})
            continue
        step = int(round((t.day - s.day).days * 24.0 / ens_params["dt_hours"]))
        members = ens.xy[:, ids.index(s.iceberg_id), step]
        row = {**base, **score_members(members, t, min_coverage)}
        if keep_members:
            row["members_xy"] = members.copy()
        results.append(row)
    return results


def summarize(rows: Sequence[dict], leads=LEADS) -> dict:
    """Per lead and config: status counts and error statistics of ``evaluated`` cases."""
    out: dict = {}
    for cfg in sorted({r["config"] for r in rows}):
        out[cfg] = {}
        for lead in leads:
            sel = [r for r in rows if r["config"] == cfg and r["lead"] == lead]
            counts: dict[str, int] = {}
            for r in sel:
                counts[r["status"]] = counts.get(r["status"], 0) + 1
            ev = np.array([r["error_km"] for r in sel if r["status"] == "evaluated"])
            cov = np.array([r["coverage"] for r in sel if "coverage" in r])
            out[cfg][lead] = {"counts": counts, "n_evaluated": int(ev.size),
                              "n_tracks": len({(r["iceberg_id"], r["season"]) for r in sel
                                               if r["status"] == "evaluated"}),
                              "error_km": _stats(ev),
                              "coverage": _stats(cov, pct=False)}
    return out


def _stats(a: np.ndarray, pct: bool = True) -> dict:
    if a.size == 0:
        return {"n": 0}
    s = {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "min": float(a.min()),
         "max": float(a.max())}
    if pct:
        s.update(p25=float(np.percentile(a, 25)), p75=float(np.percentile(a, 75)), p90=float(np.percentile(a, 90)))
    return s
