"""Leakage-safe calibration of the iceberg drift physics against official USNIC positions.

Split: whole iceberg-season *tracks* are assigned by austral season (Nov-Feb of
season Y), chronologically:

* train - seasons :data:`TRAIN_SEASONS`: parameters are fitted here (grid search);
* validation - :data:`VALIDATION_SEASONS`: chooses between the fitted candidates;
* test - :data:`TEST_SEASONS`: touched exactly once, after selection.

Every start *and* its target observation must lie in the same season, so a
track can never contribute to two splits; :func:`check_split` asserts this.
Forcing is cut to each split's dates (:meth:`DailyForcing.subset`) so later
seasons are unreachable during fitting, not just unused.

Parameters (only ones with a physical meaning in ``dx/dt = beta*u_o + alpha*u_a``):

* ``beta`` - ocean-current coupling (1 = surface current as given);
* ``alpha_scale`` - multiplies the existing wind-coefficient range (0.01-0.03).

Candidates: a 1-parameter damping ``gamma`` (``beta = alpha_scale = gamma``,
i.e. the whole deterministic drift scaled) and the 2-parameter ``(beta,
alpha_scale)``. Each is fitted on train only; validation picks one, preferring
the simpler unless the 2-parameter one is better by :data:`SIMPLER_MARGIN`.

Objective (lower is better): mean over leads 7/14/21 d of the mean position
error (km) of the ensemble mean, over every train/validation case with an
official target. A case whose members all left the grid scores
:data:`EXIT_PENALTY_KM`; it is never treated as a success.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from antarctic_routing.validation.drift_hindcast import (
    ENSEMBLE_PARAMS,
    LEADS,
    DailyForcing,
    Observation,
    pair_observations,
    run_start_group,
    season_of_day,
)

TRAIN_SEASONS = (2018, 2019, 2020)
VALIDATION_SEASONS = (2021,)
TEST_SEASONS = (2022, 2023)
EXIT_PENALTY_KM = 500.0
SIMPLER_MARGIN = 0.05          # 2-parameter model must beat the 1-parameter one by 5 % on validation


def split_of(season: int) -> str | None:
    if season in TRAIN_SEASONS:
        return "train"
    if season in VALIDATION_SEASONS:
        return "validation"
    if season in TEST_SEASONS:
        return "test"
    return None


def season_bounds(seasons: Sequence[int]) -> tuple[date, date]:
    """First and last calendar day a split may touch (Jul 1 of the first season .. Jun 30 after the last)."""
    return date(min(seasons), 7, 1), date(max(seasons) + 1, 6, 30)


@dataclass(frozen=True)
class DriftParams:
    beta: float = 1.0
    alpha_scale: float = 1.0

    def ensemble(self) -> dict:
        lo, hi = ENSEMBLE_PARAMS["alpha_range"]
        return {"beta": float(self.beta), "alpha_range": (lo * self.alpha_scale, hi * self.alpha_scale)}

    def as_dict(self) -> dict:
        lo, hi = self.ensemble()["alpha_range"]
        return {"beta": self.beta, "alpha_scale": self.alpha_scale, "alpha_range": [lo, hi]}


ORIGINAL = DriftParams()
SELECTED = DriftParams(0.1, 0.1)      # chosen on validation 2021-22 (selection.json); test 2022-24 now spent


def split_pairs(obs: Sequence[Observation], starts: Sequence[Observation], tolerance_days: int = 1):
    """``{split: [(start, lead, target), ...]}`` with whole tracks per split; targets in another season are dropped."""
    out: dict[str, list] = defaultdict(list)
    for s, lead, t in pair_observations(obs, starts, LEADS, tolerance_days):
        sp = split_of(season_of_day(s.day))
        if sp is None:
            continue
        if t is not None and season_of_day(t.day) != season_of_day(s.day):
            t = None
        out[sp].append((s, lead, t))
    check_split(out)
    return dict(out)


def tracks(pairs) -> set[tuple[str, int]]:
    return {(s.iceberg_id, season_of_day(s.day)) for s, _, _ in pairs}


def check_split(by_split: dict) -> dict:
    """Raise if any track, start or target crosses splits or falls outside its split's seasons."""
    seasons = {"train": TRAIN_SEASONS, "validation": VALIDATION_SEASONS, "test": TEST_SEASONS}
    seen: dict[tuple[str, int], str] = {}
    for sp, pairs in by_split.items():
        for s, _, t in pairs:
            for o in (s, t):
                if o is not None and season_of_day(o.day) not in seasons[sp]:
                    raise ValueError(f"{sp} contains {o.iceberg_id} {o.day} from season {season_of_day(o.day)}")
        for tr in tracks(pairs):
            if seen.setdefault(tr, sp) != sp:
                raise ValueError(f"track {tr} appears in {seen[tr]} and {sp}")
    return seen


def run_split(pairs, forcing: DailyForcing, params: DriftParams, seasons: Sequence[int],
              n_members: int = 200, min_coverage: float = 0.5, keep_members: bool = False) -> list[dict]:
    """Hindcast rows for one split with ``params``; forcing outside the split's seasons is removed first."""
    first, last = season_bounds(seasons)
    f = forcing.subset(first, last)
    by_day: dict[date, list] = defaultdict(list)
    for p in pairs:
        if not first <= p[0].day <= last or (p[2] is not None and not first <= p[2].day <= last):
            raise ValueError(f"case {p[0].iceberg_id} {p[0].day} is outside {seasons}")
        by_day[p[0].day].append(p)
    rows = []
    for day in sorted(by_day):
        group = by_day[day]
        group_starts = list({p[0].iceberg_id: p[0] for p in group}.values())
        rows += run_start_group(group_starts, group, f, day.toordinal(), n_members, min_coverage, params.ensemble(),
                                keep_members)
    return rows


def objective(rows: Sequence[dict]) -> dict:
    """Mean over leads of the per-lead mean error (km); exited cases score :data:`EXIT_PENALTY_KM`."""
    per_lead = {}
    for lead in LEADS:
        errs = [r["error_km"] if "error_km" in r else EXIT_PENALTY_KM for r in rows
                if r["lead"] == lead and r["status"] in ("evaluated", "low_coverage", "exited_grid")]
        per_lead[lead] = {"n": len(errs), "mean_error_km": float(np.mean(errs)) if errs else None,
                          "n_exited": sum(1 for r in rows if r["lead"] == lead and r["status"] == "exited_grid")}
    vals = [v["mean_error_km"] for v in per_lead.values() if v["mean_error_km"] is not None]
    return {"value": float(np.mean(vals)) if vals else float("inf"), "per_lead": per_lead}


def grid_search(pairs, forcing: DailyForcing, candidates: Sequence[DriftParams], seasons: Sequence[int],
                n_members: int = 200, evaluate=None) -> list[dict]:
    """Objective for every candidate on one split, best first (ties broken by candidate order)."""
    evaluate = evaluate or (lambda p: objective(run_split(pairs, forcing, p, seasons, n_members)))
    scored = [{"params": p.as_dict(), "objective": evaluate(p), "_p": p, "_i": i} for i, p in enumerate(candidates)]
    scored.sort(key=lambda d: (d["objective"]["value"], d["_i"]))
    return scored


def select(val_1p: float, val_2p: float, margin: float = SIMPLER_MARGIN) -> str:
    """'two_parameter' only if it beats the one-parameter candidate by more than ``margin`` on validation."""
    return "two_parameter" if val_2p < (1.0 - margin) * val_1p else "one_parameter"


# 0 .. 0.1 added after the first train-only fit put the optimum on the 0.1 edge (validation/test untouched)
GAMMA_GRID = (0.0, 0.025, 0.05, 0.075) + tuple(round(g, 2) for g in np.arange(0.10, 1.0001, 0.05))
BETA_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
ALPHA_SCALE_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)


def one_parameter_candidates(grid=GAMMA_GRID) -> list[DriftParams]:
    return [DriftParams(float(g), float(g)) for g in grid]


def two_parameter_candidates(betas=BETA_GRID, scales=ALPHA_SCALE_GRID) -> list[DriftParams]:
    return [DriftParams(b, a) for b in betas for a in scales]
