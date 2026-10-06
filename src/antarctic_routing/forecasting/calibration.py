"""Stage 4.5 - from deterministic forecasts to calibrated hazard probabilities.

1. **Residual-bootstrap ensemble.** Member k adds one whole historical forecast
   error sequence (all leads and cells of one training sample) to the forecast:

       C^(k) = clip(C_hat + (C_obs - C_hat)_{sample s_k}, 0, 1)

   Using whole residual fields keeps each member spatially and temporally
   coherent, so members are valid *joint* scenarios for route evaluation.
2. **Exceedance probability.** p_hat = (1/K) sum_k 1[C^(k) >= tau].
3. **Isotonic calibration** per lead time, fitted on validation seasons only,
   because ensemble spread is not automatically a calibrated probability.
4. **Verification** on test seasons: Brier score, reliability tables and Brier
   skill vs a climatological-frequency baseline (fitted on training seasons).

Only observed ocean cells (ocean AND NOT imputed on the target day) enter the
residual bank, the isotonic fit and the scores; residuals at imputed cells are 0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import xarray as xr
from sklearn.isotonic import IsotonicRegression

from antarctic_routing.forecasting.dataset import SequenceDataset, build_samples, issue_seasons, valid_target_mask
from antarctic_routing.preprocessing.climatology import Climatology
from antarctic_routing.validation.metrics import brier_score, reliability_bins


def residual_ensemble(pred: np.ndarray, bank: np.ndarray, n_members: int, rng: np.random.Generator) -> np.ndarray:
    """Members (K, H, ny, nx) = clip(pred + whole residual fields drawn from ``bank``)."""
    picks = rng.integers(0, bank.shape[0], size=n_members)
    return np.clip(pred[None] + bank[picks], 0.0, 1.0)


@dataclass
class IsotonicCalibrator:
    """Per-lead monotone mapping from raw to calibrated probability."""

    knots: dict[int, tuple[np.ndarray, np.ndarray]]

    @classmethod
    def fit(cls, data: Mapping[int, tuple[np.ndarray, np.ndarray]]) -> IsotonicCalibrator:
        knots = {}
        for lead, (p, o) in data.items():
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
            iso.fit(np.asarray(p, float), np.asarray(o, float))
            knots[int(lead)] = (np.asarray(iso.X_thresholds_), np.asarray(iso.y_thresholds_))
        return cls(knots)

    def transform(self, p: np.ndarray, lead: int) -> np.ndarray:
        x, y = self.knots[int(lead)]
        return np.interp(np.asarray(p, float), x, y)

    def to_dict(self) -> dict:
        return {"leads": sorted(self.knots),
                "knots": {str(k): {"x": v[0].tolist(), "y": v[1].tolist()} for k, v in self.knots.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> IsotonicCalibrator:
        return cls({int(k): (np.array(v["x"]), np.array(v["y"])) for k, v in d["knots"].items()})


def _days(ds: xr.Dataset) -> list:
    return [np.datetime64(t, "D").astype(object) for t in ds["time"].values]


def _exceedance_climatology(ds, train_seasons, season_months, tau, window_days=7) -> Climatology:
    conc = ds["ice_concentration"].values.astype(float)
    indicator = np.where(np.isfinite(conc), (np.nan_to_num(conc) >= tau).astype(float), np.nan)
    return Climatology.fit(indicator, _days(ds), train_seasons, season_months, window_days=window_days)


def climatological_exceedance(
    ds: xr.Dataset, train_seasons: Sequence[int], season_months: Sequence[int], tau: float, window_days: int = 7
) -> np.ndarray:
    """Per day-of-season and cell: training-season frequency of C >= tau."""
    return _exceedance_climatology(ds, train_seasons, season_months, tau, window_days).mean


def _split_indices(ds, history_days, lead_days, season_months, seasons) -> list[int]:
    idx = build_samples(ds, history_days, lead_days, season_months)
    wanted = set(seasons)
    return [t for t, s in zip(idx, issue_seasons(ds, idx, season_months), strict=True) if s in wanted]


def _predict(seq: SequenceDataset, predictor, indices: Sequence[int], batch_size: int = 32) -> np.ndarray:
    out = []
    for b in range(0, len(indices), batch_size):
        batch = list(indices[b: b + batch_size])
        out.append(np.asarray(predictor(np.stack([seq.inputs(t) for t in batch]), batch), float))
    return np.concatenate(out)


def evaluate_probabilities(
    ds: xr.Dataset,
    predictor,
    train_seasons: Sequence[int],
    val_seasons: Sequence[int],
    test_seasons: Sequence[int],
    history_days: int,
    lead_days: int,
    season_months: Sequence[int],
    tau: float = 0.15,
    n_members: int = 20,
    bank_size: int = 300,
    seed: int = 0,
    max_fit_points: int = 400_000,
) -> dict:
    sets = [set(train_seasons), set(val_seasons), set(test_seasons)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("train/validation/test seasons overlap")
    rng = np.random.default_rng(seed)
    H = lead_days
    days = _days(ds)
    obs = ds["ice_concentration"].values.astype(float)
    ocean = ~ds["land_mask"].values.astype(bool)
    scoreable = valid_target_mask(ds)  # (T, ny, nx): ocean and not imputed
    split = {name: _split_indices(ds, history_days, lead_days, season_months, seasons)
             for name, seasons in (("train", train_seasons), ("val", val_seasons), ("test", test_seasons))}
    seq = SequenceDataset(ds, split["train"] + split["val"] + split["test"], history_days, lead_days, season_months)

    bank_idx = sorted(rng.choice(split["train"], size=min(bank_size, len(split["train"])), replace=False).tolist())
    pred_bank = _predict(seq, predictor, bank_idx)
    target_bank = np.stack([obs[t + 1: t + H + 1] for t in bank_idx])
    valid_bank = np.stack([scoreable[t + 1: t + H + 1] for t in bank_idx])
    bank = np.where(valid_bank, np.nan_to_num(target_bank - pred_bank, nan=0.0), 0.0)

    def probabilities(indices):
        preds = _predict(seq, predictor, indices)
        p = np.empty((len(indices), H, int(ocean.sum())))
        o = np.empty_like(p)
        v = np.empty(p.shape, bool)
        for j, t in enumerate(indices):
            members = residual_ensemble(preds[j], bank, n_members, rng)
            p[j] = (members >= tau).mean(axis=0)[:, ocean]
            o[j] = (np.nan_to_num(obs[t + 1: t + H + 1]) >= tau)[:, ocean]
            v[j] = scoreable[t + 1: t + H + 1][:, ocean]
        return p, o, v

    p_val, o_val, v_val = probabilities(split["val"])
    fit_data = {}
    for h in range(H):
        pv, ov = p_val[:, h][v_val[:, h]], o_val[:, h][v_val[:, h]]
        if pv.size > max_fit_points:
            keep = rng.choice(pv.size, max_fit_points, replace=False)
            pv, ov = pv[keep], ov[keep]
        fit_data[h + 1] = (pv, ov)
    calibrator = IsotonicCalibrator.fit(fit_data)

    p_test, o_test, v_test = probabilities(split["test"])
    clim = _exceedance_climatology(ds, train_seasons, season_months, tau)
    p_clim = np.stack([
        np.stack([np.nan_to_num(clim.mean_for(days[t] + timedelta(days=h)), nan=0.0)[ocean] for h in range(1, H + 1)])
        for t in split["test"]
    ])
    p_cal = np.stack([calibrator.transform(p_test[:, h], h + 1) for h in range(H)], axis=1)

    briers = {name: [brier_score(arr[:, h][v_test[:, h]], o_test[:, h][v_test[:, h]]) for h in range(H)]
              for name, arr in (("raw", p_test), ("calibrated", p_cal), ("climatology", p_clim))}
    return {
        "tau": tau,
        "leads": list(range(1, H + 1)),
        "brier": briers,
        "brier_skill_vs_climatology": {
            name: [1.0 - b / c if c > 0 else None for b, c in zip(briers[name], briers["climatology"], strict=True)]
            for name in ("raw", "calibrated")
        },
        "reliability": {name: reliability_bins(arr[v_test], o_test[v_test])
                        for name, arr in (("raw", p_test), ("calibrated", p_cal), ("climatology", p_clim))},
        "calibrator": calibrator.to_dict(),
        "n_members": n_members,
        "bank_size": len(bank_idx),
        "n_test_forecasts": len(split["test"]),
        "train_seasons": sorted(train_seasons),
        "val_seasons": sorted(val_seasons),
        "test_seasons": sorted(test_seasons),
        "execution_mode": ds.attrs.get("execution_mode", "real"),
    }
