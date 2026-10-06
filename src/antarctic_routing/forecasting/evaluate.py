"""Stage 10 - per-lead forecast verification against the required baselines.

Every method is scored on identical test samples, cells and lead times:

* MAE / RMSE over valid cells: ocean AND NOT imputed on the target day;
* IIEE (integrated ice-edge error, Goessling et al. 2016): true area where
  forecast and observation disagree on C >= tau_e, in km^2;
* skill vs a baseline: Delta(h) = E_baseline(h) - E_model(h) (> 0 is better).

Climatology and the damped-persistence decay rho are fitted on training seasons
only; evaluation seasons must not overlap them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import timedelta

import numpy as np
import xarray as xr
from pyproj import Proj

from antarctic_routing.forecasting.baselines import damped_anomaly_persistence, fit_anomaly_decay
from antarctic_routing.forecasting.dataset import SequenceDataset, build_samples, issue_seasons, valid_target_mask
from antarctic_routing.preprocessing.climatology import Climatology, season_of

Predictor = Callable[[np.ndarray, Sequence[int]], np.ndarray]
BASELINES = ("persistence", "climatology", "damped_persistence")


def true_cell_area_km2(ds: xr.Dataset) -> np.ndarray:
    """Projected cell area divided by the EPSG:3031 areal scale factor."""
    res_km = float(ds.attrs["resolution_m"]) / 1000.0
    factors = Proj("EPSG:3031").get_factors(ds["lon"].values, ds["lat"].values)
    return res_km**2 / np.asarray(factors.areal_scale)


def fit_baselines(ds: xr.Dataset, train_seasons: Sequence[int], season_months: Sequence[int]):
    """Climatology and damped-persistence rho from training seasons, observed ocean cells only.

    Cells that are land or imputed on a given day are set to NaN before fitting, so
    they contribute neither to the climatology nor to the anomaly pairs behind rho
    (the same ``ocean AND NOT imputed`` definition used for scoring).
    """
    days = [np.datetime64(t, "D").astype(object) for t in ds["time"].values]
    conc = np.where(valid_target_mask(ds), ds["ice_concentration"].values.astype(float), np.nan)
    clim = Climatology.fit(conc, days, train_seasons, season_months, window_days=7)
    segments = []
    for s in sorted(set(train_seasons)):
        idx = [i for i, d in enumerate(days) if season_of(d, season_months) == s]
        if len(idx) > 1:
            segments.append(np.stack([clim.anomaly(conc[i], days[i]) for i in idx]))
    rho = fit_anomaly_decay(segments)
    return clim, rho


def evaluate_forecasts(
    ds: xr.Dataset,
    predictor: Predictor,
    train_seasons: Sequence[int],
    test_seasons: Sequence[int],
    history_days: int,
    lead_days: int,
    season_months: Sequence[int],
    model_name: str = "unet",
    edge_threshold: float = 0.15,
    batch_size: int = 32,
) -> dict:
    if set(train_seasons) & set(test_seasons):
        raise ValueError("train and test seasons overlap")
    clim, rho = fit_baselines(ds, train_seasons, season_months)
    idx = build_samples(ds, history_days, lead_days, season_months)
    wanted = set(test_seasons)
    idx = [t for t, s in zip(idx, issue_seasons(ds, idx, season_months), strict=True) if s in wanted]
    if not idx:
        raise ValueError("no test samples")
    seq = SequenceDataset(ds, idx, history_days, lead_days, season_months)
    obs_all = ds["ice_concentration"].values.astype(float)
    days = [np.datetime64(t, "D").astype(object) for t in ds["time"].values]
    scoreable = valid_target_mask(ds)  # (T, ny, nx): ocean and not imputed
    area = true_cell_area_km2(ds)
    methods = [model_name, *BASELINES]
    H = lead_days
    sums = {m: np.zeros((3, H)) for m in methods}  # abs, sq, count
    iiee = {m: np.zeros(H) for m in methods}
    per_season = {m: {s: np.zeros((2, H)) for s in sorted(wanted)} for m in methods}  # abs, count

    for b in range(0, len(idx), batch_size):
        batch = idx[b: b + batch_size]
        inputs = np.stack([seq.inputs(t) for t in batch])
        model_pred = np.asarray(predictor(inputs, batch), float)
        for j, t in enumerate(batch):
            c_t = obs_all[t]
            season = season_of(days[t], season_months)
            for h in range(1, H + 1):
                obs = obs_all[t + h]
                target_day = days[t] + timedelta(days=h)
                preds = {
                    model_name: model_pred[j, h - 1],
                    "persistence": c_t,
                    "climatology": clim.mean_for(target_day),
                    "damped_persistence": damped_anomaly_persistence(c_t, days[t], h, clim, rho),
                }
                for m, p in preds.items():
                    valid = scoreable[t + h] & np.isfinite(p) & np.isfinite(obs)
                    err = p[valid] - obs[valid]
                    sums[m][0, h - 1] += np.abs(err).sum()
                    sums[m][1, h - 1] += (err**2).sum()
                    sums[m][2, h - 1] += valid.sum()
                    per_season[m][season][0, h - 1] += np.abs(err).sum()
                    per_season[m][season][1, h - 1] += valid.sum()
                    disagree = valid & ((np.nan_to_num(p) >= edge_threshold) != (np.nan_to_num(obs) >= edge_threshold))
                    iiee[m][h - 1] += area[disagree].sum()

    n = len(idx)
    mae = {m: (sums[m][0] / np.maximum(sums[m][2], 1)).tolist() for m in methods}
    rmse = {m: np.sqrt(sums[m][1] / np.maximum(sums[m][2], 1)).tolist() for m in methods}
    return {
        "methods": methods,
        "leads": list(range(1, H + 1)),
        "mae": mae,
        "rmse": rmse,
        "iiee_km2": {m: (iiee[m] / n).tolist() for m in methods},
        "skill_mae_vs": {b: (np.array(mae[b]) - np.array(mae[model_name])).tolist() for b in BASELINES},
        "by_season": {m: {s: (v[0] / np.maximum(v[1], 1)).tolist() for s, v in per_season[m].items()}
                      for m in methods},
        "rho": rho,
        "n_samples": n,
        "train_seasons": sorted(train_seasons),
        "test_seasons": sorted(test_seasons),
        "edge_threshold": edge_threshold,
        "execution_mode": ds.attrs.get("execution_mode", "real"),
    }
