"""Sliding-window samples for sea-ice forecasting.

Sample issued on day t:

    X_t = [C_{t-L+1}, ..., C_t, land, sin(doy), cos(doy)]   (L + 3 channels)
    Y_t = [C_{t+1}, ..., C_{t+H}]                            (H channels)

A sample is valid only if its whole window (t-L+1 .. t+H) is daily-contiguous
and inside a single season, so no sample straddles the austral winter gap.
Concentration is already a 0-1 fraction, so no data-derived normalisation is
needed (and therefore none can leak from validation/test seasons).

Scoring mask: a target cell counts only if it is ocean and was observed, i.e.
``valid = ~land_mask & ~imputed_mask`` for that target day. Gap-filled
(imputed) values stay in the inputs exactly as stored, but are never scored as
truth. Datasets without ``imputed_mask`` are treated as fully observed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

try:  # PyTorch is the optional 'ml' extra; the API image imports this module without it.
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

from antarctic_routing.preprocessing.climatology import day_of_season, season_of


def _days(ds: xr.Dataset) -> list:
    return [np.datetime64(t, "D").astype(object) for t in ds["time"].values]


def build_samples(ds: xr.Dataset, history_days: int, lead_days: int, season_months: Sequence[int]) -> list[int]:
    """Indices t of valid issue days."""
    days = _days(ds)
    seasons = [season_of(d, season_months) for d in days]
    out = []
    for t in range(history_days - 1, len(days) - lead_days):
        lo, hi = t - history_days + 1, t + lead_days
        if seasons[lo] is None or any(seasons[i] != seasons[lo] for i in (t, hi)):
            continue
        if (days[hi] - days[lo]).days != hi - lo:
            continue
        out.append(t)
    return out


def imputed_mask(ds: xr.Dataset) -> np.ndarray:
    """(T, ny, nx) bool, True where the value was gap-filled; all False if the variable is absent."""
    if "imputed_mask" in ds:
        return ds["imputed_mask"].values.astype(bool)
    return np.zeros((ds.sizes["time"], *ds["land_mask"].shape), bool)


def valid_target_mask(ds: xr.Dataset) -> np.ndarray:
    """(T, ny, nx) bool scoring mask: ocean AND NOT imputed."""
    return ~ds["land_mask"].values.astype(bool)[None] & ~imputed_mask(ds)


def issue_seasons(ds: xr.Dataset, indices: Sequence[int], season_months: Sequence[int]) -> list[int | None]:
    days = _days(ds)
    return [season_of(days[t], season_months) for t in indices]


class SequenceDataset(Dataset):
    def __init__(
        self,
        ds: xr.Dataset,
        indices: Sequence[int],
        history_days: int,
        lead_days: int,
        season_months: Sequence[int],
    ) -> None:
        self.conc = np.nan_to_num(ds["ice_concentration"].values.astype(np.float32), nan=0.0)
        self.land = ds["land_mask"].values.astype(bool)
        self.valid = valid_target_mask(ds)
        self.indices = list(indices)
        self.L, self.H = history_days, lead_days
        days = _days(ds)
        angle = np.array([2 * np.pi * day_of_season(d, season_months) / 365.25 for d in days], np.float32)
        self.sin, self.cos = np.sin(angle), np.cos(angle)
        self._land_ch = self.land.astype(np.float32)[None]

    @property
    def in_channels(self) -> int:
        return self.L + 3

    def inputs(self, t: int) -> np.ndarray:
        ones = np.ones((1, *self.land.shape), np.float32)
        return np.concatenate([
            self.conc[t - self.L + 1: t + 1], self._land_ch, ones * self.sin[t], ones * self.cos[t],
        ])

    def targets(self, t: int) -> np.ndarray:
        return self.conc[t + 1: t + self.H + 1]

    def target_mask(self, t: int) -> np.ndarray:
        """(H, ny, nx) bool: which target cells may be scored (ocean and not imputed)."""
        return self.valid[t + 1: t + self.H + 1]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        t = self.indices[i]
        x, y, m = self.inputs(t), self.targets(t), self.target_mask(t)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(np.ascontiguousarray(m))
