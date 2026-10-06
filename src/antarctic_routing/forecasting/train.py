"""Train the sea-ice U-Net with a masked MAE loss and early stopping.

    L_MAE = sum_i M_i |C_hat_i - C_i| / sum_i M_i      (M_i = 1 on observed ocean cells)

M is per target day and cell: ocean AND NOT imputed (see ``dataset.valid_target_mask``).

The best epoch (lowest validation MAE) is checkpointed together with the
metadata needed to reproduce and audit it: seasons used, window lengths, seed,
the execution mode of the training data and the training dataset/grid
(identifier, SHA-256, grid shape, resolution, origin, source product).

Grid safety: the U-Net accepts any grid size, so a checkpoint is only used on a
dataset with the same grid shape, resolution and origin it was trained on
(:func:`check_grid`); anything else is rejected instead of silently run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from antarctic_routing.common.provenance import sha256_file, utc_now
from antarctic_routing.forecasting.dataset import SequenceDataset, build_samples, issue_seasons
from antarctic_routing.forecasting.unet import IceUNet


@dataclass
class TrainConfig:
    history_days: int = 14
    lead_days: int = 21  # same as config forecast.lead_days; the CLI passes the config value
    epochs: int = 30
    batch_size: int = 16
    base_channels: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 6
    seed: int = 0
    residual: bool = True


@dataclass
class TrainResult:
    model: IceUNet
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    checkpoint: Path | None = None


def masked_mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over cells where ``mask`` is True.

    ``mask`` may be (H, W), (B, H, W), (B, 1, H, W) or per-lead (B, lead, H, W);
    it is broadcast over channels where needed.
    """
    m = mask.to(pred.dtype)
    if m.dim() == 2:
        m = m[None, None]
    elif m.dim() == 3:
        m = m[:, None]
    m = m.expand_as(pred)
    return (torch.abs(pred - target) * m).sum() / m.sum().clamp_min(1.0)


def grid_signature(ds: xr.Dataset) -> dict:
    """What identifies the grid a model was trained on."""
    return {
        "grid_shape": [int(n) for n in ds["land_mask"].shape],
        "resolution_m": float(ds.attrs["resolution_m"]),
        "x0": float(ds["x"].values[0]),
        "y0": float(ds["y"].values[0]),
        "crs": str(ds.attrs.get("crs", "")),
    }


def check_grid(meta: dict, ds: xr.Dataset) -> None:
    """Raise ``ValueError`` unless ``ds`` is on the grid the checkpoint was trained on."""
    if "resolution_m" not in meta or "grid_origin" not in meta:
        raise ValueError("checkpoint has no grid provenance (resolution/origin); retrain it with this version")
    sig = grid_signature(ds)
    problems = []
    if list(meta["grid_shape"]) != sig["grid_shape"]:
        problems.append(f"grid shape {sig['grid_shape']} != trained {list(meta['grid_shape'])}")
    if abs(float(meta["resolution_m"]) - sig["resolution_m"]) > 1e-6:
        trained_km, given_km = float(meta["resolution_m"]) / 1000, sig["resolution_m"] / 1000
        problems.append(f"resolution {given_km:g} km != trained {trained_km:g} km")
    x0, y0 = meta["grid_origin"]
    if max(abs(x0 - sig["x0"]), abs(y0 - sig["y0"])) > 0.5 * sig["resolution_m"]:
        problems.append(f"grid origin ({sig['x0']:.0f}, {sig['y0']:.0f}) != trained ({x0:.0f}, {y0:.0f})")
    if problems:
        raise ValueError("checkpoint does not match this dataset's grid: " + "; ".join(problems))


def _subset(ds: xr.Dataset, cfg: TrainConfig, season_months, seasons: Sequence[int]) -> SequenceDataset:
    idx = build_samples(ds, cfg.history_days, cfg.lead_days, season_months)
    wanted = set(seasons)
    idx = [t for t, s in zip(idx, issue_seasons(ds, idx, season_months), strict=True) if s in wanted]
    if not idx:
        raise ValueError(f"no samples for seasons {sorted(wanted)}")
    return SequenceDataset(ds, idx, cfg.history_days, cfg.lead_days, season_months)


def _evaluate(model: IceUNet, loader: DataLoader) -> float:
    model.eval()
    total, weight = 0.0, 0.0
    with torch.no_grad():
        for x, y, mask in loader:
            m = mask.expand_as(y).float()
            total += float((torch.abs(model(x) - y) * m).sum())
            weight += float(m.sum())
    return total / max(weight, 1.0)


def train_unet(
    ds: xr.Dataset,
    train_seasons: Sequence[int],
    val_seasons: Sequence[int],
    season_months: Sequence[int],
    cfg: TrainConfig,
    out_dir: str | Path,
    dataset_path: str | Path | None = None,
    dataset_id: str | None = None,
) -> TrainResult:
    """Train and checkpoint. ``dataset_path`` (fingerprinted with SHA-256) or ``dataset_id`` names the data."""
    if set(train_seasons) & set(val_seasons):
        raise ValueError("train and validation seasons overlap")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    train = _subset(ds, cfg, season_months, train_seasons)
    val = _subset(ds, cfg, season_months, val_seasons)
    gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val, batch_size=cfg.batch_size)

    model = IceUNet(train.in_channels, cfg.lead_days, base=cfg.base_channels,
                    persistence_channel=cfg.history_days - 1 if cfg.residual else None)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = TrainResult(model=model)
    best, best_state, stale = float("inf"), None, 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, batches = 0.0, 0
        for x, y, mask in train_loader:
            opt.zero_grad()
            loss = masked_mae_loss(model(x), y, mask)
            loss.backward()
            opt.step()
            running += loss.item()
            batches += 1
        sched.step()
        val_mae = _evaluate(model, val_loader)
        result.history.append({"epoch": epoch, "train_loss": running / max(batches, 1), "val_mae": val_mae})
        if val_mae < best - 1e-6:
            best, stale, result.best_epoch = val_mae, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    sig = grid_signature(ds)
    dataset = {
        "id": str(dataset_path) if dataset_path is not None else (dataset_id or "in-memory dataset"),
        "path": str(dataset_path) if dataset_path is not None else None,
        "sha256": sha256_file(Path(dataset_path)) if dataset_path is not None else None,
        "execution_mode": ds.attrs.get("execution_mode", "real"),
        "crs": sig["crs"],
        **{k: ds.attrs[k] for k in ("source_product", "source_product_family") if k in ds.attrs},
    }
    meta = {
        "in_channels": train.in_channels, "out_channels": cfg.lead_days, "base_channels": cfg.base_channels,
        "persistence_channel": model.persistence_channel,
        "train_config": asdict(cfg), "train_seasons": sorted(train_seasons), "val_seasons": sorted(val_seasons),
        "season_months": list(season_months), "best_epoch": result.best_epoch, "val_mae": best,
        "execution_mode": ds.attrs.get("execution_mode", "real"), "grid_shape": sig["grid_shape"],
        "resolution_m": sig["resolution_m"], "grid_origin": [sig["x0"], sig["y0"]], "dataset": dataset,
        "created_at": utc_now(),
    }
    result.checkpoint = out / "best.pt"
    torch.save({"state_dict": model.state_dict(), "meta": meta}, result.checkpoint)
    (out / "history.json").write_text(json.dumps({"history": result.history, "meta": meta}, indent=2))
    return result


def load_model(path: str | Path, ds: xr.Dataset | None = None) -> tuple[IceUNet, dict]:
    """Load a checkpoint; with ``ds``, refuse it unless ``ds`` is on the training grid."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    meta = blob["meta"]
    if ds is not None:
        check_grid(meta, ds)
    model = IceUNet(meta["in_channels"], meta["out_channels"], base=meta["base_channels"],
                    persistence_channel=meta.get("persistence_channel"))
    model.load_state_dict(blob["state_dict"])
    model.grid_shape = tuple(meta["grid_shape"])
    return model.eval(), meta


def unet_predictor(model: IceUNet, batch_size: int = 32):
    """Adapter giving a U-Net the ``predictor(inputs, indices)`` interface used by evaluation."""

    expected = getattr(model, "grid_shape", None)

    def predict(inputs: np.ndarray, indices: Sequence[int]) -> np.ndarray:
        if expected is not None and tuple(np.shape(inputs)[-2:]) != tuple(expected):
            raise ValueError(f"input grid {tuple(np.shape(inputs)[-2:])} != model's training grid {tuple(expected)}")
        outs = []
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                outs.append(model(torch.from_numpy(np.asarray(inputs[i: i + batch_size]))).numpy())
        return np.concatenate(outs)

    return predict
