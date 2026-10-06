from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch

from antarctic_routing.config import load_config
from antarctic_routing.forecasting.dataset import SequenceDataset, build_samples
from antarctic_routing.forecasting.evaluate import evaluate_forecasts
from antarctic_routing.forecasting.train import TrainConfig, load_model, masked_mae_loss, train_unet
from antarctic_routing.forecasting.unet import IceUNet
from antarctic_routing.preprocessing.climatology import season_of
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.synthetic import synthetic_history

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
SEASON = CFG.project.season_months


@pytest.fixture(scope="module")
def grid():
    return PolarGrid.from_domain(CFG.domain, resolution_km=50)


@pytest.fixture(scope="module")
def history(grid):
    return synthetic_history(CFG, grid, seasons=range(2010, 2016), seed=3)


# ------------------------------------------------------------ synthetic history
def test_history_covers_only_season_days_and_is_labelled(history):
    days = [np.datetime64(t, "D").astype(object) for t in history["time"].values]
    assert {season_of(d, SEASON) for d in days} == set(range(2010, 2016))
    assert history.attrs["execution_mode"] == "controlled_synthetic"
    conc = history["ice_concentration"].values[:, ~history["land_mask"].values]
    assert np.isfinite(conc).all() and conc.min() >= 0 and conc.max() <= 1


def test_history_is_deterministic(grid):
    a = synthetic_history(CFG, grid, seasons=[2010], seed=1)["ice_concentration"].values
    b = synthetic_history(CFG, grid, seasons=[2010], seed=1)["ice_concentration"].values
    assert np.array_equal(a, b, equal_nan=True)


def test_history_has_dynamics_persistence_error_grows_with_lead(history):
    c = history["ice_concentration"].values[:100]
    ocean = ~history["land_mask"].values
    err = [np.mean(np.abs(c[h:][:, ocean] - c[:-h][:, ocean])) for h in (1, 7)]
    assert err[1] > 2 * err[0]


# ------------------------------------------------------------------- samples
def test_samples_never_cross_season_boundaries(history):
    idx = build_samples(history, history_days=5, lead_days=3, season_months=SEASON)
    times = history["time"].values
    for t in idx:
        window = [np.datetime64(x, "D").astype(object) for x in times[t - 4: t + 4]]
        assert len({season_of(d, SEASON) for d in window}) == 1
        assert (np.diff(times[t - 4: t + 4]) == np.timedelta64(1, "D")).all()
    per_season = 120 - 5 - 3 + 1
    assert len(idx) == 6 * per_season


def test_sequence_dataset_aligns_inputs_and_targets(history):
    idx = build_samples(history, 5, 3, SEASON)
    ds = SequenceDataset(history, idx, history_days=5, lead_days=3, season_months=SEASON)
    x, y, mask = ds[0]
    t = idx[0]
    conc = np.nan_to_num(history["ice_concentration"].values, nan=0.0)
    assert x.shape[0] == 5 + 3  # 5 history frames + land + sin + cos
    assert np.allclose(x[4].numpy(), conc[t])          # last input frame is "today"
    assert np.allclose(y[2].numpy(), conc[t + 3])      # target channel h-1 is t + h
    land = history["land_mask"].values
    assert mask.dtype == torch.bool and mask.shape == (3, *land.shape)  # one mask per lead
    assert not mask[:, land].any() and mask[:, ~land].all()  # imputed_mask is all False here


def _with_imputed(history, day_index, cells):
    """Copy of ``history`` with ``cells`` (bool grid) flagged imputed on one day."""
    h = history.copy(deep=True)
    h["imputed_mask"].values[day_index] = cells
    return h


def _ocean_block(history):
    cells = np.zeros(history["land_mask"].shape, bool)
    cells[5:9, 5:9] = True
    cells &= ~history["land_mask"].values
    assert cells.any()
    return cells


def test_target_mask_is_ocean_and_not_imputed_per_lead(history):
    idx = build_samples(history, 5, 3, SEASON)
    t = idx[0]
    cells = _ocean_block(history)
    h = _with_imputed(history, t + 2, cells)  # imputed on lead 2 only
    mask = SequenceDataset(h, idx, 5, 3, SEASON)[0][2].numpy()
    land = history["land_mask"].values
    assert not mask[1][cells].any()                       # lead 2: imputed cells excluded
    assert mask[0][cells].all() and mask[2][cells].all()  # other leads unaffected
    assert not mask[:, land].any()                        # land still excluded
    x_plain = SequenceDataset(history, idx, 5, 3, SEASON)[0][0]
    assert torch.equal(SequenceDataset(h, idx, 5, 3, SEASON)[0][0], x_plain)  # inputs/values untouched


def test_dataset_without_imputed_mask_variable_scores_all_ocean(history):
    h = history.drop_vars("imputed_mask")
    idx = build_samples(h, 5, 3, SEASON)
    mask = SequenceDataset(h, idx, 5, 3, SEASON)[0][2].numpy()
    assert (mask == ~h["land_mask"].values[None]).all()


# --------------------------------------------------------------------- model
@pytest.mark.parametrize("shape", [(23, 31), (32, 32), (17, 9)])
def test_unet_preserves_spatial_shape_and_outputs_fractions(shape):
    torch.manual_seed(0)
    model = IceUNet(in_channels=8, out_channels=3, base=8)
    out = model(torch.rand(2, 8, *shape))
    assert out.shape == (2, 3, *shape)
    assert out.min() >= 0 and out.max() <= 1


def test_masked_loss_ignores_land():
    pred = torch.zeros(1, 1, 2, 2)
    target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    mask = torch.tensor([[False, True], [True, True]])
    assert masked_mae_loss(pred, target, mask).item() == pytest.approx(0.0)


def test_masked_loss_ignores_imputed_cells_per_lead():
    pred = torch.zeros(1, 2, 2, 2)
    target = torch.zeros(1, 2, 2, 2)
    target[0, 1, 0, 0] = 1.0                       # wrong only on lead 2, cell (0, 0)
    mask = torch.ones(1, 2, 2, 2, dtype=torch.bool)
    assert masked_mae_loss(pred, target, mask).item() == pytest.approx(1 / 8)
    mask[0, 1, 0, 0] = False                       # ... which is imputed on that day
    assert masked_mae_loss(pred, target, mask).item() == pytest.approx(0.0)


# ---------------------------------------------------------- train & evaluate
def test_training_reduces_validation_error_and_checkpoint_roundtrips(history, tmp_path):
    tc = TrainConfig(history_days=5, lead_days=3, epochs=4, batch_size=16, base_channels=8, lr=3e-3, seed=0)
    result = train_unet(history, train_seasons=[2010, 2011, 2012, 2013], val_seasons=[2014],
                        season_months=SEASON, cfg=tc, out_dir=tmp_path)
    assert result.history[-1]["val_mae"] < result.history[0]["val_mae"]
    model, meta = load_model(tmp_path / "best.pt")
    assert meta["train_seasons"] == [2010, 2011, 2012, 2013]
    assert meta["execution_mode"] == "controlled_synthetic"
    x = torch.rand(1, 8, *history["land_mask"].shape)
    with torch.no_grad():
        assert torch.allclose(model(x), result.model.eval()(x))


def test_evaluation_reports_every_method_per_lead_and_oracle_is_perfect(history):
    conc = np.nan_to_num(history["ice_concentration"].values, nan=0.0)

    def oracle(batch_inputs, sample_index):
        return np.stack([conc[t + 1: t + 4] for t in sample_index])

    report = evaluate_forecasts(history, oracle, train_seasons=[2010, 2011, 2012, 2013], test_seasons=[2015],
                                history_days=5, lead_days=3, season_months=SEASON, model_name="oracle")
    assert set(report["methods"]) == {"oracle", "persistence", "climatology", "damped_persistence"}
    for method in report["methods"]:
        assert len(report["mae"][method]) == 3
    assert max(report["mae"]["oracle"]) < 1e-6
    assert report["mae"]["persistence"][0] < report["mae"]["persistence"][2]
    assert 0.0 <= report["rho"] <= 1.0
    assert report["iiee_km2"]["oracle"][0] == 0.0
    assert report["n_samples"] == 120 - 5 - 3 + 1
    assert report["test_seasons"] == [2015]


def test_evaluation_does_not_score_imputed_cells(history):
    conc = np.nan_to_num(history["ice_concentration"].values, nan=0.0)
    cells = _ocean_block(history)
    test_days = np.where(history["time"].values >= np.datetime64("2015-11-01"))[0]
    h = history.copy(deep=True)
    for d in test_days:
        h["imputed_mask"].values[d, cells] = True

    def wrong_at_imputed(batch_inputs, sample_index):
        pred = np.stack([conc[t + 1: t + 4] for t in sample_index])
        pred[:, :, cells] = 1.0 - pred[:, :, cells]  # badly wrong, but only where data were filled
        return pred

    kwargs = dict(train_seasons=[2010, 2011, 2012, 2013], test_seasons=[2015], history_days=5, lead_days=3,
                  season_months=SEASON, model_name="m")
    assert max(evaluate_forecasts(h, wrong_at_imputed, **kwargs)["mae"]["m"]) < 1e-6
    assert min(evaluate_forecasts(history, wrong_at_imputed, **kwargs)["mae"]["m"]) > 1e-3


def test_evaluation_rejects_overlapping_train_and_test_seasons(history):
    with pytest.raises(ValueError, match="overlap"):
        evaluate_forecasts(history, lambda b, i: None, [2010, 2015], [2015], 5, 3, SEASON)


def test_issue_date_helper():
    assert season_of(date(2027, 1, 5), SEASON) == 2026


def test_residual_unet_with_zero_head_reproduces_persistence_exactly():
    torch.manual_seed(0)
    model = IceUNet(in_channels=8, out_channels=3, base=8, persistence_channel=4).eval()
    x = torch.rand(2, 8, 19, 23)
    with torch.no_grad():
        out = model(x)
    assert torch.allclose(out, x[:, 4:5].expand_as(out))   # zero-initialised head => C_t for every lead


def test_residual_unet_output_stays_in_unit_interval():
    torch.manual_seed(1)
    model = IceUNet(in_channels=8, out_channels=3, base=8, persistence_channel=4)
    with torch.no_grad():
        model.head.weight.normal_(0, 5.0)
        model.head.bias.normal_(0, 5.0)
    out = model(torch.rand(2, 8, 16, 16))
    assert out.min() >= 0 and out.max() <= 1


# ------------------------------------------------- checkpoint provenance / grid
@pytest.fixture(scope="module")
def trained_on_file(history, tmp_path_factory):
    out = tmp_path_factory.mktemp("prov")
    path = out / "history.nc"
    history.to_netcdf(path)
    tc = TrainConfig(history_days=5, lead_days=3, epochs=1, batch_size=16, base_channels=8, seed=0)
    train_unet(history, [2010, 2011, 2012, 2013], [2014], SEASON, tc, out / "model", dataset_path=path)
    return out / "model" / "best.pt", path


def test_checkpoint_records_dataset_and_grid_provenance(history, trained_on_file):
    from antarctic_routing.common.provenance import sha256_file

    ckpt, path = trained_on_file
    _, meta = load_model(ckpt)
    assert meta["dataset"]["path"] == str(path) and meta["dataset"]["sha256"] == sha256_file(path)
    assert meta["dataset"]["execution_mode"] == "controlled_synthetic"
    assert meta["grid_shape"] == list(history["land_mask"].shape)
    assert meta["resolution_m"] == 50_000.0
    assert meta["grid_origin"] == [float(history["x"].values[0]), float(history["y"].values[0])]


def test_load_model_accepts_the_training_grid_and_rejects_another_resolution(history, trained_on_file):
    ckpt, _ = trained_on_file
    load_model(ckpt, history)  # same grid: fine
    other = synthetic_history(CFG, PolarGrid.from_domain(CFG.domain, resolution_km=25), seasons=range(2010, 2012))
    with pytest.raises(ValueError, match="resolution 25 km != trained 50 km"):
        load_model(ckpt, other)


def test_load_model_rejects_same_shape_on_a_shifted_grid(history, trained_on_file):
    ckpt, _ = trained_on_file
    shifted = history.assign_coords(x=history["x"].values + 200_000.0)
    with pytest.raises(ValueError, match="grid origin"):
        load_model(ckpt, shifted)


def test_predictor_rejects_inputs_on_another_grid(trained_on_file):
    from antarctic_routing.forecasting.train import unet_predictor

    model, meta = load_model(trained_on_file[0])
    predict = unet_predictor(model)
    assert predict(np.zeros((1, meta["in_channels"], *meta["grid_shape"]), np.float32), [0]).shape[1] == 3
    with pytest.raises(ValueError, match="training grid"):
        predict(np.zeros((1, meta["in_channels"], 40, 40), np.float32), [0])


def test_checkpoint_without_grid_provenance_is_rejected(history, trained_on_file, tmp_path):
    blob = torch.load(trained_on_file[0], weights_only=True)
    for key in ("resolution_m", "grid_origin"):
        blob["meta"].pop(key)
    legacy = tmp_path / "legacy.pt"
    torch.save(blob, legacy)
    load_model(legacy)  # loading without a dataset still works
    with pytest.raises(ValueError, match="no grid provenance"):
        load_model(legacy, history)


# ------------------------------------------------- baseline fitting on observed cells
def _train_days(history, seasons=(2010, 2011, 2012, 2013)):
    days = [np.datetime64(t, "D").astype(object) for t in history["time"].values]
    return [i for i, d in enumerate(days) if season_of(d, SEASON) in seasons]


def test_fit_baselines_ignores_values_at_imputed_cells(history):
    from antarctic_routing.forecasting.evaluate import fit_baselines

    cells = _ocean_block(history)
    flagged = _train_days(history)[::2]  # every other training day, so the cells keep some observed data

    def variant(fill, flag):
        h = history.copy(deep=True)
        for d in flagged:
            h["ice_concentration"].values[d, cells] = fill
            h["imputed_mask"].values[d, cells] = flag
        return fit_baselines(h, [2010, 2011, 2012, 2013], SEASON)

    (clim0, rho0), (clim1, rho1) = variant(0.0, True), variant(1.0, True)
    assert np.array_equal(clim0.mean, clim1.mean, equal_nan=True) and rho0 == rho1  # imputed values never used
    (clim_u, rho_u) = variant(1.0, False)  # control: the same values, treated as observed, do change the fit
    assert not np.array_equal(clim0.mean, clim_u.mean, equal_nan=True) and rho0 != rho_u


def test_fit_baselines_unchanged_when_nothing_is_imputed(history):
    from antarctic_routing.forecasting.baselines import fit_anomaly_decay
    from antarctic_routing.forecasting.evaluate import fit_baselines
    from antarctic_routing.preprocessing.climatology import Climatology

    assert not history["imputed_mask"].values.any()
    train = [2010, 2011, 2012, 2013]
    clim, rho = fit_baselines(history, train, SEASON)
    days = [np.datetime64(t, "D").astype(object) for t in history["time"].values]
    conc = history["ice_concentration"].values.astype(float)
    ref = Climatology.fit(conc, days, train, SEASON, window_days=7)  # previous behaviour: raw values
    segments = [np.stack([ref.anomaly(conc[i], days[i]) for i in _train_days(history, (s,))]) for s in train]
    assert np.array_equal(clim.mean, ref.mean, equal_nan=True)
    assert rho == pytest.approx(fit_anomaly_decay(segments), abs=1e-12)
