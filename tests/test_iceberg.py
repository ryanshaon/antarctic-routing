from datetime import datetime

import numpy as np
import pytest
from pyproj import Geod

from antarctic_routing.config import DomainSection
from antarctic_routing.iceberg.drift import (
    ForcingField,
    ResidualCorrector,
    density,
    drift_ensemble,
    drift_trajectory,
    group_split,
    haversine_m,
    presence_layers,
)
from antarctic_routing.preprocessing.grid import PolarGrid

DOMAIN = DomainSection(lat_min=-66.0, lat_max=-55.0, lon_min=-72.0, lon_max=-52.0)
GEOD = Geod(ellps="WGS84")


@pytest.fixture(scope="module")
def grid():
    return PolarGrid.from_domain(DOMAIN, resolution_km=25)


def test_haversine_hand_value():
    # one degree of latitude on a 6371 km sphere
    assert haversine_m(-60.0, -60.0, -61.0, -60.0) == pytest.approx(6371e3 * np.pi / 180, rel=1e-9)


def test_constant_current_displacement_matches_analytic_ground_distance(grid):
    """0.25 m/s for 6 h must move the berg 5.4 km over the ground (projection scale handled)."""
    current = ForcingField.constant(grid, u=0.25, v=0.0, hours=12)
    x0, y0 = grid.to_xy(-60.0, -60.0)
    traj = drift_trajectory(x0, y0, current, wind=None, alpha=0.0, hours=6, dt_hours=0.5)
    lat0, lon0 = grid.to_latlon(*traj.xy[0])
    lat1, lon1 = grid.to_latlon(*traj.xy[-1])
    ground = GEOD.inv(lon0, lat0, lon1, lat1)[2]
    assert ground == pytest.approx(5400.0, rel=2e-3)
    projected = np.hypot(*(traj.xy[-1] - traj.xy[0]))
    assert projected > ground  # EPSG:3031 stretches distances north of 71S


def test_wind_term_scales_with_alpha(grid):
    current = ForcingField.constant(grid, 0.0, 0.0, hours=24)
    wind = ForcingField.constant(grid, 10.0, 0.0, hours=24)
    x0, y0 = grid.to_xy(-60.0, -60.0)
    d1 = np.hypot(*np.diff(drift_trajectory(x0, y0, current, wind, alpha=0.01, hours=10).xy[[0, -1]], axis=0)[0])
    d2 = np.hypot(*np.diff(drift_trajectory(x0, y0, current, wind, alpha=0.02, hours=10).xy[[0, -1]], axis=0)[0])
    assert d2 == pytest.approx(2 * d1, rel=1e-3)


def test_time_varying_forcing_is_interpolated_in_time(grid):
    u = np.stack([np.full(grid.shape, 0.0), np.full(grid.shape, 0.2)])
    current = ForcingField(np.array([0.0, 10.0]), grid, u, np.zeros_like(u))
    x0, y0 = grid.to_xy(-60.0, -60.0)
    traj = drift_trajectory(x0, y0, current, None, 0.0, hours=10, dt_hours=0.1)
    ux, _ = current.sample(np.array([5.0]), np.array([x0]), np.array([y0]))
    assert ux[0] == pytest.approx(0.1)
    assert traj.xy[-1, 0] > traj.xy[0, 0]


def test_berg_leaving_the_domain_is_flagged(grid):
    current = ForcingField.constant(grid, 2.0, 0.0, hours=500)
    x0, y0 = grid.x[-2], grid.y[grid.y.size // 2]
    traj = drift_trajectory(x0, y0, current, None, 0.0, hours=400)
    assert traj.exited and np.isnan(traj.xy[-1]).all()


def test_ensemble_spread_and_presence_density(grid):
    current = ForcingField.constant(grid, 0.2, 0.0, hours=72)
    wind = ForcingField.constant(grid, 8.0, 0.0, hours=72)
    bergs = [("B1", -60.0, -60.0)]
    ens = drift_ensemble(bergs, current, wind, n_members=40, hours=72, rng=np.random.default_rng(0),
                         sigma_pos_m=2000.0, alpha_range=(0.01, 0.03), velocity_noise=0.03)
    assert ens.xy.shape == (40, 1, ens.times_h.size, 2)
    spread0 = np.nanstd(ens.xy[:, 0, 0, 0])
    spread_end = np.nanstd(ens.xy[:, 0, -1, 0])
    assert spread_end > spread0  # uncertainty grows with time
    pres = presence_layers(ens, grid, n_layers=3, layer_hours=24.0, radius_m=0.0)
    assert pres.shape == (40, 3, *grid.shape) and pres.dtype == bool
    assert pres[:, 0].reshape(40, -1).any(axis=1).all()  # every member has the berg somewhere on day 0
    p = density(pres)
    assert p.min() >= 0 and p.max() <= 1 and p[0].max() > 0


def test_presence_radius_marks_neighbouring_cells(grid):
    current = ForcingField.constant(grid, 0.0, 0.0, hours=24)
    ens = drift_ensemble([("B", -60.0, -60.0)], current, None, 1, 24, np.random.default_rng(0),
                         sigma_pos_m=0.0, alpha_range=(0.0, 0.0), velocity_noise=0.0)
    small = presence_layers(ens, grid, 1, 24.0, radius_m=0.0)[0, 0].sum()
    large = presence_layers(ens, grid, 1, 24.0, radius_m=60_000.0)[0, 0].sum()
    assert small == 1 and large > 9


def test_group_split_keeps_each_iceberg_on_one_side():
    ids = np.array(["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5)
    train, test = group_split(ids, test_fraction=0.5, seed=0)
    assert set(ids[train]).isdisjoint(set(ids[test]))
    assert train.sum() + test.sum() == ids.size and test.any() and train.any()


def _tracks(grid, n_bergs, rng, alpha_true=0.035, days=12):
    """Synthetic daily 'observed' tracks with a stronger wind response than the physics prior."""
    current = ForcingField.constant(grid, 0.1, 0.05, hours=24 * days)
    winds = []
    rows = []
    for b in range(n_bergs):
        uw, vw = rng.normal(6, 3), rng.normal(0, 4)
        wind = ForcingField.constant(grid, uw, vw, hours=24 * days)
        x0, y0 = grid.to_xy(rng.uniform(-62, -59), rng.uniform(-64, -58))
        traj = drift_trajectory(x0, y0, current, wind, alpha_true, hours=24 * days, dt_hours=1.0)
        for d in range(days + 1):
            xy = traj.xy[int(d * 24)]
            rows.append((f"B{b}", d, xy[0], xy[1], uw, vw))
        winds.append(wind)
    return current, rows


def test_learned_correction_beats_physics_on_held_out_icebergs(grid):
    rng = np.random.default_rng(0)
    current, rows = _tracks(grid, 24, rng)
    corrector, report = ResidualCorrector.fit_evaluate(rows, grid, current, alpha_physics=0.015, seed=0)
    assert report["test_icebergs"] and set(report["test_icebergs"]).isdisjoint(report["train_icebergs"])
    err = report["mean_error_km"]
    assert err["corrected"] < err["physics"] < err["no_move"]


def test_forcing_from_scenarios_uses_daily_layers(grid):
    from antarctic_routing.synthetic import ScenarioSet

    world = ScenarioSet(grid=grid, start=datetime(2026, 12, 1), time_step_hours=24.0,
                        land=np.zeros(grid.shape, bool), conc=np.zeros((2, 3, *grid.shape), np.float32),
                        current_x=np.ones((3, *grid.shape)) * 0.3, current_y=np.zeros((3, *grid.shape)),
                        execution_mode="controlled_synthetic", description="t")
    f = ForcingField.from_scenarios_current(world)
    assert f.times_h.tolist() == [0.0, 24.0, 48.0]


def test_add_iceberg_hazard_attaches_one_member_per_scenario():
    from datetime import date
    from pathlib import Path

    from antarctic_routing.config import load_config
    from antarctic_routing.iceberg.drift import add_iceberg_hazard
    from antarctic_routing.synthetic import generate_synthetic

    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    g = PolarGrid.from_domain(cfg.domain, resolution_km=25)
    world = generate_synthetic(cfg, g, date(2026, 12, 20), n_days=3, n_scenarios=12, seed=0)
    assert world.wind_x is not None and world.wind_x.shape == world.current_x.shape
    out = add_iceberg_hazard(world, [("A23A", -60.5, -62.0)], rng=np.random.default_rng(0), radius_m=10_000)
    assert out.berg.shape == (12, 3, *g.shape)
    assert out.berg[:, 0].reshape(12, -1).any(axis=1).all()
    assert "iceberg" in out.description.lower()


def test_router_avoids_an_iceberg_on_the_direct_line():
    from _worlds import A, B, _vessel, _world
    from antarctic_routing.iceberg.drift import add_iceberg_hazard
    from antarctic_routing.routing.candidates import plan_candidates

    world = _world(k=100)
    world.wind_x = np.zeros_like(world.current_x)
    world.wind_y = np.zeros_like(world.current_x)
    r, c = 5, 10
    lat, lon = world.grid.cell_latlon(r, c)
    world = add_iceberg_hazard(world, [("B1", lat, lon)], rng=np.random.default_rng(0), sigma_pos_m=0.0,
                               alpha_range=(0.0, 0.0), velocity_noise=0.0, radius_m=15_000)
    result = plan_candidates(world, _vessel(), A, B, risk_budget=0.05, risk_weights=[0, 100], n_scenario_routes=0)
    shortest = next(c for c in result.candidates if "shortest_distance" in c.labels)
    assert shortest.evaluation.p_breach == 1.0
    assert result.status == "feasible" and result.recommended.evaluation.p_breach == 0.0
    assert (r, c) not in result.recommended.route.cells


def test_spread_scaling_keeps_the_ensemble_mean_trajectory(grid):
    from antarctic_routing.iceberg.drift import scale_ensemble_spread

    current = ForcingField.constant(grid, u=0.1, v=0.05, hours=48)
    ens = drift_ensemble([("A", -60.0, -60.0), ("B", -62.0, -58.0)], current, None, 30, 48,
                         np.random.default_rng(1), beta=0.1)
    ens.xy[3, 0, 10:] = np.nan                                     # one member leaves the grid
    out = scale_ensemble_spread(ens, 0.6053)
    np.testing.assert_allclose(np.nanmean(out.xy, 0), np.nanmean(ens.xy, 0), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(np.isnan(out.xy), np.isnan(ens.xy))
    dev_in = ens.xy - np.nanmean(ens.xy, 0, keepdims=True)
    np.testing.assert_allclose(out.xy - np.nanmean(out.xy, 0, keepdims=True), 0.6053 * dev_in, atol=1e-6)
    assert scale_ensemble_spread(ens, 1.0) is ens


def test_add_iceberg_hazard_defaults_are_the_uncalibrated_model():
    from datetime import date
    from pathlib import Path

    from antarctic_routing.config import load_config
    from antarctic_routing.iceberg.drift import add_iceberg_hazard
    from antarctic_routing.synthetic import generate_synthetic

    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    g = PolarGrid.from_domain(cfg.domain, resolution_km=25)
    world = generate_synthetic(cfg, g, date(2026, 12, 20), n_days=4, n_scenarios=40, seed=0)
    bergs = [("A23A", -60.5, -62.0)]
    a = add_iceberg_hazard(world, bergs, rng=np.random.default_rng(0), radius_m=10_000)
    b = add_iceberg_hazard(world, bergs, rng=np.random.default_rng(0), radius_m=10_000, beta=1.0,
                           alpha_range=(0.01, 0.03), spread_factor=1.0)
    c = add_iceberg_hazard(world, bergs, rng=np.random.default_rng(0), radius_m=10_000, beta=0.1,
                           alpha_range=(0.001, 0.003), spread_factor=0.6053)
    np.testing.assert_array_equal(a.berg, b.berg)
    assert c.berg[:, -1].sum() < a.berg[:, -1].sum()               # damped, tighter cloud covers fewer cells
