"""Read-only real-data bundles for the API (publish.py)."""

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from antarctic_routing.config import load_config
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.publish import (
    DATA_STATUSES,
    BundleError,
    canonical_sha256,
    load_bundle,
    map_layers,
    source_statuses,
    strip_paths,
    to_percent,
    write_bundle,
    write_compact_json,
)
from antarctic_routing.synthetic import generate_synthetic

CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def small_world():
    """Synthetic world whose layers are labelled like a forecast issue (layer 0 observed)."""
    cfg = load_config(CONFIG)
    w = generate_synthetic(cfg, PolarGrid.from_domain(cfg.domain, 50.0), date(2026, 12, 20), 4, 20, seed=1)
    return replace(w, layer_source=["observed"] + ["forecast"] * (w.n_times - 1))


@pytest.fixture(scope="module")
def world():
    return small_world()


def plan_payload(mode="controlled_synthetic", forcing_status="schematic"):
    """A minimal plan_window-shaped payload for bundle tests (fixture data, not a result)."""
    return {
        "issue": "2026-12-20", "execution_mode": mode, "selected": "2026-12-20", "n_members": 20,
        "layer_source": ["observed", "forecast", "forecast"], "explanation": "fixture",
        "forcing": {"wind_forcing": "/abs/dir/winds.nc", "current_forcing": "schematic (no --current-forcing given)"},
        "forcing_provenance": {"winds": "ERA5 daily", "currents": "schematic", "status": forcing_status,
                               "wind_dates": ["2026-12-20", "2026-12-22"],
                               "sources": {"winds": {"forcing_file": "/abs/dir/winds.nc"}}},
        "options": [{"departure": "2026-12-20", "p_breach_upper": 0.02, "feasible": True}],
        "selected_route": {"departure": "2026-12-20", "cells_row_col": [[1, 2], [2, 3]], "evaluation": {}},
        "icebergs": [], "iceberg_source": None,
    }


def make_bundle(tmp_path, world, mode="controlled_synthetic", forcing_status="schematic"):
    src = tmp_path / "run"
    src.mkdir()
    (src / "plan_window.json").write_text(json.dumps(plan_payload(mode, forcing_status)))
    write_compact_json(src / "forecast_maps.json", map_layers(world, 0.15))
    write_bundle(tmp_path / "bundle", src / "plan_window.json", src / "forecast_maps.json",
                 {"inputs": {"sea_ice": {"path": "/data/sea_ice.nc", "sha256": "ab"}}}, limitations=["fixture"])
    return tmp_path / "bundle"


def test_missing_values_are_none_never_zero():
    assert to_percent(np.array([0.0, 0.155, np.nan, 1.0])) == [0, 16, None, 100]


def test_map_layers_label_every_layer_and_hide_land(world):
    maps = map_layers(world, 0.15)
    assert maps["execution_mode"] == "controlled_synthetic" and len(maps["layers"]) == world.n_times
    assert [la["source"] for la in maps["layers"]] == world.layer_source
    assert maps["layers"][1]["date"] == "2026-12-21" and maps["observed"]["date"] == "2026-12-20"
    land = np.array(maps["land"], bool)
    p = maps["layers"][2]["p_ice_ge_limit_pct"]
    assert land.any() and all(p[i] is None for i in np.flatnonzero(land))
    assert all(v is not None and 0 <= v <= 100 for i, v in enumerate(p) if not land[i])
    assert maps["layers"][0]["p_berg_pct"] is None
    assert len(maps["grid"]["x_km"]) == maps["grid"]["nx"] and len(maps["grid"]["y_km"]) == maps["grid"]["ny"]


def test_canonical_checksum_ignores_where_the_files_were():
    a = plan_payload()
    b = json.loads(json.dumps(a).replace("/abs/dir/", "/elsewhere/deeper/"))
    assert a != b and canonical_sha256(a) == canonical_sha256(b)
    assert strip_paths({"p": "/x/y/z.nc", "s": "schematic (no --forcing given)"}) == {
        "p": "z.nc", "s": "schematic (no --forcing given)"}
    c = plan_payload()
    c["selected"] = "2026-12-21"
    assert canonical_sha256(c) != canonical_sha256(a)


def test_bundle_round_trip_strips_paths_and_verifies(tmp_path, world):
    b = load_bundle(make_bundle(tmp_path, world))
    assert b.plan_window["forcing"]["wind_forcing"] == "winds.nc"
    assert b.manifest["inputs"]["sea_ice"]["path"] == "sea_ice.nc"
    assert b.index["limitations"] == ["fixture"] and b.figure is None
    assert "/abs/" not in json.dumps(b.plan_window) + json.dumps(b.manifest)


def test_synthetic_bundle_is_never_labelled_real(tmp_path, world):
    statuses = source_statuses(load_bundle(make_bundle(tmp_path, world)))
    assert all(s["status"] in DATA_STATUSES for s in statuses)
    assert not [s for s in statuses if s["status"] in ("real", "historical")]


def test_real_bundle_labels_schematic_currents_as_schematic(tmp_path, world):
    by_name = {s["name"]: s for s in source_statuses(load_bundle(make_bundle(tmp_path, world, "real", "mixed")))}
    assert by_name["winds"]["status"] == "historical" and by_name["currents"]["status"] == "schematic"
    assert by_name["sea_ice_forecast"]["status"] == "forecast"
    assert by_name["icebergs"]["status"] == "unavailable"


@pytest.mark.parametrize("damage, reason", [
    (lambda d: (d / "forecast_maps.json").write_text("{}"), "checksum mismatch for forecast_maps.json"),
    (lambda d: (d / "manifest.json").unlink(), "manifest.json is missing"),
    (lambda d: (d / "bundle.json").unlink(), "no bundle.json"),
    (lambda d: (d / "bundle.json").write_text("{"), "not readable JSON"),
])
def test_damaged_bundles_are_rejected_with_the_reason(tmp_path, world, damage, reason):
    root = make_bundle(tmp_path, world)
    damage(root)
    with pytest.raises(BundleError, match=reason):
        load_bundle(root)


def test_bundle_index_cannot_smuggle_other_files(tmp_path, world):
    root = make_bundle(tmp_path, world)
    idx = json.loads((root / "bundle.json").read_text())
    idx["files"]["../secret.txt"] = {"sha256": "0"}
    (root / "bundle.json").write_text(json.dumps(idx))
    with pytest.raises(BundleError, match="unexpected file"):
        load_bundle(root)


def test_edited_plan_that_keeps_file_checksums_consistent_is_still_rejected(tmp_path, world):
    from antarctic_routing.common.provenance import sha256_file

    root = make_bundle(tmp_path, world)
    plan = json.loads((root / "plan_window.json").read_text())
    plan["selected"] = "2026-12-24"
    (root / "plan_window.json").write_text(json.dumps(plan))
    idx = json.loads((root / "bundle.json").read_text())
    idx["files"]["plan_window.json"]["sha256"] = sha256_file(root / "plan_window.json")
    (root / "bundle.json").write_text(json.dumps(idx))
    with pytest.raises(BundleError, match="canonical checksum"):
        load_bundle(root)


def test_iceberg_tracks_reproduce_the_planner_ensemble(world):
    from antarctic_routing.iceberg.drift import add_iceberg_hazard
    from antarctic_routing.publish import iceberg_tracks

    bergs = [("B1", -60.2, -62.6), ("B2", -61.0, -60.0)]
    kw = {"beta": 0.1, "alpha_range": (0.001, 0.003), "spread_factor": 0.6053}
    hazard = add_iceberg_hazard(world, bergs, rng=np.random.default_rng(7), radius_m=10_000.0, **kw)
    tracks = iceberg_tracks(hazard, bergs, 7, 10_000.0, **kw)
    assert [t["id"] for t in tracks] == ["B1", "B2"] and len(tracks[0]["daily_mean"]) == world.n_times
    first = tracks[0]["daily_mean"][0]
    assert abs(first["lat"] - -60.2) < 0.1 and abs(first["lon"] - -62.6) < 0.2 and first["on_grid_fraction"] == 1.0
    for other in (replace(hazard, berg=np.zeros_like(hazard.berg)), replace(hazard, berg=None)):
        with pytest.raises(ValueError, match="does not match"):     # not the hazard the planner used
            iceberg_tracks(other, bergs, 7, 10_000.0, **kw)
