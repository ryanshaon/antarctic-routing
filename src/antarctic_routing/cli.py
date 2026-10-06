"""Command-line entry point: ``antroute`` (or ``python -m antarctic_routing.cli``).

Commands
--------
validate-config   Validate config.yaml and print the scenario summary.
demo              Plan risk-budgeted routes for one departure (controlled-synthetic data).
departures        Sweep a departure window and select a date under the risk budget.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from antarctic_routing import DISCLAIMER, __version__
from antarctic_routing.common.provenance import (
    StageResult,
    file_record,
    sha256_file,
    utc_now,
    write_json_artifact,
)
from antarctic_routing.config import ProjectConfig, load_config, min_scenarios_for_budget
from antarctic_routing.export import route_to_csv, route_to_geojson
from antarctic_routing.locations import route_horizon_days
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.routing.candidates import plan_candidates
from antarctic_routing.routing.departure import sweep_departures
from antarctic_routing.routing.fuel import VesselModel
from antarctic_routing.synthetic import generate_synthetic
from antarctic_routing.viz import plot_departures, plot_plan

DEFAULT_CONFIG = "config/config.yaml"
SOFTWARE = {"name": "antarctic-routing", "version": __version__}


def _horizon_days(cfg: ProjectConfig) -> int:
    """Horizon of the configured route (the API resolves other routes with :mod:`antarctic_routing.locations`)."""
    o, d = cfg.route.origin, cfg.route.destination
    return route_horizon_days(o.lat, o.lon, d.lat, d.lon, cfg.vessel.cruise_speed_kmh, cfg.grid.time_step_hours,
                              cfg.forecast.lead_days)


def _setup(args):
    cfg = load_config(args.config)
    grid = PolarGrid.from_domain(cfg.domain, args.resolution_km or cfg.grid.resolution_km)
    n_scen = args.scenarios or cfg.forecast.route_scenarios
    o = grid.cell_of(cfg.route.origin.lat, cfg.route.origin.lon)
    d = grid.cell_of(cfg.route.destination.lat, cfg.route.destination.lon)
    return cfg, grid, n_scen, o, d


def _parse_bergs(args) -> list[tuple[str, float, float]]:
    """--iceberg ID:LAT:LON entries plus the latest position per berg from a USNIC list."""
    bergs: dict[str, tuple[str, float, float]] = {}
    if getattr(args, "icebergs", None):
        from antarctic_routing.ingestion.icebergs import read_iceberg_positions

        for row in sorted(read_iceberg_positions(args.icebergs), key=lambda r: r["date"]):
            bergs[row["iceberg_id"]] = (row["iceberg_id"], row["lat"], row["lon"])
    for spec in getattr(args, "iceberg", None) or []:
        bid, lat, lon = spec.split(":")
        bergs[bid.upper()] = (bid.upper(), float(lat), float(lon))
    return list(bergs.values())


def _bergs_in_grid(bergs, grid) -> tuple[list, list]:
    """``(inside, outside)`` the polar routing grid, so only bergs on the grid enter drift."""
    from antarctic_routing.ingestion.icebergs import in_grid_mask

    if not bergs:
        return [], []
    mask = in_grid_mask([b[1] for b in bergs], [b[2] for b in bergs], grid)
    return [b for b, m in zip(bergs, mask, strict=True) if m], [b for b, m in zip(bergs, mask, strict=True) if not m]


def _iceberg_source(args) -> dict | None:
    """Provenance and quality flags of the whole USNIC file, including bergs outside the grid."""
    if not getattr(args, "icebergs", None):
        return None
    from antarctic_routing.ingestion.icebergs import iceberg_provenance, read_iceberg_positions

    return iceberg_provenance(args.icebergs, read_iceberg_positions(args.icebergs))


def _plan_dict(plan, world, bergs=()) -> dict:
    return {
        "icebergs": [{"id": b[0], "lat": b[1], "lon": b[2]} for b in bergs],
        "status": plan.status,
        "explanation": plan.explanation,
        "risk_budget": plan.risk_budget,
        "estimator": plan.estimator,
        "execution_mode": world.execution_mode,
        "data_description": world.description,
        "departure_utc": world.start.isoformat() + "Z",
        "n_scenarios": world.n_scenarios,
        "recommended": plan.recommended.labels if plan.recommended else None,
        "candidates": [
            {"labels": c.labels, "tags": c.tags, "feasible": c.feasible,
             "risk_statistic": c.risk_statistic, **c.evaluation.summary()}
            for c in plan.candidates
        ],
        "warnings": plan.warnings,
        "disclaimer": DISCLAIMER,
    }


def cmd_validate(args) -> int:
    cfg = load_config(args.config)
    v, r = cfg.vessel, cfg.routing
    print(f"Config OK: {args.config}")
    print(f"  Region      : {cfg.project.region} ({cfg.domain.crs}), season months {cfg.project.season_months}")
    print(f"  Route       : ({cfg.route.origin.lat}, {cfg.route.origin.lon}) -> "
          f"({cfg.route.destination.lat}, {cfg.route.destination.lon})")
    print(f"  Departures  : {cfg.route.departure_start} .. {cfg.route.departure_end}")
    print(f"  Vessel      : {v.name}, {v.cruise_speed_knots} kn ({v.cruise_speed_kmh:.1f} km/h), "
          f"ice limit tau_v={v.max_ice_concentration}")
    print(f"  Risk budget : {r.risk_budget:.1%} via {r.risk_estimator} at {r.confidence:.0%} confidence; "
          f"needs >= {min_scenarios_for_budget(r.risk_budget, r.confidence)} scenarios "
          f"(configured {cfg.forecast.route_scenarios})")
    if "REPLACE" in v.ice_class:
        print("  WARNING     : vessel.ice_class is a placeholder - replace with the verified class.")
    return 0


def cmd_demo(args) -> int:
    t0 = time.time()
    started = utc_now()
    cfg, grid, n_scen, o, d = _setup(args)
    dep = args.departure or cfg.route.departure_start
    world = generate_synthetic(cfg, grid, dep, _horizon_days(cfg), n_scen, seed=args.seed)
    bergs = _parse_bergs(args)
    if bergs:
        import numpy as np

        from antarctic_routing.iceberg.drift import add_iceberg_hazard

        world = add_iceberg_hazard(world, bergs, rng=np.random.default_rng(args.seed),
                                   radius_m=args.berg_radius_km * 1e3)
    vessel = VesselModel.from_config(cfg)
    plan = plan_candidates(
        world, vessel, o, d, cfg.routing.risk_budget, cfg.routing.risk_weights,
        cfg.routing.risk_estimator, cfg.routing.connectivity, args.scenario_routes,
        cfg.routing.confidence, args.seed,
    )
    out = Path(args.out)
    outputs = [write_json_artifact(out / "plan.json", _plan_dict(plan, world, bergs))]
    outputs.append(plot_plan(world, plan, vessel.tau, out / "route_map.png",
                             f"Antarctic route plan - departure {dep.isoformat()} ({n_scen} joint scenarios)"))
    cfg_sha = sha256_file(Path(args.config))
    if plan.recommended:
        gj = route_to_geojson(plan.recommended, world, issued=world.start.isoformat() + "Z", config_sha256=cfg_sha)
        outputs.append(write_json_artifact(out / "recommended_route.geojson", gj))
        outputs.append(route_to_csv(plan.recommended, world, out / "recommended_route.csv"))

    rec = plan.recommended.evaluation.summary() if plan.recommended else {}
    stage = StageResult(
        stage="route_planning", status="passed", execution_mode=world.execution_mode, software=SOFTWARE,
        inputs=[file_record(Path(args.config), "scenario configuration")],
        parameters={"departure": dep.isoformat(), "n_scenarios": n_scen, "seed": args.seed,
                    "resolution_km": grid.resolution_m / 1000, "risk_budget": cfg.routing.risk_budget,
                    "estimator": cfg.routing.risk_estimator},
        outputs=[file_record(p, "route_planning") for p in outputs],
        metrics={"plan_status": plan.status, "n_candidates": len(plan.candidates), **rec},
        command=" ".join(["antroute", *args.argv]),
        started_at=started, finished_at=utc_now(), duration_seconds=round(time.time() - t0, 3),
        warnings=plan.warnings + ["controlled_synthetic data: not a real forecast"],
    )
    write_json_artifact(out / "stage-result.json", stage.to_dict())
    print(plan.explanation)
    print(f"Artifacts written to {out}/")
    return 0


def cmd_departures(args) -> int:
    cfg, grid, n_scen, o, d = _setup(args)
    start = args.start or cfg.route.departure_start
    end = args.end or cfg.route.departure_end
    dates, cur = [], start
    while cur <= end:
        dates.append(cur)
        cur += timedelta(days=args.step_days)
    days = _horizon_days(cfg)
    sweep = sweep_departures(
        dates, lambda dd: generate_synthetic(cfg, grid, dd, days, n_scen, seed=args.seed),
        VesselModel.from_config(cfg), o, d, cfg.routing.risk_budget, cfg.routing.risk_weights,
        cfg.routing.risk_estimator, cfg.routing.connectivity, args.scenario_routes,
        cfg.routing.confidence, args.seed,
    )
    out = Path(args.out)
    payload = {**sweep.to_dict(), "execution_mode": "controlled_synthetic", "n_scenarios": n_scen,
               "disclaimer": DISCLAIMER}
    write_json_artifact(out / "departures.json", payload)
    plot_departures(sweep, cfg.routing.risk_budget, out / "departure_chart.png",
                    f"Departure window {start} .. {end} (controlled-synthetic, {n_scen} scenarios/date)")
    for opt in sweep.options:
        flag = "OK " if opt.feasible else "-- "
        print(f"  {flag}{opt.departure}  P(breach) UB {opt.p_breach_upper:6.1%}  "
              f"E[t] {opt.expected_hours:6.1f} h  E[fuel] {opt.expected_fuel:7.0f}")
    print(sweep.explanation)
    return 0


def cmd_fetch_sea_ice(args) -> int:
    from antarctic_routing.ingestion import sea_ice
    from antarctic_routing.ingestion.base import run_download

    cfg = load_config(args.config)
    d = cfg.domain
    root = Path(args.root)
    results = []
    for req in sea_ice.daily_requests(args.start, args.end, (d.lat_min, d.lat_max, d.lon_min, d.lon_max)):
        res = run_download(req, root, lambda r, dest: sea_ice.fetch_osisaf(r, dest))
        results.append({"date": req.start.isoformat(), "status": res.status,
                        "outputs": res.outputs, "warnings": res.warnings})
        print(f"  {req.start}  {res.status:8s} {'; '.join(res.warnings)[:100]}")
    write_json_artifact(root / "sea_ice" / "fetch-summary.json", {"product": sea_ice.PRODUCT, "results": results})
    ok = all(r["status"] == "passed" for r in results)
    print("All files available." if ok else "Some downloads did not complete - see fetch-summary.json.")
    return 0 if ok else 1


def cmd_build_dataset(args) -> int:
    import glob

    from antarctic_routing.ingestion.osisaf_reader import build_sea_ice_dataset, product_family

    t0, started = time.time(), utc_now()
    cfg = load_config(args.config)
    grid = PolarGrid.from_domain(cfg.domain, args.resolution_km or cfg.grid.resolution_km)
    paths = sorted({p for pattern in args.inputs for p in glob.glob(pattern, recursive=True)})
    if not paths:
        print("No input files matched.")
        return 1
    ds = build_sea_ice_dataset(paths, grid)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    imputed = float(ds["imputed_mask"].mean())
    products = dict(zip(ds.attrs["source_files"].split(","),
                        zip(ds.attrs["source_product_ids"].split(","),
                            ds.attrs["source_product_versions"].split(","), strict=True), strict=True))
    inputs = []
    for p in paths:
        product_id, version = products[Path(p).name]
        record = file_record(Path(p), product_id)
        record.update(product_family=product_family(product_id), product_version=version)
        inputs.append(record)
    stage = StageResult(
        stage="harmonise_sea_ice", status="passed", execution_mode="real", software=SOFTWARE,
        inputs=inputs,
        parameters={"resolution_km": grid.resolution_m / 1000, "crs": "EPSG:3031",
                    "product_family": ds.attrs["source_product_family"],
                    "source_product": ds.attrs["source_product"]},
        outputs=[file_record(out, "harmonised sea-ice dataset")],
        metrics={"days": int(ds.sizes["time"]), "imputed_fraction": imputed,
                 "land_fraction": float(ds["land_mask"].mean())},
        command=" ".join(["antroute", *args.argv]), started_at=started, finished_at=utc_now(),
        duration_seconds=round(time.time() - t0, 3),
    )
    write_json_artifact(out.with_suffix(".stage-result.json"), stage.to_dict())
    print(f"Wrote {out} ({ds.sizes['time']} days, {imputed:.1%} imputed cells)")
    return 0


def _forecast_data(args):
    """Load real harmonised data (--data) or generate a synthetic history; split seasons."""
    import xarray as xr

    from antarctic_routing.forecasting.dataset import issue_seasons
    from antarctic_routing.preprocessing.climatology import chronological_split
    from antarctic_routing.synthetic import synthetic_history

    cfg = load_config(args.config)
    if args.data:
        ds = xr.load_dataset(args.data)
    else:
        a, b = (int(v) for v in args.synthetic_seasons.split(":"))
        grid = PolarGrid.from_domain(cfg.domain, args.resolution_km or 25.0)
        ds = synthetic_history(cfg, grid, range(a, b), seed=args.seed)
    months = cfg.project.season_months
    seasons = sorted({s for s in issue_seasons(ds, range(ds.sizes["time"]), months) if s is not None})
    split = chronological_split(seasons, args.n_val, args.n_test)
    return cfg, ds, months, split


def cmd_train_forecast(args) -> int:
    from antarctic_routing.forecasting.train import TrainConfig, train_unet

    cfg, ds, months, split = _forecast_data(args)
    tc = TrainConfig(history_days=args.history_days or cfg.forecast.history_days,
                     lead_days=args.lead_days or cfg.forecast.lead_days,
                     epochs=args.epochs, base_channels=args.base_channels, seed=args.seed,
                     residual=not args.direct)
    print(f"Training on seasons {split['train']}, validating on {split['val']} "
          f"({ds.attrs.get('execution_mode', 'real')} data; {tc.history_days} d history, {tc.lead_days} d leads)")
    synthetic_id = None if args.data else f"synthetic_history:{args.synthetic_seasons}:seed={args.seed}"
    result = train_unet(ds, split["train"], split["val"], months, tc, args.out,
                        dataset_path=args.data, dataset_id=synthetic_id)
    for row in result.history:
        print(f"  epoch {row['epoch']:3d}  train {row['train_loss']:.4f}  val MAE {row['val_mae']:.4f}")
    print(f"Best epoch {result.best_epoch}; checkpoint {result.checkpoint}")
    return 0


def _load_checkpoint(args, ds):
    """Load ``--weights`` for ``ds``: the grid must match and an explicit --lead-days must agree."""
    from antarctic_routing.forecasting.train import load_model

    try:
        model, meta = load_model(args.weights, ds)
    except ValueError as exc:
        raise SystemExit(f"refusing to use {args.weights}: {exc}") from None
    trained = meta["train_config"]["lead_days"]
    if args.lead_days is not None and args.lead_days != trained:
        raise SystemExit(f"--lead-days {args.lead_days} does not match the checkpoint's {trained}-day horizon")
    return model, meta


def cmd_evaluate_forecast(args) -> int:
    from antarctic_routing.forecasting.evaluate import evaluate_forecasts
    from antarctic_routing.forecasting.train import unet_predictor
    from antarctic_routing.viz import plot_forecast_skill

    t0, started = time.time(), utc_now()
    cfg, ds, months, split = _forecast_data(args)
    model, meta = _load_checkpoint(args, ds)
    hd, ld = meta["train_config"]["history_days"], meta["train_config"]["lead_days"]
    if set(meta["train_seasons"]) & set(split["test"]):
        raise SystemExit("refusing to evaluate: checkpoint was trained on a test season")
    report = evaluate_forecasts(ds, unet_predictor(model), split["train"], split["test"], hd, ld, months)
    out = Path(args.out)
    outputs = [write_json_artifact(out / "forecast_eval.json", report)]
    outputs.append(plot_forecast_skill(report, out / "forecast_skill.png", "Sea-ice U-Net vs baselines"))
    stage = StageResult(
        stage="forecast_evaluation", status="passed", execution_mode=report["execution_mode"], software=SOFTWARE,
        inputs=[file_record(Path(args.weights), "U-Net checkpoint")],
        parameters={"history_days": hd, "lead_days": ld, "train_seasons": split["train"],
                    "test_seasons": split["test"]},
        outputs=[file_record(p, "forecast_evaluation") for p in outputs],
        metrics={"mae": report["mae"], "skill_mae_vs": report["skill_mae_vs"]},
        command=" ".join(["antroute", *args.argv]), started_at=started, finished_at=utc_now(),
        duration_seconds=round(time.time() - t0, 3),
    )
    write_json_artifact(out / "stage-result.json", stage.to_dict())
    header = "lead  " + "  ".join(f"{m[:12]:>12s}" for m in report["methods"])
    print(header)
    for i, h in enumerate(report["leads"]):
        print(f"{h:4d}  " + "  ".join(f"{report['mae'][m][i]:12.4f}" for m in report["methods"]))
    return 0


def cmd_calibrate_forecast(args) -> int:
    from antarctic_routing.forecasting.calibration import evaluate_probabilities
    from antarctic_routing.forecasting.train import unet_predictor
    from antarctic_routing.viz import plot_reliability

    cfg, ds, months, split = _forecast_data(args)
    model, meta = _load_checkpoint(args, ds)
    if set(meta["train_seasons"]) & (set(split["val"]) | set(split["test"])):
        raise SystemExit("refusing to calibrate: checkpoint was trained on a validation/test season")
    tc = meta["train_config"]
    report = evaluate_probabilities(
        ds, unet_predictor(model), split["train"], split["val"], split["test"], tc["history_days"],
        tc["lead_days"], months, tau=cfg.vessel.max_ice_concentration, n_members=args.members, seed=args.seed,
    )
    out = Path(args.out)
    write_json_artifact(out / "calibrator.json", {**report["calibrator"], "tau": report["tau"],
                                                  "weights_sha256": sha256_file(Path(args.weights))})
    write_json_artifact(out / "probability_eval.json", report)
    plot_reliability(report, out / "reliability.png", "Sea-ice hazard probability calibration")
    print("lead   Brier raw   calibrated   climatology")
    for i, h in enumerate(report["leads"]):
        print(f"{h:4d}  {report['brier']['raw'][i]:10.4f}  {report['brier']['calibrated'][i]:11.4f}"
              f"  {report['brier']['climatology'][i]:12.4f}")
    return 0


def cmd_trust_horizon(args) -> int:
    import json

    from antarctic_routing.forecasting.trust import trust_horizon
    from antarctic_routing.viz import plot_trust_horizon

    report = json.loads(Path(args.report).read_text())
    if "by_season" not in report:
        raise SystemExit("report has no per-season errors; re-run evaluate-forecast")
    model = report["methods"][0]
    by_baseline = {b: trust_horizon(report["by_season"], model, b, n_boot=args.n_boot, alpha=args.alpha,
                                    min_delta=args.min_delta, seed=args.seed) for b in args.baselines}
    first = next(iter(by_baseline.values()))
    result = {
        "model": model, "trust_horizon_days": min(t["trust_horizon_days"] for t in by_baseline.values()),
        "rule": "minimum over baselines of the last lead with every lower bound above min_delta",
        "by_baseline": by_baseline, "leads": first["leads"], "min_delta": args.min_delta, "alpha": args.alpha,
        "n_seasons": first["n_seasons"], "method": first["method"],
        "execution_mode": report.get("execution_mode", "real"), "source_report": str(args.report),
    }
    out = Path(args.out)
    write_json_artifact(out / "trust_horizon.json", result)
    plot_trust_horizon(result, out / "trust_horizon.png", "Forecast trust horizon")
    for b, t in by_baseline.items():
        print(f"  vs {b:20s} trust horizon {t['trust_horizon_days']} d"
              f"{' (limited by evaluated leads)' if t['limited_by_max_lead'] else ''}")
    print(f"Trust horizon: {result['trust_horizon_days']} day(s)")
    return 0


def _forcing_record(path, f, kind: str) -> dict:
    from antarctic_routing.common.provenance import sha256_file

    a = f.attrs
    return {"forcing_file": str(path), "forcing_sha256": sha256_file(path), "product": kind,
            "time_mode": f.wind_time_mode if kind == "ERA5" else f.current_time_mode,
            "source_files": a.get("source_files"), "source_sha256": a.get("source_sha256"),
            "averaging": a.get("averaging"), "crs": a.get("crs"), "resolution_m": a.get("resolution_m")}


def _load_forcing_args(args, shape) -> dict:
    """ForecastContext forcing kwargs from ``--forcing`` or the separate ``--wind-forcing``/``--current-forcing``.

    A field that is not supplied stays schematic and is labelled so. A separate
    file must contain the field it is given for; nothing is substituted silently.
    """
    from antarctic_routing.ingestion.forcing import load_forcing_fields

    combined = getattr(args, "forcing", None)
    wind_path, current_path = getattr(args, "wind_forcing", None), getattr(args, "current_forcing", None)
    if combined and (wind_path or current_path):
        raise ValueError("use either --forcing or --wind-forcing/--current-forcing, not both")
    out = {"currents": None, "winds": None, "wind_times": None, "current_times": None,
           "forcing_sources": {}, "forcing_provenance": {}}
    picks = []
    if combined:
        picks = [(combined, "winds"), (combined, "currents")]
    else:
        picks = [(p, k) for p, k in ((wind_path, "winds"), (current_path, "currents")) if p]
    for path, kind in picks:
        f = load_forcing_fields(path, expected_shape=shape)
        if kind == "winds":
            if f.winds is None:
                if combined:
                    continue
                raise ValueError(f"--wind-forcing {path} has no wind_x/wind_y")
            out.update(winds=f.winds, wind_times=f.wind_times)
            out["forcing_sources"]["winds"] = "ERA5 daily" if f.wind_times is not None else "ERA5 time-mean"
            out["forcing_provenance"]["winds"] = _forcing_record(path, f, "ERA5")
        else:
            if f.currents is None:
                if combined:
                    continue
                raise ValueError(f"--current-forcing {path} has no current_x/current_y")
            out.update(currents=f.currents, current_times=f.current_times)
            out["forcing_sources"]["currents"] = "CMEMS daily" if f.current_times is not None else "CMEMS time-mean"
            out["forcing_provenance"]["currents"] = _forcing_record(path, f, "CMEMS")
    return out


def _forecast_context(args, ds, months, split, n_bank: int = 300):
    from antarctic_routing.forecasting.scenarios import ForecastContext
    from antarctic_routing.forecasting.train import unet_predictor

    model, meta = _load_checkpoint(args, ds)
    tc = meta["train_config"]
    forcing = _load_forcing_args(args, ds["land_mask"].shape)
    if forcing["currents"] is not None:
        land = ds["land_mask"].values.astype(bool)
        c = forcing["currents"]
        forcing["currents"] = (np.where(land, 0.0, c[0]), np.where(land, 0.0, c[1]))
    ctx = ForecastContext.build(ds, unet_predictor(model), tc["history_days"], tc["lead_days"], months,
                                meta["train_seasons"], bank_size=n_bank, seed=args.seed, **forcing)
    return ctx, meta


def _forcing_arg_summary(args):
    """The forcing file argument(s) as given (unchanged string for the single ``--forcing`` form)."""
    wind, current = getattr(args, "wind_forcing", None), getattr(args, "current_forcing", None)
    if wind or current:
        return {"wind_forcing": wind or "schematic (no --wind-forcing given)",
                "current_forcing": current or "schematic (no --current-forcing given)"}
    return args.forcing or "schematic (no --forcing given)"


def _forcing_gate(args, ctx, ds) -> tuple[int | None, list[str]]:
    """Exit code 2 when ``--require-real-forcing`` meets schematic/mixed forcing; otherwise the warnings to record.

    Real sea-ice data run with schematic winds or currents is allowed (and labelled
    ``mixed``/``schematic`` in the output) but is announced on stderr, never silent.
    """
    from antarctic_routing.forecasting.scenarios import forcing_status

    src = ctx.forcing_sources
    status = forcing_status(src)
    if status == "real":
        return None, []
    problem = f"forcing is {status} (winds: {src.get('winds')}, currents: {src.get('currents')})"
    if getattr(args, "require_real_forcing", False):
        print(f"BLOCKED: {problem}; --require-real-forcing needs real winds and currents", file=sys.stderr)
        return 2, []
    if ds.attrs.get("execution_mode", "real") != "real":
        return None, []
    msg = f"real sea-ice data with {problem}; schematic fields are not real data"
    print(f"WARNING: {msg}", file=sys.stderr)
    return None, [msg]


def _drift_kwargs(args) -> dict:
    """Iceberg drift ensemble parameters for add_iceberg_hazard; defaults are the uncalibrated model."""
    a = float(getattr(args, "drift_alpha_scale", 1.0))
    return {"beta": float(getattr(args, "drift_beta", 1.0)), "alpha_range": (0.01 * a, 0.03 * a),
            "spread_factor": float(getattr(args, "drift_spread_factor", 1.0))}


def _selected_route(sweep, world) -> dict | None:
    """Coordinates, evaluation and iceberg-presence statistics of the selected departure's route."""
    opt = sweep.selected
    if opt is None:
        return None
    plan = opt.plan
    cand = plan.recommended or next((c for c in plan.candidates if "lowest_risk" in c.tags), None)
    rows = np.array([c[0] for c in cand.route.cells])
    cols = np.array([c[1] for c in cand.route.cells])
    lat, lon = world.grid.lat2d[rows, cols], world.grid.lon2d[rows, cols]
    out = {"departure": opt.departure.isoformat(), "labels": list(cand.labels),
           "evaluation": cand.evaluation.summary(),
           "max_segment_breach_prob": float(max(cand.evaluation.segment_breach_prob, default=0.0)),
           "cells_row_col": [[int(r), int(c)] for r, c in zip(rows, cols, strict=True)],
           "coordinates_lat_lon": [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lat, lon, strict=True)]}
    if world.berg is not None:
        step = world.time_step_hours
        first = int((opt.lead_days or 0) * 24 // step)
        last = min(world.n_times - 1, int(((opt.lead_days or 0) * 24 + opt.expected_hours) // step))
        dens = world.berg[:, first:last + 1].mean(0)                      # (T, ny, nx) member fraction
        out["iceberg_presence_on_route"] = {
            "layers": [first, last], "max_member_fraction_on_route_cells": float(dens[:, rows, cols].max()),
            "route_cells_with_any_presence": int((dens[:, rows, cols] > 0).any(0).sum())}
    return out


def _iceberg_hazard_stats(world) -> dict | None:
    if world.berg is None:
        return None
    dens = world.berg.mean(0)                                            # (T, ny, nx)
    return {"cells_with_any_presence_per_layer": [int((d > 0).sum()) for d in dens],
            "cells_with_presence_ge_5pct_per_layer": [int((d >= 0.05).sum()) for d in dens],
            "max_member_fraction_per_layer": [float(d.max()) for d in dens]}


def cmd_plan_window(args) -> int:
    import json

    import numpy as np

    from antarctic_routing.forecasting.scenarios import grid_of
    from antarctic_routing.routing.departure import plan_from_issue
    from antarctic_routing.viz import plot_departures

    cfg, ds, months, split = _forecast_data(args)
    ctx, meta = _forecast_context(args, ds, months, split)
    code, forcing_warnings = _forcing_gate(args, ctx, ds)
    if code is not None:
        return code
    trust = None
    if args.trust_report:
        trust = int(json.loads(Path(args.trust_report).read_text())["trust_horizon_days"])
    n_days = args.window_days + _horizon_days(cfg)
    world = ctx.scenarios(args.issue, n_days, args.members, np.random.default_rng(args.seed))
    grid = grid_of(ds)
    bergs, outside = _bergs_in_grid(_parse_bergs(args), grid)
    if bergs:
        from antarctic_routing.iceberg.drift import add_iceberg_hazard

        world = add_iceberg_hazard(world, bergs, rng=np.random.default_rng(args.seed),
                                   radius_m=args.berg_radius_km * 1e3, **_drift_kwargs(args))
    o = grid.cell_of(cfg.route.origin.lat, cfg.route.origin.lon)
    d = grid.cell_of(cfg.route.destination.lat, cfg.route.destination.lon)
    sweep = plan_from_issue(
        world, list(range(args.window_days)), VesselModel.from_config(cfg), o, d, cfg.routing.risk_budget,
        cfg.routing.risk_weights, cfg.routing.risk_estimator, cfg.routing.connectivity, args.scenario_routes,
        cfg.routing.confidence, args.seed, trust_horizon_days=trust, require_trusted=args.require_trusted,
    )
    out = Path(args.out)
    payload = {**sweep.to_dict(), "issue": args.issue.isoformat(), "trust_horizon_days": trust,
               "layer_source": world.layer_source, "n_members": world.n_scenarios,
               "forecast_lead_days": ctx.lead_days, "model_train_seasons": meta["train_seasons"],
               "forcing": _forcing_arg_summary(args),
               "forcing_provenance": world.meta["forcing"],
               "icebergs": [{"id": b[0], "lat": b[1], "lon": b[2]} for b in bergs],
               "icebergs_outside_grid": [{"id": b[0], "lat": b[1], "lon": b[2]} for b in outside],
               "iceberg_source": _iceberg_source(args),
               "iceberg_drift": {**_drift_kwargs(args), "radius_m": args.berg_radius_km * 1e3,
                                 "drifted": bool(bergs)},
               "iceberg_hazard": _iceberg_hazard_stats(world),
               "selected_route": _selected_route(sweep, world),
               "execution_mode": world.execution_mode, "data_description": world.description,
               "disclaimer": DISCLAIMER}
    if forcing_warnings:
        payload["warnings"] = forcing_warnings
    write_json_artifact(out / "plan_window.json", payload)
    if args.export_maps:
        from antarctic_routing.publish import iceberg_tracks, map_layers, write_compact_json

        maps = map_layers(world, VesselModel.from_config(cfg).tau)
        if bergs:
            drift = _drift_kwargs(args)
            maps["iceberg_tracks"] = iceberg_tracks(world, bergs, args.seed, args.berg_radius_km * 1e3, **drift)
        write_compact_json(out / "forecast_maps.json", maps)
    plot_departures(sweep, cfg.routing.risk_budget, out / "departure_window.png",
                    f"Departure window from forecast issued {args.issue} ({world.n_scenarios} joint scenarios)",
                    issue=args.issue, forecast_days=ctx.lead_days, trust_horizon_days=trust)
    for opt in sweep.options:
        flag = "OK " if opt.feasible else "-- "
        trusted = "" if opt.within_trust_horizon is None else (" trusted" if opt.within_trust_horizon else " UNTRUSTED")
        print(f"  {flag}+{opt.lead_days:2d} d {opt.departure}  UB {opt.p_breach_upper:6.1%}  "
              f"E[t] {opt.expected_hours:5.1f} h  {opt.support}{trusted}")
    print(sweep.explanation)
    if bergs or outside:
        print(f"Icebergs: {len(bergs)} of {len(bergs) + len(outside)} inside the routing grid drifted; "
              f"{len(outside)} outside it excluded")
    fm = world.meta["forcing"]
    print(f"Forcing: {fm['label']} ({fm['status']})")
    return 0


def cmd_replay(args) -> int:
    import json

    import numpy as np

    from antarctic_routing.forecasting.scenarios import grid_of
    from antarctic_routing.replay import run_replay
    from antarctic_routing.routing.replan import ReplanPolicy
    from antarctic_routing.viz import plot_replay

    cfg, ds, months, split = _forecast_data(args)
    ctx, meta = _forecast_context(args, ds, months, split)
    code, forcing_warnings = _forcing_gate(args, ctx, ds)
    if code is not None:
        return code
    trust = int(json.loads(Path(args.trust_report).read_text())["trust_horizon_days"]) if args.trust_report else None
    grid = grid_of(ds)
    o = grid.cell_of(cfg.route.origin.lat, cfg.route.origin.lon)
    d = grid.cell_of(cfg.route.destination.lat, cfg.route.destination.lon)
    out = Path(args.out)
    audit = out / "audit.jsonl"
    audit.unlink(missing_ok=True)
    policy = ReplanPolicy(cfg.routing.risk_budget, cfg.routing.risk_estimator, cfg.routing.confidence)
    result = run_replay(ctx, VesselModel.from_config(cfg), o, d, args.start, args.window_days, args.max_wait_days,
                        args.members, policy, cfg.routing.risk_weights, _horizon_days(cfg),
                        np.random.default_rng(args.seed), audit_path=audit, trust_horizon_days=trust,
                        connectivity=cfg.routing.connectivity, scenario_routes=args.scenario_routes, seed=args.seed)
    result["model_train_seasons"] = meta["train_seasons"]
    if forcing_warnings:
        result["forcing_warnings"] = forcing_warnings
    from antarctic_routing.preprocessing.climatology import season_of

    replay_season = season_of(args.start, months)
    if replay_season in set(meta["train_seasons"]):
        result["warning"] = f"replay season {replay_season} was used to train the model (in-sample replay)"
    write_json_artifact(out / "replay.json", result)
    plot_replay(result, ctx, cfg.vessel.max_ice_concentration, out / "replay.png",
                f"Voyage replay from {args.start}")
    for rec in result["days"]:
        print(f"  {rec['day']}  {rec['phase']:8s} {rec['action']:18s} {', '.join(rec.get('triggers') or [])}")
    if "truth" in result:
        for name in ("planner", "naive"):
            t = result["truth"][name]
            print(f"  {name:8s} left {t['departure']}  {t['hours']:.1f} h  fuel {t['fuel_index']:.0f}  "
                  f"observed-ice cells {t['observed_breach_cells']}")
    return 0


def cmd_backtest(args) -> int:
    import csv

    from antarctic_routing.forecasting.scenarios import grid_of
    from antarctic_routing.routing.replan import ReplanPolicy
    from antarctic_routing.validation.backtest import run_backtest
    from antarctic_routing.viz import plot_backtest

    cfg, ds, months, split = _forecast_data(args)
    ctx, meta = _forecast_context(args, ds, months, split)
    code, forcing_warnings = _forcing_gate(args, ctx, ds)
    if code is not None:
        return code
    seasons = args.seasons or split["test"]
    starts = []
    for s in seasons:
        for md in args.start_days:
            m, d = (int(v) for v in md.split("-"))
            starts.append(date(s if m >= months[0] else s + 1, m, d))
    grid = grid_of(ds)
    o = grid.cell_of(cfg.route.origin.lat, cfg.route.origin.lon)
    dst = grid.cell_of(cfg.route.destination.lat, cfg.route.destination.lon)
    policy = ReplanPolicy(cfg.routing.risk_budget, cfg.routing.risk_estimator, cfg.routing.confidence)
    result = run_backtest(ctx, VesselModel.from_config(cfg), o, dst, starts, args.window_days, args.max_wait_days,
                          args.members, policy, cfg.routing.risk_weights, _horizon_days(cfg),
                          buffer_km=args.buffer_km, seed=args.seed, scenario_routes=args.scenario_routes,
                          connectivity=cfg.routing.connectivity)
    if forcing_warnings:
        result["forcing_warnings"] = forcing_warnings
    out = Path(args.out)
    write_json_artifact(out / "backtest.json", result)
    with (out / "backtest_voyages.csv").open("w", newline="") as fh:
        cols = ["start", "season", "method", "departure", "days_waited", "breached", "observed_breach_cells",
                "hazard_hours", "sail_hours", "elapsed_hours", "fuel_index"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(result["voyages"])
    plot_backtest(result, out / "backtest.png", "Replay backtest vs baselines")
    print(f"{'method':16s} {'n':>3s} {'breach%':>8s} {'hazard h':>9s} {'wait d':>7s} {'elapsed h':>10s} {'fuel':>7s}")
    for m, sm in result["summary"].items():
        def f(v, fmt):
            return format(v, fmt) if v is not None else "-"
        print(f"{m:16s} {sm['n']:3d} {f(None if sm['breach_rate'] is None else 100 * sm['breach_rate'], '8.0f')} "
              f"{f(sm['mean_hazard_hours'], '9.1f')} {f(sm['mean_days_waited'], '7.1f')} "
              f"{f(sm['mean_elapsed_hours'], '10.1f')} {f(sm['mean_fuel_index'], '7.0f')}")
    return 0


def cmd_sensitivity(args) -> int:
    from antarctic_routing.validation.sensitivity import run_sensitivity
    from antarctic_routing.viz import plot_sensitivity

    cfg, grid, n_scen, o, d = _setup(args)
    dep = args.departure or cfg.route.departure_start
    world = generate_synthetic(cfg, grid, dep, _horizon_days(cfg), n_scen, seed=args.seed)
    result = run_sensitivity(world, VesselModel.from_config(cfg), o, d, args.lambdas, args.ks,
                             cfg.routing.risk_budget, cfg.routing.risk_weights, cfg.routing.risk_estimator,
                             cfg.routing.connectivity)
    out = Path(args.out)
    write_json_artifact(out / "sensitivity.json", result)
    plot_sensitivity(result, out / "sensitivity.png", f"Fuel/speed assumption sensitivity, departure {dep}")
    print(f"Recommendation unchanged in {result['stable_fraction']:.0%} of {len(result['grid'])} settings")
    for r in result["grid"]:
        if r["route_changed"]:
            print(f"  changes at lambda={r['lambda']:g}, k={r['k']:g} (moves {r['deviation_km'] or 0:.0f} km)")
    return 0


def cmd_serve(args) -> int:
    import logging
    import os

    import uvicorn

    from antarctic_routing.api.main import create_app

    logging.basicConfig(level=os.environ.get("ANTROUTE_LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(f"Serving dashboard and API on http://{args.host}:{args.port}  (docs: /docs)")
    # one worker process: jobs and voyages are held in this process's memory
    uvicorn.run(create_app(args.config, args.artifacts_dir), host=args.host, port=args.port, log_level="info")
    return 0


def cmd_brief(args) -> int:
    import json
    import tempfile

    from antarctic_routing.brief import write_brief

    cfg, grid, n_scen, o, d = _setup(args)
    dep = args.departure or cfg.route.departure_start
    world = generate_synthetic(cfg, grid, dep, _horizon_days(cfg), n_scen, seed=args.seed)
    plan = plan_candidates(world, VesselModel.from_config(cfg), o, d, cfg.routing.risk_budget,
                           cfg.routing.risk_weights, cfg.routing.risk_estimator, cfg.routing.connectivity,
                           args.scenario_routes, cfg.routing.confidence, args.seed)
    trust, limited = None, False
    if args.trust_report:
        report = json.loads(Path(args.trust_report).read_text())
        trust = int(report["trust_horizon_days"])
        limited = any(b.get("limited_by_max_lead") for b in report.get("by_baseline", {}).values())
    with tempfile.TemporaryDirectory() as tmp:
        png = plot_plan(world, plan, cfg.vessel.max_ice_concentration, Path(tmp) / "map.png",
                        f"Route plan - departure {dep.isoformat()}")
        out = write_brief(args.out, cfg, world, plan, png, trust, limited)
    print(f"Wrote {out}: {plan.explanation}")
    return 0


def cmd_fetch_forcing(args) -> int:
    from antarctic_routing.ingestion import cmems, era5
    from antarctic_routing.ingestion.base import run_download

    cfg = load_config(args.config)
    root = Path(args.root)
    jobs = []
    grid = PolarGrid.from_domain(cfg.domain, args.resolution_km or cfg.grid.resolution_km)
    if not args.skip_era5:
        margin = era5.AREA_MARGIN_DEG if args.era5_margin_deg is None else args.era5_margin_deg
        jobs.append(("era5", era5.request_for(cfg.domain, args.start, args.end, grid, margin),
                     era5.make_fetcher(cfg.domain, grid, margin)))
    if not args.skip_cmems:
        margin = cmems.AREA_MARGIN_DEG if args.cmems_margin_deg is None else args.cmems_margin_deg
        jobs.append(("cmems", cmems.request_for(cfg.domain, args.start, args.end, args.cmems_dataset, grid, margin),
                     cmems.make_fetcher(cfg.domain, args.cmems_dataset, grid, margin)))
    results = []
    for name, req, fetch in jobs:
        res = run_download(req, root, fetch)
        results.append({"source": name, "status": res.status, "outputs": res.outputs, "warnings": res.warnings})
        print(f"  {name:6s} {res.status:8s} {'; '.join(res.warnings)[:140]}")
    write_json_artifact(root / "forcing-summary.json", {"results": results})
    ok = all(r["status"] == "passed" for r in results)
    print("Forcing files ready." if ok else "Some forcing downloads did not complete - see forcing-summary.json.")
    return 0 if ok else 1


def cmd_build_forcing(args) -> int:
    from antarctic_routing.ingestion.forcing import build_forcing

    t0, started = time.time(), utc_now()
    cfg = load_config(args.config)
    grid = PolarGrid.from_domain(cfg.domain, args.resolution_km or cfg.grid.resolution_km)
    ds = build_forcing(grid, era5_path=args.era5, cmems_path=args.cmems, wind_time_mode=args.wind_time_mode,
                       current_time_mode=args.current_time_mode)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    inputs = [file_record(Path(p), name) for p, name in ((args.era5, "ERA5"), (args.cmems, "CMEMS")) if p]
    stage = StageResult(
        stage="harmonise_forcing", status="passed", execution_mode="real", software=SOFTWARE, inputs=inputs,
        parameters={"resolution_km": grid.resolution_m / 1000, "averaging": ds.attrs["averaging"],
                    **({"wind_time_mode": args.wind_time_mode} if args.era5 else {}),
                    **({"current_time_mode": args.current_time_mode} if args.cmems else {})},
        outputs=[file_record(out, "forcing fields")],
        metrics={k: v for k, v in ds.attrs.items() if k.endswith("filled_fraction")},
        command=" ".join(["antroute", *args.argv]), started_at=started, finished_at=utc_now(),
        duration_seconds=round(time.time() - t0, 3),
    )
    write_json_artifact(out.with_suffix(".stage-result.json"), stage.to_dict())
    print(f"Wrote {out} ({', '.join(ds.data_vars)})")
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="antroute", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"antarctic-routing {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, out_default):
        sp.add_argument("--config", default=DEFAULT_CONFIG)
        sp.add_argument("--out", default=out_default)
        sp.add_argument("--scenarios", type=int, default=None, help="override forecast.route_scenarios")
        sp.add_argument("--resolution-km", type=float, default=None, help="override grid.resolution_km")
        sp.add_argument("--scenario-routes", type=int, default=3, help="per-scenario optimal candidates")
        sp.add_argument("--seed", type=int, default=42)

    v = sub.add_parser("validate-config", help="validate configuration")
    v.add_argument("--config", default=DEFAULT_CONFIG)
    v.set_defaults(func=cmd_validate)

    dm = sub.add_parser("demo", help="plan routes for one departure")
    common(dm, "artifacts/demo")
    dm.add_argument("--departure", type=date.fromisoformat, default=None)
    dm.add_argument("--iceberg", action="append", metavar="ID:LAT:LON", help="tracked iceberg (repeatable)")
    dm.add_argument("--icebergs", default=None, help="USNIC iceberg list CSV (latest position per berg)")
    dm.add_argument("--berg-radius-km", type=float, default=10.0, help="safety radius around each berg")
    dm.set_defaults(func=cmd_demo)

    dp = sub.add_parser("departures", help="sweep a departure window")
    common(dp, "artifacts/departures")
    dp.add_argument("--start", type=date.fromisoformat, default=None)
    dp.add_argument("--end", type=date.fromisoformat, default=None)
    dp.add_argument("--step-days", type=int, default=1)
    dp.set_defaults(func=cmd_departures)

    fs = sub.add_parser("fetch-sea-ice", help="download OSI SAF OSI-401-b daily files")
    fs.add_argument("--config", default=DEFAULT_CONFIG)
    fs.add_argument("--start", type=date.fromisoformat, required=True)
    fs.add_argument("--end", type=date.fromisoformat, required=True)
    fs.add_argument("--root", default="data/raw")
    fs.set_defaults(func=cmd_fetch_sea_ice)

    bd = sub.add_parser("build-dataset", help="harmonise OSI SAF files onto the EPSG:3031 grid")
    bd.add_argument("--config", default=DEFAULT_CONFIG)
    bd.add_argument("--inputs", nargs="+", required=True, help="file globs, e.g. 'data/raw/sea_ice/**/*.nc'")
    bd.add_argument("--out", default="data/processed/sea_ice.nc")
    bd.add_argument("--resolution-km", type=float, default=None)
    bd.set_defaults(func=cmd_build_dataset)

    ff = sub.add_parser("fetch-forcing", help="download ERA5 winds and CMEMS currents (needs credentials)")
    ff.add_argument("--config", default=DEFAULT_CONFIG)
    ff.add_argument("--start", type=date.fromisoformat, required=True)
    ff.add_argument("--end", type=date.fromisoformat, required=True)
    ff.add_argument("--root", default="data/raw")
    ff.add_argument("--cmems-dataset", default="cmems_mod_glo_phy_my_0.083deg_P1D-m")
    ff.add_argument("--resolution-km", type=float, default=None,
                    help="routing grid whose full lat/lon extent the ERA5 area must cover (default: config)")
    ff.add_argument("--era5-margin-deg", type=float, default=None,
                    help="margin around the grid extent for the ERA5 area, degrees (default 0.5)")
    ff.add_argument("--cmems-margin-deg", type=float, default=None,
                    help="margin around the grid extent for the CMEMS subset box, degrees (default 0.5)")
    ff.add_argument("--skip-era5", action="store_true")
    ff.add_argument("--skip-cmems", action="store_true")
    ff.set_defaults(func=cmd_fetch_forcing)

    bf = sub.add_parser("build-forcing", help="ERA5/CMEMS NetCDF -> wind/current fields on the polar grid")
    bf.add_argument("--config", default=DEFAULT_CONFIG)
    bf.add_argument("--era5", default=None)
    bf.add_argument("--cmems", default=None)
    bf.add_argument("--resolution-km", type=float, default=None)
    bf.add_argument("--wind-time-mode", choices=("mean", "daily"), default="mean",
                    help="ERA5 wind: one time mean (default) or UTC-daily means with a time axis")
    bf.add_argument("--current-time-mode", choices=("mean", "daily"), default="mean",
                    help="CMEMS currents: one time mean (default) or UTC-daily means with a time axis")
    bf.add_argument("--out", default="data/processed/forcing.nc")
    bf.set_defaults(func=cmd_build_forcing)

    def forecast_common(sp):
        sp.add_argument("--config", default=DEFAULT_CONFIG)
        src = sp.add_mutually_exclusive_group()
        src.add_argument("--data", default=None, help="harmonised NetCDF from build-dataset")
        src.add_argument("--synthetic-seasons", default="2004:2024", help="start:end (end exclusive)")
        sp.add_argument("--resolution-km", type=float, default=None)
        sp.add_argument("--history-days", type=int, default=None)
        sp.add_argument("--lead-days", type=int, default=None,
                        help="forecast horizon; default: config forecast.lead_days (training) or the "
                             "checkpoint's horizon (evaluation), which must agree if given")
        sp.add_argument("--n-val", type=int, default=3)
        sp.add_argument("--n-test", type=int, default=3)
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--forcing", default=None, help="forcing.nc from build-forcing (real winds/currents)")
        sp.add_argument("--wind-forcing", default=None,
                        help="build-forcing file whose winds (2-D or daily) are used; not with --forcing")
        sp.add_argument("--current-forcing", default=None,
                        help="build-forcing file whose currents (2-D or daily) are used; not with --forcing")

    tf = sub.add_parser("train-forecast", help="train the sea-ice U-Net")
    forecast_common(tf)
    tf.add_argument("--epochs", type=int, default=30)
    tf.add_argument("--base-channels", type=int, default=16)
    tf.add_argument("--direct", action="store_true", help="predict C directly instead of C_t + change")
    tf.add_argument("--out", default="models/unet")
    tf.set_defaults(func=cmd_train_forecast)

    ef = sub.add_parser("evaluate-forecast", help="score the U-Net against baselines per lead")
    forecast_common(ef)
    ef.add_argument("--weights", default="models/unet/best.pt")
    ef.add_argument("--out", default="reports/forecast")
    ef.set_defaults(func=cmd_evaluate_forecast)

    cf = sub.add_parser("calibrate-forecast", help="ensemble probabilities + isotonic calibration")
    forecast_common(cf)
    cf.add_argument("--weights", default="models/unet/best.pt")
    cf.add_argument("--members", type=int, default=20)
    cf.add_argument("--out", default="reports/calibration")
    cf.set_defaults(func=cmd_calibrate_forecast)

    th = sub.add_parser("trust-horizon", help="season-blocked bootstrap trust horizon from an evaluation report")
    th.add_argument("--report", default="reports/forecast/forecast_eval.json")
    th.add_argument("--baselines", nargs="+", default=["damped_persistence", "climatology"])
    th.add_argument("--min-delta", type=float, default=0.0)
    th.add_argument("--alpha", type=float, default=0.05)
    th.add_argument("--n-boot", type=int, default=2000)
    th.add_argument("--seed", type=int, default=0)
    th.add_argument("--out", default="reports/trust")
    th.set_defaults(func=cmd_trust_horizon)

    pw = sub.add_parser("plan-window", help="departure window from one forecast issue")
    forecast_common(pw)
    pw.add_argument("--weights", default="models/unet/best.pt")
    pw.add_argument("--issue", type=date.fromisoformat, required=True)
    pw.add_argument("--window-days", type=int, default=14)
    pw.add_argument("--members", type=int, default=200)
    pw.add_argument("--scenario-routes", type=int, default=2)
    pw.add_argument("--trust-report", default=None, help="trust_horizon.json from trust-horizon")
    pw.add_argument("--require-trusted", action="store_true")
    pw.add_argument("--iceberg", action="append", metavar="ID:LAT:LON", help="tracked iceberg (repeatable)")
    pw.add_argument("--icebergs", default=None, help="USNIC iceberg list CSV (latest position per berg)")
    pw.add_argument("--berg-radius-km", type=float, default=10.0)
    pw.add_argument("--drift-beta", type=float, default=1.0,
                    help="iceberg current coupling beta (default 1 = uncalibrated)")
    pw.add_argument("--drift-alpha-scale", type=float, default=1.0,
                    help="multiplies the iceberg wind-coefficient range 0.01-0.03 (default 1)")
    pw.add_argument("--drift-spread-factor", type=float, default=1.0,
                    help="iceberg ensemble spread about its mean (default 1 = uncalibrated)")
    pw.add_argument("--require-real-forcing", action="store_true",
                    help="exit with code 2 unless both winds and currents come from real forcing files")
    pw.add_argument("--export-maps", action="store_true",
                    help="also write forecast_maps.json (per-layer maps for the API/dashboard)")
    pw.add_argument("--out", default="reports/window")
    pw.set_defaults(func=cmd_plan_window)

    rp = sub.add_parser("replay", help="day-by-day historical voyage replay with replanning")
    forecast_common(rp)
    rp.add_argument("--weights", default="models/unet/best.pt")
    rp.add_argument("--start", type=date.fromisoformat, required=True)
    rp.add_argument("--window-days", type=int, default=7)
    rp.add_argument("--max-wait-days", type=int, default=30)
    rp.add_argument("--members", type=int, default=200)
    rp.add_argument("--scenario-routes", type=int, default=1)
    rp.add_argument("--trust-report", default=None)
    rp.add_argument("--require-real-forcing", action="store_true",
                    help="exit with code 2 unless both winds and currents come from real forcing files")
    rp.add_argument("--out", default="reports/replay")
    rp.set_defaults(func=cmd_replay)

    bt = sub.add_parser("backtest", help="multi-season replay backtest vs naive and ice-edge-buffer baselines")
    forecast_common(bt)
    bt.add_argument("--weights", default="models/unet/best.pt")
    bt.add_argument("--seasons", type=int, nargs="*", default=None, help="default: the held-out test seasons")
    bt.add_argument("--start-days", nargs="+", default=["11-15", "11-25", "12-05", "12-15"],
                    help="MM-DD; needs history-days of in-season data before each start")
    bt.add_argument("--window-days", type=int, default=7)
    bt.add_argument("--max-wait-days", type=int, default=40)
    bt.add_argument("--members", type=int, default=120)
    bt.add_argument("--buffer-km", type=float, default=50.0)
    bt.add_argument("--scenario-routes", type=int, default=0)
    bt.add_argument("--require-real-forcing", action="store_true",
                    help="exit with code 2 unless both winds and currents come from real forcing files")
    bt.add_argument("--out", default="reports/backtest")
    bt.set_defaults(func=cmd_backtest)

    se = sub.add_parser("sensitivity", help="does the recommendation change with fuel/speed assumptions?")
    common(se, "reports/sensitivity")
    se.add_argument("--departure", type=date.fromisoformat, default=None)
    se.add_argument("--lambdas", type=float, nargs="+", default=[0, 1, 2, 4, 8, 16, 32])
    se.add_argument("--ks", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    se.set_defaults(func=cmd_sensitivity)

    br = sub.add_parser("brief", help="PDF voyage brief for one departure")
    common(br, "reports/voyage_brief.pdf")
    br.add_argument("--departure", type=date.fromisoformat, default=None)
    br.add_argument("--trust-report", default=None)
    br.set_defaults(func=cmd_brief)

    sv = sub.add_parser("serve", help="run the API and dashboard")
    sv.add_argument("--config", default=DEFAULT_CONFIG)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--artifacts-dir", default=None,
                    help="verified real-data bundle to serve read-only (default: $ANTROUTE_ARTIFACTS_DIR)")
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)
    args.argv = argv
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
