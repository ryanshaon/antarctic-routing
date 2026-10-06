import json
from pathlib import Path

import numpy as np
import pytest

from antarctic_routing.cli import main

CONFIG = str(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
FAST = ["--resolution-km", "25", "--scenarios", "80"]


def test_validate_config_prints_summary(capsys):
    assert main(["validate-config", "--config", CONFIG]) == 0
    out = capsys.readouterr().out
    assert "EPSG:3031" in out and "risk budget" in out.lower()


def test_demo_writes_all_artifacts(tmp_path):
    rc = main(["demo", "--config", CONFIG, "--departure", "2027-01-10", "--out", str(tmp_path), *FAST])
    assert rc == 0
    for name in ("plan.json", "route_map.png", "stage-result.json"):
        assert (tmp_path / name).is_file(), name
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["status"] in {"feasible", "infeasible"}
    assert plan["execution_mode"] == "controlled_synthetic"
    stage = json.loads((tmp_path / "stage-result.json").read_text())
    assert stage["execution_mode"] == "controlled_synthetic"
    assert stage["status"] in {"passed", "fallback"}
    if plan["status"] == "feasible":
        gj = json.loads((tmp_path / "recommended_route.geojson").read_text())
        assert gj["type"] == "FeatureCollection"
        assert (tmp_path / "recommended_route.csv").is_file()


def test_departures_sweep_writes_chart_and_table(tmp_path):
    rc = main(["departures", "--config", CONFIG, "--start", "2026-11-20", "--end", "2027-01-10",
               "--step-days", "17", "--out", str(tmp_path), *FAST])
    assert rc == 0
    sweep = json.loads((tmp_path / "departures.json").read_text())
    assert len(sweep["options"]) == 4
    assert (tmp_path / "departure_chart.png").is_file()


def test_build_dataset_from_osisaf_files(tmp_path):
    from datetime import date

    import xarray as xr

    from _osisaf_fixture import write_osisaf

    raw = tmp_path / "raw"
    raw.mkdir()
    for d in (1, 2):
        write_osisaf(raw / f"ice_{d}.nc", date(2024, 12, d))
    out = tmp_path / "sea_ice.nc"
    rc = main(["build-dataset", "--config", CONFIG, "--inputs", str(raw / "*.nc"), "--out", str(out),
               "--resolution-km", "25"])
    assert rc == 0
    with xr.open_dataset(out) as ds:
        assert ds.sizes["time"] == 2 and ds.attrs["execution_mode"] == "real"
    stage = json.loads(out.with_suffix(".stage-result.json").read_text())
    assert stage["status"] == "passed" and len(stage["inputs"]) == 2


def test_build_dataset_stage_records_product_family_and_actual_product(tmp_path):
    """Provenance keeps the family (OSI-401) apart from the product each file reports (OSI-401-b/-d)."""
    from datetime import date

    from _osisaf_fixture import write_osisaf

    raw = tmp_path / "raw"
    raw.mkdir()
    write_osisaf(raw / "ice_b.nc", date(2024, 12, 1))
    write_osisaf(raw / "ice_d.nc", date(2026, 9, 15), layout="osi401d")
    out = tmp_path / "sea_ice.nc"
    rc = main(["build-dataset", "--config", CONFIG, "--inputs", str(raw / "*.nc"), "--out", str(out),
               "--resolution-km", "25"])
    assert rc == 0
    stage = json.loads(out.with_suffix(".stage-result.json").read_text())
    by_name = {Path(r["path"]).name: r for r in stage["inputs"]}
    assert by_name["ice_b.nc"]["source"] == "OSI-401-b" and by_name["ice_b.nc"]["product_version"] == ""
    assert by_name["ice_d.nc"]["source"] == "OSI-401-d" and by_name["ice_d.nc"]["product_version"] == "4.1"
    assert {r["product_family"] for r in stage["inputs"]} == {"OSI-401"}
    assert stage["parameters"]["product_family"] == "OSI-401"
    assert stage["parameters"]["source_product"] == "OSI-401-b,OSI-401-d"


def test_fetch_sea_ice_reports_blocked_or_failed_without_network(tmp_path, monkeypatch):
    import antarctic_routing.ingestion.sea_ice as sea_ice

    def offline(request, dest, timeout=60.0):
        raise ConnectionError("network unavailable")

    monkeypatch.setattr(sea_ice, "fetch_osisaf", offline)
    rc = main(["fetch-sea-ice", "--config", CONFIG, "--start", "2024-12-01", "--end", "2024-12-02",
               "--root", str(tmp_path)])
    assert rc == 1
    summary = json.loads((tmp_path / "sea_ice" / "fetch-summary.json").read_text())
    assert [r["status"] for r in summary["results"]] == ["failed", "failed"]


def test_train_and_evaluate_forecast_on_synthetic_history(tmp_path):
    common = ["--config", CONFIG, "--synthetic-seasons", "2010:2016", "--resolution-km", "50",
              "--history-days", "5", "--lead-days", "3", "--n-val", "1", "--n-test", "1"]
    rc = main(["train-forecast", *common, "--epochs", "2", "--base-channels", "8", "--out", str(tmp_path / "model")])
    assert rc == 0
    assert (tmp_path / "model" / "best.pt").is_file()
    rc = main(["evaluate-forecast", *common, "--weights", str(tmp_path / "model" / "best.pt"),
               "--out", str(tmp_path / "report")])
    assert rc == 0
    report = json.loads((tmp_path / "report" / "forecast_eval.json").read_text())
    assert report["test_seasons"] == [2015] and report["execution_mode"] == "controlled_synthetic"
    assert (tmp_path / "report" / "forecast_skill.png").is_file()
    stage = json.loads((tmp_path / "report" / "stage-result.json").read_text())
    assert stage["stage"] == "forecast_evaluation"


def test_calibrate_forecast_writes_report_calibrator_and_reliability_plot(tmp_path):
    common = ["--config", CONFIG, "--synthetic-seasons", "2010:2016", "--resolution-km", "50",
              "--history-days", "5", "--lead-days", "3", "--n-val", "1", "--n-test", "1"]
    assert main(["train-forecast", *common, "--epochs", "1", "--base-channels", "8",
                 "--out", str(tmp_path / "model")]) == 0
    rc = main(["calibrate-forecast", *common, "--weights", str(tmp_path / "model" / "best.pt"),
               "--members", "8", "--out", str(tmp_path / "cal")])
    assert rc == 0
    report = json.loads((tmp_path / "cal" / "probability_eval.json").read_text())
    assert report["val_seasons"] == [2014] and report["test_seasons"] == [2015]
    assert json.loads((tmp_path / "cal" / "calibrator.json").read_text())["leads"] == [1, 2, 3]
    assert (tmp_path / "cal" / "reliability.png").is_file()


def test_demo_with_iceberg_records_it_in_plan(tmp_path):
    rc = main(["demo", "--config", CONFIG, "--departure", "2027-01-10", "--out", str(tmp_path), *FAST,
               "--iceberg", "A23A:-60.0:-63.0"])
    assert rc == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["icebergs"] == [{"id": "A23A", "lat": -60.0, "lon": -63.0}]
    assert "iceberg" in plan["data_description"].lower()


def test_demo_reads_latest_usnic_positions(tmp_path):
    csv_path = tmp_path / "usnic.csv"
    csv_path.write_text("Iceberg,Length (NM),Width (NM),Latitude,Longitude,Updated\n"
                        "A23A,38,32,60 0S,63 0W,12/01/2026\n"
                        "A23A,38,32,60 30S,62 0W,12/08/2026\n")
    rc = main(["demo", "--config", CONFIG, "--departure", "2027-01-10", "--out", str(tmp_path / "o"), *FAST,
               "--icebergs", str(csv_path)])
    assert rc == 0
    plan = json.loads((tmp_path / "o" / "plan.json").read_text())
    assert plan["icebergs"] == [{"id": "A23A", "lat": -60.5, "lon": -62.0}]


def test_trust_horizon_from_evaluation_report(tmp_path):
    seasons = {str(s): None for s in (2021, 2022, 2023, 2024, 2025)}
    report = {
        "methods": ["unet", "persistence", "climatology", "damped_persistence"], "leads": [1, 2, 3],
        "execution_mode": "controlled_synthetic",
        "by_season": {
            "unet": {s: [0.010, 0.015, 0.030] for s in seasons},
            "persistence": {s: [0.012, 0.020, 0.028] for s in seasons},
            "damped_persistence": {s: [0.011, 0.018, 0.027] for s in seasons},
            "climatology": {s: [0.030, 0.030, 0.030] for s in seasons},
        },
    }
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(report))
    assert main(["trust-horizon", "--report", str(path), "--out", str(tmp_path / "t")]) == 0
    th = json.loads((tmp_path / "t" / "trust_horizon.json").read_text())
    assert th["by_baseline"]["damped_persistence"]["trust_horizon_days"] == 2
    assert th["by_baseline"]["climatology"]["trust_horizon_days"] == 2
    assert th["trust_horizon_days"] == 2
    assert (tmp_path / "t" / "trust_horizon.png").is_file()


def test_plan_window_from_one_forecast_issue(tmp_path):
    common = ["--config", CONFIG, "--synthetic-seasons", "2010:2016", "--resolution-km", "50",
              "--history-days", "5", "--lead-days", "3", "--n-val", "1", "--n-test", "1"]
    assert main(["train-forecast", *common, "--epochs", "1", "--base-channels", "8",
                 "--out", str(tmp_path / "model")]) == 0
    trust = tmp_path / "trust.json"
    trust.write_text(json.dumps({"trust_horizon_days": 2}))
    rc = main(["plan-window", *common, "--weights", str(tmp_path / "model" / "best.pt"),
               "--issue", "2015-12-20", "--window-days", "4", "--members", "80",
               "--trust-report", str(trust), "--out", str(tmp_path / "w")])
    assert rc == 0
    out = json.loads((tmp_path / "w" / "plan_window.json").read_text())
    assert [o["lead_days"] for o in out["options"]] == [0, 1, 2, 3]
    assert out["issue"] == "2015-12-20" and out["trust_horizon_days"] == 2
    assert out["layer_source"][0] == "observed" and "climatology" in out["layer_source"]
    assert all(o["support"] in ("forecast-supported", "climatology-dominated") for o in out["options"])
    assert (tmp_path / "w" / "departure_window.png").is_file()


def test_replay_cli_writes_log_figure_and_summary(tmp_path):
    common = ["--config", CONFIG, "--synthetic-seasons", "2010:2016", "--resolution-km", "50",
              "--history-days", "5", "--lead-days", "3", "--n-val", "1", "--n-test", "1"]
    assert main(["train-forecast", *common, "--epochs", "1", "--base-channels", "8",
                 "--out", str(tmp_path / "model")]) == 0
    rc = main(["replay", *common, "--weights", str(tmp_path / "model" / "best.pt"), "--start", "2015-11-20",
               "--window-days", "4", "--members", "80", "--out", str(tmp_path / "r")])
    assert rc == 0
    summary = json.loads((tmp_path / "r" / "replay.json").read_text())
    assert summary["start"] == "2015-11-20" and "truth" in summary
    assert (tmp_path / "r" / "audit.jsonl").is_file()
    assert (tmp_path / "r" / "replay.png").is_file()


def test_backtest_cli_writes_summary_table_and_figure(tmp_path):
    common = ["--config", CONFIG, "--synthetic-seasons", "2010:2016", "--resolution-km", "50",
              "--history-days", "5", "--lead-days", "3", "--n-val", "1", "--n-test", "1"]
    assert main(["train-forecast", *common, "--epochs", "1", "--base-channels", "8",
                 "--out", str(tmp_path / "model")]) == 0
    rc = main(["backtest", *common, "--weights", str(tmp_path / "model" / "best.pt"),
               "--start-days", "11-18", "12-02", "--window-days", "4", "--members", "80",
               "--out", str(tmp_path / "b")])
    assert rc == 0
    out = json.loads((tmp_path / "b" / "backtest.json").read_text())
    assert out["starts"] == ["2015-11-18", "2015-12-02"]          # test season only
    assert set(out["summary"]) == {"planner", "naive", "ice_edge_buffer"}
    assert (tmp_path / "b" / "backtest.png").is_file()
    assert (tmp_path / "b" / "backtest_voyages.csv").is_file()


def test_sensitivity_cli_writes_grid_and_heatmap(tmp_path):
    rc = main(["sensitivity", "--config", CONFIG, "--departure", "2026-12-20", *FAST,
               "--lambdas", "0", "4", "--ks", "0.5", "0.7", "--out", str(tmp_path)])
    assert rc == 0
    out = json.loads((tmp_path / "sensitivity.json").read_text())
    assert len(out["grid"]) == 4 and out["baseline"]["lambda"] == 4.0
    assert (tmp_path / "sensitivity.png").is_file()


def test_voyage_brief_pdf(tmp_path):
    out = tmp_path / "brief.pdf"
    rc = main(["brief", "--config", CONFIG, "--departure", "2027-01-10", *FAST, "--out", str(out)])
    assert rc == 0
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    assert data.count(b"/Type /Page") - data.count(b"/Type /Pages") >= 2


def test_build_forcing_and_use_it_in_plan_window(tmp_path):
    from test_forcing import write_cmems, write_era5

    era5, cmems = write_era5(tmp_path / "era5.nc"), write_cmems(tmp_path / "cmems.nc")
    forcing = tmp_path / "forcing.nc"
    assert main(["build-forcing", "--config", CONFIG, "--era5", str(era5), "--cmems", str(cmems),
                 "--resolution-km", "50", "--out", str(forcing)]) == 0
    assert forcing.is_file() and forcing.with_suffix(".stage-result.json").is_file()
    common = ["--config", CONFIG, "--synthetic-seasons", "2010:2016", "--resolution-km", "50",
              "--history-days", "5", "--lead-days", "3", "--n-val", "1", "--n-test", "1"]
    assert main(["train-forecast", *common, "--epochs", "1", "--base-channels", "8",
                 "--out", str(tmp_path / "model")]) == 0
    rc = main(["plan-window", *common, "--weights", str(tmp_path / "model" / "best.pt"), "--forcing", str(forcing),
               "--issue", "2015-12-20", "--window-days", "2", "--members", "80",
               "--iceberg", "B1:-60.2:-62.6", "--out", str(tmp_path / "w")])
    assert rc == 0
    out = json.loads((tmp_path / "w" / "plan_window.json").read_text())
    assert out["forcing"] == str(forcing)
    assert out["icebergs"] == [{"id": "B1", "lat": -60.2, "lon": -62.6}]


def test_build_forcing_daily_wind_mode_is_recorded(tmp_path):
    import xarray as xr

    from test_forcing import two_day_hourly, write_hourly_era5

    u, v = two_day_hourly()
    era5 = write_hourly_era5(tmp_path / "era5.nc", u, v)
    out = tmp_path / "forcing.nc"
    assert main(["build-forcing", "--config", CONFIG, "--era5", str(era5), "--resolution-km", "50",
                 "--wind-time-mode", "daily", "--out", str(out)]) == 0
    stage = json.loads(out.with_suffix(".stage-result.json").read_text())
    assert stage["parameters"]["wind_time_mode"] == "daily"
    with xr.open_dataset(out) as ds:
        assert ds["wind_x"].dims == ("time", "y", "x") and ds.attrs["wind_time_mode"] == "daily"


PW_COMMON = ["--config", CONFIG, "--synthetic-seasons", "2010:2016", "--resolution-km", "50",
             "--history-days", "5", "--lead-days", "3", "--n-val", "1", "--n-test", "1"]


@pytest.fixture(scope="module")
def pw_weights(tmp_path_factory):
    out = tmp_path_factory.mktemp("model")
    assert main(["train-forecast", *PW_COMMON, "--epochs", "1", "--base-channels", "8", "--out", str(out)]) == 0
    return out / "best.pt"


def _daily_era5_forcing(tmp_path, start, days):
    """Daily-mode forcing from an hourly ERA5-format file (fixture data, not a real-data claim)."""
    from test_forcing import write_hourly_era5

    era5 = write_hourly_era5(tmp_path / "era5.nc", np.full(24 * days, 8.0), np.full(24 * days, 1.0), start=start)
    out = tmp_path / "forcing.nc"
    assert main(["build-forcing", "--config", CONFIG, "--era5", str(era5), "--resolution-km", "50",
                 "--wind-time-mode", "daily", "--out", str(out)]) == 0
    return out


def test_plan_window_with_daily_forcing_and_mixed_provenance(tmp_path, pw_weights):
    from datetime import datetime

    # issue 2015-12-20 with --window-days 2 needs 2 + horizon (6) = 8 days: Dec 20 .. Dec 27
    forcing = _daily_era5_forcing(tmp_path, datetime(2015, 12, 18), 11)
    rc = main(["plan-window", *PW_COMMON, "--weights", str(pw_weights), "--forcing", str(forcing),
               "--issue", "2015-12-20", "--window-days", "2", "--members", "80",
               "--iceberg", "B1:-60.2:-62.6", "--out", str(tmp_path / "w")])
    assert rc == 0
    fp = json.loads((tmp_path / "w" / "plan_window.json").read_text())["forcing_provenance"]
    assert fp["winds"] == "ERA5 daily" and fp["currents"] == "schematic"
    assert fp["status"] == "mixed" and fp["label"] == "ERA5 daily winds + schematic currents"
    assert fp["wind_dates"] == ["2015-12-20", "2015-12-27"]


def test_plan_window_with_insufficient_daily_forcing_fails_clearly(tmp_path, pw_weights):
    from datetime import datetime

    forcing = _daily_era5_forcing(tmp_path, datetime(2015, 12, 18), 8)      # ends Dec 25
    with pytest.raises(ValueError, match=r"missing 2 date\(s\): 2015-12-26, 2015-12-27"):
        main(["plan-window", *PW_COMMON, "--weights", str(pw_weights), "--forcing", str(forcing),
              "--issue", "2015-12-20", "--window-days", "2", "--members", "80",
              "--iceberg", "B1:-60.2:-62.6", "--out", str(tmp_path / "w")])
    assert not (tmp_path / "w" / "plan_window.json").exists()


def test_plan_window_drifts_only_bergs_on_the_routing_grid(tmp_path, pw_weights, monkeypatch):
    from antarctic_routing.iceberg import drift
    from antarctic_routing.ingestion.icebergs import iceberg_provenance, read_iceberg_positions

    seen = []
    real = drift.add_iceberg_hazard

    def spy(world, bergs, **kw):
        seen.append([b[0] for b in bergs])
        return real(world, bergs, **kw)

    monkeypatch.setattr(drift, "add_iceberg_hazard", spy)
    # real 2023-11-09 USNIC rows: two on the polar grid, A74A south of it, B22A far away
    csv = tmp_path / "AntarcticIcebergs_20231109.csv"
    csv.write_bytes(("\ufeffIceberg,Length (NM),Width (NM),Latitude,Longitude,Remarks,Last Update\r\n"
                     "A23A,40,32,-63.13,-51.67,,11/09/2023\r\n"
                     "A74A,30,18,-71.51,-55.01,Calved from north Brunt,11/09/2023\r\n"
                     "A76B,20,7,-65.31,-58.29,,11/09/2023\r\n"
                     "B22A,32,22,-74.0,-108.0,,11/09/2023\r\n").encode("utf-8"))
    out = tmp_path / "w"
    rc = main(["plan-window", *PW_COMMON, "--weights", str(pw_weights), "--issue", "2015-12-20",
               "--window-days", "2", "--members", "80", "--icebergs", str(csv), "--out", str(out)])
    assert rc == 0
    assert seen == [["A23A", "A76B"]]                       # excluded before drift, input order kept
    pw = json.loads((out / "plan_window.json").read_text())
    assert [b["id"] for b in pw["icebergs"]] == ["A23A", "A76B"]
    assert [b["id"] for b in pw["icebergs_outside_grid"]] == ["A74A", "B22A"]
    assert pw["iceberg_source"] == json.loads(json.dumps(iceberg_provenance(csv, read_iceberg_positions(csv)),
                                                         default=str))   # whole file, flags included
    assert pw["iceberg_source"]["n_rows"] == 4
    assert pw["iceberg_drift"] == {"beta": 1.0, "alpha_range": [0.01, 0.03], "spread_factor": 1.0,
                                   "radius_m": 10_000.0, "drifted": True}


def test_plan_window_passes_calibrated_drift_parameters(tmp_path, pw_weights, monkeypatch):
    from antarctic_routing.iceberg import drift

    seen = []
    real = drift.add_iceberg_hazard

    def spy(world, bergs, **kw):
        seen.append({k: kw[k] for k in ("beta", "alpha_range", "spread_factor")})
        return real(world, bergs, **kw)

    monkeypatch.setattr(drift, "add_iceberg_hazard", spy)
    out = tmp_path / "w"
    rc = main(["plan-window", *PW_COMMON, "--weights", str(pw_weights), "--issue", "2015-12-20",
               "--window-days", "2", "--members", "40", "--iceberg", "A23A:-63.13:-51.67",
               "--drift-beta", "0.1", "--drift-alpha-scale", "0.1", "--drift-spread-factor", "0.6053",
               "--out", str(out)])
    assert rc == 0
    assert seen[0]["beta"] == 0.1 and seen[0]["spread_factor"] == 0.6053
    assert seen[0]["alpha_range"] == pytest.approx((0.001, 0.003))
    pw = json.loads((out / "plan_window.json").read_text())
    assert pw["iceberg_drift"]["beta"] == 0.1 and pw["iceberg_drift"]["spread_factor"] == 0.6053
    assert len(pw["iceberg_hazard"]["max_member_fraction_per_layer"]) == len(pw["layer_source"])
    if pw["selected"]:
        r = pw["selected_route"]
        assert r["departure"] == pw["selected"] and len(r["coordinates_lat_lon"]) == len(r["cells_row_col"]) >= 2
        assert r["evaluation"]["p_breach_upper"] == next(o for o in pw["options"]
                                                         if o["departure"] == pw["selected"])["p_breach_upper"]
        assert "iceberg_presence_on_route" in r
    else:
        assert pw["selected_route"] is None


def _daily_cmems_forcing(tmp_path, start, days):
    """Daily-mode current forcing from a CMEMS-format file (fixture data, not a real-data claim)."""
    from test_forcing import write_daily_cmems

    cm = write_daily_cmems(tmp_path / "cmems.nc", [0.2] * days, [0.05] * days, start=start)
    out = tmp_path / "currents.nc"
    assert main(["build-forcing", "--config", CONFIG, "--cmems", str(cm), "--resolution-km", "50",
                 "--current-time-mode", "daily", "--out", str(out)]) == 0
    return out


def test_plan_window_with_separate_daily_wind_and_current_forcing_is_real(tmp_path, pw_weights):
    from datetime import datetime

    winds = _daily_era5_forcing(tmp_path, datetime(2015, 12, 18), 11)
    currents = _daily_cmems_forcing(tmp_path, datetime(2015, 12, 19), 10)
    rc = main(["plan-window", *PW_COMMON, "--weights", str(pw_weights), "--wind-forcing", str(winds),
               "--current-forcing", str(currents), "--issue", "2015-12-20", "--window-days", "2",
               "--members", "80", "--iceberg", "B1:-60.2:-62.6", "--out", str(tmp_path / "w")])
    assert rc == 0
    pw = json.loads((tmp_path / "w" / "plan_window.json").read_text())
    fp = pw["forcing_provenance"]
    assert fp["winds"] == "ERA5 daily" and fp["currents"] == "CMEMS daily" and fp["status"] == "real"
    assert fp["label"] == "ERA5 daily winds + CMEMS daily currents"
    assert fp["wind_dates"] == fp["current_dates"] == ["2015-12-20", "2015-12-27"]
    assert fp["sources"]["winds"]["forcing_file"] == str(winds) and fp["sources"]["winds"]["time_mode"] == "daily"
    assert fp["sources"]["currents"]["forcing_file"] == str(currents)
    assert fp["sources"]["currents"]["product"] == "CMEMS" and fp["sources"]["currents"]["source_sha256"]
    assert pw["forcing"] == {"wind_forcing": str(winds), "current_forcing": str(currents)}


def test_plan_window_with_missing_current_dates_fails_clearly(tmp_path, pw_weights):
    from datetime import datetime

    winds = _daily_era5_forcing(tmp_path, datetime(2015, 12, 18), 11)
    currents = _daily_cmems_forcing(tmp_path, datetime(2015, 12, 21), 10)       # starts a day late
    with pytest.raises(ValueError, match=r"daily current forcing .*missing 1 date\(s\): 2015-12-20"):
        main(["plan-window", *PW_COMMON, "--weights", str(pw_weights), "--wind-forcing", str(winds),
              "--current-forcing", str(currents), "--issue", "2015-12-20", "--window-days", "2",
              "--members", "80", "--out", str(tmp_path / "w")])
    assert not (tmp_path / "w" / "plan_window.json").exists()


def test_forcing_arguments_are_checked(tmp_path, pw_weights):
    from datetime import datetime

    winds = _daily_era5_forcing(tmp_path, datetime(2015, 12, 18), 11)
    base = ["plan-window", *PW_COMMON, "--weights", str(pw_weights), "--issue", "2015-12-20",
            "--window-days", "2", "--members", "80", "--out", str(tmp_path / "w")]
    with pytest.raises(ValueError, match="either --forcing or"):
        main([*base, "--forcing", str(winds), "--current-forcing", str(winds)])
    with pytest.raises(ValueError, match="has no current_x"):
        main([*base, "--current-forcing", str(winds)])                 # a wind-only file given as currents


def test_fetch_forcing_without_credentials_is_blocked(tmp_path, monkeypatch):
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_USERNAME", raising=False)
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_PASSWORD", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    rc = main(["fetch-forcing", "--config", CONFIG, "--start", "2024-12-01", "--end", "2024-12-03",
               "--root", str(tmp_path / "raw")])
    assert rc == 1
    summary = json.loads((tmp_path / "raw" / "forcing-summary.json").read_text())
    assert {r["status"] for r in summary["results"]} <= {"blocked", "failed"}


def test_forecast_horizon_defaults_to_config_lead_days_end_to_end(tmp_path):
    """No --lead-days: config forecast.lead_days drives the samples, the model output and the evaluation."""
    import pytest
    import torch

    from antarctic_routing.config import load_config
    from antarctic_routing.forecasting.train import load_model

    lead = load_config(CONFIG).forecast.lead_days
    assert lead == 21
    common = ["--config", CONFIG, "--synthetic-seasons", "2010:2016", "--resolution-km", "50",
              "--history-days", "5", "--n-val", "1", "--n-test", "1"]
    assert main(["train-forecast", *common, "--epochs", "1", "--base-channels", "8",
                 "--out", str(tmp_path / "model")]) == 0
    model, meta = load_model(tmp_path / "model" / "best.pt")
    assert meta["train_config"]["lead_days"] == lead and meta["out_channels"] == lead
    with torch.no_grad():
        out = model(torch.zeros(1, meta["in_channels"], *meta["grid_shape"]))
    assert out.shape[1] == lead

    assert main(["evaluate-forecast", *common, "--weights", str(tmp_path / "model" / "best.pt"),
                 "--out", str(tmp_path / "report")]) == 0
    report = json.loads((tmp_path / "report" / "forecast_eval.json").read_text())
    assert report["leads"] == list(range(1, lead + 1))
    assert report["n_samples"] == 120 - 5 - lead + 1  # one test season, windows built with the 21-day horizon
    with pytest.raises(SystemExit, match="does not match the checkpoint"):
        main(["evaluate-forecast", *common, "--lead-days", "7", "--weights", str(tmp_path / "model" / "best.pt"),
              "--out", str(tmp_path / "report7")])


def test_evaluate_refuses_checkpoint_trained_on_another_resolution(tmp_path):
    import pytest

    common = ["--config", CONFIG, "--synthetic-seasons", "2010:2016", "--history-days", "5", "--lead-days", "3",
              "--n-val", "1", "--n-test", "1"]
    assert main(["train-forecast", *common, "--resolution-km", "50", "--epochs", "1", "--base-channels", "8",
                 "--out", str(tmp_path / "model")]) == 0
    with pytest.raises(SystemExit, match="resolution 100 km != trained 50 km"):
        main(["evaluate-forecast", *common, "--resolution-km", "100",
              "--weights", str(tmp_path / "model" / "best.pt"), "--out", str(tmp_path / "report")])


def test_plan_window_require_real_forcing_blocks_schematic_forcing(tmp_path, pw_weights, capsys):
    rc = main(["plan-window", *PW_COMMON, "--weights", str(pw_weights), "--issue", "2015-12-20",
               "--window-days", "2", "--members", "80", "--require-real-forcing", "--out", str(tmp_path / "w")])
    assert rc == 2
    assert "BLOCKED: forcing is schematic" in capsys.readouterr().err
    assert not (tmp_path / "w" / "plan_window.json").exists()


def test_replay_and_backtest_require_real_forcing_block_too(tmp_path, pw_weights):
    assert main(["replay", *PW_COMMON, "--weights", str(pw_weights), "--start", "2015-12-20", "--window-days", "2",
                 "--require-real-forcing", "--out", str(tmp_path / "r")]) == 2
    assert main(["backtest", *PW_COMMON, "--weights", str(pw_weights), "--require-real-forcing",
                 "--out", str(tmp_path / "b")]) == 2
    assert not (tmp_path / "r" / "replay.json").exists() and not (tmp_path / "b" / "backtest.json").exists()


def test_plan_window_with_real_forcing_passes_the_gate_and_exports_maps(tmp_path, pw_weights, capsys):
    from datetime import datetime

    winds = _daily_era5_forcing(tmp_path, datetime(2015, 12, 18), 11)
    currents = _daily_cmems_forcing(tmp_path, datetime(2015, 12, 19), 10)
    rc = main(["plan-window", *PW_COMMON, "--weights", str(pw_weights), "--wind-forcing", str(winds),
               "--current-forcing", str(currents), "--issue", "2015-12-20", "--window-days", "2",
               "--members", "80", "--iceberg", "B1:-60.2:-62.6", "--require-real-forcing", "--export-maps",
               "--out", str(tmp_path / "w")])
    assert rc == 0 and "WARNING" not in capsys.readouterr().err
    pw = json.loads((tmp_path / "w" / "plan_window.json").read_text())
    assert "warnings" not in pw
    maps = json.loads((tmp_path / "w" / "forecast_maps.json").read_text())
    assert [la["source"] for la in maps["layers"]] == pw["layer_source"]
    assert maps["layers"][0]["date"] == "2015-12-20" and maps["observed"]["date"] == "2015-12-20"
    assert maps["n_members"] == 80 and maps["tau"] == 0.15
    assert all(la["p_berg_pct"] is not None for la in maps["layers"])
    assert [t["id"] for t in maps["iceberg_tracks"]] == ["B1"]
    assert len(maps["iceberg_tracks"][0]["daily_mean"]) == len(maps["layers"])
    land = maps["land"]
    assert all(v is None for v, lnd in zip(maps["layers"][3]["p_ice_ge_limit_pct"], land, strict=True) if lnd)


def test_plan_window_on_real_sea_ice_with_schematic_forcing_warns(tmp_path, pw_weights, monkeypatch, capsys):
    from antarctic_routing import cli

    original = cli._forecast_data

    def relabelled(args):          # exercise the real-data path; the fixture data itself is synthetic
        cfg, ds, months, split = original(args)
        ds.attrs["execution_mode"] = "real"
        return cfg, ds, months, split

    monkeypatch.setattr(cli, "_forecast_data", relabelled)
    rc = main(["plan-window", *PW_COMMON, "--weights", str(pw_weights), "--issue", "2015-12-20",
               "--window-days", "2", "--members", "80", "--out", str(tmp_path / "w")])
    assert rc == 0
    assert "WARNING: real sea-ice data with forcing is schematic" in capsys.readouterr().err
    pw = json.loads((tmp_path / "w" / "plan_window.json").read_text())
    assert pw["forcing_provenance"]["status"] == "schematic" and "schematic" in pw["warnings"][0]
    assert not (tmp_path / "w" / "forecast_maps.json").exists()
