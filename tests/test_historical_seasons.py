"""Additional real seasons in the Real Historical Data archive (``sea_ice_additional``).

A later season is appended to the frozen sea-ice file in memory. These tests use synthetic stand-in files and
need no real data; tests/test_historical_real.py runs the real 2024-25 season when the archive is present.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from antarctic_routing.common.provenance import sha256_file
from antarctic_routing.config import load_config
from antarctic_routing.forecasting.scenarios import ForecastContext
from antarctic_routing.historical import (
    DailyForcing,
    HistoricalArchive,
    HistoricalUnavailable,
    UsnicArchive,
    provenance,
)
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.synthetic import synthetic_history

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "config.yaml"
SPEC = REPO / "config" / "real_historical.json"
CFG = load_config(CONFIG)
MONTHS = CFG.project.season_months
L = 5


def _real(ds: xr.Dataset, label: str) -> xr.Dataset:
    return ds.assign_attrs(execution_mode="real", source_product=label)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """A 'frozen' file with seasons 2020-21 and 2021-22, and a separate later 2022-23 file (stand-ins)."""
    tmp = tmp_path_factory.mktemp("seasons")
    grid = PolarGrid.from_domain(CFG.domain, 100)
    full = synthetic_history(CFG, grid, range(2020, 2023), seed=1)
    t = full["time"].values.astype("datetime64[D]")
    cut = np.datetime64("2022-11-01")
    frozen, later = _real(full.isel(time=t < cut), "frozen"), _real(full.isel(time=t >= cut), "later")
    frozen.to_netcdf(tmp / "frozen.nc")
    later.to_netcdf(tmp / "later.nc")
    inp = {"sea_ice": {"path": "frozen.nc", "sha256": sha256_file(tmp / "frozen.nc")},
           "sea_ice_additional": [{"path": "later.nc", "sha256": sha256_file(tmp / "later.nc"), "season": 2022,
                                   "source": "stand-in later season", "provenance": "later-provenance.json"}]}
    return tmp, inp, full


def test_pinned_spec_registers_2024_25_with_real_forcing_and_icebergs():
    spec = json.loads(SPEC.read_text())
    inp = spec["inputs"]
    (extra,) = inp["sea_ice_additional"]
    assert extra["path"] == "sea-ice-2024-25/processed/sea_ice_25km_2024_25.nc"
    assert extra["sha256"].startswith("a4ea4918830f09f5") and extra["season"] == 2024
    assert "OSI-430-a v3.0" in extra["source"] and extra["provenance"].endswith("PROVENANCE-sea-ice-2024-25.json")
    assert inp["sea_ice"]["path"] == "processed/sea_ice_25km.nc"                     # the frozen file is unchanged
    winds = [w for w in inp["wind_forcing"] if w["first"] >= "2024-11-01"]
    currents = [c for c in inp["current_forcing"] if c["first"] >= "2024-11-01"]
    assert [(w["first"], w["last"]) for w in winds] == [("2024-11-01", "2024-12-31"), ("2025-01-01", "2025-02-28")]
    assert [(c["first"], c["last"]) for c in currents] == [("2024-11-01", "2025-02-28")]
    assert all("era5" in w["path"] for w in winds) and all("cmems" in c["path"] for c in currents)
    bergs = [b["date"] for b in inp["icebergs"] if b["date"] >= "2024-11-01"]
    assert len(bergs) == 17 and bergs[0] == "2024-11-07" and bergs[-1] == "2025-02-28"
    for split in spec["splits"].values():
        assert split["independent_evaluation"] == [2024]
        assert 2024 not in split["train"] + split["validation"] + split["test"]


def test_additional_season_is_appended_without_touching_the_frozen_data(files):
    tmp, inp, full = files
    frozen = xr.load_dataset(tmp / "frozen.nc")
    ds, recs = HistoricalArchive._join_sea_ice(frozen, inp, tmp)
    assert ds.sizes["time"] == full.sizes["time"]
    np.testing.assert_array_equal(ds["ice_concentration"].values, full["ice_concentration"].values)
    n = frozen.sizes["time"]
    np.testing.assert_array_equal(ds["ice_concentration"].values[:n], frozen["ice_concentration"].values)
    assert ds.attrs == frozen.attrs and (ds["land_mask"].values == frozen["land_mask"].values).all()
    assert [(r["file"], r["first"], r["last"]) for r in recs] == [
        ("frozen.nc", "2020-11-01", "2022-02-28"), ("later.nc", "2022-11-01", "2023-02-28")]
    assert recs[1]["source"] == "stand-in later season" and recs[1]["provenance"] == "later-provenance.json"
    assert HistoricalArchive._join_sea_ice(frozen, {"sea_ice": inp["sea_ice"]}, tmp)[0] is frozen


@pytest.mark.parametrize("change, match", [
    (lambda d: d.assign_attrs(execution_mode="controlled_synthetic"), "not real data"),
    (lambda d: d.assign_coords(x=d["x"] + 1.0), "not on the frozen grid"),
    (lambda d: d.assign(land_mask=~d["land_mask"]), "different land mask"),
    (lambda d: d.assign_coords(time=d["time"] - np.timedelta64(400, "D")), "overlaps"),
])
def test_an_additional_season_that_does_not_match_fails(files, tmp_path, change, match):
    tmp, inp, _ = files
    change(xr.load_dataset(tmp / "later.nc")).to_netcdf(tmp_path / "bad.nc")
    (tmp_path / "frozen.nc").write_bytes((tmp / "frozen.nc").read_bytes())
    bad = {**inp, "sea_ice_additional": [{**inp["sea_ice_additional"][0], "path": "bad.nc"}]}
    with pytest.raises(HistoricalUnavailable, match=match) as e:
        HistoricalArchive._join_sea_ice(xr.load_dataset(tmp / "frozen.nc"), bad, tmp_path)
    assert e.value.status == "failed"


def test_load_verifies_the_additional_season_like_every_other_input(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    spec = json.loads(SPEC.read_text())
    for name, body in (("sea_ice", b"ice"), ("checkpoint", b"model"),
                       ("drift_selection", json.dumps({"selected_params": {"beta": 0.1, "alpha_scale": 0.1}})),
                       ("spread_calibration", json.dumps({"selected_model": {"c": 0.6053, "q": 0.0}}))):
        p = root / name
        p.write_bytes(body if isinstance(body, bytes) else body.encode())
        spec["inputs"][name] = {"path": name, "sha256": sha256_file(p)}
    for name in ("wind_forcing", "current_forcing", "icebergs"):
        spec["inputs"][name] = []
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(HistoricalUnavailable, match="sea_ice_additional not found") as e:
        HistoricalArchive.load(root, spec_path, MONTHS)
    assert e.value.status == "blocked"
    (root / "sea-ice-2024-25" / "processed").mkdir(parents=True)
    (root / spec["inputs"]["sea_ice_additional"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(HistoricalUnavailable, match="sea_ice_additional .* has SHA-256") as e:
        HistoricalArchive.load(root, spec_path, MONTHS)
    assert e.value.status == "failed"


def test_frozen_seasons_forecast_identically_after_the_join(files):
    """Climatology, residual bank and anomalies come from the training seasons only, so a forecast issued in
    an existing season is the same with or without the appended season."""
    tmp, inp, _ = files
    frozen = xr.load_dataset(tmp / "frozen.nc")
    joined, _ = HistoricalArchive._join_sea_ice(frozen, inp, tmp)

    def persistence(inputs, idx):
        return np.repeat(inputs[:, L - 1: L], 21, axis=1)

    a = ForecastContext.build(frozen, persistence, L, 21, MONTHS, [2020], bank_size=20, seed=42)
    b = ForecastContext.build(joined, persistence, L, 21, MONTHS, [2020], bank_size=20, seed=42)
    np.testing.assert_array_equal(a.bank, b.bank)
    assert a.bank_seasons == b.bank_seasons and a.anomalies.keys() == b.anomalies.keys()
    wa = a.scenarios(date(2021, 12, 1), 6, 20, np.random.default_rng(42))
    wb = b.scenarios(date(2021, 12, 1), 6, 20, np.random.default_rng(42))
    np.testing.assert_array_equal(wa.conc, wb.conc)
    wc = b.scenarios(date(2022, 12, 1), 6, 20, np.random.default_rng(42))      # the appended season forecasts
    assert np.isfinite(wc.conc).any()


def test_the_appended_season_is_discoverable_labelled_and_traced(files):
    tmp, inp, _ = files
    joined, recs = HistoricalArchive._join_sea_ice(xr.load_dataset(tmp / "frozen.nc"), inp, tmp)
    t = joined["time"].values.astype("datetime64[D]")
    zeros = np.zeros((t.size, *joined["land_mask"].shape))
    p = tmp / "AntarcticIcebergs_20221201.csv"
    p.write_text("Iceberg,Length (NM),Width (NM),Latitude,Longitude,Remarks,Last Update\n"
                 "A23A,20,5,-60.0,-60.0,,12/01/2022\n")
    spec = json.loads(SPEC.read_text())
    spec["splits"] = {k: {**v, "independent_evaluation": [2022]} for k, v in spec["splits"].items()}
    spec["splits"]["forecast_model"]["test"] = [2021]
    spec["splits"]["iceberg_drift"]["test"] = [2021]
    a = HistoricalArchive(spec, tmp, joined, DailyForcing("winds", "ERA5", t, zeros, zeros, ()),
                          DailyForcing("currents", "CMEMS", t, zeros, zeros, ()),
                          UsnicArchive([{"path": p.name, "sha256": sha256_file(p), "date": "2022-12-01"}], tmp),
                          MONTHS, sea_ice_files=recs)
    cov = a.coverage(7)
    assert [c["season"] for c in cov] == ["2022-23"]            # only the season with an iceberg list in range
    s = cov[0]
    assert (s["first"], s["forecast_model"], s["iceberg_drift"]) == ("2022-12-01", "independent_evaluation",
                                                                     "independent_evaluation")
    assert s["out_of_sample"] and s["independent_evaluation"] and len(s["evaluation_notes"]) == 2
    assert all("nothing fitted" in n for n in s["evaluation_notes"])
    assert not a.season_info(2021)["independent_evaluation"]
    assert a.problems(date(2022, 12, 1), 7) == []
    assert "not in the sea-ice archive" in a.problems(date(2022, 3, 15), 7)[0]
    prov = provenance(a, CFG, date(2022, 12, 1), 7, None)
    assert prov["sea_ice"] == {"source": "stand-in later season", "file": "later.nc", "sha256": recs[1]["sha256"],
                               "observed_through": "2022-12-01", "provenance": "later-provenance.json"}
    assert prov["season"]["season"] == "2022-23" and prov["execution_mode"] == "real"
    old = provenance(a, CFG, date(2021, 12, 1), 7, None)["sea_ice"]
    assert old["file"] == "sea_ice_25km.nc" and old["sha256"] == spec["inputs"]["sea_ice"]["sha256"]
