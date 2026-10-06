"""Historical seasonal analogue: any date after the archive that the forecast pathway cannot serve.

These tests use a stand-in archive and stand-in off-season files in the real file formats (all labelled
controlled_synthetic, like the archive they extend); what is tested is the pathway choice, the analogue and member
rules, the iceberg-source rule, the labels and that every date used is recorded. tests/test_historical_real.py runs
the seasonal analogue on the real data when it is present.
"""

import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from antarctic_routing.api import historical as hist_api
from antarctic_routing.api.main import create_app
from antarctic_routing.common.provenance import sha256_file
from antarctic_routing.config import load_config
from antarctic_routing.forecast_mode import forecast_usnic
from antarctic_routing.forecasting.scenarios import ForecastContext
from antarctic_routing.historical import (
    DailyForcing,
    HistoricalArchive,
    HistoricalPlanner,
    HistoricalUnavailable,
    UsnicArchive,
    drift_kwargs,
)
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.seasonal_analogue import (
    AnalogueData,
    AnaloguePlanner,
    AnalogueUnavailable,
    resolve,
)
from antarctic_routing.synthetic import synthetic_history

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "config.yaml"
SPEC = REPO / "config" / "real_historical.json"
CFG = load_config(CONFIG)
HEADER = "Iceberg,Length (NM),Width (NM),Latitude,Longitude,Remarks,Last Update\n"
L = 5


def _list(tmp: Path, day: date, rows: list[tuple]) -> dict:
    p = tmp / f"AntarcticIcebergs_{day:%Y%m%d}.csv"
    p.write_text(HEADER + "".join(f"{i},20,5,{lat},{lon},,{d}\n" for i, lat, lon, d in rows))
    return {"path": p.name, "sha256": sha256_file(p), "date": day.isoformat()}


def _days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=k) for k in range((last - first).days + 1)]


def _offseason_ice(tmp: Path, ref: xr.Dataset, year: int) -> dict:
    """Mar-Oct of ``year`` on the archive's grid: concentration rising into winter (a stand-in in the file format)."""
    days = _days(date(year, 3, 1), date(year, 10, 31))
    land = ref["land_mask"].values
    base = np.clip(-(ref["lat"].values + 60.0) / 10.0, 0.0, 1.0)            # more ice further south
    conc = np.stack([np.clip(base * (0.3 + 0.6 * np.sin(np.pi * k / len(days))) + 0.01 * (year - 2020), 0, 1)
                     for k in range(len(days))]).astype(np.float32)
    conc[:, land] = np.nan
    ds = xr.Dataset({"ice_concentration": (("time", "y", "x"), conc), "land_mask": (("y", "x"), land)},
                    coords={"time": np.array(days, dtype="datetime64[ns]"), "y": ref["y"].values,
                            "x": ref["x"].values},
                    attrs={"execution_mode": ref.attrs["execution_mode"], "resolution_m": ref.attrs["resolution_m"]})
    p = tmp / f"ice_{year}.nc"
    ds.to_netcdf(p)
    return {"path": p.name, "sha256": sha256_file(p), "first": str(days[0]), "last": str(days[-1]),
            "source": "stand-in off-season sea ice", "season": year}


def _offseason_forcing(tmp: Path, ref: xr.Dataset, kind: str, year: int) -> dict:
    days = _days(date(year, 3, 1), date(year, 10, 31))
    shape = (len(days), *ref["land_mask"].shape)
    names = ("wind_x", "wind_y") if kind == "winds" else ("current_x", "current_y")
    ds = xr.Dataset({names[0]: (("time", "y", "x"), np.full(shape, 0.1)),
                     names[1]: (("time", "y", "x"), np.zeros(shape))},
                    coords={"time": np.array(days, dtype="datetime64[ns]"), "y": ref["y"].values,
                            "x": ref["x"].values},
                    attrs={f"{'wind' if kind == 'winds' else 'current'}_time_mode": "daily"})
    p = tmp / f"{kind}_{year}.nc"
    ds.to_netcdf(p)
    return {"path": p.name, "sha256": sha256_file(p), "first": str(days[0]), "last": str(days[-1])}


def _archive(tmp: Path, res_km: int = 100, offseason: bool = True) -> HistoricalArchive:
    """Seasons 2020-21 .. 2022-23 (archive ends 2023-02-28); off-season sea ice 2021 and 2022, forcing and one
    iceberg lists in Mar-Oct 2022; one 'forecast' list on 2023-11-20."""
    grid = PolarGrid.from_domain(CFG.domain, res_km)
    ds = synthetic_history(CFG, grid, range(2020, 2023), seed=3)
    t = ds["time"].values.astype("datetime64[D]")
    zeros = np.zeros((t.size, *ds["land_mask"].shape))
    spec = json.loads(SPEC.read_text())
    spec["inputs"]["icebergs"] = [_list(tmp, date(2022, 11, 25), [("A23A", -60.0, -60.0, "11/25/2022")])]
    spec["inputs"]["forecast_icebergs"] = [_list(tmp, date(2023, 11, 20), [("A23A", -61.0, -61.0, "11/20/2023")])]
    if offseason:
        spec["inputs"]["analogue_sea_ice"] = [_offseason_ice(tmp, ds, y) for y in (2021, 2022)]
        spec["inputs"]["analogue_wind_forcing"] = [_offseason_forcing(tmp, ds, "winds", 2022)]
        spec["inputs"]["analogue_current_forcing"] = [_offseason_forcing(tmp, ds, "currents", 2022)]
        spec["inputs"]["analogue_icebergs"] = [_list(tmp, date(2022, 3, 1), [("A23A", -60.2, -60.2, "03/01/2022")]),
                                               _list(tmp, date(2022, 8, 10), [("A23A", -60.5, -60.5, "08/10/2022"),
                                                                            ("B09", -70.0, 100.0, "08/10/2022")])]
    else:
        for k in ("analogue_sea_ice", "analogue_wind_forcing", "analogue_current_forcing", "analogue_icebergs"):
            spec["inputs"].pop(k, None)
    return HistoricalArchive(spec, tmp, ds, DailyForcing("winds", "ERA5", t, zeros, zeros, ()),
                             DailyForcing("currents", "CMEMS", t, zeros, zeros, ()),
                             UsnicArchive(spec["inputs"]["icebergs"], tmp), CFG.project.season_months)


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    return _archive(tmp_path_factory.mktemp("an"))


@pytest.fixture(scope="module")
def data(archive):
    return AnalogueData.load(archive)


# ------------------------------------------------------------------ data and rules
def test_off_season_inputs_join_the_archive_into_year_round_observations(data, archive):
    days = data.days
    assert date(2022, 8, 14) in days and date(2021, 6, 1) in days and date(2022, 12, 1) in days
    assert days == sorted(days) and days[-1] == archive.days[-1]
    assert data.winds.missing(date(2022, 3, 1), 245) == [] and data.currents.missing(date(2022, 3, 1), 245) == []
    assert data.winds.missing(date(2021, 8, 14), 1) == ["2021-08-14"]          # forcing only for the analogue year


def test_missing_off_season_inputs_are_a_blocked_state_never_a_fallback(tmp_path):
    with pytest.raises(HistoricalUnavailable) as e:
        AnalogueData.load(_archive(tmp_path, offseason=False))
    assert e.value.status == "blocked"


def test_a_changed_off_season_file_fails_its_checksum(tmp_path):
    a = _archive(tmp_path)
    (tmp_path / a.spec["inputs"]["analogue_icebergs"][0]["path"]).write_text("tampered")
    with pytest.raises(HistoricalUnavailable) as e:
        AnalogueData.load(a)
    assert e.value.status == "failed" and "SHA-256" in e.value.reason


@pytest.mark.parametrize("requested", [date(2027, 8, 14), date(2030, 8, 14), date(2035, 8, 14)])
def test_any_future_august_date_gets_the_latest_august_with_data(data, requested):
    s = resolve(data, requested, 9, None)
    assert s.analogue == date(2022, 8, 14) and s.requested_day(s.analogue) == requested
    assert s.member_years == (2021,) and s.shifts == tuple(range(-7, 8))
    assert s.iceberg_kind == "analogue_year"


def test_a_recent_official_list_is_used_when_it_is_at_most_120_days_old(data, archive):
    s = resolve(data, date(2024, 3, 15), 9, forecast_usnic(archive))       # 2023-11-20 list: 116 days
    assert s.analogue == date(2022, 3, 15) and s.iceberg_kind == "recent_official"
    old = resolve(data, date(2024, 8, 12), 9, forecast_usnic(archive))     # 266 days: the analogue year's list
    assert (old.analogue, old.iceberg_kind) == (date(2022, 8, 12), "analogue_year")
    assert old.iceberg_usnic.list_for(old.analogue, old.iceberg_max_age_days)[0] == date(2022, 8, 10)


def test_a_date_no_year_can_serve_is_refused_with_the_reason(data):
    with pytest.raises(AnalogueUnavailable, match="no year with real winds"):
        resolve(data, date(2027, 6, 1), 300, None)          # 300 days from June: no year has data for all


def test_world_members_are_real_observations_of_other_years_never_the_analogue_year(data, archive):
    s = resolve(data, date(2027, 8, 14), 9, None)
    planner = AnaloguePlanner(data, s, drift_kwargs(archive.params))
    world, snap = planner.world(s.analogue, 9)
    assert world.n_scenarios == 200 and world.layer_source == ["analogue_observed"] * 9
    pool = planner.pool(s.analogue, 9)
    assert len(pool) == 15 and {y for y, _, _ in pool} == {2021}
    land = world.land
    starts = [st for _, _, st in pool]
    for k in range(5):                                      # each member is one observed 9-day sequence
        m = world.conc[k][:, ~land]
        assert any(np.allclose(m, data.conc(st, 9)[:, ~land]) for st in starts)
    assert not any(np.allclose(world.conc[0][:, ~land], data.conc(s.analogue, 9)[:, ~land]) for _ in [0])
    assert snap.list_date == date(2022, 8, 10) and [b[0] for b in snap.bergs] == ["A23A"]


# ------------------------------------------------------------------ API
def persistence(inputs, idx):
    return np.repeat(inputs[:, L - 1: L], 21, axis=1)


class StandInPlanner:
    """The historical planner's interface on the stand-in archive (persistence instead of the U-Net)."""

    def __init__(self, archive):
        self.archive, self.lead_days, self.meta = archive, 21, None
        self.ctx = ForecastContext.build(archive.ds, persistence, L, 21, CFG.project.season_months, [2020, 2021],
                                         bank_size=20, seed=0)

    drift_kwargs = HistoricalPlanner.drift_kwargs
    world = HistoricalPlanner.world

    def observed_bergs(self, day):
        return self.archive.usnic.positions(day, 14)[3]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("anapi")
    archive = _archive(tmp, 25)
    planner = StandInPlanner(archive)
    mp = pytest.MonkeyPatch()
    mp.setenv("ANTROUTE_DATA_ROOT", str(tmp))
    mp.setattr(hist_api.HistoricalArchive, "load", classmethod(lambda cls, *a, **k: archive))
    mp.setattr(hist_api, "HistoricalPlanner", lambda a: planner)
    with TestClient(create_app(CONFIG)) as c:
        yield c
    mp.undo()


BODY = {"origin": {"preset": "drake_passage"}, "destination": {"preset": "bransfield_strait"}, "window_days": 3,
        "issue": "2027-08-14"}


@pytest.fixture(scope="module")
def plan(client):
    r = client.post("/real/plan", json=BODY)
    assert r.status_code == 200, r.text
    return r


def test_an_off_season_future_date_is_a_labelled_seasonal_analogue_estimate(plan):
    j = plan.json()
    m = j["metadata"]
    assert (m["mode"], m["execution_mode"], m["data_status"]) == ("forecast", "modelled", "forecast_estimate")
    assert m["banners"][0] == "FORECAST / HACKATHON ESTIMATE" and "seasonal analogue" in m["banners"][1]
    assert m["requested_date"] == "2027-08-14" and "not a meteorological" in m["forecast_disclosure"]
    f = m["forecast"]
    assert (f["pathway"], f["pathway_label"]) == ("seasonal_analogue", "Historical seasonal analogue")
    assert "forecast mode is unavailable" in f["pathway_reason"] or "outside the Nov-Feb" in f["pathway_reason"]
    assert j["departure"]["options"][0]["departure"] == "2027-08-14"
    assert [d["date"] for d in j["daily"]][0] >= "2027-08-14"


def test_the_analogue_records_every_historical_date_it_used(plan):
    f = plan.json()["metadata"]["forecast"]
    assert f["analogue_date"] == "2022-08-14"
    dm = f["date_mapping"]
    assert dm["engine_days"][0] == "2022-08-14" and dm["shown_as"][0] == "2027-08-14"
    assert dm["offset_days"] == (date(2027, 8, 14) - date(2022, 8, 14)).days
    assert f["forcing"]["status"] == "historical_reanalysis_analogue"
    assert f["forcing"]["winds"]["dates_used"][0] == f["forcing"]["currents"]["dates_used"][0] == "2022-08-14"
    sea = f["sea_ice"]
    assert sea["status"] == "historical_analogue_ensemble" and sea["member_years"] == [2021]
    assert sea["distinct_sequences"] == 15 and len(sea["member_windows"]) == 15
    assert {w["first"] for w in sea["member_windows"] if w["shift_days"] == 0} == {"2021-08-14"}
    b = f["icebergs"]
    assert (b["kind"], b["snapshot_date"], b["file"]) == ("analogue_year", "2022-08-10",
                                                          "AntarcticIcebergs_20220810.csv")
    assert b["n_source_bergs"] == 2 and b["n_in_grid"] == 1
    assert (b["age_days_at_analogue_date"], b["max_age_days"]) == (4, 14)          # 2022-08-10 -> 2022-08-14


def test_confidence_is_exposed_not_used_to_refuse(plan):
    c = plan.json()["metadata"]["forecast"]["confidence"]
    assert c["level"] == "very low" and c["member_years"] == [2021]          # one analogue year only
    assert any("Not a forecast" in t for t in c["caveats"])
    assert any("years after the latest real observation" in t for t in c["caveats"])
    assert any("analogue date" in t and "icebergs" in t for t in c["caveats"])


def test_analogue_layers_are_labelled_and_no_observation_is_claimed_for_requested_dates(plan):
    j = plan.json()
    assert j["metadata"]["layer_source"] == ["analogue_observed"] * j["metadata"]["scenario_days"]
    late = re.compile(r"202[7-9]-|20[3-9]\d-")
    for path, v in _walk(j):
        if isinstance(v, str) and late.search(v):
            assert "observ" not in path.lower() or "observations_for_requested_dates" in path, path
    assert j["icebergs"]["list_date"] == "2022-08-10"                     # a real list date, never shifted


def _walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, f"{path}/{i}")
    else:
        yield path, o


def test_analogue_plans_are_deterministic(client, plan):
    assert client.post("/real/plan", json=BODY).content == plan.content


def test_the_forecast_pathway_still_serves_its_own_dates(client):
    r = client.post("/real/plan", json={**BODY, "issue": "2023-12-01"})
    assert r.status_code == 200, r.text
    f = r.json()["metadata"]["forecast"]
    assert f["pathway"] == "proxy_forecast" and f["date_mapping"]["analogue_start_date"] == "2022-12-01"


@pytest.mark.parametrize("issue", ["2023-06-15", "2025-01-10", "2028-03-03", "2030-08-14"])
def test_every_future_date_is_accepted(client, issue):
    r = client.post("/real/plan", json={**BODY, "issue": issue})
    assert r.status_code == 200, (issue, r.text)
    assert r.json()["metadata"]["forecast"]["pathway"] == "seasonal_analogue"


def test_past_dates_outside_the_archive_stay_refused(client):
    r = client.post("/real/plan", json={**BODY, "issue": "2022-06-15"})
    assert r.status_code == 422 and r.json()["detail"]["status"] == "out_of_coverage"


def test_dates_endpoint_says_any_later_date_is_accepted(client):
    d = client.get("/real/historical/dates?origin=drake_passage&destination=bransfield_strait").json()
    e = d["estimate"]
    assert e["any_date_after"] == "2023-02-28" and e["analogue"]["available"] is True
    assert set(e["pathways"]) == {"proxy_forecast", "seasonal_analogue"}


def test_analogue_simulation_sails_the_analogue_years_ice_and_says_so(client, plan):
    r = client.post("/real/simulate?wait=true", json=BODY).json()
    assert r["status"] == "done", r.get("error")
    s = r["result"]
    if s["status"] == "not_simulated":                                     # stand-in winter ice can block a route
        assert s["reason"]
        return
    m = s["metadata"]
    assert m["mode"] == "forecast" and "Seasonal-analogue voyage simulation" in m["simulation_note"]
    assert s["frames"][0]["date"] >= "2027-08-14"
    assert all(f["observed"]["sea_ice_source"] == "proxy_analogue_observed" for f in s["frames"])
    assert "2022" in s["summary"]["notes"]["forecast_mode"]
