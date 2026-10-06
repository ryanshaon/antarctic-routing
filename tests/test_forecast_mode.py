"""Forecast / hackathon-estimate mode: dates after the real archive, planned from a labelled analogue start.

These tests use a synthetic stand-in archive (the real archive and PyTorch are kept outside Git); what is tested
is the mode switch, the analogue and iceberg-snapshot rules, the date mapping and the labels.
tests/test_historical_real.py runs forecast mode on the real archive.
"""

import json
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from antarctic_routing.api import historical as hist_api
from antarctic_routing.api.main import create_app
from antarctic_routing.common.provenance import sha256_file
from antarctic_routing.config import load_config
from antarctic_routing.forecast_mode import (
    BANNERS,
    ForecastUnavailable,
    forecast_usnic,
    is_forecast_date,
    resolve,
    shift_dates,
)
from antarctic_routing.forecasting.scenarios import ForecastContext
from antarctic_routing.historical import DailyForcing, HistoricalArchive, HistoricalPlanner, UsnicArchive
from antarctic_routing.preprocessing.grid import PolarGrid
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


def _archive(tmp: Path, res_km: int = 100) -> HistoricalArchive:
    """Seasons 2021-22 and 2022-23 (archive ends 2023-02-28), one archive list and one later 'forecast' list."""
    grid = PolarGrid.from_domain(CFG.domain, res_km)
    ds = synthetic_history(CFG, grid, range(2021, 2023), seed=3)
    t = ds["time"].values.astype("datetime64[D]")
    zeros = np.zeros((t.size, *ds["land_mask"].shape))
    spec = json.loads(SPEC.read_text())
    spec["inputs"]["icebergs"] = [_list(tmp, date(2022, 11, 25), [("A23A", -60.0, -60.0, "11/25/2022")])]
    spec["inputs"]["forecast_icebergs"] = [_list(tmp, date(2023, 11, 20), [("A23A", -61.0, -61.0, "11/20/2023"),
                                                                          ("B09", -70.0, 100.0, "11/20/2023"),
                                                                          ("C99", -62.0, -62.0, "12/05/2023")])]
    return HistoricalArchive(spec, tmp, ds, DailyForcing("winds", "ERA5", t, zeros, zeros, ()),
                             DailyForcing("currents", "CMEMS", t, zeros, zeros, ()),
                             UsnicArchive(spec["inputs"]["icebergs"], tmp), CFG.project.season_months)


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    return _archive(tmp_path_factory.mktemp("fc"))


# ------------------------------------------------------------------ mode switch and analogue rules
def test_dates_after_the_archive_are_forecast_dates(archive):
    assert archive.days[-1] == date(2023, 2, 28)
    assert not is_forecast_date(archive, date(2023, 2, 28)) and not is_forecast_date(archive, date(2022, 12, 1))
    assert is_forecast_date(archive, date(2023, 11, 1)) and is_forecast_date(archive, date(2023, 12, 1))
    assert not is_forecast_date(archive, date(2023, 3, 1))           # off season: historical mode refuses it


def test_the_analogue_is_the_same_calendar_day_in_the_latest_covering_season(archive):
    s = resolve(archive, date(2023, 12, 1), 7, forecast_usnic(archive))
    assert (s.analogue, s.offset_days, s.requested_day(date(2022, 12, 3))) == \
        (date(2022, 12, 1), 365, date(2023, 12, 3))
    assert (s.snapshot_date, s.snapshot_file, s.max_age_days) == (date(2023, 11, 20),
                                                                  "AntarcticIcebergs_20231120.csv", 120)


def test_dates_forecast_mode_cannot_support_are_refused_with_the_reason(archive):
    usnic = forecast_usnic(archive)
    for day, why in ((date(2022, 12, 1), "inside the real archive"),
                     (date(2023, 6, 15), "outside the Nov-Feb season"),
                     (date(2023, 11, 13), "no archive season can start"),            # history too short every year
                     (date(2025, 1, 10), "no recent official iceberg list")):       # latest list is 2023-11-20
        with pytest.raises(ForecastUnavailable, match=why):
            resolve(archive, day, 7, usnic)


def test_the_analogue_ignores_the_archive_iceberg_lists_only(archive):
    """The analogue day needs the sea-ice, ERA5 and CMEMS rules; its own USNIC age is irrelevant because the
    forecast-mode snapshot is the latest list before the requested date."""
    day = date(2023, 1, 20)                                    # archive list 2022-11-25 is 56 days old
    assert any("days old" in p for p in archive.problems(day, 7))
    assert archive.problems(day, 7, require_usnic=False) == []


def test_forecast_lists_are_verified_and_never_invented(tmp_path):
    a = _archive(tmp_path)
    (tmp_path / "AntarcticIcebergs_20231120.csv").write_text(HEADER)          # changed after pinning
    with pytest.raises(ForecastUnavailable, match="SHA-256"):
        forecast_usnic(a)
    (tmp_path / "AntarcticIcebergs_20231120.csv").unlink()
    with pytest.raises(ForecastUnavailable, match="not found"):
        forecast_usnic(a)


def test_snapshot_positions_never_use_reports_dated_after_the_requested_day(archive):
    usnic = forecast_usnic(archive)
    _, _, _, bergs = usnic.positions(date(2023, 12, 1), 120)
    assert [b[0] for b in bergs] == ["A23A", "B09"]                         # C99 is dated 2023-12-05


# ------------------------------------------------------------------ date mapping
def test_shift_dates_moves_engine_dates_and_keeps_real_world_dates():
    obj = {"date": "2022-12-01", "eta_utc": "2022-12-02T13:09Z", "text": "depart 2022-12-01, not 2022-02-30",
           "n": 20221201, "id": "x20221201", "list": ["2022-12-31"], "icebergs": {"list_date": "2022-11-25"},
           "provenance": {"observed_through": "2022-12-01"}}
    out = shift_dates(obj, 365, frozenset({"icebergs", "provenance"}))
    assert out == {"date": "2023-12-01", "eta_utc": "2023-12-02T13:09Z", "text": "depart 2023-12-01, not 2022-02-30",
                   "n": 20221201, "id": "x20221201", "list": ["2023-12-31"], "icebergs": {"list_date": "2022-11-25"},
                   "provenance": {"observed_through": "2022-12-01"}}
    assert obj["date"] == "2022-12-01"                                       # input is not modified


# ------------------------------------------------------------------ API (stand-in planner)
def persistence(inputs, idx):
    return np.repeat(inputs[:, L - 1: L], 21, axis=1)


class StandInPlanner:
    """The real planner's interface on the stand-in archive (persistence instead of the U-Net)."""

    def __init__(self, archive):
        self.archive, self.lead_days, self.meta = archive, 21, None
        self.ctx = ForecastContext.build(archive.ds, persistence, L, 21, CFG.project.season_months, [2021],
                                         bank_size=20, seed=0)

    drift_kwargs = HistoricalPlanner.drift_kwargs

    world = HistoricalPlanner.world                           # historical mode, unchanged

    def observed_bergs(self, day):
        return self.archive.usnic.positions(day, 14)[3]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fcapi")
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
        "issue": "2023-12-01"}


def _walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, f"{path}/{i}")
    else:
        yield path, o


@pytest.fixture(scope="module")
def plan(client):
    r = client.post("/real/plan", json=BODY)
    assert r.status_code == 200, r.text
    return r


def test_a_future_date_is_planned_in_forecast_mode_with_its_labels(plan):
    j = plan.json()
    m = j["metadata"]
    assert (m["mode"], m["execution_mode"], m["data_status"]) == ("forecast", "modelled", "forecast_estimate")
    assert m["banners"] == BANNERS and m["hindsight_forcing"] is False and m["proxy_forcing"] is True
    assert m["requested_date"] == m["issue_date"] == "2023-12-01" and "not a certified" in m["forecast_disclosure"]
    assert "provenance" not in m and m["label"] == "Forecast / hackathon estimate"
    assert j["departure"]["options"][0]["departure"] == "2023-12-01"
    assert j["route"]["departure_date"] >= "2023-12-01" and [d["date"] for d in j["daily"]][0] >= "2023-12-01"


def test_forecast_metadata_records_the_actual_data_and_proxy_dates(plan):
    f = plan.json()["metadata"]["forecast"]
    assert f["forecast_mode"] is True and f["archive_last_date"] == "2023-02-28"
    dm = f["date_mapping"]
    assert (dm["analogue_start_date"], dm["analogue_season"], dm["offset_days"]) == ("2022-12-01", "2022-23", 365)
    assert dm["engine_days"] == ["2022-12-01", "2022-12-09"] and dm["shown_as"] == ["2023-12-01", "2023-12-09"]
    assert f["sea_ice"]["status"] == "proxy_analogue" and f["sea_ice"]["observed_window_used"] == \
        ["2022-11-18", "2022-12-01"]
    assert f["sea_ice_forecast"]["status"] == "forecast"
    assert f["forcing"]["status"] == "proxy_analogue_reanalysis"
    assert f["forcing"]["winds"]["dates_used"] == f["forcing"]["currents"]["dates_used"] == \
        ["2022-12-01", "2022-12-09"]
    b = f["icebergs"]
    assert (b["snapshot_date"], b["file"], b["age_days_at_requested_date"]) == \
        ("2023-11-20", "AntarcticIcebergs_20231120.csv", 11)
    assert b["n_source_bergs"] == 2 and b["drift_model"]["beta"] == 0.1 and b["drift_model"]["members"] == 200
    assert f["engine_provenance"]["issue"] == "2022-12-01"            # real dates of the analogue inputs


def test_forecast_limitations_state_the_snapshot_age_rule_not_the_historical_one(plan):
    f = plan.json()["metadata"]["forecast"]
    engine = f["engine_provenance"]["limitations"]
    assert len(f["limitations"]) == len(engine)
    assert not any("up to 14 days old" in t for t in f["limitations"])
    assert any(t.startswith(f"The iceberg snapshot may be up to {f['icebergs']['max_age_days']} days old; its actual "
                            "age is shown in the result") for t in f["limitations"])
    assert [t for t in f["limitations"] if "iceberg snapshot" not in t] == \
        [t for t in engine if not t.startswith("Iceberg positions come from")]


def test_no_observation_is_claimed_for_the_requested_dates(plan):
    j = plan.json()
    late = re.compile(r"2023-(0[3-9]|1[0-2])-|202[4-9]-")             # after the archive's last day
    for path, v in _walk(j):
        if isinstance(v, str) and late.search(v) and "/icebergs" not in path:
            assert "observ" not in path.lower(), path
        assert v != "observed", path                                     # proxy layers are labelled proxy
    assert j["icebergs"]["list_date"] == "2023-11-20"                     # the real list date, never shifted
    assert j["metadata"]["layer_source"][0] == "proxy_analogue_observed"


def test_forecast_plan_and_window_are_deterministic(client, plan):
    again = client.post("/real/plan", json=BODY)
    assert again.content == plan.content


def test_historical_dates_stay_in_historical_mode(client):
    r = client.post("/real/plan", json={**BODY, "issue": "2022-12-01"})
    assert r.status_code == 200, r.text
    m = r.json()["metadata"]
    assert (m["mode"], m["execution_mode"]) == ("historical", "real") and "forecast" not in m
    assert m["provenance"]["issue"] == "2022-12-01"


def test_unsupported_future_dates_are_refused_never_substituted(client):
    # Any later date falls back to a historical seasonal analogue; this stand-in has no off-season inputs, so the
    # analogue is blocked and the date is refused with both reasons (tests/test_seasonal_analogue.py serves it).
    for issue in ("2023-06-15", "2025-01-10"):
        r = client.post("/real/plan", json={**BODY, "issue": issue})
        assert r.status_code == 422 and r.json()["detail"]["status"] == "forecast_unavailable", issue
        assert "no historical seasonal analogue: blocked" in r.json()["detail"]["reason"], issue
        assert client.post("/real/simulate?wait=true", json={**BODY, "issue": issue}).status_code == 422
    r = client.post("/real/plan", json={**BODY, "issue": "2022-06-15"})          # past and outside the archive
    assert r.status_code == 422 and r.json()["detail"]["status"] == "out_of_coverage"


def test_dates_endpoint_lists_the_forecast_range(client):
    d = client.get("/real/historical/dates?origin=drake_passage&destination=bransfield_strait").json()
    f = d["forecast"]
    assert f["available"] and f["mode"] == "forecast" and f["banners"] == BANNERS
    assert "2023-12-01" in f["dates"] and f["first"] > d["window_dates"][-1]
    assert all(date.fromisoformat(x) > date(2023, 2, 28) for x in f["dates"])


def test_forecast_simulation_runs_on_the_proxy_and_says_so(client, plan):
    r = client.post("/real/simulate?wait=true", json=BODY).json()
    assert r["status"] == "done", r.get("error")
    s = r["result"]
    m = s["metadata"]
    assert m["mode"] == "forecast" and m["execution_mode"] == "modelled" and "analogue" in m["simulation_note"]
    assert s["frames"][0]["date"] >= "2023-12-01"
    assert all(f["observed"]["sea_ice_source"] == "proxy_analogue_observed" for f in s["frames"])
    assert all(f["map"]["source"] != "observed" for f in s["frames"] if f.get("map"))
    assert "forecast_mode" in s["summary"]["notes"]
    datetime.fromisoformat(s["summary"]["departure_utc"].replace("Z", "+00:00"))
    again = client.post("/real/simulate?wait=true", json=BODY).json()["result"]
    assert again == s
