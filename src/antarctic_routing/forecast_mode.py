"""Forecast / hackathon-estimate mode: plans for dates after the real archive (e.g. a future departure).

There are no observations or operational forecasts for such dates in the runtime data, so the unchanged real-data
engine runs from an **analogue start** and every proxy input is labelled:

* **sea ice** - the real OSI SAF observations of the same calendar day in the latest archive season that covers
  the run (the *analogue date*), forecast with the frozen U-Net. This is a proxy starting state, not an
  observation of the requested year;
* **winds and currents** - ERA5/CMEMS reanalysis of the analogue days: proxy forcing, not a forecast;
* **icebergs** - the latest official USNIC list on or before the requested date. No newer position exists, so
  each berg is held at its last reported position until the requested date, then drifted with the calibrated
  ensemble (the same drift model and parameters as the historical mode).

The engine works on the analogue dates; :func:`shift_dates` maps them one-to-one onto the requested dates by a
fixed day offset. Nothing is fitted, tuned or synthesised; a date the rules cannot support raises
:class:`ForecastUnavailable` with the reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from antarctic_routing.historical import (
    HistoricalArchive,
    HistoricalPlanner,
    HistoricalUnavailable,
    IcebergSnapshot,
    OutOfCoverage,
    UsnicArchive,
    _verify,
    season_label,
)
from antarctic_routing.preprocessing.climatology import season_of

MODE = "forecast"
DATA_STATUS = "forecast_estimate"
EXECUTION_MODE = "modelled"           # real inputs and frozen models, proxy timing: not observed data
BANNERS = ["FORECAST / HACKATHON ESTIMATE", "Proxy sea ice and forcing from an analogue season",
           "Research estimate, not certified navigation"]


class ForecastUnavailable(ValueError):
    """The requested date cannot be planned in forecast mode (the reason says why)."""


def is_forecast_date(archive: HistoricalArchive, day: date) -> bool:
    """In-season dates after the last day of the real archive go to forecast mode; every other date stays in
    historical mode (and an uncovered one is refused there, unchanged)."""
    return day > archive.days[-1] and season_of(day, archive.season_months) is not None


def _shift_years(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year + years)
    except ValueError:            # 29 February -> 28 February
        return day.replace(year=day.year + years, day=28)


@dataclass(frozen=True)
class ForecastSetup:
    """How one requested date is planned: the analogue start and the iceberg snapshot."""

    requested: date
    analogue: date
    n_days: int
    usnic: UsnicArchive               # archive lists + the forecast-mode snapshots (all official USNIC)
    snapshot_date: date
    snapshot_file: str
    snapshot_sha256: str
    max_age_days: int

    @property
    def offset_days(self) -> int:
        return (self.requested - self.analogue).days

    def requested_day(self, engine_day: date) -> date:
        return engine_day + timedelta(days=self.offset_days)


def forecast_usnic(archive: HistoricalArchive) -> UsnicArchive:
    """The archive's USNIC lists plus the forecast-mode snapshots, verified (forecast mode only)."""
    recs = archive.spec["inputs"].get("forecast_icebergs", [])
    try:
        for rec in recs:
            _verify(archive.root, rec, "forecast_icebergs")
    except HistoricalUnavailable as exc:
        raise ForecastUnavailable(f"forecast mode is unavailable: {exc.reason}") from None
    return UsnicArchive(list(archive.spec["inputs"]["icebergs"]) + list(recs), archive.root)


def resolve(archive: HistoricalArchive, requested: date, n_days: int, usnic: UsnicArchive) -> ForecastSetup:
    """The analogue start and iceberg snapshot for ``requested`` (raises :class:`ForecastUnavailable`)."""
    if season_of(requested, archive.season_months) is None:
        raise ForecastUnavailable(
            f"{requested} is outside the Nov-Feb season that the archive and the frozen sea-ice model cover")
    if not is_forecast_date(archive, requested):
        raise ForecastUnavailable(f"{requested} is inside the real archive; it is planned in historical mode")
    analogue, tried = None, []
    for k in range(1, requested.year - archive.days[0].year + 2):
        cand = _shift_years(requested, -k)
        if cand > archive.days[-1]:
            continue
        problems = archive.problems(cand, n_days, require_usnic=False)
        if not problems:
            analogue = cand
            break
        tried.append(f"{cand}: {problems[0]}")
        if len(tried) == 3:
            break
    if analogue is None:
        raise ForecastUnavailable(f"no archive season can start a {n_days}-day forecast on the calendar day of "
                                  f"{requested} (" + "; ".join(tried or ["no earlier season"]) + ")")
    max_age = int(archive.spec.get("forecast_mode", {}).get("usnic_max_age_days", 120))
    try:
        d, path, sha = usnic.list_for(requested, max_age)
    except OutOfCoverage as exc:
        raise ForecastUnavailable(f"no recent official iceberg list for {requested}: {exc}") from None
    return ForecastSetup(requested, analogue, n_days, usnic, d, Path(path).name, sha, max_age)


class ForecastPlanner:
    """The historical planner's interface on the analogue dates, with the forecast-mode iceberg snapshot.

    Everything except the iceberg source is the unchanged :class:`HistoricalPlanner` (frozen U-Net, residual
    bank, seed, calibrated drift parameters)."""

    def __init__(self, planner: HistoricalPlanner, setup: ForecastSetup) -> None:
        self.base, self.setup = planner, setup
        self.archive, self.ctx, self.meta, self.lead_days = planner.archive, planner.ctx, \
            getattr(planner, "meta", None), planner.lead_days

    def drift_kwargs(self) -> dict:
        return self.base.drift_kwargs()

    def problems(self, engine_day: date, n_days: int) -> list[str]:
        return self.archive.problems(engine_day, n_days, require_usnic=False)

    def snapshot(self, engine_day: date, grid) -> IcebergSnapshot:
        s = self.setup
        return s.usnic.snapshot(s.requested_day(engine_day), grid, s.max_age_days)

    def world(self, engine_day: date, n_days: int):
        from antarctic_routing.iceberg.drift import add_iceberg_hazard

        p = self.archive.params
        problems = self.problems(engine_day, n_days)
        if problems:
            raise OutOfCoverage(f"no forecast-mode run from analogue day {engine_day}: " + "; ".join(problems))
        world = self.ctx.scenarios(engine_day, n_days, p["members"], np.random.default_rng(p["seed"]))
        snap = self.snapshot(engine_day, world.grid)
        if snap.bergs:
            world = add_iceberg_hazard(world, snap.bergs, rng=np.random.default_rng(p["seed"]),
                                       radius_m=p["berg_radius_km"] * 1e3, **self.drift_kwargs())
        return world, snap

    def observed_bergs(self, engine_day: date) -> list[tuple[str, float, float]]:
        """Last reported positions on or before the requested-timeline day (held; no newer report exists)."""
        s = self.setup
        return s.usnic.positions(s.requested_day(engine_day), s.max_age_days)[3]

    def usnic_positions(self, engine_day: date):
        """(list date, file, age in days on the requested timeline, positions) for the simulation display."""
        s = self.setup
        day = s.requested_day(engine_day)
        d, path, _, bergs = s.usnic.positions(day, s.max_age_days)
        return d, path, (day - d).days, bergs


# --------------------------------------------------------------------------- dates and labels
_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")


def shift_dates(obj, offset_days: int, skip: frozenset = frozenset()):
    """Copy of ``obj`` with every ISO date (also inside datetimes and text) moved by ``offset_days``.

    Keys in ``skip`` (real-world dates such as an iceberg list's date, or the provenance of the analogue
    inputs) are copied unchanged."""
    def move(m: re.Match) -> str:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return m.group(0)
        return (d + timedelta(days=offset_days)).isoformat()

    def walk(o):
        if isinstance(o, dict):
            return {k: (v if k in skip else walk(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, str):
            return _DATE.sub(move, o)
        return o

    return walk(obj)


def forecast_limitations(limitations: list[str], max_age_days: int) -> list[str]:
    """The engine's limitations, with the historical iceberg-age line restated for a held forecast snapshot.

    Wording only: the snapshot rule itself (``max_age_days``) and every calculation are unchanged."""
    line = (f"The iceberg snapshot may be up to {max_age_days} days old; its actual age is shown in the result. "
            "Positions are the last official USNIC report, held until the requested date and then drifted. "
            "Only large tracked icebergs are included.")
    return [line if t.startswith("Iceberg positions come from") else t for t in limitations]


def forecast_metadata(archive: HistoricalArchive, setup: ForecastSetup, snap: IcebergSnapshot, engine_provenance:
                      dict, n_scenarios: int) -> dict:
    """What a forecast-mode result was computed from: every input with its status and the dates actually used."""
    a, p, s = archive, archive.params, setup
    first, last = s.analogue, s.analogue + timedelta(days=s.n_days - 1)
    hist = s.analogue - timedelta(days=a.history_days - 1)
    sea = a.sea_ice_file(s.analogue) or {}
    fm = a.spec.get("forecast_mode", {})
    files = {f.kind: f.files_for(s.analogue, s.n_days) for f in (a.winds, a.currents)}
    return {
        "forecast_mode": True,
        "requested_date": s.requested.isoformat(),
        "requested_date_relation": "after_archive",
        "archive_last_date": a.days[-1].isoformat(),
        "observations_for_requested_dates": "none: no sea-ice, wind, current or iceberg observation dated on or "
                                            "after the requested date is used",
        "date_mapping": {
            "rule": "analogue = the requested calendar day in the latest archive season that covers the run; "
                    "engine day k is shown as requested date + k",
            "analogue_start_date": s.analogue.isoformat(),
            "analogue_season": season_label(season_of(s.analogue, a.season_months)),
            "offset_days": s.offset_days,
            "engine_days": [first.isoformat(), last.isoformat()],
            "shown_as": [s.requested.isoformat(), (s.requested + timedelta(days=s.n_days - 1)).isoformat()],
        },
        "sea_ice": {
            "status": "proxy_analogue",
            "starting_state": "real OSI SAF observations of the analogue days, NOT observations of the requested "
                              "year",
            "observed_window_used": [hist.isoformat(), s.analogue.isoformat()],
            "source": sea.get("source"), "file": sea.get("file"), "sha256": sea.get("sha256"),
        },
        "sea_ice_forecast": {"status": "forecast", "model": engine_provenance["forecast_model"],
                             "note": "frozen U-Net and residual-bank scenarios from the proxy starting state"},
        "forcing": {
            "status": "proxy_analogue_reanalysis",
            "note": "ERA5/CMEMS reanalysis of the analogue days stands in for the requested days; it is NOT a "
                    "forecast and NOT a measurement of the requested year",
            "winds": {"product": "ERA5 daily reanalysis", "dates_used": [first.isoformat(), last.isoformat()],
                      "files": files["winds"]},
            "currents": {"product": "CMEMS GLOBAL reanalysis daily", "dates_used": [first.isoformat(),
                                                                                    last.isoformat()],
                         "files": files["currents"]},
        },
        "icebergs": {
            "status": "observed_snapshot_held" if snap.bergs else "observed_snapshot_none_in_grid",
            "source": "U.S. National Ice Center (USNIC), official",
            "snapshot_date": s.snapshot_date.isoformat(), "file": s.snapshot_file, "sha256": s.snapshot_sha256,
            "age_days_at_requested_date": (s.requested - s.snapshot_date).days,
            "max_age_days": s.max_age_days,
            "n_source_bergs": len(snap.bergs) + len(snap.outside_grid), "n_in_grid": len(snap.bergs),
            "ids_in_grid": [b[0] for b in snap.bergs],
            "drift_model": {"model": "calibrated ensemble drift (historical-mode parameters, unchanged)",
                            "beta": p["drift_beta"], "alpha_scale": p["drift_alpha_scale"],
                            "spread_factor": p["drift_spread_factor"], "members": n_scenarios,
                            "forcing": "proxy analogue reanalysis (above)"},
            "note": "positions are the last official report, held unchanged until the requested date (no newer "
                    "report exists), then drifted forward",
        },
        "label": fm.get("label", "Forecast / hackathon estimate"),
        "disclosure": fm.get("disclosure"), "method": fm.get("method"),
        "limitations": forecast_limitations(engine_provenance.get("limitations", []), s.max_age_days),
        "engine_provenance": engine_provenance,
    }
