"""Historical seasonal analogue estimate: a route estimate for any future date the forecast mode cannot serve.

Forecast mode (:mod:`antarctic_routing.forecast_mode`) needs an in-season analogue start for the frozen U-Net and an
official iceberg list at most 120 days old, so it covers only a few months after the latest list. Every other date
after the real archive (another month, another year, 2030) is planned here from **real historical observations of
the same calendar period**. Nothing is forecast, fitted or synthesised:

* **sea ice** - an analogue ensemble: the real OSI SAF observations of the same calendar days in the most recent
  earlier years (default 7), each also shifted by -7..+7 days, so members differ by year and by a week of timing.
  No model is run, so the frozen U-Net (trained on Nov-Feb only) is never applied outside its season;
* **winds and currents** - the ERA5/CMEMS reanalysis of the *analogue date* (the requested calendar day in the most
  recent year with complete data), the same proxy forcing role as in forecast mode;
* **icebergs** - the latest official USNIC list when it is at most 120 days older than the requested date (held,
  then drifted, as in forecast mode); otherwise the official list of the analogue date (that year's icebergs,
  drifted with the calibrated ensemble), disclosed as such.

The analogue year itself is left out of the sea-ice members, so the voyage simulation can sail through its real
observed ice without the plan having seen it. Off-season inputs (Mar-Oct sea ice, forcing and iceberg lists) are
listed with their SHA-256 under ``inputs.analogue_*`` in ``config/real_historical.json`` and verified before use;
they are kept apart from the frozen archive, so historical mode, forecast mode and the frozen demo are unchanged.
A date no analogue can serve raises :class:`AnalogueUnavailable` with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from antarctic_routing.forecasting.scenarios import ForecastContext, grid_of
from antarctic_routing.historical import (
    DailyForcing,
    HistoricalArchive,
    HistoricalUnavailable,
    IcebergSnapshot,
    OutOfCoverage,
    UsnicArchive,
    _verify,
    season_label,
)
from antarctic_routing.preprocessing.climatology import season_of
from antarctic_routing.synthetic import ScenarioSet

PATHWAY = "seasonal_analogue"
LABEL = "Historical seasonal analogue"
YEAR_ROUND = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]   # "season" = July-June, so every month is contiguous


class AnalogueUnavailable(ValueError):
    """No historical seasonal analogue can serve the date (the reason says why)."""


def _shift_years(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year + years)
    except ValueError:            # 29 February -> 28 February
        return day.replace(year=day.year + years, day=28)


def _merge(a: DailyForcing, b: DailyForcing) -> DailyForcing:
    """Two daily forcing records joined by date (they must not overlap)."""
    days = np.concatenate([a.times, b.times])
    order = np.argsort(days, kind="stable")
    days = days[order]
    if np.any(np.diff(days.astype(np.int64)) == 0):
        raise HistoricalUnavailable("failed", f"analogue {a.kind} files overlap the archive's dates")
    return DailyForcing(a.kind, a.product, days, np.concatenate([a.x, b.x])[order],
                        np.concatenate([a.y, b.y])[order], a.files + b.files)


# --------------------------------------------------------------------------- data
@dataclass
class AnalogueData:
    """Year-round real observations: the archive's Nov-Feb seasons plus the verified off-season inputs."""

    archive: HistoricalArchive
    ds: xr.Dataset                       # observed sea ice, every available day, sorted (land_mask of the archive)
    winds: DailyForcing
    currents: DailyForcing
    usnic: UsnicArchive                  # archive + off-season official lists (the analogue-year icebergs)
    files: dict
    settings: dict
    _index: dict[date, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {np.datetime64(t, "D").astype(object): i for i, t in enumerate(self.ds["time"].values)}

    @classmethod
    def load(cls, archive: HistoricalArchive) -> AnalogueData:
        spec, root = archive.spec, archive.root
        inp = spec["inputs"]
        settings = spec.get("seasonal_analogue")
        names = ("analogue_sea_ice", "analogue_wind_forcing", "analogue_current_forcing", "analogue_icebergs")
        if not settings or not all(inp.get(n) for n in names):
            raise HistoricalUnavailable("blocked", "no off-season inputs are listed for the seasonal analogue")
        for n in names:
            for rec in inp[n]:
                _verify(root, rec, n)
        ref = archive.ds
        keep = ref["time"].values >= np.datetime64(settings.get("sea_ice_from", "2016-11-01"))
        parts = [ref["ice_concentration"].isel(time=keep).astype(np.float32)]
        for rec in inp["analogue_sea_ice"]:
            new = xr.load_dataset(root / rec["path"])
            name = Path(rec["path"]).name
            if new.attrs.get("execution_mode") != ref.attrs.get("execution_mode", "real"):     # never mix kinds
                raise HistoricalUnavailable("failed", f"the sea-ice file {name} is not "
                                                      f"{ref.attrs.get('execution_mode', 'real')} data")
            if not (np.array_equal(new["x"], ref["x"]) and np.array_equal(new["y"], ref["y"])):
                raise HistoricalUnavailable("failed", f"the sea-ice file {name} is not on the frozen grid")
            if not (new["land_mask"].values == ref["land_mask"].values).all():
                raise HistoricalUnavailable("failed", f"the sea-ice file {name} has a different land mask")
            parts.append(new["ice_concentration"].astype(np.float32))
        conc = xr.concat(parts, dim="time").sortby("time")
        t = conc["time"].values.astype("datetime64[D]")
        if np.any(np.diff(t.astype(np.int64)) == 0):
            raise HistoricalUnavailable("failed", "the off-season sea-ice files overlap the archive's dates")
        ds = xr.Dataset({"ice_concentration": conc, "land_mask": ref["land_mask"]}, attrs=dict(ref.attrs))
        shape = ref["land_mask"].shape
        land = ref["land_mask"].values.astype(bool)
        try:
            winds = _merge(archive.winds, DailyForcing.load(inp["analogue_wind_forcing"], root, "winds", shape))
            cur = DailyForcing.load(inp["analogue_current_forcing"], root, "currents", shape)
        except ValueError as exc:
            raise HistoricalUnavailable("failed", str(exc)) from None
        cur = DailyForcing(cur.kind, cur.product, cur.times, np.where(land, 0.0, cur.x),   # as the archive does
                           np.where(land, 0.0, cur.y), cur.files)
        currents = _merge(archive.currents, cur)
        usnic = UsnicArchive(list(inp["icebergs"]) + list(inp["analogue_icebergs"]), root)
        files = {"sea_ice": [{"file": Path(r["path"]).name, "sha256": r["sha256"], "first": r["first"],
                              "last": r["last"], "source": r["source"]} for r in inp["analogue_sea_ice"]]}
        return cls(archive, ds, winds, currents, usnic, files, settings)

    # ------------------------------------------------------------ coverage
    @property
    def days(self) -> list[date]:
        return list(self._index)

    def has_ice(self, start: date, n_days: int) -> bool:
        return all(start + timedelta(days=k) in self._index for k in range(n_days))

    def forcing_missing(self, start: date, n_days: int) -> list[str]:
        return [f"daily {f.product} {f.kind} are missing for {len(m)} of the {n_days} days ({m[0]} .. {m[-1]})"
                for f in (self.winds, self.currents) if (m := f.missing(start, n_days))]

    def conc(self, start: date, n_days: int) -> np.ndarray:
        idx = [self._index[start + timedelta(days=k)] for k in range(n_days)]
        return self.ds["ice_concentration"].values[idx]

    def context(self) -> ForecastContext:
        """A model-free context over the year-round observations, for the voyage simulation's 'observed' ice."""
        return ForecastContext(
            ds=self.ds, predictor=None, history_days=1, lead_days=0, season_months=YEAR_ROUND, train_seasons=[],
            clim=None, bank=np.zeros((0,), np.float32), bank_seasons=[], anomalies={},
            currents=(self.currents.x, self.currents.y), winds=(self.winds.x, self.winds.y),
            wind_times=self.winds.times, current_times=self.currents.times,
            forcing_sources={"winds": "ERA5 daily", "currents": "CMEMS daily"})


# --------------------------------------------------------------------------- one date
@dataclass(frozen=True)
class AnalogueSetup:
    """How one requested date is estimated: the analogue date, the sea-ice members and the iceberg source."""

    requested: date
    analogue: date                       # winds, currents, analogue-year icebergs and the simulation's ice
    n_days: int
    member_years: tuple[int, ...]        # year offsets are taken from the analogue date's calendar year
    shifts: tuple[int, ...]
    iceberg_kind: str                    # "recent_official" | "analogue_year"
    iceberg_usnic: UsnicArchive
    iceberg_max_age_days: int
    forecast_reason: str | None          # why the forecast pathway could not serve the date

    @property
    def offset_days(self) -> int:
        return (self.requested - self.analogue).days

    def requested_day(self, engine_day: date) -> date:
        return engine_day + timedelta(days=self.offset_days)

    def member_start(self, engine_day: date, year: int, shift: int) -> date:
        return _shift_years(engine_day, year - self.analogue.year) + timedelta(days=shift)


def resolve(data: AnalogueData, requested: date, n_days: int, recent_usnic: UsnicArchive | None,
            forecast_reason: str | None = None) -> AnalogueSetup:
    """The analogue for ``requested`` (raises :class:`AnalogueUnavailable`).

    Analogue date: the requested calendar day in the most recent earlier year whose ``n_days`` have ERA5 winds and
    CMEMS currents and earlier years with sea ice. Members: the same days in up to ``member_years`` years before it,
    each shifted by every day in ``-max_shift..max_shift`` that has observed sea ice for the whole run. The analogue
    year's own sea ice is not a member (it is only the voyage simulation's 'observed' ice), so a day missing upstream
    there does not stop a plan."""
    st = data.settings
    n_years, max_shift = int(st.get("member_years", 7)), int(st.get("max_shift_days", 7))
    tried = []
    analogue = None
    last = data.days[-1]
    def member_years(cand: date) -> list[int]:
        years = []
        for y in range(cand.year - 1, data.days[0].year - 1, -1):
            if data.has_ice(_shift_years(cand, y - cand.year), n_days):
                years.append(y)
            if len(years) == n_years:
                break
        return years

    years: list[int] = []
    for k in range(1, requested.year - data.days[0].year + 2):
        cand = _shift_years(requested, -k)
        if cand + timedelta(days=n_days - 1) > last:
            continue
        problems = data.forcing_missing(cand, n_days)
        years = [] if problems else member_years(cand)
        if not problems and not years:
            problems = ["no earlier year has observed sea ice for every day"]
        if not problems:
            analogue = cand
            break
        tried.append(f"{cand}: {problems[0]}")
        if len(tried) == 3:
            break
    if analogue is None:
        raise AnalogueUnavailable(f"no year with real winds, currents and earlier-year sea ice for the {n_days} days "
                                  f"from the calendar day of {requested} (" + "; ".join(tried or ["no earlier year"])
                                  + ")")
    max_age = int(st.get("recent_iceberg_max_age_days", 120))
    kind, usnic = "analogue_year", data.usnic
    if recent_usnic is not None:
        try:
            recent_usnic.list_for(requested, max_age)
            kind, usnic = "recent_official", recent_usnic
        except OutOfCoverage:
            pass
    if kind == "analogue_year":          # the weekly list of the analogue date; an older one only if none is
        limits = (int(data.archive.params["usnic_max_age_days"]), max_age)          # recent (age disclosed)
        for limit in limits:
            try:
                usnic.list_for(analogue, limit)
                max_age = limit
                break
            except OutOfCoverage as exc:
                why = exc
        else:
            raise AnalogueUnavailable(f"no official iceberg list for the analogue date {analogue}: {why}")
    return AnalogueSetup(requested, analogue, n_days, tuple(years), tuple(range(-max_shift, max_shift + 1)), kind,
                         usnic, max_age, forecast_reason)


# --------------------------------------------------------------------------- planner
class _ArchiveView:
    """What the voyage simulator reads from an archive, over the year-round observations."""

    def __init__(self, data: AnalogueData) -> None:
        self.data, self.ds, self.spec = data, data.ds, data.archive.spec
        self.params, self.usnic, self.root = data.archive.params, data.usnic, data.archive.root
        self.season_months, self.history_days = YEAR_ROUND, 1
        self.winds, self.currents = data.winds, data.currents

    @property
    def days(self) -> list[date]:
        return self.data.days


class AnaloguePlanner:
    """The planner interface (``world``, ``problems``, iceberg lookups) for a seasonal-analogue estimate."""

    def __init__(self, data: AnalogueData, setup: AnalogueSetup, drift_kwargs: dict) -> None:
        self.data, self.setup, self._drift = data, setup, drift_kwargs
        self.archive = _ArchiveView(data)
        self.ctx = data.context()
        self.meta, self.lead_days = None, 0

    def drift_kwargs(self) -> dict:
        return dict(self._drift)

    def pool(self, engine_day: date, n_days: int) -> list[tuple[int, int, date]]:
        """Distinct (year, shift, start) sea-ice sequences with observations for the whole run."""
        s = self.setup
        out = []
        for y in s.member_years:
            for sh in s.shifts:
                start = s.member_start(engine_day, y, sh)
                if self.data.has_ice(start, n_days):
                    out.append((y, sh, start))
        return out

    def problems(self, engine_day: date, n_days: int) -> list[str]:
        out = list(self.data.forcing_missing(engine_day, n_days))
        if not self.pool(engine_day, n_days):
            out.append(f"no analogue-year sea ice for the {n_days} days from {engine_day}")
        return out

    def truth_missing(self, start: date, n_days: int) -> list[date]:
        """Days with no observed sea ice in the analogue year (the voyage simulation sails through that ice)."""
        return [d for k in range(n_days) if (d := start + timedelta(days=k)) not in self.data._index]

    def members(self, engine_day: date, n_days: int) -> list[tuple[int, int, date]]:
        """The ``members`` scenarios: the pool in a fixed shuffled order (seed), repeated when it is smaller."""
        p = self.data.archive.params
        pool = self.pool(engine_day, n_days)
        order = np.random.default_rng(p["seed"]).permutation(len(pool))
        return [pool[order[i % len(pool)]] for i in range(p["members"])]

    def _positions(self, engine_day: date):
        s = self.setup
        if s.iceberg_kind == "recent_official":      # held at the last official report, as in forecast mode
            day = s.requested_day(engine_day)
            d, path, sha, bergs = s.iceberg_usnic.positions(day, s.iceberg_max_age_days + s.n_days)
            return d, path, sha, bergs, (day - d).days
        d, path, sha, bergs = s.iceberg_usnic.positions(engine_day, s.iceberg_max_age_days)
        return d, path, sha, bergs, (engine_day - d).days

    def snapshot(self, engine_day: date, grid) -> IcebergSnapshot:
        from antarctic_routing.ingestion.icebergs import in_grid_mask

        d, path, sha, bergs, age = self._positions(engine_day)
        inside = in_grid_mask([b[1] for b in bergs], [b[2] for b in bergs], grid) if bergs else np.zeros(0, bool)
        return IcebergSnapshot(d, path.name, sha, age, [b for b, m in zip(bergs, inside, strict=True) if m],
                               [b[0] for b, m in zip(bergs, inside, strict=True) if not m])

    def observed_bergs(self, engine_day: date) -> list[tuple[str, float, float]]:
        return self._positions(engine_day)[3]

    def usnic_positions(self, engine_day: date):
        d, path, _, bergs, age = self._positions(engine_day)
        return d, path, age, bergs

    def world(self, engine_day: date, n_days: int) -> tuple[ScenarioSet, IcebergSnapshot]:
        from antarctic_routing.iceberg.drift import add_iceberg_hazard

        problems = self.problems(engine_day, n_days)
        if problems:
            raise OutOfCoverage(f"no seasonal-analogue run from analogue day {engine_day}: " + "; ".join(problems))
        p, data = self.data.archive.params, self.data
        members = self.members(engine_day, n_days)
        land = data.ds["land_mask"].values.astype(bool)
        cache: dict[date, np.ndarray] = {}
        conc = np.empty((len(members), n_days, *land.shape), np.float32)
        for k, (_, _, start) in enumerate(members):
            if start not in cache:
                cache[start] = data.conc(start, n_days)
            conc[k] = cache[start]
        conc[:, :, land] = np.nan
        need = np.datetime64(engine_day, "D") + np.arange(n_days)
        wi = np.searchsorted(data.winds.times, need)
        ci = np.searchsorted(data.currents.times, need)
        years = sorted({m[0] for m in members})
        world = ScenarioSet(
            grid=grid_of(data.ds), start=datetime(engine_day.year, engine_day.month, engine_day.day),
            time_step_hours=24.0, land=land, conc=conc,
            current_x=data.currents.x[ci], current_y=data.currents.y[ci],
            execution_mode="real",
            description=(f"Seasonal analogue issued {engine_day}: {len(members)} members from {len(cache)} observed "
                         f"sea-ice sequences ({years[0]}-{years[-1]}); no model forecast."),
            wind_x=data.winds.x[wi], wind_y=data.winds.y[wi],
            layer_source=["analogue_observed"] * n_days,
            meta={"issue": engine_day.isoformat(), "forecast_days": 0, "analogue_sequences": len(cache)},
        )
        snap = self.snapshot(engine_day, world.grid)
        if snap.bergs:
            world = add_iceberg_hazard(world, snap.bergs, rng=np.random.default_rng(p["seed"]),
                                       radius_m=p["berg_radius_km"] * 1e3, **self.drift_kwargs())
        return world, snap


# --------------------------------------------------------------------------- disclosure
def ice_spread(data: AnalogueData, setup: AnalogueSetup) -> dict:
    """How much the analogue years disagree: ocean-mean sea-ice concentration on the first day, per year."""
    land = data.ds["land_mask"].values.astype(bool)
    per = {}
    for y in setup.member_years:
        c = data.conc(setup.member_start(setup.analogue, y, 0), 1)[0]
        per[str(y)] = round(float(np.nanmean(np.where(land, np.nan, c))), 4)
    v = list(per.values())
    return {"mean_concentration_by_year": per, "min": min(v), "max": max(v), "range": round(max(v) - min(v), 4)}


def confidence(data: AnalogueData, setup: AnalogueSetup, latest_observation: date) -> dict:
    """A plain confidence level with the caveats behind it (never a reason to refuse the date)."""
    spread = ice_spread(data, setup)
    n_years = len(setup.member_years)
    lead_years = (setup.requested - latest_observation).days / 365.25
    caveats = [
        "Not a forecast: no observation or model run exists for the requested dates. Sea ice is what happened on "
        f"the same calendar days in {setup.member_years[-1]}-{setup.member_years[0]}; winds and currents are the "
        f"reanalysis of {setup.analogue}.",
        f"Risk percentages are how often the route met ice or icebergs across {n_years} analogue years "
        f"(x {len(setup.shifts)} day shifts), not calibrated probabilities for the requested year.",
    ]
    if setup.iceberg_kind == "analogue_year":
        caveats.append("No official iceberg list falls within 120 days of the requested date, so the icebergs are "
                       f"the official list of the analogue date ({setup.analogue.year}): typical of the season, not "
                       "the bergs that will be there.")
    if lead_years > 1:
        caveats.append(f"The requested date is {lead_years:.1f} years after the latest real observation "
                       f"({latest_observation}); year-to-year and climate changes since then are not predicted.")
    if spread["range"] >= 0.1:
        caveats.append("The analogue years disagree strongly about the sea ice on these days (mean ocean "
                       f"concentration {spread['min'] * 100:.0f}-{spread['max'] * 100:.0f} %); expect large "
                       "uncertainty.")
    level = "very low" if n_years < 5 or (setup.iceberg_kind == "analogue_year" and spread["range"] >= 0.1) \
        else "low"
    return {"level": level, "caveats": caveats, "member_years": list(setup.member_years),
            "lead_years_after_latest_observation": round(lead_years, 2), "sea_ice_spread": spread}


def analogue_metadata(data: AnalogueData, setup: AnalogueSetup, snap: IcebergSnapshot, n_scenarios: int,
                      latest_observation: date) -> dict:
    """What a seasonal-analogue estimate was computed from: every input, its role and the dates actually used."""
    a, p, s = data.archive, data.archive.params, setup
    st = data.settings
    first, last = s.analogue, s.analogue + timedelta(days=s.n_days - 1)
    files = {f.kind: f.files_for(s.analogue, s.n_days) for f in (data.winds, data.currents)}
    pool = AnaloguePlanner(data, s, {}).pool(s.analogue, s.n_days)
    recent = s.iceberg_kind == "recent_official"
    return {
        "forecast_mode": True,
        "pathway": PATHWAY, "pathway_label": LABEL,
        "pathway_reason": s.forecast_reason,
        "requested_date": s.requested.isoformat(),
        "requested_date_relation": "after_archive",
        "archive_last_date": a.days[-1].isoformat(),
        "observations_for_requested_dates": "none: no sea-ice, wind, current or iceberg observation dated on or "
                                            "after the requested date is used",
        "analogue_date": s.analogue.isoformat(),
        "date_mapping": {
            "rule": "analogue = the requested calendar day in the most recent earlier year with real sea ice, winds "
                    "and currents for the whole run; engine day k is shown as requested date + k",
            "analogue_start_date": s.analogue.isoformat(),
            "analogue_season": (season_label(season) if (season := season_of(s.analogue, a.season_months)) is not None
                                else f"{s.analogue.year} (outside the Nov-Feb archive seasons)"),
            "offset_days": s.offset_days,
            "engine_days": [first.isoformat(), last.isoformat()],
            "shown_as": [s.requested.isoformat(), (s.requested + timedelta(days=s.n_days - 1)).isoformat()],
        },
        "sea_ice": {
            "status": "historical_analogue_ensemble",
            "method": "no model: each member is the real OSI SAF observation of the same calendar days in an earlier "
                      "year, shifted by up to "
                      f"{max(s.shifts)} days",
            "member_years": list(s.member_years),
            "day_shifts": [min(s.shifts), max(s.shifts)],
            "distinct_sequences": len(pool),
            "members": n_scenarios,
            "member_windows": [{"year": y, "shift_days": sh, "first": st0.isoformat(),
                                "last": (st0 + timedelta(days=s.n_days - 1)).isoformat()} for y, sh, st0 in pool],
            "analogue_year_excluded": "the analogue date's own year is not a member; it is the 'observed' ice of "
                                      "the voyage simulation",
            "source": "OSI SAF OSI-450-a / OSI-430-a daily sea-ice concentration, 25 km",
            "off_season_files": data.files["sea_ice"],
        },
        "forcing": {
            "status": "historical_reanalysis_analogue",
            "note": "ERA5/CMEMS reanalysis of the analogue days stands in for the requested days; it is NOT a "
                    "forecast and NOT a measurement of the requested year",
            "winds": {"product": "ERA5 daily reanalysis", "dates_used": [first.isoformat(), last.isoformat()],
                      "files": files["winds"]},
            "currents": {"product": "CMEMS GLOBAL reanalysis daily", "dates_used": [first.isoformat(),
                                                                                    last.isoformat()],
                         "files": files["currents"]},
        },
        "icebergs": {
            "status": ("observed_snapshot_held" if recent else "analogue_year_snapshot") if snap.bergs
            else "observed_snapshot_none_in_grid",
            "kind": s.iceberg_kind,
            "source": "U.S. National Ice Center (USNIC), official",
            "snapshot_date": snap.list_date.isoformat(), "file": snap.file, "sha256": snap.sha256,
            "age_days_at_requested_date": (s.requested - snap.list_date).days,
            "age_days_at_analogue_date": None if recent else (s.analogue - snap.list_date).days,
            "max_age_days": s.iceberg_max_age_days,
            "n_source_bergs": len(snap.bergs) + len(snap.outside_grid), "n_in_grid": len(snap.bergs),
            "ids_in_grid": [b[0] for b in snap.bergs],
            "drift_model": {"model": "calibrated ensemble drift (historical-mode parameters, unchanged)",
                            "beta": p["drift_beta"], "alpha_scale": p["drift_alpha_scale"],
                            "spread_factor": p["drift_spread_factor"], "members": n_scenarios,
                            "forcing": "historical reanalysis analogue (above)"},
            "note": ("positions are the last official report, held unchanged until the requested date (no newer "
                     "report exists), then drifted forward") if recent else
                    (f"no official list falls within {int(st.get('recent_iceberg_max_age_days', 120))} days of the "
                     "requested date: these are the official positions on the analogue date, drifted forward"),
        },
        "confidence": confidence(data, s, latest_observation),
        "label": "Forecast / hackathon estimate",
        "disclosure": st.get("disclosure"), "method": st.get("method"),
        "limitations": [t for t in a.spec["limitations"] if not t.startswith("Iceberg positions come from")
                        and not t.startswith("The sea-ice forecast does not beat persistence")] + [
            "Seasonal analogue: sea ice, winds, currents and (when no recent list exists) icebergs come from earlier "
            "years; the result shows what those years would have meant for this route, not a prediction."],
    }
