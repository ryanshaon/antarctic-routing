"""Real Historical Data mode - interactive planning on past seasons with the frozen real-data pipeline.

This wraps the steps ``antroute plan-window`` runs so the API can call them for any issue date that the
local real-data archive covers:

* sea ice: the OSI SAF 25 km history and the frozen U-Net (``ForecastContext``, residual bank, seed 42); later
  seasons listed under ``sea_ice_additional`` (same grid, product and land mask) are appended in memory only,
  after the frozen file, so the climatology and residual bank (training seasons only) are unchanged;
* forcing: daily ERA5 winds and CMEMS reanalysis currents, joined from several files by exact date;
* icebergs: the latest weekly USNIC list on or before the issue date, drifted with the calibrated
  parameters (gamma = 0.1, spread factor c = 0.6053);
* routing: the unchanged risk-budgeted planner, departure window and replanning.

Every input is listed with its SHA-256 in ``config/real_historical.json`` and verified before use. Nothing
is fitted, tuned or substituted: a missing input, PyTorch or a date the archive cannot support raises
:class:`HistoricalUnavailable` or :class:`OutOfCoverage`, never a synthetic fallback.

**Hindsight forcing.** For days after the issue date the winds and currents are reanalysis, known only
afterwards; results are a historical hindcast, not what a live forecast could have provided.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from antarctic_routing.common.provenance import sha256_file
from antarctic_routing.config import ProjectConfig
from antarctic_routing.forecasting.scenarios import ForecastContext
from antarctic_routing.ingestion.forcing import load_forcing_fields
from antarctic_routing.ingestion.icebergs import USNIC_SOURCE, in_grid_mask, read_iceberg_positions
from antarctic_routing.preprocessing.climatology import season_of
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.synthetic import ScenarioSet

LABEL = "Real Historical Data"
DATA_STATUS = "historical"
BLOCKED, FAILED = "blocked", "failed"


class HistoricalUnavailable(RuntimeError):
    """The mode cannot run: ``status`` is ``blocked`` (missing input or dependency) or ``failed`` (bad input)."""

    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status, self.reason = status, reason


class OutOfCoverage(ValueError):
    """The archive cannot support this issue date (no data is ever substituted)."""


def season_label(season: int) -> str:
    return f"{season}-{(season + 1) % 100:02d}"


# --------------------------------------------------------------------------- forcing
@dataclass(frozen=True)
class DailyForcing:
    """One daily forcing field (winds or currents) joined from several files by exact UTC date."""

    kind: str                      # "winds" | "currents"
    product: str                   # "ERA5" | "CMEMS"
    times: np.ndarray              # datetime64[D], strictly increasing
    x: np.ndarray                  # (T, ny, nx) grid-x component
    y: np.ndarray                  # (T, ny, nx) grid-y component
    files: tuple[dict, ...]        # {"file", "sha256", "first", "last"} per source file

    @classmethod
    def load(cls, records: Sequence[dict], root: Path, kind: str, shape: tuple[int, int]) -> DailyForcing:
        product = "ERA5" if kind == "winds" else "CMEMS"
        parts, files = [], []
        for rec in records:
            path = root / rec["path"]
            f = load_forcing_fields(path, expected_shape=shape)
            xy, times = (f.winds, f.wind_times) if kind == "winds" else (f.currents, f.current_times)
            if xy is None or times is None or np.ndim(xy[0]) != 3:
                raise ValueError(f"{path.name} has no daily {kind}")
            days = np.asarray(times).astype("datetime64[D]")
            parts.append((days, np.asarray(xy[0]), np.asarray(xy[1])))
            files.append({"file": path.name, "sha256": rec["sha256"], "first": str(days[0]), "last": str(days[-1])})
        if not parts:
            raise ValueError(f"no daily {kind} files listed")
        days = np.concatenate([p[0] for p in parts])
        order = np.argsort(days, kind="stable")
        days = days[order]
        dup = days[1:][np.diff(days.astype(np.int64)) == 0]
        if dup.size:
            raise ValueError(f"daily {kind} files overlap on {dup.size} date(s), e.g. {dup[0]}; list each date once")
        x = np.concatenate([p[1] for p in parts])[order]
        y = np.concatenate([p[2] for p in parts])[order]
        return cls(kind, product, days, x, y, tuple(files))

    def missing(self, start: date, n_days: int) -> list[str]:
        need = np.datetime64(start, "D") + np.arange(n_days)
        return [str(d) for d in need[~np.isin(need, self.times)]]

    def files_for(self, start: date, n_days: int) -> list[dict]:
        lo, hi = str(start), str(start + timedelta(days=n_days - 1))
        return [f for f in self.files if f["first"] <= hi and f["last"] >= lo]


# --------------------------------------------------------------------------- icebergs
@dataclass(frozen=True)
class IcebergSnapshot:
    """USNIC positions known on the issue date: the latest weekly list on or before it."""

    list_date: date
    file: str
    sha256: str
    age_days: int
    bergs: list[tuple[str, float, float]]     # on the routing grid, drifted
    outside_grid: list[str]

    def summary(self) -> dict:
        return {"source": USNIC_SOURCE, "list_date": self.list_date.isoformat(), "age_days": self.age_days,
                "file": self.file, "sha256": self.sha256, "drifted": [b[0] for b in self.bergs],
                "positions": [{"id": b[0], "lat": b[1], "lon": b[2]} for b in self.bergs],
                "outside_grid": self.outside_grid}


class UsnicArchive:
    """Weekly USNIC Antarctic iceberg lists, looked up by date with no future information."""

    def __init__(self, records: Sequence[dict], root: Path) -> None:
        self.lists = sorted((date.fromisoformat(r["date"]), root / r["path"], r["sha256"]) for r in records)

    def list_for(self, day: date, max_age_days: int) -> tuple[date, Path, str]:
        known = [x for x in self.lists if x[0] <= day]
        if not known:
            raise OutOfCoverage(f"no USNIC iceberg list on or before {day}")
        d, path, sha = known[-1]
        if (day - d).days > max_age_days:
            raise OutOfCoverage(f"the latest USNIC iceberg list before {day} is from {d} "
                                f"({(day - d).days} days old; limit {max_age_days})")
        return d, path, sha

    def positions(self, day: date, max_age_days: int) -> tuple[date, Path, str, list[tuple[str, float, float]]]:
        """Latest position per berg (as ``plan-window --icebergs``), dated no later than ``day``."""
        d, path, sha = self.list_for(day, max_age_days)
        latest: dict[str, tuple[str, float, float]] = {}
        for row in sorted(read_iceberg_positions(path), key=lambda r: r["date"]):
            if row["date"] <= day:
                latest[row["iceberg_id"]] = (row["iceberg_id"], row["lat"], row["lon"])
        return d, path, sha, list(latest.values())

    def snapshot(self, day: date, grid: PolarGrid, max_age_days: int) -> IcebergSnapshot:
        d, path, sha, bergs = self.positions(day, max_age_days)
        inside = in_grid_mask([b[1] for b in bergs], [b[2] for b in bergs], grid) if bergs else np.zeros(0, bool)
        return IcebergSnapshot(d, path.name, sha, (day - d).days,
                               [b for b, m in zip(bergs, inside, strict=True) if m],
                               [b[0] for b, m in zip(bergs, inside, strict=True) if not m])


# --------------------------------------------------------------------------- archive
def _verify(root: Path, rec: dict, name: str) -> Path:
    path = root / rec["path"]
    if not path.is_file():
        raise HistoricalUnavailable(BLOCKED, f"input {name} not found at $ANTROUTE_DATA_ROOT/{rec['path']}")
    got = sha256_file(path)
    if got != rec["sha256"]:
        raise HistoricalUnavailable(FAILED, f"input {name} ({rec['path']}) has SHA-256 {got[:16]}…, "
                                            f"expected {rec['sha256'][:16]}…")
    return path


@dataclass
class HistoricalArchive:
    """Verified real-data inputs and the issue dates they support (cheap; no model is loaded)."""

    spec: dict
    root: Path
    ds: xr.Dataset
    winds: DailyForcing
    currents: DailyForcing
    usnic: UsnicArchive
    season_months: list[int]
    history_days: int = 14
    sea_ice_files: list[dict] = field(default_factory=list)
    _dates: dict[int, list[date]] = field(default_factory=dict, repr=False)
    _days: list[date] = field(default_factory=list, repr=False)
    _index: dict[date, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._days = [np.datetime64(t, "D").astype(object) for t in self.ds["time"].values]
        self._index = {d: i for i, d in enumerate(self._days)}

    @classmethod
    def load(cls, data_root: str | Path, spec_path: str | Path, season_months: Sequence[int]) -> HistoricalArchive:
        spec = json.loads(Path(spec_path).read_text())
        root = Path(data_root)
        if not root.is_dir():
            raise HistoricalUnavailable(BLOCKED, f"data root {root} does not exist")
        inp = spec["inputs"]
        for name in ("sea_ice", "checkpoint", "drift_selection", "spread_calibration"):
            _verify(root, inp[name], name)
        for name in ("sea_ice_additional", "wind_forcing", "current_forcing", "icebergs"):
            for rec in inp.get(name, []):
                _verify(root, rec, name)
        cls._check_parameters(spec, root)
        ds = xr.load_dataset(root / inp["sea_ice"]["path"])
        if ds.attrs.get("execution_mode", "real") != "real":
            raise HistoricalUnavailable(FAILED, "the sea-ice file is not real data")
        ds, sea_ice_files = cls._join_sea_ice(ds, inp, root)
        shape = ds["land_mask"].shape
        try:
            winds = DailyForcing.load(inp["wind_forcing"], root, "winds", shape)
            currents = DailyForcing.load(inp["current_forcing"], root, "currents", shape)
        except ValueError as exc:
            raise HistoricalUnavailable(FAILED, str(exc)) from None
        land = ds["land_mask"].values.astype(bool)
        currents = DailyForcing(currents.kind, currents.product, currents.times,       # as plan-window does
                                np.where(land, 0.0, currents.x), np.where(land, 0.0, currents.y), currents.files)
        return cls(spec, root, ds, winds, currents, UsnicArchive(inp["icebergs"], root), list(season_months),
                   sea_ice_files=sea_ice_files)

    @staticmethod
    def _join_sea_ice(ds: xr.Dataset, inp: dict, root: Path) -> tuple[xr.Dataset, list[dict]]:
        """Append the ``sea_ice_additional`` seasons after the frozen file, in memory (no file is written).

        Each must be real data on the frozen grid with the frozen land mask and start after the data before it;
        the frozen file's attributes and land mask are kept, so the model and its training seasons see the
        same inputs as before."""
        first = inp["sea_ice"]
        files = [{"file": Path(first["path"]).name, "sha256": first["sha256"],
                  "first": str(ds["time"].values[0])[:10], "last": str(ds["time"].values[-1])[:10],
                  "source": "OSI SAF OSI-450-a / OSI-430-a daily sea-ice concentration, 25 km"}]
        extra = inp.get("sea_ice_additional", [])
        if not extra:
            return ds, files
        parts = [ds]
        for rec in extra:
            new = xr.load_dataset(root / rec["path"])
            name = Path(rec["path"]).name
            if new.attrs.get("execution_mode") != "real":
                raise HistoricalUnavailable(FAILED, f"the sea-ice file {name} is not real data")
            if not (np.array_equal(new["x"], ds["x"]) and np.array_equal(new["y"], ds["y"])):
                raise HistoricalUnavailable(FAILED, f"the sea-ice file {name} is not on the frozen grid")
            if not (new["land_mask"].values == ds["land_mask"].values).all():
                raise HistoricalUnavailable(FAILED, f"the sea-ice file {name} has a different land mask")
            if new["time"].values[0] <= parts[-1]["time"].values[-1]:
                raise HistoricalUnavailable(FAILED, f"the sea-ice file {name} overlaps the data before it")
            parts.append(new)
            files.append({"file": name, "sha256": rec["sha256"], "first": str(new["time"].values[0])[:10],
                          "last": str(new["time"].values[-1])[:10], "source": rec["source"],
                          "provenance": rec.get("provenance")})
        joined = xr.concat(parts, dim="time", data_vars="minimal", coords="minimal", compat="override")
        joined["land_mask"] = ds["land_mask"]
        joined.attrs = dict(ds.attrs)
        return joined, files

    def sea_ice_file(self, day: date) -> dict | None:
        """The sea-ice file holding ``day`` (its provenance as listed in the input spec)."""
        return next((f for f in self.sea_ice_files if f["first"] <= day.isoformat() <= f["last"]), None)

    @staticmethod
    def _check_parameters(spec: dict, root: Path) -> None:
        """The drift parameters must equal the calibration files' selections (as the frozen reproduction checks)."""
        p, inp = spec["parameters"], spec["inputs"]
        sel = json.loads((root / inp["drift_selection"]["path"]).read_text())["selected_params"]
        spr = json.loads((root / inp["spread_calibration"]["path"]).read_text())["selected_model"]
        if (sel["beta"], sel["alpha_scale"]) != (p["drift_beta"], p["drift_alpha_scale"]):
            raise HistoricalUnavailable(FAILED, f"drift selection {sel} differs from the frozen parameters")
        if (spr["c"], spr["q"]) != (p["drift_spread_factor"], 0.0):
            raise HistoricalUnavailable(FAILED, f"spread calibration {spr} differs from the frozen parameters")

    # ------------------------------------------------------------ coverage
    @property
    def params(self) -> dict:
        return self.spec["parameters"]

    @property
    def days(self) -> list[date]:
        return self._days

    def problems(self, issue: date, n_days: int, require_usnic: bool = True) -> list[str]:
        """Why ``issue`` cannot be used for ``n_days`` scenario days (empty when it can).

        ``require_usnic=False`` leaves out the iceberg-list rule, for forecast mode, which brings its own
        iceberg snapshot (:mod:`antarctic_routing.forecast_mode`)."""
        days, months, L = self.days, self.season_months, self.history_days
        out = []
        if issue not in self._index:
            return [f"{issue} is not in the sea-ice archive "
                    f"({days[0]} .. {days[-1]}, Nov-Feb only, missing upstream dates excluded)"]
        t = self._index[issue]
        lo = t - L + 1
        if lo < 0 or (days[t] - days[lo]).days != L - 1 or season_of(days[lo], months) != season_of(issue, months):
            out.append(f"the forecast needs {L} contiguous in-season days of observed sea ice ending on {issue}")
        last = issue + timedelta(days=n_days - 1)
        if season_of(last, months) != season_of(issue, months):
            out.append(f"the {n_days} scenario days ({issue} .. {last}) run past the end of the season")
        for f in (self.winds, self.currents):
            miss = f.missing(issue, n_days)
            if miss:
                out.append(f"daily {f.product} {f.kind} are missing for {len(miss)} of the {n_days} days "
                           f"({miss[0]} .. {miss[-1]})")
        if require_usnic:
            try:
                self.usnic.list_for(issue, self.params["usnic_max_age_days"])
            except OutOfCoverage as exc:
                out.append(str(exc))
        return out

    def check(self, issue: date, n_days: int) -> None:
        problems = self.problems(issue, n_days)
        if problems:
            raise OutOfCoverage(f"no Real Historical Data run for {issue}: " + "; ".join(problems))

    def available(self, n_days: int) -> list[date]:
        if n_days not in self._dates:
            self._dates[n_days] = [d for d in self.days if not self.problems(d, n_days)]
        return self._dates[n_days]

    def season_info(self, season: int) -> dict:
        """Which fitted components used ``season`` (results there are not independent evidence)."""
        roles = {}
        for comp, split in self.spec["splits"].items():
            roles[comp] = next((name for name in ("train", "validation", "test", "independent_evaluation")
                                if season in split.get(name, [])), "not used")
        use = {"train": "training", "validation": "validation (model selection)"}
        notes = []
        if roles["forecast_model"] in use:
            notes.append(f"the sea-ice U-Net used this season for {use[roles['forecast_model']]}")
        if roles["iceberg_drift"] in use:
            notes.append(f"the iceberg drift and spread calibration used this season for {use[roles['iceberg_drift']]}")
        independent = [comp for comp, role in roles.items() if role == "independent_evaluation"]
        return {"season": season_label(season), "start_year": season, "forecast_model": roles["forecast_model"],
                "iceberg_drift": roles["iceberg_drift"], "out_of_sample": not notes,
                "in_sample_notes": notes, "independent_evaluation": bool(independent),
                "evaluation_notes": [_EVAL_NOTES[c] for c in independent]}

    def coverage(self, n_days: int) -> list[dict]:
        by: dict[int, list[date]] = {}
        for d in self.available(n_days):
            by.setdefault(season_of(d, self.season_months), []).append(d)
        return [{**self.season_info(s), "first": v[0].isoformat(), "last": v[-1].isoformat(), "n_dates": len(v)}
                for s, v in sorted(by.items())]


_EVAL_NOTES = {
    "forecast_model": "the frozen sea-ice U-Net and its calibration were scored on this season after freezing "
                      "(nothing fitted)",
    "iceberg_drift": "the frozen iceberg drift and spread parameters were scored on this season after freezing "
                     "(nothing fitted)",
}


# --------------------------------------------------------------------------- planner
def drift_kwargs(p: dict) -> dict:
    """The calibrated drift parameters of ``p`` (the input spec's ``parameters``), as ``plan-window`` passes them."""
    a = float(p["drift_alpha_scale"])
    return {"beta": float(p["drift_beta"]), "alpha_range": (0.01 * a, 0.03 * a),
            "spread_factor": float(p["drift_spread_factor"])}


class HistoricalPlanner:
    """The frozen U-Net forecast context over the verified archive (heavy: built once, then reused)."""

    def __init__(self, archive: HistoricalArchive) -> None:
        try:
            from antarctic_routing.forecasting.train import load_model, unet_predictor
        except ImportError as exc:
            raise HistoricalUnavailable(BLOCKED, f"PyTorch is not installed ({exc}); the frozen U-Net needs the "
                                                 "'ml' extra (pip install -e '.[ml]')") from None
        a, spec = archive, archive.spec
        try:
            model, meta = load_model(a.root / spec["inputs"]["checkpoint"]["path"], a.ds)
        except ValueError as exc:
            raise HistoricalUnavailable(FAILED, f"checkpoint rejected: {exc}") from None
        split = spec["splits"]["forecast_model"]
        tc = meta["train_config"]
        if list(meta["train_seasons"]) != split["train"] or list(meta.get("val_seasons", [])) != split["validation"]:
            raise HistoricalUnavailable(FAILED, "checkpoint seasons differ from the pinned forecast-model split")
        a.history_days = tc["history_days"]
        self.archive, self.meta, self.lead_days = a, meta, tc["lead_days"]
        prov = {f.kind: {"product": f"{f.product} daily", "files": list(f.files)} for f in (a.winds, a.currents)}
        self.ctx = ForecastContext.build(
            a.ds, unet_predictor(model), tc["history_days"], tc["lead_days"], a.season_months, meta["train_seasons"],
            bank_size=a.params["bank_size"], seed=a.params["seed"],
            winds=(a.winds.x, a.winds.y), wind_times=a.winds.times,
            currents=(a.currents.x, a.currents.y), current_times=a.currents.times,
            forcing_sources={"winds": "ERA5 daily", "currents": "CMEMS daily"}, forcing_provenance=prov,
        )

    def drift_kwargs(self) -> dict:
        """The calibrated drift parameters, exactly as ``plan-window --drift-*`` passes them."""
        return drift_kwargs(self.archive.params)

    def world(self, issue: date, n_days: int) -> tuple[ScenarioSet, IcebergSnapshot]:
        """Joint sea-ice + iceberg scenarios issued on ``issue`` (same draws as ``plan-window``, seed 42)."""
        from antarctic_routing.iceberg.drift import add_iceberg_hazard

        a, p = self.archive, self.archive.params
        a.check(issue, n_days)
        world = self.ctx.scenarios(issue, n_days, p["members"], np.random.default_rng(p["seed"]))
        snap = a.usnic.snapshot(issue, world.grid, p["usnic_max_age_days"])
        if snap.bergs:
            world = add_iceberg_hazard(world, snap.bergs, rng=np.random.default_rng(p["seed"]),
                                       radius_m=p["berg_radius_km"] * 1e3, **self.drift_kwargs())
        return world, snap

    def berg_hazard(self, world: ScenarioSet, day: date) -> ScenarioSet:
        """Add the iceberg hazard known on ``day`` to a world issued that day (replay)."""
        from antarctic_routing.iceberg.drift import add_iceberg_hazard

        p = self.archive.params
        snap = self.archive.usnic.snapshot(day, world.grid, p["usnic_max_age_days"])
        if not snap.bergs:
            return world
        return add_iceberg_hazard(world, snap.bergs, rng=np.random.default_rng(p["seed"]),
                                  radius_m=p["berg_radius_km"] * 1e3, **self.drift_kwargs())

    def observed_bergs(self, day: date) -> list[tuple[str, float, float]]:
        """USNIC positions reported on or before ``day`` (replay truth; positions are weekly)."""
        return self.archive.usnic.positions(day, self.archive.params["usnic_max_age_days"])[3]


def _sea_ice_provenance(archive: HistoricalArchive, spec: dict, issue: date) -> dict:
    f = archive.sea_ice_file(issue)
    if f is None or f is archive.sea_ice_files[0]:                    # the frozen file: unchanged record
        return {"source": "OSI SAF OSI-450-a / OSI-430-a daily sea-ice concentration, 25 km",
                "file": Path(spec["inputs"]["sea_ice"]["path"]).name, "sha256": spec["inputs"]["sea_ice"]["sha256"],
                "observed_through": issue.isoformat()}
    return {"source": f["source"], "file": f["file"], "sha256": f["sha256"], "observed_through": issue.isoformat(),
            "provenance": f.get("provenance")}


def provenance(archive: HistoricalArchive, cfg: ProjectConfig, issue: date, n_days: int,
               snap: IcebergSnapshot | None, planner: HistoricalPlanner | None = None) -> dict:
    """What a Real Historical Data result was computed from, for the API response."""
    spec, p = archive.spec, archive.params
    season = season_of(issue, archive.season_months)
    ck = spec["inputs"]["checkpoint"]
    return {
        "label": LABEL, "data_status": DATA_STATUS, "execution_mode": "real",
        "issue": issue.isoformat(),
        "scenario_days": [issue.isoformat(), (issue + timedelta(days=n_days - 1)).isoformat()],
        "season": archive.season_info(season) if season is not None else None,
        "hindsight_forcing": spec["hindsight_forcing"],
        "sea_ice": _sea_ice_provenance(archive, spec, issue),
        "forecast_model": {"id": Path(ck["path"]).parent.name, "sha256": ck["sha256"],
                           "lead_days": planner.lead_days if planner else None,
                           "train_seasons": spec["splits"]["forecast_model"]["train"]},
        "forcing": {f.kind: {"product": f"{f.product} daily (reanalysis)", "files": f.files_for(issue, n_days)}
                    for f in (archive.winds, archive.currents)},
        "icebergs": snap.summary() if snap is not None else None,
        "parameters": {k: p[k] for k in ("members", "seed", "berg_radius_km", "drift_beta", "drift_alpha_scale",
                                         "drift_spread_factor", "usnic_max_age_days")},
        "probability_calibration": {"applied_in_route_risk": False},
        "vessel": {"name": cfg.vessel.name, "ice_class": cfg.vessel.ice_class,
                   "max_ice_concentration": cfg.vessel.max_ice_concentration,
                   "note": "generic placeholder vessel; real data, illustrative vessel"},
        "limitations": spec["limitations"],
    }
