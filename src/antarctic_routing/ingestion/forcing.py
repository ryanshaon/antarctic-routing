"""Real forcing fields: ERA5 10 m wind and CMEMS surface currents on the polar grid.

Both products arrive on regular lat/lon grids with east/north vector
components. They are time-averaged over the downloaded period, interpolated to
the EPSG:3031 cell centres and *rotated* into grid x/y components (see
``preprocessing.grid``). Cells outside the source coverage (or on land in
CMEMS) are filled with zero velocity and the filled fraction is recorded.

The result feeds :class:`forecasting.scenarios.ForecastContext` (``--forcing``)
in place of the schematic currents and westerlies. A time mean is the default
(``wind_time_mode="mean"``). ``wind_time_mode="daily"`` instead groups the
hourly ERA5 ``u10``/``v10`` by UTC day, averages each component separately and
keeps ``wind_x/wind_y(time, y, x)`` with one timestamp per day (00:00 UTC).
``current_time_mode="daily"`` does the same for CMEMS ``uo``/``vo`` (already
daily means in the ``P1D-m`` products, so normally one sample per day) and keeps
``current_x/current_y(time, y, x)``. When both are daily they must share the
same days. ForecastContext does not consume daily files yet: :func:`load_forcing`
rejects them explicitly, and :func:`load_forcing_fields` reads both layouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from antarctic_routing.common.provenance import sha256_file
from antarctic_routing.preprocessing.grid import CRS, PolarGrid
from antarctic_routing.preprocessing.harmonize import regrid_vector_latlon

_TIME_NAMES = ("valid_time", "time")
_LAT_NAMES = ("latitude", "lat")
_LON_NAMES = ("longitude", "lon")
WIND_TIME_MODES = ("mean", "daily")
CURRENT_TIME_MODES = WIND_TIME_MODES


def _name(ds: xr.Dataset, options) -> str:
    for n in options:
        if n in ds.dims or n in ds.coords:
            return n
    raise ValueError(f"none of {options} found in dataset")


def read_latlon_vector(path: str | Path, u_name: str, v_name: str):
    """Time-mean (and surface-level) east/north components with their lat/lon axes."""
    with xr.open_dataset(path) as ds:
        u, v = ds[u_name], ds[v_name]
        if "depth" in u.dims:
            u, v = u.isel(depth=0), v.isel(depth=0)
        t = next((n for n in _TIME_NAMES if n in u.dims), None)
        if t:
            u, v = u.mean(t, skipna=True), v.mean(t, skipna=True)
        lat = ds[_name(ds, _LAT_NAMES)].values.astype(float)
        lon = ds[_name(ds, _LON_NAMES)].values.astype(float)
        lon = np.where(lon > 180, lon - 360, lon)
        return u.values.astype(float), v.values.astype(float), lat, lon


def read_latlon_vector_daily(path: str | Path, u_name: str, v_name: str):
    """UTC-daily means of east/north components, each averaged separately.

    Returns ``(u, v, lat, lon, days, hours_per_day)`` with ``u, v`` shaped
    ``(n_days, n_lat, n_lon)`` and ``days`` as ``datetime64[ns]`` at 00:00 UTC.
    """
    with xr.open_dataset(path) as ds:
        u, v = ds[u_name], ds[v_name]
        if "depth" in u.dims:
            u, v = u.isel(depth=0), v.isel(depth=0)
        t = next((n for n in _TIME_NAMES if n in u.dims), None)
        if t is None:
            raise ValueError(f"{Path(path).name}: daily mode needs a time axis ({' or '.join(_TIME_NAMES)})")
        day = u[t].dt.floor("D").rename("day")
        u_day = u.groupby(day).mean(t, skipna=True)
        v_day = v.groupby(day).mean(t, skipna=True)
        hours = u[t].groupby(day).count()
        lat = ds[_name(ds, _LAT_NAMES)].values.astype(float)
        lon = ds[_name(ds, _LON_NAMES)].values.astype(float)
        lon = np.where(lon > 180, lon - 360, lon)
        return (u_day.transpose("day", ...).values.astype(float), v_day.transpose("day", ...).values.astype(float),
                lat, lon, u_day["day"].values, hours.values.astype(int))


def _regrid_daily(path, u_name, v_name, grid: PolarGrid):
    u, v, lat, lon, days, hours = read_latlon_vector_daily(path, u_name, v_name)
    order = np.argsort(lon)
    xs, ys = [], []
    for k in range(len(days)):
        x, y = regrid_vector_latlon(u[k][:, order], v[k][:, order], lat, lon[order], grid)
        xs.append(x)
        ys.append(y)
    x, y = np.stack(xs), np.stack(ys)
    missing = ~(np.isfinite(x) & np.isfinite(y))
    return np.where(missing, 0.0, x), np.where(missing, 0.0, y), float(missing.mean()), days, hours


def missing_days(days: np.ndarray) -> int:
    """Calendar days absent between the first and last timestamp of a daily axis."""
    if len(days) == 0:
        return 0
    d = np.asarray(days).astype("datetime64[D]")
    return int((d[-1] - d[0]).astype(int) + 1 - np.unique(d).size)


def _regrid(path, u_name, v_name, grid: PolarGrid):
    u, v, lat, lon = read_latlon_vector(path, u_name, v_name)
    order = np.argsort(lon)
    x, y = regrid_vector_latlon(u[:, order], v[:, order], lat, lon[order], grid)
    missing = ~(np.isfinite(x) & np.isfinite(y))
    return np.where(missing, 0.0, x), np.where(missing, 0.0, y), float(missing.mean())


def build_forcing(grid: PolarGrid, era5_path: str | Path | None = None,
                  cmems_path: str | Path | None = None, wind_time_mode: str = "mean",
                  current_time_mode: str = "mean") -> xr.Dataset:
    if era5_path is None and cmems_path is None:
        raise ValueError("provide an ERA5 and/or a CMEMS file")
    if wind_time_mode not in WIND_TIME_MODES:
        raise ValueError(f"wind_time_mode must be one of {WIND_TIME_MODES}")
    if current_time_mode not in CURRENT_TIME_MODES:
        raise ValueError(f"current_time_mode must be one of {CURRENT_TIME_MODES}")
    data_vars, attrs, sources, coords = {}, {}, [], {"y": grid.y, "x": grid.x}
    averaging = "time mean over the source files"
    if era5_path is not None:
        if wind_time_mode == "daily":
            wx, wy, filled, days, hours = _regrid_daily(era5_path, "u10", "v10", grid)
            data_vars.update(wind_x=(("time", "y", "x"), wx), wind_y=(("time", "y", "x"), wy))
            coords["time"] = days
            attrs.update(wind_hours_per_day_min=int(hours.min()), wind_hours_per_day_max=int(hours.max()))
            averaging = ("wind: UTC-daily mean of hourly u10/v10, each component averaged separately"
                         + ("; currents: time mean over the source file" if cmems_path is not None else ""))
        else:
            wx, wy, filled = _regrid(era5_path, "u10", "v10", grid)
            data_vars.update(wind_x=(("y", "x"), wx), wind_y=(("y", "x"), wy))
        attrs["wind_filled_fraction"] = filled
        attrs["wind_time_mode"] = wind_time_mode
        sources.append(Path(era5_path))
    if cmems_path is not None:
        if current_time_mode == "daily":
            cx, cy, filled, days, samples = _regrid_daily(cmems_path, "uo", "vo", grid)
            if "time" in coords and not np.array_equal(np.asarray(coords["time"]), days):
                raise ValueError("daily ERA5 and CMEMS files cover different days; download the same period")
            data_vars.update(current_x=(("time", "y", "x"), cx), current_y=(("time", "y", "x"), cy))
            coords["time"] = days
            attrs.update(current_samples_per_day_min=int(samples.min()),
                         current_samples_per_day_max=int(samples.max()),
                         current_missing_days=missing_days(days))
            current_avg = "currents: UTC-daily mean of surface uo/vo, each component averaged separately"
        else:
            cx, cy, filled = _regrid(cmems_path, "uo", "vo", grid)
            data_vars.update(current_x=(("y", "x"), cx), current_y=(("y", "x"), cy))
            current_avg = "currents: time mean over the source file"
        if era5_path is not None and wind_time_mode == "daily":
            averaging = averaging.split("; currents:")[0] + "; " + current_avg
        elif current_time_mode == "daily":
            averaging = current_avg + ("; wind: time mean over the source file" if era5_path is not None else "")
        attrs["current_filled_fraction"] = filled
        attrs["current_time_mode"] = current_time_mode
        sources.append(Path(cmems_path))
    for name in data_vars:
        data_vars[name] = (*data_vars[name], {"units": "m s-1", "long_name": f"{name} (EPSG:3031 grid component)"})
    return xr.Dataset(
        data_vars,
        coords=coords,
        attrs={**attrs, "crs": CRS, "resolution_m": grid.resolution_m, "execution_mode": "real",
               "averaging": averaging,
               "source_files": ",".join(p.name for p in sources),
               "source_sha256": ",".join(sha256_file(p) for p in sources)},
    )


@dataclass(frozen=True)
class ForcingFields:
    """Forcing read back from :func:`build_forcing` output, old 2-D or daily layout."""

    currents: tuple[np.ndarray, np.ndarray] | None
    winds: tuple[np.ndarray, np.ndarray] | None   # (ny, nx) each, or (T, ny, nx) when wind_times is set
    wind_times: np.ndarray | None                 # datetime64 per wind layer; None for a time mean
    wind_time_mode: str
    current_times: np.ndarray | None = None       # datetime64 per current layer; None for a time mean
    current_time_mode: str = "mean"
    attrs: dict = field(default_factory=dict)     # global attributes of the forcing file (provenance)


def load_forcing_fields(path: str | Path, expected_shape: tuple[int, int]) -> ForcingFields:
    with xr.open_dataset(path) as ds:
        shape = (ds.sizes["y"], ds.sizes["x"])
        if shape != tuple(expected_shape):
            raise ValueError(f"forcing grid {shape} does not match the dataset grid {tuple(expected_shape)}; "
                             "rebuild it with the same --resolution-km")
        currents = (ds["current_x"].values, ds["current_y"].values) if "current_x" in ds else None
        winds = (ds["wind_x"].values, ds["wind_y"].values) if "wind_x" in ds else None
        timed = "wind_x" in ds and "time" in ds["wind_x"].dims
        times = ds["time"].values if timed else None
        mode = ds.attrs.get("wind_time_mode", "daily" if timed else "mean")
        c_timed = "current_x" in ds and "time" in ds["current_x"].dims
        c_times = ds["time"].values if c_timed else None
        c_mode = ds.attrs.get("current_time_mode", "daily" if c_timed else "mean")
        attrs = dict(ds.attrs)
    return ForcingFields(currents, winds, times, mode, c_times, c_mode, attrs)


def load_forcing(path: str | Path, expected_shape: tuple[int, int]):
    """Return ``(currents, winds)`` tuples (each ``(x, y)`` arrays or None) for ForecastContext.

    Only time-mean (2-D) forcing is accepted here; ForecastContext cannot use
    daily winds yet, so a daily file is rejected rather than misused.
    """
    f = load_forcing_fields(path, expected_shape)
    if f.wind_times is not None:
        raise ValueError(f"{Path(path).name} has daily (time-dependent) winds, which ForecastContext does not "
                         "support yet; rebuild with --wind-time-mode mean")
    if f.current_times is not None:
        raise ValueError(f"{Path(path).name} has daily (time-dependent) currents, which ForecastContext does not "
                         "support yet; rebuild with --current-time-mode mean")
    return f.currents, f.winds
