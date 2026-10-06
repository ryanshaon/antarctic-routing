"""Real forcing: ERA5 10 m wind and CMEMS surface currents onto the polar grid."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from antarctic_routing.config import DomainSection
from antarctic_routing.ingestion.forcing import build_forcing, load_forcing
from antarctic_routing.preprocessing.grid import PolarGrid

DOMAIN = DomainSection(lat_min=-66.0, lat_max=-55.0, lon_min=-72.0, lon_max=-52.0)
LAT = np.arange(-45.0, -75.25, -0.25)   # ERA5 style: descending latitude
LON = np.arange(-85.0, -34.75, 0.25)


def write_era5(path: Path, u=10.0, v=0.0, time_name="valid_time"):
    t = [datetime(2024, 12, 1) + timedelta(hours=6 * k) for k in range(8)]
    shape = (len(t), LAT.size, LON.size)
    xr.Dataset({"u10": ((time_name, "latitude", "longitude"), np.full(shape, u, np.float32)),
                "v10": ((time_name, "latitude", "longitude"), np.full(shape, v, np.float32))},
               coords={time_name: t, "latitude": LAT, "longitude": LON}).to_netcdf(path)
    return path


def write_cmems(path: Path, uo=0.3, vo=0.0):
    t = [datetime(2024, 12, 1) + timedelta(days=k) for k in range(3)]
    lat = LAT[::-1]                         # CMEMS style: ascending latitude, with a depth axis
    shape = (len(t), 2, lat.size, LON.size)
    data = np.full(shape, uo, np.float32)
    data[:, 1] = 99.0                       # deeper level must be ignored
    xr.Dataset({"uo": (("time", "depth", "latitude", "longitude"), data),
                "vo": (("time", "depth", "latitude", "longitude"), np.full(shape, vo, np.float32))},
               coords={"time": t, "depth": [0.49, 1.54], "latitude": lat, "longitude": LON}).to_netcdf(path)
    return path


@pytest.fixture(scope="module")
def grid():
    return PolarGrid.from_domain(DOMAIN, resolution_km=25)


def test_wind_and_current_are_regridded_and_rotated(tmp_path, grid):
    ds = build_forcing(grid, era5_path=write_era5(tmp_path / "e.nc"), cmems_path=write_cmems(tmp_path / "c.nc"))
    lam = np.deg2rad(grid.lon2d)
    assert np.allclose(ds["wind_x"].values, 10.0 * np.cos(lam), atol=1e-4)
    assert np.allclose(ds["wind_y"].values, -10.0 * np.sin(lam), atol=1e-4)
    assert np.allclose(np.hypot(ds["current_x"], ds["current_y"]), 0.3, atol=1e-5)  # surface level only
    assert ds.attrs["execution_mode"] == "real" and ds.attrs["crs"] == "EPSG:3031"
    assert len(ds.attrs["source_sha256"].split(",")) == 2


def test_old_era5_time_name_is_supported(tmp_path, grid):
    ds = build_forcing(grid, era5_path=write_era5(tmp_path / "e.nc", u=0.0, v=5.0, time_name="time"))
    assert np.allclose(np.hypot(ds["wind_x"], ds["wind_y"]), 5.0, atol=1e-4)
    assert "current_x" not in ds


def test_missing_coverage_is_zero_filled_and_reported(tmp_path, grid):
    path = tmp_path / "c.nc"
    write_cmems(path)
    with xr.open_dataset(path) as src:
        small = src.sel(latitude=slice(-62, -55)).load()   # does not cover the whole grid
    small.to_netcdf(tmp_path / "small.nc")
    ds = build_forcing(grid, cmems_path=tmp_path / "small.nc")
    assert np.isfinite(ds["current_x"].values).all()
    assert 0 < ds.attrs["current_filled_fraction"] < 1


def test_load_forcing_returns_tuples_matching_grid(tmp_path, grid):
    out = tmp_path / "forcing.nc"
    ds = build_forcing(grid, era5_path=write_era5(tmp_path / "e.nc"), cmems_path=write_cmems(tmp_path / "c.nc"))
    ds.to_netcdf(out)
    currents, winds = load_forcing(out, expected_shape=grid.shape)
    assert currents[0].shape == grid.shape and winds[1].shape == grid.shape
    with pytest.raises(ValueError, match="grid"):
        load_forcing(out, expected_shape=(3, 3))


def test_build_forcing_requires_at_least_one_source(grid):
    with pytest.raises(ValueError):
        build_forcing(grid)


# ------------------------------------------------------------- daily wind mode
def write_hourly_era5(path: Path, u: np.ndarray, v: np.ndarray, start=datetime(2023, 11, 3), time_name="valid_time"):
    """Hourly u10/v10 where every grid point carries the per-hour value u[k], v[k]."""
    t = [start + timedelta(hours=k) for k in range(len(u))]
    shape = (len(t), LAT.size, LON.size)
    xr.Dataset({"u10": ((time_name, "latitude", "longitude"),
                        np.broadcast_to(np.asarray(u, np.float32)[:, None, None], shape).copy()),
                "v10": ((time_name, "latitude", "longitude"),
                        np.broadcast_to(np.asarray(v, np.float32)[:, None, None], shape).copy())},
               coords={time_name: t, "latitude": LAT, "longitude": LON}).to_netcdf(path)
    return path


def two_day_hourly():
    # day 1: u flips +10/-10 every hour (mean 0), v = 4 -> daily vector (0, 4), not a 10 m/s "speed mean"
    # day 2: u = 6, v = -2
    u = np.r_[np.tile([10.0, -10.0], 12), np.full(24, 6.0)]
    v = np.r_[np.full(24, 4.0), np.full(24, -2.0)]
    return u, v


def rotated(grid, ue, vn):
    lam = np.deg2rad(grid.lon2d)
    return ue * np.cos(lam) + vn * np.sin(lam), -ue * np.sin(lam) + vn * np.cos(lam)


def test_daily_mode_averages_each_component_per_utc_day(tmp_path, grid):
    u, v = two_day_hourly()
    ds = build_forcing(grid, era5_path=write_hourly_era5(tmp_path / "e.nc", u, v), wind_time_mode="daily")
    for k, (ue, vn) in enumerate([(0.0, 4.0), (6.0, -2.0)]):
        ex, ey = rotated(grid, ue, vn)
        assert np.allclose(ds["wind_x"].values[k], ex, atol=1e-4)
        assert np.allclose(ds["wind_y"].values[k], ey, atol=1e-4)
    assert np.allclose(np.hypot(ds["wind_x"][0], ds["wind_y"][0]), 4.0, atol=1e-4)  # components, not speeds


@pytest.mark.parametrize("time_name", ["valid_time", "time"])
def test_daily_mode_reads_valid_time_and_old_time_name(tmp_path, grid, time_name):
    u, v = two_day_hourly()
    ds = build_forcing(grid, era5_path=write_hourly_era5(tmp_path / "e.nc", u, v, time_name=time_name),
                       wind_time_mode="daily")
    assert ds["wind_x"].dims == ("time", "y", "x")
    assert ds.sizes["time"] == 2


def test_daily_mode_keeps_one_midnight_utc_timestamp_per_day(tmp_path, grid):
    u, v = np.full(24 * 3 - 5, 1.0), np.zeros(24 * 3 - 5)       # last day is partial (19 hours)
    ds = build_forcing(grid, era5_path=write_hourly_era5(tmp_path / "e.nc", u, v), wind_time_mode="daily")
    expected = np.array(["2023-11-03", "2023-11-04", "2023-11-05"], dtype="datetime64[ns]")
    assert (ds["time"].values == expected).all()
    assert ds.attrs["wind_hours_per_day_min"] == 19 and ds.attrs["wind_hours_per_day_max"] == 24


def test_daily_output_shape_and_metadata(tmp_path, grid):
    u, v = two_day_hourly()
    ds = build_forcing(grid, era5_path=write_hourly_era5(tmp_path / "e.nc", u, v),
                       cmems_path=write_cmems(tmp_path / "c.nc"), wind_time_mode="daily")
    assert ds["wind_x"].shape == ds["wind_y"].shape == (2, *grid.shape)
    assert ds["current_x"].shape == grid.shape                     # CMEMS stays a 2-D time mean
    assert ds.attrs["wind_time_mode"] == "daily"
    assert "UTC-daily mean" in ds.attrs["averaging"] and "currents: time mean" in ds.attrs["averaging"]
    assert np.isfinite(ds["wind_x"].values).all()


def test_mean_mode_is_the_unchanged_default(tmp_path, grid):
    u, v = two_day_hourly()
    path = write_hourly_era5(tmp_path / "e.nc", u, v)
    default, mean = build_forcing(grid, era5_path=path), build_forcing(grid, era5_path=path, wind_time_mode="mean")
    assert default["wind_x"].dims == ("y", "x") and "time" not in default.coords
    assert default.attrs["averaging"] == "time mean over the source files"
    assert default.attrs["wind_time_mode"] == "mean"
    xr.testing.assert_identical(default, mean)
    ex, ey = rotated(grid, u.mean(), v.mean())
    assert np.allclose(default["wind_x"], ex, atol=1e-4) and np.allclose(default["wind_y"], ey, atol=1e-4)
    # with whole days only, the mean of the daily layers is the overall time mean
    daily = build_forcing(grid, era5_path=path, wind_time_mode="daily")
    assert np.allclose(daily["wind_x"].mean("time"), default["wind_x"], atol=1e-5)


def test_daily_mode_rejects_bad_inputs(tmp_path, grid):
    with pytest.raises(ValueError, match="wind_time_mode"):
        build_forcing(grid, era5_path=write_era5(tmp_path / "e.nc"), wind_time_mode="hourly")
    no_time = tmp_path / "nt.nc"
    with xr.open_dataset(write_era5(tmp_path / "e2.nc")) as src:
        src.isel(valid_time=0, drop=True).load().to_netcdf(no_time)
    with pytest.raises(ValueError, match="time axis"):
        build_forcing(grid, era5_path=no_time, wind_time_mode="daily")


def test_load_old_2d_and_new_daily_forcing(tmp_path, grid):
    from antarctic_routing.ingestion.forcing import load_forcing_fields

    u, v = two_day_hourly()
    era5 = write_hourly_era5(tmp_path / "e.nc", u, v)
    old = tmp_path / "old.nc"                      # as written before wind_time_mode existed
    legacy = build_forcing(grid, era5_path=era5, cmems_path=write_cmems(tmp_path / "c.nc"))
    del legacy.attrs["wind_time_mode"]
    legacy.to_netcdf(old)
    currents, winds = load_forcing(old, expected_shape=grid.shape)
    assert winds[0].shape == grid.shape and currents[0].shape == grid.shape
    f = load_forcing_fields(old, expected_shape=grid.shape)
    assert f.wind_times is None and f.wind_time_mode == "mean"

    new = tmp_path / "daily.nc"
    build_forcing(grid, era5_path=era5, wind_time_mode="daily").to_netcdf(new)
    f = load_forcing_fields(new, expected_shape=grid.shape)
    assert f.wind_time_mode == "daily" and f.currents is None
    assert f.winds[0].shape == (2, *grid.shape)
    assert list(f.wind_times.astype("datetime64[D]").astype(str)) == ["2023-11-03", "2023-11-04"]
    with pytest.raises(ValueError, match="daily"):          # ForecastContext path refuses, never misuses
        load_forcing(new, expected_shape=grid.shape)
    with pytest.raises(ValueError, match="grid"):
        load_forcing_fields(new, expected_shape=(3, 3))


REAL_ERA5 = os.environ.get("ANTROUTE_REAL_ERA5_SAMPLE")


@pytest.mark.skipif(not REAL_ERA5 or not Path(REAL_ERA5).is_file(),
                    reason="set ANTROUTE_REAL_ERA5_SAMPLE to a real hourly ERA5 u10/v10 NetCDF")
def test_daily_mode_on_real_era5_sample(grid):
    with xr.open_dataset(REAL_ERA5) as src:
        t = next(n for n in ("valid_time", "time") if n in src.dims)
        days = np.unique(src[t].values.astype("datetime64[D]"))
        first = src.sel({t: src[t].dt.floor("D") == src[t].values[0].astype("datetime64[D]")})
        u0, v0 = first["u10"].mean(t).values, first["v10"].mean(t).values
        lat, lon = src["latitude"].values, src["longitude"].values
    ds = build_forcing(grid, era5_path=REAL_ERA5, wind_time_mode="daily")
    assert ds["wind_x"].shape == (days.size, *grid.shape)
    assert (ds["time"].values.astype("datetime64[D]") == days).all()
    # day 1 regridded independently from the raw hourly file
    from antarctic_routing.preprocessing.harmonize import regrid_vector_latlon
    ex, ey = regrid_vector_latlon(u0, v0, lat, lon, grid)
    ok = np.isfinite(ex)
    assert np.allclose(ds["wind_x"].values[0][ok], ex[ok], atol=1e-4)
    assert np.allclose(ds["wind_y"].values[0][ok], ey[ok], atol=1e-4)


# ---------------------------------------------------------- daily current mode
def write_daily_cmems(path: Path, uo, vo, start=datetime(2023, 11, 3), days=None):
    """Daily CMEMS-style uo/vo (ascending lat, two depth levels) with per-day values uo[k], vo[k]."""
    days = days if days is not None else [start + timedelta(days=k) for k in range(len(uo))]
    lat = LAT[::-1]
    shape = (len(days), 2, lat.size, LON.size)
    u = np.broadcast_to(np.asarray(uo, np.float32)[:, None, None, None], shape).copy()
    v = np.broadcast_to(np.asarray(vo, np.float32)[:, None, None, None], shape).copy()
    u[:, 1] = v[:, 1] = 99.0                     # deeper level must be ignored
    u[:, :, :4, :] = v[:, :, :4, :] = np.nan      # CMEMS land/ice-shelf cells are NaN
    xr.Dataset({"uo": (("time", "depth", "latitude", "longitude"), u),
                "vo": (("time", "depth", "latitude", "longitude"), v)},
               coords={"time": days, "depth": [0.494, 1.54], "latitude": lat, "longitude": LON}).to_netcdf(path)
    return path


def test_daily_currents_keep_each_day_and_the_surface_level(tmp_path, grid):
    path = write_daily_cmems(tmp_path / "c.nc", uo=[0.2, -0.1, 0.0], vo=[0.0, 0.3, -0.4])
    ds = build_forcing(grid, cmems_path=path, current_time_mode="daily")
    assert ds["current_x"].dims == ("time", "y", "x") and ds["current_x"].shape == (3, *grid.shape)
    expected = np.array(["2023-11-03", "2023-11-04", "2023-11-05"], dtype="datetime64[ns]")
    assert (ds["time"].values == expected).all()
    for k, (ue, vn) in enumerate([(0.2, 0.0), (-0.1, 0.3), (0.0, -0.4)]):
        ex, ey = rotated(grid, ue, vn)
        assert np.allclose(ds["current_x"].values[k], ex, atol=1e-5)
        assert np.allclose(ds["current_y"].values[k], ey, atol=1e-5)
    assert ds.attrs["current_time_mode"] == "daily" and ds.attrs["current_missing_days"] == 0
    assert ds.attrs["current_samples_per_day_min"] == ds.attrs["current_samples_per_day_max"] == 1
    assert "currents: UTC-daily mean" in ds.attrs["averaging"]
    assert np.isfinite(ds["current_x"].values).all()


def test_daily_currents_report_missing_days_and_zero_fill(tmp_path, grid):
    days = [datetime(2023, 11, 3), datetime(2023, 11, 4), datetime(2023, 11, 7)]   # 5th and 6th absent
    ds = build_forcing(grid, cmems_path=write_daily_cmems(tmp_path / "c.nc", [0.1] * 3, [0.0] * 3, days=days),
                       current_time_mode="daily")
    assert ds.attrs["current_missing_days"] == 2
    small = tmp_path / "small.nc"                      # source box smaller than the grid -> zero-filled cells
    with xr.open_dataset(write_daily_cmems(tmp_path / "c2.nc", [0.1], [0.0])) as src:
        src.sel(latitude=slice(-62, -58), longitude=slice(-65, -58)).load().to_netcdf(small)
    ds = build_forcing(grid, cmems_path=small, current_time_mode="daily")
    assert 0 < ds.attrs["current_filled_fraction"] < 1
    assert np.isfinite(ds["current_x"].values).all()


def test_missing_days_helper():
    from antarctic_routing.ingestion.forcing import missing_days

    d = np.array(["2023-11-01", "2023-11-02", "2023-11-05"], dtype="datetime64[ns]")
    assert missing_days(d) == 2 and missing_days(d[:2]) == 0 and missing_days(np.array([])) == 0


def test_daily_winds_and_daily_currents_share_one_time_axis(tmp_path, grid):
    u, v = two_day_hourly()
    era5 = write_hourly_era5(tmp_path / "e.nc", u, v)
    ds = build_forcing(grid, era5_path=era5, cmems_path=write_daily_cmems(tmp_path / "c.nc", [0.1, 0.2], [0.0, 0.0]),
                       wind_time_mode="daily", current_time_mode="daily")
    assert ds["wind_x"].shape == ds["current_x"].shape == (2, *grid.shape)
    assert "wind: UTC-daily" in ds.attrs["averaging"] and "currents: UTC-daily" in ds.attrs["averaging"]
    other = write_daily_cmems(tmp_path / "c3.nc", [0.1, 0.2], [0.0, 0.0], start=datetime(2023, 12, 1))
    with pytest.raises(ValueError, match="different days"):
        build_forcing(grid, era5_path=era5, cmems_path=other, wind_time_mode="daily", current_time_mode="daily")
    with pytest.raises(ValueError, match="current_time_mode"):
        build_forcing(grid, cmems_path=other, current_time_mode="hourly")


def test_current_mean_mode_is_the_unchanged_default(tmp_path, grid):
    path = write_daily_cmems(tmp_path / "c.nc", uo=[0.2, 0.4], vo=[0.0, 0.0])
    default = build_forcing(grid, cmems_path=path)
    assert default["current_x"].dims == ("y", "x") and "time" not in default.coords
    assert default.attrs["averaging"] == "time mean over the source files"
    assert default.attrs["current_time_mode"] == "mean"
    daily = build_forcing(grid, cmems_path=path, current_time_mode="daily")
    assert np.allclose(daily["current_x"].mean("time"), default["current_x"], atol=1e-6)


def test_load_daily_currents(tmp_path, grid):
    from antarctic_routing.ingestion.forcing import load_forcing_fields

    path = tmp_path / "daily_c.nc"
    build_forcing(grid, cmems_path=write_daily_cmems(tmp_path / "c.nc", [0.1, 0.2], [0.0, 0.0]),
                  current_time_mode="daily").to_netcdf(path)
    f = load_forcing_fields(path, expected_shape=grid.shape)
    assert f.current_time_mode == "daily" and f.winds is None and f.wind_times is None
    assert f.currents[0].shape == (2, *grid.shape)
    assert list(f.current_times.astype("datetime64[D]").astype(str)) == ["2023-11-03", "2023-11-04"]
    with pytest.raises(ValueError, match="daily .*currents"):   # ForecastContext path refuses, never misuses
        load_forcing(path, expected_shape=grid.shape)
    old = tmp_path / "old_c.nc"
    build_forcing(grid, cmems_path=write_cmems(tmp_path / "c2.nc")).to_netcdf(old)
    f = load_forcing_fields(old, expected_shape=grid.shape)
    assert f.current_times is None and f.current_time_mode == "mean"


REAL_CMEMS = os.environ.get("ANTROUTE_REAL_CMEMS_SAMPLE")


@pytest.mark.skipif(not REAL_CMEMS or not Path(REAL_CMEMS).is_file(),
                    reason="set ANTROUTE_REAL_CMEMS_SAMPLE to a real daily CMEMS uo/vo NetCDF")
def test_daily_currents_on_real_cmems_sample(grid):
    from antarctic_routing.preprocessing.harmonize import regrid_vector_latlon

    with xr.open_dataset(REAL_CMEMS) as src:
        days = src["time"].values.astype("datetime64[D]")
        u0, v0 = src["uo"].isel(time=0, depth=0).values, src["vo"].isel(time=0, depth=0).values
        lat, lon = src["latitude"].values, src["longitude"].values
    ds = build_forcing(grid, cmems_path=REAL_CMEMS, current_time_mode="daily")
    assert ds["current_x"].shape == (days.size, *grid.shape)
    assert (ds["time"].values.astype("datetime64[D]") == days).all()
    ex, ey = regrid_vector_latlon(u0, v0, lat, lon, grid)
    ok = np.isfinite(ex)
    assert np.allclose(ds["current_x"].values[0][ok], ex[ok], atol=1e-6)
    assert np.allclose(ds["current_y"].values[0][ok], ey[ok], atol=1e-6)
    assert np.hypot(ds["current_x"], ds["current_y"]).max() < 3.0
