"""OSI SAF fixture files following the product layout (not real data).

``layout="cf"`` uses CF ``flag_values``/``flag_meanings``. ``layout="osi401d"``
mirrors the structure of a real THREDDS file checked for 2026-09-15
(``product_id = OSI-401-d``): float32 bit-coded ``status_flag`` described only
by free-text ``flag_descriptions``, with ice_conc NaN over land and missing.
"""

from datetime import date

import numpy as np
import xarray as xr
from pyproj import Transformer

from antarctic_routing.ingestion.osisaf_reader import OSI_PROJ4

# Copied verbatim from the real file's status_flag attributes.
OSI401D_FLAG_DESCRIPTIONS = (
    "no bits set (0): nominal value of the sea ice concentration (sic) retrieved by the algorithm\n"
    "bit 2 (10): Open water filter screening applied \n"
    " bit 3 (100): NWP skin temperature screening applied \n"
    " bit 4 (1000): 37 GHz polarization difference screening applied \n"
    " bit 5 (10000): Maximum sea ice climatology screening applied \n"
    " bit 6 (100000): Lake mask \n"
    " bit 7 (1000000): Land mask \n"
    " bit 8 (10000000): Near coast grid points \n"
    " bit 9 (100000000): Missing value"
)
OSI401D_BITS = {"open_water": 2, "lake": 32, "land": 64, "near_coast": 128, "missing": 256}


def _fixture_conc(lat):
    """Percent concentration, linear in latitude: 0% at -58, 100% at -64."""
    return np.clip((-58.0 - lat) / 6.0 * 100.0, 0, 100)


def write_osisaf(path, day: date, land_box=None, missing_box=None, layout="cf", coast_box=None,
                 status_attrs=None):
    to_osi = Transformer.from_crs("EPSG:4326", OSI_PROJ4, always_xy=True)
    xs, ys = to_osi.transform([-80, -80, -40, -40, -60], [-70, -50, -70, -50, -60])
    xc = np.arange(np.floor(min(xs) / 1e4) * 10, np.ceil(max(xs) / 1e4) * 10 + 10, 10.0)  # km
    yc = np.arange(np.ceil(max(ys) / 1e4) * 10, np.floor(min(ys) / 1e4) * 10 - 10, -10.0)  # descending, km
    xx, yy = np.meshgrid(xc * 1000, yc * 1000)
    lon, lat = Transformer.from_crs(OSI_PROJ4, "EPSG:4326", always_xy=True).transform(xx, yy)
    conc = _fixture_conc(lat)
    status = np.zeros(conc.shape, np.int8)
    def box(b):  # (lat_min, lat_max, lon_min, lon_max)
        return (lat >= b[0]) & (lat <= b[1]) & (lon >= b[2]) & (lon <= b[3])

    if layout == "osi401d":
        status = np.zeros(conc.shape, np.float32)
        status[conc == 0] = OSI401D_BITS["open_water"]
        if coast_box:
            status[box(coast_box)] += OSI401D_BITS["near_coast"]  # concentration stays valid
        if land_box:
            status[box(land_box)] = OSI401D_BITS["land"]
            conc[box(land_box)] = np.nan
        if missing_box:
            status[box(missing_box)] = OSI401D_BITS["missing"]
            conc[box(missing_box)] = np.nan
        flag_attrs = {"units": "1", "grid_mapping": "Polar_Stereographic_Grid",
                      "flag_descriptions": OSI401D_FLAG_DESCRIPTIONS}
        product_id, product_version = "OSI-401-d", "4.1"
    else:
        if land_box:
            status[box(land_box)] = 1
            conc[box(land_box)] = np.nan
        if missing_box:
            conc[box(missing_box)] = np.nan
        flag_attrs = {"flag_values": np.array([0, 1, 2], np.int8), "flag_meanings": "nominal land lake"}
        product_id, product_version = "OSI-401-b", None
    if status_attrs is not None:
        flag_attrs = status_attrs
    ds = xr.Dataset(
        {
            "ice_conc": (("time", "yc", "xc"), conc[None].astype("float32"),
                         {"units": "%", "grid_mapping": "Polar_Stereographic_Grid"}),
            "status_flag": (("time", "yc", "xc"), status[None], flag_attrs),
            "Polar_Stereographic_Grid": ((), np.int32(0), {
                "grid_mapping_name": "polar_stereographic",
                "straight_vertical_longitude_from_pole": 0.0,
                "latitude_of_projection_origin": -90.0,
                "standard_parallel": -70.0,
                "false_easting": 0.0, "false_northing": 0.0,
                "semi_major_axis": 6378273.0, "semi_minor_axis": 6356889.44891,
                "proj4_string": OSI_PROJ4,
            }),
        },
        coords={"time": [np.datetime64(f"{day.isoformat()}T12:00")],
                "xc": ("xc", xc, {"units": "km"}), "yc": ("yc", yc, {"units": "km"})},
        attrs={"product_id": product_id, **({"product_version": product_version} if product_version else {})},
    )
    ds.to_netcdf(path)
    return path
