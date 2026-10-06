# Reproducing the frozen real-data demo

Research / decision-support only. Not certified for navigation or safety-critical operational use.

The frozen demo is one real-data planning window: a sea-ice forecast issued on **2023-11-14**, 14 candidate
departure days, a 6-day routing horizon and 200 joint sea-ice + iceberg scenarios. Nothing is trained, fitted or
tuned when it is reproduced.

## 1. Inputs (not in Git)

The inputs are large or licensed, so they live outside the repository under one data root
(`ANTROUTE_DATA_ROOT`). Paths and SHA-256 checksums are pinned in
[`frozen_demo/frozen_demo_2023-11-14.json`](frozen_demo/frozen_demo_2023-11-14.json).

| Input | Path under the data root | Source |
|---|---|---|
| Sea ice | `processed/sea_ice_25km.nc` | OSI SAF OSI-450-a + OSI-430-a, regridded with `antroute build-dataset` |
| Forecast model | `models/unet_real25_v1/best.pt` | `antroute train-forecast` on 2004-17, validated on 2018-20 |
| Winds | `iceberg/forcing/processed/forcing_era5_daily_25km_20231101_20231231.nc` | ERA5 u10/v10 via the CDS API, `antroute build-forcing --wind-time-mode daily` |
| Currents | `iceberg/forcing/processed/forcing_cmems_daily_25km_20231101_20231231.nc` | CMEMS `cmems_mod_glo_phy_my_0.083deg_P1D-m` (0.494 m), `antroute build-forcing --current-time-mode daily` |
| Icebergs | `iceberg/raw/archive/AntarcticIcebergs_20231109.csv` | Official USNIC Antarctic iceberg list |
| Drift calibration | `iceberg/hindcast/calibration/selection.json` | γ = 0.1 (selected on validation 2021-22) |
| Spread calibration | `iceberg/hindcast/uncertainty/spread_calibration_results.json` | c = 0.6053, q = 0 |

The configuration is the repository's `config/config.yaml`, also pinned by checksum.

## 2. Run

```bash
pip install -e ".[ml]"
export ANTROUTE_DATA_ROOT=/path/to/real-data
python scripts/reproduce_frozen_demo.py --out reports/frozen-demo --bundle artifacts/bundle
# to update the bundle the API image serves: cp artifacts/bundle/* deploy/bundle/
```

The script:

1. checks every input exists (**exit 2, BLOCKED** if not) and matches its checksum (**exit 3**);
2. reads γ and c back from the calibration files and checks them against the command (**exit 3** on a difference);
3. runs `antroute plan-window` with `--require-real-forcing` (schematic winds or currents are refused) and
   `--export-maps` (**exit 4** if it fails);
4. compares the result with the pinned expectation (**exit 5** on any difference);
5. with `--bundle`, writes a checksum-verified, read-only bundle for the API (`ANTROUTE_ARTIFACTS_DIR`).

Use a local disk for `--out` and `--bundle`. A synced network folder can serve a stale copy of a file just written;
the bundle loader's checksum check rejects such a bundle rather than serving it.

## 3. Expected result

| Quantity | Value |
|---|---|
| Selected departure | 2023-11-14 |
| Expected time | 37.5 h |
| Expected fuel index | 837.8 |
| Distance | 837.8 km |
| Breaches | 0 of 200 joint scenarios |
| Wilson 95 % upper bound | 1.88 % (budget 5 %) |
| Route | 36 grid cells, no iceberg presence on the route |
| Forcing | ERA5 daily winds + CMEMS daily currents (`real`) |
| `plan_window.json` canonical SHA-256 | `659bcedf3cef50f6d11e49432b3073c2b69295f04f5b01618ee25881379e7430` |

The canonical checksum is taken with file paths reduced to file names, so it is the same on any machine. When the
data root is the original `/mnt/project-files/real-data`, the file is also byte-identical to the original run
(`0d6ef85abec8e170…`). `departure_window.png` is byte-identical too, but only on a store that keeps bytes: the
project's shared file folder adds a C2PA content-credentials chunk to PNG files, so compare a copy taken from
there after removing that chunk. Other matplotlib or libpng versions can also change the PNG bytes.

## 4. Outputs

| File | Content |
|---|---|
| `plan_window.json` | Departure options, selected route, forcing and iceberg provenance |
| `forecast_maps.json` | Per-day observed / forecast maps for the dashboard (display only) |
| `departure_window.png` | Departure-window figure |
| `reproduction_report.json` | Input checksums, frozen parameters, environment, comparison |
| `bundle/bundle.json` | Bundle index with the SHA-256 of every bundle file and the limitations |

## 5. Environment

The reference run used Python 3.11.15, numpy 2.4.6, xarray 2026.9.0, torch 2.14.1 (CPU inference) and
netCDF4 1.7.4. `reproduction_report.json` records the versions of every run. Other library versions can change
floating-point results; the script then reports exit 5 rather than accepting a different answer.

## 6. What the result does not show

- The isotonic forecast-probability calibrator is **not** used in route risk; this result is not evidence that
  calibrated probabilities are used.
- 1.88 % is the Wilson upper bound for 0 breaches in 200 scenarios. It is a scenario-count floor, not evidence of
  forecast skill. The real-data trust horizon is 0 days against persistence.
- In this window no iceberg reaches the route, so the iceberg model does not affect the decision.
- The vessel is generic and its ice class is a placeholder.
