# Deployment bundle

`deploy/bundle/` holds the real-data bundle the API image serves (`ANTROUTE_ARTIFACTS_DIR=/app/deploy/bundle`).
It is the output of `scripts/reproduce_frozen_demo.py --bundle`: `bundle.json` lists the SHA-256 of every file,
and the API refuses to serve the bundle if any file differs (`/ready` then returns 503).

- **Contents:** `bundle.json`, `plan_window.json`, `forecast_maps.json`, `manifest.json`, `departure_window.png`
  (about 0.8 MB). These are derived results of one frozen real-data run, not raw datasets or model weights.
- **Sources** (credited in the dashboard): OSI SAF sea ice, ERA5 winds (Copernicus Climate Change Service),
  CMEMS currents (Copernicus Marine Service), USNIC iceberg positions.
- **Refresh:** a worker rebuilds it on local disk and copies the five files here unchanged:
  ```bash
  ANTROUTE_DATA_ROOT=/path/to/real-data python scripts/reproduce_frozen_demo.py --out /tmp/run --bundle /tmp/bundle
  cp /tmp/bundle/* deploy/bundle/
  ```
- **Do not edit or re-encode the files.** Any byte change, including metadata an image tool or a file-sync
  service adds to the PNG, fails verification.

Without `deploy/bundle/` the image still builds. With `ANTROUTE_ARTIFACTS_DIR=/app/deploy/bundle` (as on Render),
the API then reports the real data as failed and `/ready` stays 503, so the deploy never goes live.
