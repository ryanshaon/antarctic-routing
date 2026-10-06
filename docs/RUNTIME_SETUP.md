# Runtime setup for teammates

This guide runs the full product from a fresh clone: Real Historical Data, the 2026 forecast estimate, and the
historical seasonal analogue for any later date. The real data is not in Git. It ships as one runtime archive,
published as a GitHub Release asset.

**No personal API keys or accounts are needed to run the application.** The archive holds processed data only, and the
application reads local files. It does not contact Copernicus, ECMWF, OSI SAF or USNIC.

| Item | Value |
|---|---|
| Code | `https://github.com/HarshKunap/antarctic-routing`, branch `real-data/interactive-historical-mode`, commit `2115ddb` or later |
| Runtime release | tag `runtime-2115ddb`: <https://github.com/HarshKunap/antarctic-routing/releases/tag/runtime-2115ddb> |
| Archive | `antroute-real-data-runtime-full-2115ddb.zip`: 161 MB download, 262 MB extracted, 198 data files |
| Archive SHA-256 | `6de9a2cd826f20bf68f1bcf4715aa2849bab7187873010921213a41c1dd994b8` |
| Python | 3.11 or newer |

## 1. Clone the repository

Clone with `core.autocrlf=false`. `config/config.yaml` and the files in `deploy/bundle/` are checked by SHA-256,
and Windows line-ending conversion would change their bytes so the checks fail.

```bash
git clone -c core.autocrlf=false --branch real-data/interactive-historical-mode \
    https://github.com/HarshKunap/antarctic-routing
cd antarctic-routing
```

## 2. Download the runtime release asset

From the `runtime-2115ddb` release, download:
- `antroute-real-data-runtime-full-2115ddb.zip`;
- `antroute-real-data-runtime-full-2115ddb.zip.sha256`.

Check the archive before extracting it:

```bash
sha256sum -c antroute-real-data-runtime-full-2115ddb.zip.sha256          # Linux: "OK" (macOS: shasum -a 256 -c ...)
```
```powershell
(Get-FileHash .\antroute-real-data-runtime-full-2115ddb.zip -Algorithm SHA256).Hash   # Windows: compare with the table
```

## 3. Extract it

Extract it anywhere outside the repository. It creates one folder, `antroute-real-data-runtime-full-2115ddb/`, which
contains `processed/`, `models/`, `iceberg/`, `sea-ice-2024-25/` and `seasonal-analogue/`, plus:
- `RUNTIME-MANIFEST.json`, with every file's size and SHA-256 and the code commit the archive matches;
- `SHA256SUMS.txt`.

## 4. Set `ANTROUTE_DATA_ROOT`

Point it at the extracted folder: the one that contains `processed/`.

```bash
export ANTROUTE_DATA_ROOT=/path/to/antroute-real-data-runtime-full-2115ddb
```
```powershell
$env:ANTROUTE_DATA_ROOT = "C:\path\to\antroute-real-data-runtime-full-2115ddb"
```

## 5. Set `ANTROUTE_ARTIFACTS_DIR`

This is the verified frozen-demo bundle that ships in the repository.

```bash
export ANTROUTE_ARTIFACTS_DIR="$PWD/deploy/bundle"
```
```powershell
$env:ANTROUTE_ARTIFACTS_DIR = "$PWD\deploy\bundle"
```

## 6. Install the dependencies

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,ml]"   # API, NetCDF reader, PyTorch (CPU is enough), tests
```

## 7. Start the dashboard

Run this from the repository root, in the same shell as steps 4–6:

```bash
antroute serve --artifacts-dir deploy/bundle
```

The first plan takes longer, because the archive is loaded and checksum-verified once (about half a minute here).

## 8. Open the local URL

Open <http://127.0.0.1:8000>. Choose **Drake Passage** → **Bransfield Strait** and try these dates:

| Date | What you get |
|---|---|
| 2023-11-14 | Real Historical Data: the frozen run, 37.5 h, 0 of 200 scenarios breach |
| 2026-11-19 | Forecast / hackathon estimate: proxy forecast with the 2026-10-01 USNIC list |
| 2027-08-14 | Historical seasonal analogue (analogue date 2024-08-14). Winter ice means "not recommended", with very low confidence |
| 2027-03-15 | Historical seasonal analogue (analogue date 2024-03-15) |
| 2027-12-15 | Historical seasonal analogue (analogue date 2024-12-15) |

Any future date is accepted. A date the proxy forecast cannot serve becomes a labelled seasonal-analogue estimate,
and the dates it used are listed under **Data & Confidence**.

## 9. No personal API keys are needed to run it

The archive contains the processed inputs that the application verifies by SHA-256 at start. Nothing in it is a
credential, and the application never asks for one. Leave `.env` empty of data-service credentials.

## 10. Credentials are only for refreshing upstream data

You need your own accounts only if you want to download or refresh the scientific data yourself, outside this
application:
- **ERA5 winds:** a Copernicus Climate Data Store account and API key.
- **CMEMS currents:** a Copernicus Marine account.
- **OSI SAF sea ice and USNIC iceberg lists:** public, no account needed.

Keep such credentials in your own user settings or a git-ignored `.env`. Never commit them, and never put them in a
runtime archive. Refreshed data has new SHA-256 values, so `config/real_historical.json` and a new runtime release
would have to be made together.

## Optional checks

```bash
python scripts/reproduce_frozen_demo.py --out /tmp/frozen-run   # expect "REPRODUCED: selected 2023-11-14 ..."
python -m pytest                                                 # the real-data tests run when ANTROUTE_DATA_ROOT is set
```

## Troubleshooting

| Message | Fix |
|---|---|
| `blocked: input … not found at $ANTROUTE_DATA_ROOT/…` | `ANTROUTE_DATA_ROOT` points at the wrong folder. It must be the folder that contains `processed/` |
| checksum or `failed` message for a data file | Re-download the archive and check its SHA-256 |
| checksum message for `config/…` or `deploy/bundle/…` | The clone converted line endings. Re-clone with `-c core.autocrlf=false` |
| `no historical seasonal analogue: blocked` | `seasonal-analogue/` is missing from `ANTROUTE_DATA_ROOT`. Use this full archive, not an older runtime bundle |
