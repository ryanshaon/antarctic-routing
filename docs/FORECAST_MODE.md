# Forecast mode (hackathon estimate for future dates)

> **Forecast estimate:** future-year Antarctic observations/forcing are not fully available in this demo runtime.
> Results use the latest available real seasonal data and explicitly labelled proxy/estimated forcing where
> required. This is a research/hackathon estimate, not a certified navigation route.

## What it is

`POST /real/plan` and `POST /real/simulate` accept **any** departure date after the end of the real archive, such
as 2026-11-19, 2027-08-14 or 2030-08-14. There is no maximum date and no seasonal restriction. The mode is chosen
from the date alone:

| requested date | mode | `metadata.mode` | `execution_mode` | `data_status` |
|---|---|---|---|---|
| inside the real archive (last day 2025-02-28) | Real Historical Data | `historical` | `real` | `historical` |
| any date after the archive | Forecast / hackathon estimate | `forecast` | `modelled` | `forecast_estimate` |

A forecast estimate is served by one of two pathways, named in `metadata.forecast.pathway`:

1. **`proxy_forecast`** (below): tried first. It serves in-season dates whose calendar day an archive season can
   run, when an official iceberg list at most 120 days old exists.
2. **`seasonal_analogue`** ([Historical seasonal analogue](#historical-seasonal-analogue)): used automatically for
   every other date. `pathway_reason` records why the proxy forecast could not serve it.

Only when neither pathway can serve a date is it refused (HTTP 422 `forecast_unavailable`, with both reasons).

Historical requests run exactly as before. Their output is unchanged byte for byte.

```bash
curl -s -X POST localhost:8000/real/plan -H 'content-type: application/json' \
  -d '{"origin":{"preset":"drake_passage"},"destination":{"preset":"bransfield_strait"},"issue":"2026-11-19"}'
```

Forecast mode adds no new routing, risk or departure-window logic. The same planner, Wilson risk, A* routing,
departure window and simulation run unchanged, from an **analogue start**:

1. **Analogue date.** The analogue is the requested calendar day in the latest archive season that can run the
   whole window. For 2026-11-19 this is 2024-11-19, an offset of 730 days.
2. **Engine run.** The engine runs on the analogue days, using:
   - the frozen U-Net with 200 members and seed 42;
   - ERA5 and CMEMS reanalysis of those days.
3. **Icebergs.** The latest official USNIC list on or before the requested date is used, and it may be at most
   120 days old. Each berg is held at its last reported position, then drifted with the calibrated ensemble
   (beta 0.1, alpha scale 0.1, spread factor 0.6053).
4. **Date shift.** Engine dates are shown on the requested timeline (engine day + offset). Real-world dates,
   such as the iceberg list date and the provenance of the analogue inputs, are never shifted.

`GET /real/historical/dates` returns `forecast.{dates, first, last}`: the dates the proxy forecast serves. For
Drake Passage → Bransfield Strait this is 2026-11-14 … 2027-01-29 (77 dates). It also returns
`estimate.{any_date_after, pathways, analogue.available}`: every date after `any_date_after` (2025-02-28) is
accepted, and the dates outside the proxy-forecast range get a historical seasonal analogue.

A past date inside the archive period that the archive does not cover (e.g. 2024-06-15) stays in historical mode
and is refused there as before (`out_of_coverage`). It is never replaced by another date.

## What is real and what is proxy

`metadata.forecast` records every input with its status and the dates actually used:

| input | status | what is used |
|---|---|---|
| sea-ice starting state | `proxy_analogue` | real OSI SAF observations of the analogue days (`observed_window_used`), **not** observations of the requested year |
| sea-ice forecast | `forecast` | the frozen U-Net and residual-bank scenarios from that proxy start |
| winds / currents | `proxy_analogue_reanalysis` | ERA5 / CMEMS reanalysis of the analogue days (`dates_used`, files and SHA-256). Not a forecast, and not a measurement of the requested year |
| icebergs | `observed_snapshot_held` | an official USNIC list (`snapshot_date`, `file`, `sha256`, age in days, number of source bergs, ids in the grid), held and then drifted |

Three places in the output that would otherwise say "observed" are relabelled `proxy_analogue_observed`:
- the scenario layer 0;
- the sailed ice in a simulation;
- each simulated day's `sea_ice_on_segment`, which also carries its `analogue_date`.

`observations_for_requested_dates` states that no observation dated on or after the requested date is used.
`metadata.forecast.limitations` repeats the engine's limitations, with the historical iceberg-age line ("up to 14 days
old") restated for the held snapshot: it may be up to 120 days old, and its actual age is in `icebergs`. This is
wording only; no rule or calculation changes.

The two 2026 USNIC lists are pinned in `config/real_historical.json` under `inputs.forecast_icebergs`:
- `AntarcticIcebergs_20260924.csv`;
- `AntarcticIcebergs_20261001.csv`.

They are verified only when forecast mode is used, so historical mode does not depend on them.

The dashboard shows a **REAL HISTORICAL DATA** or **FORECAST / HACKATHON ESTIMATE** badge next to the date. The
date field is the only date control and has no `min` or `max`: it takes any listed historical date and any date after
the archive, and the mode follows from the date. Its default is 2026-11-19 when the route's forecast range includes
it; a date the user chose is never replaced.
A forecast result carries:
- the disclosure above;
- the forecast banners;
- a Data & Confidence panel that names every proxy and its dates.

Nothing in this mode changes the frozen datasets, checkpoint, calibration, parameters or historical results.

## Historical seasonal analogue

For a date the proxy forecast cannot serve (off season, years ahead, or no recent iceberg list), the server builds
a **model-free** estimate from real data of earlier years (`src/antarctic_routing/seasonal_analogue.py`):

| input | status | what is used |
|---|---|---|
| analogue date | `analogue_date` | the requested calendar day in the most recent earlier year with real ERA5 winds and CMEMS currents for the whole run and real sea ice in earlier years (2027-08-14 → 2024-08-14; 2028-03-03 → 2024-03-03) |
| sea ice | `historical_analogue_ensemble` | real OSI SAF daily observations of the same calendar days in up to 7 earlier years (the analogue year excluded), each shifted −7…+7 days; 200 members drawn from those sequences. No sea-ice model runs. `member_windows` lists every sequence used |
| winds / currents | `historical_reanalysis_analogue` | ERA5 / CMEMS reanalysis of the analogue days (`dates_used`, files and SHA-256). Not a forecast |
| icebergs | `observed_snapshot_held` or `analogue_year_snapshot` | the latest official USNIC list if it is at most 120 days old at the requested date (held, then drifted); otherwise the analogue date's own official list (at most 14 days old at the analogue date, else 120), drifted with the calibrated ensemble |

Every engine day is shown on the requested timeline (`date_mapping`), and real-world dates (list dates, analogue
dates, files) are never shifted. `confidence.level` is `low`, or `very low` when fewer than 5 analogue years exist
or when the analogue-year icebergs are used and the years disagree strongly about the ice (spread ≥ 0.10).
`confidence.caveats` explains each reason. Low confidence is shown, never used to refuse a date.

The off-season data (March–October 2017–2024 sea ice, 2024 ERA5 and CMEMS, and 36 USNIC lists of 2024) are pinned
in `config/real_historical.json` under `inputs.analogue_*` and verified only when this pathway is used.

Example results for Drake Passage → Bransfield Strait:
- **2027-08-14**: analogue 2024-08-14; winter ice at Bransfield, so no departure meets the 5% budget. The
  least-risky option (24 Aug) is shown for reference only, with confidence very low.
- **2028-03-03**: analogue 2024-03-03; departure recommended on the day, 37.8 h.

The voyage simulation sails through the analogue year's observed ice. OSI SAF has no data for 2024-09-12 … 17,
so a simulation whose voyage needs those days returns `not_simulated` with that reason. The plan itself still works.

## Why a 2026 result is not an operational forecast

- **Sea ice and weather come from another year.** The ice, winds and currents are those of 2024-25. They show
  what a typical season can look like on that calendar day, not what will happen in 2026.
- **The bergs are not tracked to the departure date.** Their positions are weeks old, and the drift between the
  list date and the departure is not modelled. The bergs are only held in place over that gap.
- **The method has not been validated.** The analogue approach has not been checked against what actually
  happened in any season.
- **The vessel is a placeholder.** The vessel and the fuel index are generic, and nothing is certified.

## What true live forecasting would need

1. **Near-real-time sea ice.** Daily OSI SAF OSI-401 sea-ice concentration up to the issue date, which the
   OSI-401-d compatibility work prepares. This would replace the analogue start.
2. **Operational weather and ocean forecasts.** Winds from ECMWF HRES/ENS and currents from the CMEMS analysis &
   forecast products, issued on the issue date, instead of reanalysis.
3. **Current iceberg positions.** The current USNIC list, or satellite iceberg detections, each issue day, with
   drift starting from that list's date.
4. **Live validation.** Validation of the whole chain on live, issue-time inputs, run through a season, before
   any skill claim is made.
5. **Operational infrastructure.** Data ingestion, monitoring and fall-back handling, together with
   vessel-specific limits and certification, and qualified human oversight.
