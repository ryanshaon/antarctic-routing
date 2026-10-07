# One-call route plan: `POST /real/plan` (Real Historical Data)

`POST /real/plan` runs the whole existing real-data chain server-side and returns one result for a product page:

```
S1 locations -> route horizon -> joint scenarios (OSI SAF ice + frozen U-Net + ERA5/CMEMS daily forcing + USNIC bergs
with calibrated drift) -> departure window (time-dependent A*, Wilson risk budget) -> selected route -> risk split
-> daily timeline + map layers
```

It adds no new science. It calls the same functions as `/real/historical/departures` (`HistoricalPlanner.world`, `plan_from_issue`) with the frozen parameters (200 members, seed 42). For the configured route it therefore reproduces the frozen 2023-11-14 result exactly.

## Request

```json
{"origin": {"preset": "drake_passage"}, "destination": {"preset": "bransfield_strait"},
 "issue": "2023-11-14", "window_days": 14, "include_layers": true}
```

- `origin` / `destination` (required) take a preset id or `{"lat", "lon", "name"?}`. See [LOCATIONS.md](LOCATIONS.md) for presets and snapping.
- `issue` (required) is the forecast issue date. Sea ice is observed up to this day.
- `window_days` (1..14, default 14) is the number of candidate departure days, starting on the issue day.
- `include_layers` (default true) adds daily P(ice ≥ vessel limit) and P(iceberg) maps for the voyage days.

## Response (top level)

| key | content |
|---|---|
| `status` | One of three values:<br>• `recommended`: a departure meets the risk budget.<br>• `no_feasible_departure`: no departure does. The least risky option is returned with `route.recommended=false`.<br>• `no_route`: there is no route at all. |
| `metadata` | `mode: "historical"`, `hindsight_forcing: true`, `hindsight_disclosure` (text), `banners` ("HISTORICAL MODE", "ERA5/CMEMS hindsight forcing", "Research estimate, not certified navigation"), `disclaimer`, `issue_date`, `window_days`, `horizon_days`, `scenario_days`, `n_scenarios`, `layer_source`, `provenance` (input files, checksums, model, parameters, limitations) |
| `locations` | `origin` and `destination`, each with `requested`, `resolved` (cell centre, row and col), `snapped`, `distance_km` and `reason` (from S1) |
| `horizon` | Route-specific horizon (S1) |
| `route` | `departure_date`, `departure_utc` (00:00 UTC; the forecast layers are daily), `expected_hours` with `hours_p10` / `hours_p90`, `eta_utc`, `distance_km`, `fuel_index` {expected, p10, p90, unit: relative index}, `support` / `forecast_fraction`, `beyond_horizon_fraction`, `impassable_scenarios`, `cells` / `latlon` / `xy_km`, and two per-waypoint lists for route exports: `segment_fuel_index` (expected fuel index of the leg ending at that waypoint, 0 at the start; the list sums to `fuel_index.expected`) and `waypoint_hours` (expected hours from departure, over the scenarios that reach the waypoint; the last value equals `expected_hours`) |
| `risk` | `combined` (authoritative), `sea_ice`, `iceberg` and `iceberg_only_increment`. Each has breaches, n_scenarios, p_breach and the Wilson 95 % `p_breach_upper`; most also carry per-cell `segment_breach_prob`. Also `within_budget`, `risk_budget` and `definitions`. |
| `departure` | `recommended` (date or null), `rule`, `explanation`, all `options` (one per departure day, from the same scenario set), `depart_on_issue_date` |
| `alternatives` | Candidate routes for the selected departure |
| `daily` | One entry per voyage day. See below. |
| `layers` | `grid` geometry, `land`, `vessel_limit`, and per voyage day `p_ice_ge_limit_pct` and `p_berg_pct` (integer %, null on land) |
| `icebergs`, `iceberg_tracks` | The USNIC list used (date, age, positions) and the daily drift tracks the planner used |
| `plan_id` | Id under which this result was saved (a UUID derived from the result's content, so the same plan always has the same id). See [Saved plans](#saved-plans-and-the-pdf-brief) |

### Risk metrics

The same route is sailed through the same scenarios with the same arrival times. Icebergs and the vessel limit do not change ship speed, so the timing is identical in every pass.

- **combined**: the planner's own evaluation. A scenario breaches if it meets ice ≥ the vessel limit, an iceberg footprint, or an impassable leg. **This is the decision value**, compared with the 5 % budget through its Wilson upper bound.
- **sea_ice**: the same evaluation with the icebergs removed.
- **iceberg**: the same evaluation with the ice limit switched off. Iceberg footprints and impassable legs still count.
- **iceberg_only_increment**: scenarios that breach only because of icebergs. `combined = sea_ice + increment` (counts).

The sea-ice and iceberg components overlap, so they need not add up to the combined risk; `combined ≥ max(sea_ice, iceberg)` always holds.

### Daily entries

Each entry has:
- `date` and `day_of_voyage`;
- `scenario_layer` and `layer_source` (observed / forecast / climatology);
- `nominal_hours_since_departure`;
- the route cells for that day (`route_cell_index`, `cells`, `latlon`), `position_end_of_day` and `distance_km`;
- `sea_ice`: ensemble-mean and max concentration on those cells, and the max P(conc ≥ limit);
- `iceberg`: the max P(presence);
- `risk`: the max per-cell breach probability for the combined, sea-ice and iceberg passes.

**How exact the timeline is:**
- The per-cell probabilities are exact, because each scenario is scored at its own arrival time.
- The assignment of cells to days is *nominal*: arrival time is taken as proportional to the distance sailed, scaled to the expected voyage time. It is used for display only.

## Saved plans and the PDF brief

Every `POST /real/plan` result is saved with the request that produced it.

| call | answer |
|---|---|
| `GET /real/plans?limit=20` | `storage` (`memory` or `supabase`) and `plans`, newest first: `id`, `created_at`, `origin`, `destination`, `issue_date`, `mode`, `status`, `departure_date`, `p_breach_upper`, `expected_hours`, `distance_km` |
| `GET /real/plans/{id}` | `plan_id`, `request` and `plan` (the response exactly as it was returned, without `plan_id`) |
| `GET /real/plans/{id}/brief` | A PDF: the decision with its risk split and limitations, the route map, the departure window and the candidate routes. Every number is read from the saved response; nothing is recomputed |

An unknown id answers 404 (`detail.status: "unknown_plan"`); a malformed one 422.

- **Without configuration** the last 32 plans are held in the server's memory and are lost when it restarts.
- **With `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`** they are also written to the `plans` table of a Supabase project
  and survive restarts. See [DEPLOYMENT.md](DEPLOYMENT.md#14-saved-plans-supabase).

Saving never changes or delays a plan's content: a storage failure is logged and the plan is still answered.

## Errors (never a synthetic fallback)

| HTTP | `detail.status` | when |
|---|---|---|
| 422 | `invalid_location` | Unknown preset; outside the grid; inland beyond 60 km of navigable water; origin = destination cell; a route longer than the 21-day model horizon |
| 422 | `out_of_coverage` | The issue date is not in the archive or off season; less than 14 days of in-season history; window + horizon run past the season; missing ERA5 or CMEMS days; no USNIC list ≤ 14 days old |
| 422 | `forecast_unavailable` | A date after the archive that neither the proxy forecast nor a historical seasonal analogue can serve (e.g. the off-season data are missing); both reasons are given |
| 422 | (validation) | Missing fields, or `window_days` outside 1..14 |
| 503 | `unavailable` / `blocked` / `failed` | No archive configured; inputs missing or changed (checksums); model or PyTorch unavailable |

## Performance

One request builds one scenario set (about 10–15 s on the real archive after the model context is cached). The archive and the U-Net context load once per process. The response is about 95 kB with layers.

## Dates after the archive (forecast mode)

Any date after the real archive (e.g. `"issue": "2026-11-19"`, `"2027-08-14"` or `"2030-08-14"`) returns the same response shape with `metadata.mode: "forecast"`, `execution_mode: "modelled"`, `data_status: "forecast_estimate"`, forecast banners and disclosure, and `metadata.forecast` listing every proxy input and the dates actually used. `metadata.forecast.pathway` is `proxy_forecast` or `seasonal_analogue`; the analogue records `analogue_date`, the sea-ice member years and windows, and its `confidence`. See [FORECAST_MODE.md](FORECAST_MODE.md).
