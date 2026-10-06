# `POST /real/simulate`: historical voyage simulation

**What it is:** the voyage that [`POST /real/plan`](PLAN_API.md) chose, sailed day by day on the Real Historical Data archive. The response is a list of playback frames.

**No new science.** The simulation is the existing historical replay loop ([`replay.py`](../src/antarctic_routing/replay.py)) applied to the planned route. Its pieces:
- the frozen U-Net, ERA5/CMEMS hindsight forcing, USNIC icebergs with the calibrated drift, the Wilson risk budget and A*;
- the existing replanning rules ([`routing/replan.py`](../src/antarctic_routing/routing/replan.py));
- the module [`simulation.py`](../src/antarctic_routing/simulation.py), which records what each step returns.

## Request

```json
POST /real/simulate[?wait=true]
{"origin": {"preset": "drake_passage"}, "destination": {"preset": "bransfield_strait"},
 "issue": "2023-11-14", "window_days": 14, "departure": "2023-11-14"}
```

- **Fields:** the same as `/real/plan`, plus an optional `departure`. If `departure` is given, it must equal the plan's chosen departure; otherwise the response is `status: "not_simulated"` with the reason.
- **Job:** the endpoint starts a background job and is polled at `/jobs/{id}`, like the other long historical runs. With `?wait=true` it answers when the job is done.
- **Errors before any work:**
  - an invalid location is a 422 `invalid_location`;
  - an uncovered issue date is a 422 `out_of_coverage`;
  - no archive is a 503.
- **Errors during the job:**
  - a missing model or input fails the job, with the reason in `error`;
  - nothing synthetic is substituted.

## How the voyage is simulated

1. **The plan.** The plan is the one `/real/plan` returns for the same inputs. It is reused from a small in-memory cache when `/real/plan` was just called, and otherwise recomputed with the same deterministic calls.
2. **Coverage check.** The archive must be able to issue a forecast for every day the planned voyage needs. Otherwise the job returns `not_simulated` and says which days are missing.
3. **Each day at sea:**
   - **Sail.** The vessel sails 24 h along its active route through the **observed** sea ice (`replay.sail_day`, the replay's sailing step).
   - **Forecast.** At the next 00:00 UTC a new forecast is issued from that day: `HistoricalPlanner.world(day, 1 + horizon)`, the same call as `/real/historical/voyages/{id}/replan`.
   - **Decide.** `replan()` decides `keep`, `switch` or `no_feasible_route` by its own rules:
     - the route ahead exceeds the budget;
     - a material risk reduction (at least 2 percentage points, for at most 5 % extra fuel);
     - a material fuel saving (at least 5 %);
     - stale input.
4. **Arrival.** The sailed track is scored through the observed ice with `replay.score_track`.

## Response (`status: "simulated"`)

| key | content |
|---|---|
| `metadata` | Mode `historical`, `execution_mode: "real"`, the banners, the hindsight disclosure, the disclaimer, `departure_date`, `simulation_note` and `provenance`. |
| `plan` | The plan's status, explanation, route summary and risk (the simulation's starting point). |
| `grid` | The grid geometry, the land mask and the vessel limit, for the map. |
| `frames[]` | One frame per day boundary, plus the arrival (see below). |
| `events[]` | `departed`, `replan`, `no_feasible_route`, `arrived` or `stopped`, with the frame index. A replan carries the old and new route, each with its geometry, P(breach) upper bound, expected hours, fuel index and distance; it also carries `change`, the trigger list and the reasons. |
| `summary` | Status, arrival, days at sea, the number of replans, `replan_note`, the sailed track's replay score, the planned values, and the final forecast risk. |
| `sailed_track` | The cells sailed, with xy and lat/lon. |

### Frame fields

**Time and position:**
- `index`, `day_of_voyage`, `phase` (`departure`, `at_sea`, `arrived` or `stopped`);
- `timestamp_utc`, `date`;
- `position`: lat, lon, row, col, xy_km (the last route cell reached).

**Progress and geometry:**
- `progress`: km sailed, km remaining, fraction of the way, cells sailed;
- `segment`: the cells sailed since the previous frame;
- `track`: the track so far;
- `route`: the active route ahead, with its `version`.

**Forecast and decision:**
- `forecast`, made at that frame:
  - `issued`, the date the forecast was issued;
  - `route_ahead`: P(breach), its upper bound, hours, fuel and distance;
  - `risk`: combined, sea-ice and iceberg, each with an upper bound, plus `within_budget`;
  - `eta_utc`.
- `decision`: action, triggers, explanation, alert, deviation and the plain-language reasons.
- `replanned`, `route_version`.

**Observations:**
- `observed`:
  - the observed concentration on the cells sailed that day;
  - the USNIC list in force, with the in-grid bergs and the nearest berg.
- `map`: the observed sea-ice concentration for that date, in percent.

## Known approximations (disclosed in the response)

- **Positions are 25 km route cells.** Each day the vessel stops at the last cell it reaches and waits there until the next forecast, as the replay does. `summary.held_hours` reports that wait; the replay score re-sails the track continuously.
- **Hindsight forcing.** Winds and currents for the days after each issue date are reanalysis.
- **Research estimate, not certified navigation.**

## Dates after the archive (forecast mode)

A date after the real archive is simulated the same way on the analogue days of [forecast mode](FORECAST_MODE.md): the sea ice sailed through is the analogue season's real observation, labelled `proxy_analogue_observed` (each day's `sea_ice_on_segment` carries its `analogue_date`), icebergs come from the latest official USNIC list held at their last positions, and dates are shown on the requested timeline. `metadata.mode` is `forecast` and `summary.notes.forecast_mode` says so.
