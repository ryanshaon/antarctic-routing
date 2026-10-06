# Origin/destination locations and the route horizon

The Real Historical Data endpoints can plan between any two locations in the routing domain. Without
`origin`/`destination`, they use the configured route (`config/config.yaml`) exactly as before.
Code: `src/antarctic_routing/locations.py`.

## Presets

`GET /real/locations` lists the presets together with how each one resolves on the routing grid.

| id | name | lat | lon | on the 25 km grid |
|---|---|---|---|---|
| `drake_passage` | Drake Passage (south of Cape Horn), the configured origin | -56.30 | -66.00 | in an ocean cell |
| `bransfield_strait` | Bransfield Strait (central), the configured destination | -63.00 | -59.00 | in an ocean cell |
| `king_george_island` | King George Island (Maxwell Bay) | -62.22 | -58.86 | in an ocean cell |
| `deception_island` | Deception Island | -62.97 | -60.65 | in an ocean cell |
| `esperanza` | Esperanza Base (Hope Bay) | -63.40 | -56.99 | in an ocean cell |
| `palmer_station` | Palmer Station (Anvers Island) | -64.77 | -64.05 | in an ocean cell |
| `vernadsky` | Vernadsky Station (Argentine Islands) | -65.25 | -64.26 | in an ocean cell |
| `rothera` | Rothera Research Station (Adelaide Island) | -67.57 | -68.13 | land cell, so it is snapped ~18 km |
| `signy_island` | Signy Island (South Orkney Islands) | -60.71 | -45.60 | in an ocean cell |

These presets are research waypoints for decision support. They are not approved anchorages, and they are not navigation advice.

## Location resolution and snapping

A location is either a preset (`{"preset": "palmer_station"}`) or a point (`{"lat": -64.5, "lon": -61.5, "name": "optional"}`).
It is resolved on the grid the planner uses: the real archive's 25 km EPSG:3031 grid and its land mask.

1. **Outside the grid:** the request is refused with HTTP 422 `invalid_location`. The point is never clamped to the grid edge.
2. **Navigable cell:** if the cell that contains the point is navigable, that cell is used and `snapped: false`.
   - *Navigable* means an ocean cell that belongs to the largest 4-connected ocean region.
   - 4-connectivity matches the routing graph, which only allows diagonal moves when the cells they cross are ocean.
3. **Not navigable:** otherwise the point moves to the nearest navigable cell centre, measured by WGS84 geodesic distance.
   - The response sets `snapped: true` and gives the `distance_km` and a `reason` (on land, or ocean cut off from the open sea).
   - If two cells are equally near, the lowest (row, col) is chosen, so the result is deterministic.
   - A location is never placed on land.
   - A point more than 60 km from any navigable cell is refused rather than moved.
4. **Same cell:** if the origin and the destination resolve to the same cell, the request is refused.

Each resolved location reports `requested` {lat, lon}, `resolved` {lat, lon, row, col} (the cell centre), `snapped`, `distance_km` and `reason`.

## Route-specific forecast horizon

The horizon is computed from the great-circle distance between the two resolved cell centres. It uses the same rule as the configured route:

    planning_hours = 1.5 x distance / (0.5 x cruise speed)        # generous: detours + ice slowdown
    required_days  = ceil(planning_hours / time step) + 1
    horizon_days   = required_days, refused (422) if it exceeds the model's lead days (21)

The configured route keeps its horizon of 6 days. A forecast issued on day *D* needs `1 + horizon` scenario days for a route, or `window_days + horizon` for a departure window. Every day in that span must be inside the archive's coverage: in-season sea ice with 14 days of history, daily ERA5 and CMEMS, and a USNIC list no more than 14 days old. Otherwise the request is refused with HTTP 422 `out_of_coverage` and the reason. Nothing is requested beyond the available data.

## Endpoints

- `GET /real/locations` lists the presets with their resolution, the snapping rule and the grid.
- `POST /real/locations/resolve` takes `{"origin": .., "destination": .., "issue"?: date, "window_days"?: 1..14}`. It returns:
  - the resolved ends;
  - `horizon` {great_circle_km, planning_hours, required_days, horizon_days, lead_days, supported};
  - `scenario_days` {route, window};
  - with `issue`, the coverage {ok, problems} for a route and for a window.
- `GET /real/historical/dates?origin=<preset>&destination=<preset>` returns the issue dates valid for that route's horizon.
- `POST /real/historical/routes | departures | voyages | replay` accept optional `origin` and `destination` (give both or neither). When they are given, the response includes `route` (the resolved ends and the horizon), and its origin and destination positions are the resolved cells.
