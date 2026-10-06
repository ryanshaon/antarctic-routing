"""Named origin/destination presets, snapping to the routing grid, and the route-specific horizon.

A location is resolved on the grid the planner actually uses (the real archive's 25 km grid):

1. The requested point must lie inside the grid; otherwise :class:`LocationError` (never clamped).
2. If the cell containing it is navigable, that cell is used (``snapped`` is false).
3. Otherwise the nearest navigable cell centre by WGS84 geodesic distance is used, up to
   ``max_snap_km``; ties go to the lowest (row, col), so the result is deterministic.
   ``snapped`` is true and the response says why. A location is never placed on land.

"Navigable" means ocean (not in the land mask) *and* in the largest 4-connected ocean region.
4-connectivity matches the routing graph, whose diagonal and knight moves are only valid when the
orthogonal cells they cross are navigable, so a lake-like ocean cell cut off by land is never chosen.

The forecast horizon of a route uses the same rule as the fixed-route CLI (:func:`route_horizon_days`):
``ceil(1.5 * distance / (0.5 * cruise speed) / time step) + 1`` days, at most the model's lead days.
A route whose generous time estimate needs more days than the model forecasts is refused, not truncated.

Presets are real, named places inside the routing domain; they are research waypoints for the
hackathon demo, not approved anchorages or navigation advice.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from pyproj import Geod
from scipy import ndimage

from antarctic_routing.preprocessing.grid import PolarGrid

_GEOD = Geod(ellps="WGS84")

MAX_SNAP_KM = 60.0     # a little over two 25 km cells; farther means the point is inland, not near a coast


class LocationError(ValueError):
    """A location that cannot be used for routing (unknown preset, outside the grid, inland, ...)."""


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    lat: float
    lon: float
    region: str
    note: str


# Real places in the Drake Passage / Antarctic Peninsula / Scotia Sea routing domain. Coordinates are the
# station or named point; snapping moves any that fall on a 25 km land cell to the nearest open water.
PRESETS: tuple[Preset, ...] = (
    Preset("drake_passage", "Drake Passage (south of Cape Horn)", -56.3, -66.0, "Drake Passage",
           "Default origin of the frozen configuration"),
    Preset("bransfield_strait", "Bransfield Strait (central)", -63.0, -59.0, "South Shetland Islands",
           "Default destination of the frozen configuration"),
    Preset("king_george_island", "King George Island (Maxwell Bay)", -62.22, -58.86, "South Shetland Islands",
           "Station cluster on Maxwell Bay"),
    Preset("deception_island", "Deception Island", -62.97, -60.65, "South Shetland Islands",
           "Caldera island; at 25 km it is open-water cells next to land"),
    Preset("esperanza", "Esperanza Base (Hope Bay)", -63.40, -56.99, "Antarctic Peninsula tip", "Argentine station"),
    Preset("palmer_station", "Palmer Station (Anvers Island)", -64.77, -64.05, "Antarctic Peninsula west",
           "US station"),
    Preset("vernadsky", "Vernadsky Station (Argentine Islands)", -65.25, -64.26, "Antarctic Peninsula west",
           "Ukrainian station"),
    Preset("rothera", "Rothera Research Station (Adelaide Island)", -67.57, -68.13, "Marguerite Bay",
           "UK station; on land at 25 km, so it snaps to nearby open water"),
    Preset("signy_island", "Signy Island (South Orkney Islands)", -60.71, -45.60, "Scotia Sea", "UK station"),
)
PRESETS_BY_ID = {p.id: p for p in PRESETS}


def navigable_mask(land: np.ndarray) -> np.ndarray:
    """Ocean cells in the largest 4-connected ocean region (the cells a route can start or end on)."""
    ocean = ~np.asarray(land, dtype=bool)
    labels, n = ndimage.label(ocean)          # default structure = 4-connectivity
    if n == 0:
        return np.zeros_like(ocean)
    sizes = np.bincount(labels.ravel())[1:]
    return labels == (int(np.argmax(sizes)) + 1)  # argmax: the first (lowest label) on a size tie


@dataclass(frozen=True)
class ResolvedLocation:
    id: str | None
    name: str
    requested_lat: float
    requested_lon: float
    cell: tuple[int, int]
    lat: float               # centre of the routing cell actually used
    lon: float
    snapped: bool            # True when the requested cell was not navigable and the point was moved
    distance_km: float       # requested point -> centre of the cell used
    reason: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cell"] = list(self.cell)
        d["requested"] = {"lat": d.pop("requested_lat"), "lon": d.pop("requested_lon")}
        d["resolved"] = {"lat": round(d.pop("lat"), 5), "lon": round(d.pop("lon"), 5), "row": self.cell[0],
                         "col": self.cell[1]}
        d["distance_km"] = round(self.distance_km, 1)
        return d


def resolve_point(grid: PolarGrid, land: np.ndarray, navigable: np.ndarray, lat: float, lon: float, *,
                  name: str | None = None, preset_id: str | None = None,
                  max_snap_km: float = MAX_SNAP_KM) -> ResolvedLocation:
    """Resolve a WGS84 point to the routing cell a route will start or end on (see the module docstring)."""
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0) or not (math.isfinite(lat) and math.isfinite(lon)):
        raise LocationError(f"({lat}, {lon}) is not a valid latitude/longitude")
    label = name or f"{lat:.4f}, {lon:.4f}"
    try:
        r, c = grid.cell_of(lat, lon)
    except ValueError:
        raise LocationError(f"{label} ({lat}, {lon}) is outside the routing grid") from None
    if navigable[r, c]:
        clat, clon = grid.cell_latlon(r, c)
        dist = _GEOD.inv(lon, lat, clon, clat)[2] / 1000.0
        return ResolvedLocation(preset_id, label, lat, lon, (r, c), clat, clon, False, dist,
                                "the requested point lies in a navigable routing cell")
    rows, cols = np.nonzero(navigable)        # row-major order: argmin's first hit = lowest (row, col)
    if rows.size == 0:
        raise LocationError("the routing grid has no navigable cells")
    lats, lons = grid.lat2d[rows, cols], grid.lon2d[rows, cols]
    d_km = _GEOD.inv(np.full(rows.size, lon), np.full(rows.size, lat), lons, lats)[2] / 1000.0
    k = int(np.argmin(d_km))
    if d_km[k] > max_snap_km:
        raise LocationError(f"{label} ({lat}, {lon}) is {d_km[k]:.0f} km from the nearest navigable routing cell "
                            f"(limit {max_snap_km:.0f} km); choose a point at sea")
    why = ("the requested point lies on land in the grid's land mask" if land[r, c]
           else "the requested cell is ocean cut off from the open sea at this grid resolution")
    return ResolvedLocation(preset_id, label, lat, lon, (int(rows[k]), int(cols[k])), float(lats[k]),
                            float(lons[k]), True, float(d_km[k]),
                            f"{why}; moved to the nearest navigable cell ({d_km[k]:.0f} km away)")


class LocationResolver:
    """Resolves presets and points on one routing grid + land mask (build once per grid)."""

    def __init__(self, grid: PolarGrid, land: np.ndarray, max_snap_km: float = MAX_SNAP_KM) -> None:
        self.grid = grid
        self.land = np.asarray(land, dtype=bool)
        if self.land.shape != grid.shape:
            raise ValueError(f"land mask shape {self.land.shape} does not match the grid {grid.shape}")
        self.navigable = navigable_mask(self.land)
        self.max_snap_km = float(max_snap_km)

    def resolve(self, spec: str | dict | Preset) -> ResolvedLocation:
        """``spec`` is a preset id, ``{"preset": id}`` or ``{"lat": .., "lon": .., "name"?: ..}``."""
        if isinstance(spec, Preset):
            return self._resolve(spec.lat, spec.lon, spec.name, spec.id)
        if isinstance(spec, str):
            spec = {"preset": spec}
        if not isinstance(spec, dict):
            raise LocationError("a location is a preset id or an object with 'preset' or 'lat' and 'lon'")
        if spec.get("preset") is not None:
            if "lat" in spec or "lon" in spec:
                raise LocationError("give either 'preset' or 'lat'/'lon', not both")
            p = PRESETS_BY_ID.get(spec["preset"])
            if p is None:
                raise LocationError(f"unknown preset {spec['preset']!r}; choose one of {sorted(PRESETS_BY_ID)}")
            return self._resolve(p.lat, p.lon, p.name, p.id)
        if spec.get("lat") is None or spec.get("lon") is None:
            raise LocationError("a location needs 'preset' or both 'lat' and 'lon'")
        return self._resolve(float(spec["lat"]), float(spec["lon"]), spec.get("name"), None)

    def _resolve(self, lat: float, lon: float, name: str | None, preset_id: str | None) -> ResolvedLocation:
        return resolve_point(self.grid, self.land, self.navigable, lat, lon, name=name, preset_id=preset_id,
                             max_snap_km=self.max_snap_km)

    def presets(self) -> list[dict]:
        """Every preset with its resolution on this grid (presets that cannot be resolved are reported, not hidden)."""
        out = []
        for p in PRESETS:
            row = {"id": p.id, "name": p.name, "region": p.region, "note": p.note,
                   "requested": {"lat": p.lat, "lon": p.lon}}
            try:
                r = self.resolve(p)
                row.update(available=True, resolved=r.to_dict()["resolved"], snapped=r.snapped,
                           distance_km=round(r.distance_km, 1), reason=r.reason)
            except LocationError as exc:
                row.update(available=False, resolved=None, snapped=None, distance_km=None, reason=str(exc))
            out.append(row)
        return out


# --------------------------------------------------------------------------- horizon
@dataclass(frozen=True)
class RouteHorizon:
    great_circle_km: float
    planning_hours: float      # generous time allowance: 1.5 x distance at half cruise speed
    required_days: int         # scenario days after the issue day the allowance needs (uncapped)
    horizon_days: int          # min(required, lead_days): what the planner uses
    lead_days: int
    supported: bool            # False when the allowance needs more days than the model forecasts

    def to_dict(self) -> dict:
        d = asdict(self)
        d["great_circle_km"] = round(self.great_circle_km, 1)
        d["planning_hours"] = round(self.planning_hours, 1)
        return d


def route_horizon(distance_km: float, cruise_speed_kmh: float, time_step_hours: float, lead_days: int) -> RouteHorizon:
    """The forecast horizon a route needs. Same rule as the fixed-route CLI, so the default route is unchanged."""
    hours = 1.5 * distance_km / (0.5 * cruise_speed_kmh)   # generous: detours + ice slowdown
    required = int(math.ceil(hours / time_step_hours) + 1)
    return RouteHorizon(float(distance_km), float(hours), required, int(min(lead_days, required)), int(lead_days),
                        required <= lead_days)


def route_horizon_days(o_lat: float, o_lon: float, d_lat: float, d_lon: float, cruise_speed_kmh: float,
                       time_step_hours: float, lead_days: int) -> int:
    """Horizon in days for the great-circle distance between two points (capped at ``lead_days``)."""
    km = _GEOD.inv(o_lon, o_lat, d_lon, d_lat)[2] / 1000.0
    return route_horizon(km, cruise_speed_kmh, time_step_hours, lead_days).horizon_days


@dataclass(frozen=True)
class RouteSpec:
    """Everything a planner needs about *where*: resolved endpoints and the route-specific horizon."""

    origin: ResolvedLocation
    destination: ResolvedLocation
    horizon: RouteHorizon

    @property
    def cells(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return self.origin.cell, self.destination.cell

    @property
    def horizon_days(self) -> int:
        return self.horizon.horizon_days

    def scenario_days(self, window_days: int | None = None) -> int:
        """Scenario days one forecast issue needs: ``1 + horizon`` for a route leaving on the issue day,
        ``window_days + horizon`` for a departure window (as the historical endpoints count them)."""
        return (1 if window_days is None else int(window_days)) + self.horizon.horizon_days

    def to_dict(self) -> dict:
        return {"origin": self.origin.to_dict(), "destination": self.destination.to_dict(),
                "horizon": self.horizon.to_dict()}


def build_route(resolver: LocationResolver, origin, destination, cruise_speed_kmh: float, time_step_hours: float,
                lead_days: int) -> RouteSpec:
    """Resolve both ends and compute the horizon from the cells the route will actually connect."""
    o, d = resolver.resolve(origin), resolver.resolve(destination)
    if o.cell == d.cell:
        raise LocationError(f"origin and destination resolve to the same routing cell {list(o.cell)}; "
                            "choose locations farther apart")
    km = resolver.grid.geodesic_distance_m(*o.cell, *d.cell) / 1000.0
    hz = route_horizon(km, cruise_speed_kmh, time_step_hours, lead_days)
    if not hz.supported:
        raise LocationError(f"{o.name} -> {d.name} ({km:.0f} km) needs a {hz.required_days}-day forecast horizon; "
                            f"the frozen model forecasts {lead_days} days")
    return RouteSpec(o, d, hz)
