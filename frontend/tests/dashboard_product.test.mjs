// Browser tests for the dashboard's Plan Route product view, against a mocked API.
//
//   node --test frontend/tests/            (needs Playwright with Chromium; skipped when it is not installed)
//
// Every request the page makes is answered here: the dashboard files from src/antarctic_routing/dashboard and the
// API from small stand-in payloads shaped like GET /real/locations, GET /real/historical/dates and POST /real/plan.
// The stand-in plan is a 6 x 5 cell test grid, not real data; tests/test_historical_real.py and the S3 smoke run
// cover the real archive. Set ANTROUTE_SMOKE_URL=http://host:port to also run one real Plan Route end to end.
import { test } from "node:test";
import assert from "node:assert/strict";
import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const DASH = resolve(here, "..", "..", "src", "antarctic_routing", "dashboard");
const ORIGIN = "http://dashboard.test";

function loadPlaywright() {
  const require = createRequire(import.meta.url);
  try { return require("playwright"); } catch { /* fall through to a global install */ }
  try {
    const root = execSync("npm root -g", { stdio: ["ignore", "pipe", "ignore"] }).toString().trim();
    return require(join(root, "playwright"));
  } catch { return null; }
}
const pw = loadPlaywright();
const skip = pw ? false : "Playwright is not installed";

/* ------------------------------------------------------------- stand-in API payloads */
const DISCLOSURE = "Hindsight forcing: test disclosure text for the stand-in plan.";
const PRESETS = [
  { id: "alpha", name: "Alpha Point", region: "North", note: "", requested: { lat: -60, lon: -60 }, available: true,
    resolved: { lat: -60.01, lon: -60.02, row: 1, col: 1 }, snapped: false, distance_km: 2.0, reason: "in a navigable cell" },
  { id: "bravo", name: "Bravo Bay", region: "North", note: "", requested: { lat: -61, lon: -59 }, available: true,
    resolved: { lat: -61.0, lon: -59.0, row: 3, col: 4 }, snapped: false, distance_km: 1.0, reason: "in a navigable cell" },
  { id: "charlie", name: "Charlie Station", region: "South", note: "", requested: { lat: -62, lon: -61 }, available: true,
    resolved: { lat: -61.9, lon: -60.7 , row: 2, col: 3 }, snapped: true, distance_km: 12.4,
    reason: "the requested point lies on land in the grid's land mask; moved to the nearest navigable cell (12 km away)" },
];
const DATES = (() => {
  const out = [];
  for (let d = new Date("2023-11-01T00:00:00Z"); d <= new Date("2023-12-20T00:00:00Z"); d.setUTCDate(d.getUTCDate() + 1)) {
    out.push(d.toISOString().slice(0, 10));
  }
  return out;
})();
const datesPayload = {
  label: "Real Historical Data", horizon_days: 6, window_days_max: 14, route_dates: DATES, window_dates: DATES,
  seasons: { route: [{ season: "2023-24", first: DATES[0], last: DATES.at(-1), n_dates: DATES.length, out_of_sample: true, in_sample_notes: [] }],
    window: [{ season: "2023-24", first: DATES[0], last: DATES.at(-1), n_dates: DATES.length, out_of_sample: true, in_sample_notes: [] }] },
};

const NX = 6, NY = 5;
const grid = { crs: "EPSG:3031", nx: NX, ny: NY, res_km: 25, x0_km: 0, y0_km: 0, rotation_rad: 0,
  x_km: [...Array(NX).keys()].map((i) => 12.5 + 25 * i), y_km: [...Array(NY).keys()].map((j) => 12.5 + 25 * j) };
const land = Array(NX * NY).fill(0); land[NX * NY - 1] = 1;
// GET /real/locations' "map": the grid, its land mask and each cell centre's WGS84 position (stand-in values)
const pickMap = { ...grid, land, navigable: land.map((l) => 1 - l),
  lat: [...Array(NX * NY).keys()].map((k) => -60 - 0.25 * Math.floor(k / NX)),
  lon: [...Array(NX * NY).keys()].map((k) => -62 + 0.4 * (k % NX)) };
const layer = (k, ice) => ({ scenario_layer: k, date: `2023-11-${14 + k}`, source: k ? "forecast" : "observed",
  p_ice_ge_limit_pct: land.map((l, i) => (l ? null : (i * ice) % 100)), p_berg_pct: land.map((l, i) => (l ? null : i === 8 ? 40 : 0)) });

function riskPart(breaches, upper) {
  return { breaches, n_scenarios: 200, p_breach: breaches / 200, p_breach_upper: upper, segment_breach_prob: [0, 0, 0, 0],
    impassable_scenarios: 0 };
}
function option(day, feasible, upper, hours) {
  return { departure: `2023-11-${day}`, status: feasible ? "feasible" : "exceeds_budget", feasible, expected_hours: hours,
    expected_fuel: 100 + day, p_breach: upper / 2, p_breach_upper: upper, beyond_horizon_fraction: 0, route_labels: [],
    lead_days: day - 14, forecast_fraction: 1, support: "forecast-supported", within_trust_horizon: null };
}

function makePlan(status = "recommended") {
  const rec = status === "recommended";
  const loc = (p) => ({ id: p.id, name: p.name, cell: [p.resolved.row, p.resolved.col], snapped: p.snapped,
    distance_km: p.distance_km, reason: p.reason, requested: p.requested, resolved: p.resolved });
  return {
    status,
    explanation: rec ? "Stand-in explanation: selected 2023-11-14." : "Stand-in explanation: no departure date satisfies the budget.",
    metadata: { mode: "historical", label: "Real Historical Data", execution_mode: "real", data_status: "historical",
      issue_date: "2023-11-14", hindsight_forcing: true, hindsight_disclosure: DISCLOSURE,
      banners: ["HISTORICAL MODE", "ERA5/CMEMS hindsight forcing", "Research estimate, not certified navigation"],
      disclaimer: "Stand-in disclaimer.", window_days: 14, horizon_days: 6, scenario_days: 20, n_scenarios: 200,
      layer_source: ["observed", "forecast"],
      provenance: { limitations: ["Stand-in limitation one."], season: datesPayload.seasons.window[0] } },
    locations: { origin: loc(PRESETS[0]), destination: loc(PRESETS[1]) },
    horizon: { great_circle_km: 120, planning_hours: 20, required_days: 6, horizon_days: 6, lead_days: 21, supported: true },
    route: { recommended: rec, departure_date: rec ? "2023-11-14" : "2023-11-16", departure_utc: rec ? "2023-11-14T00:00:00Z" : "2023-11-16T00:00:00Z",
      time_resolution: "24 h forecast layers", lead_days: rec ? 0 : 2, expected_hours: 30.25, hours_p10: 29.5, hours_p90: 31.0,
      eta_utc: rec ? "2023-11-15T06:15Z" : "2023-11-17T06:15Z", distance_km: 123.4,
      fuel_index: { expected: 456.7, p10: 450, p90: 460, unit: "relative fuel index" }, labels: [], tags: [],
      forecast_fraction: 1, support: "forecast-supported", beyond_horizon_fraction: 0, impassable_scenarios: 0,
      cells: [[1, 1], [1, 2], [2, 3], [3, 4]], latlon: [[-60, -60], [-60.3, -59.8], [-60.6, -59.5], [-61, -59]],
      xy_km: [[37.5, 37.5], [62.5, 37.5], [87.5, 62.5], [112.5, 87.5]] },
    risk: { authoritative: "combined", risk_budget: 0.05, confidence: 0.95, estimator: "wilson_upper",
      combined: { ...riskPart(rec ? 2 : 150, rec ? 0.0311 : 0.7911), within_budget: rec },
      sea_ice: riskPart(rec ? 1 : 140, rec ? 0.0222 : 0.7422), iceberg: riskPart(rec ? 1 : 10, rec ? 0.0133 : 0.0833),
      iceberg_only_increment: riskPart(1, 0.02),
      definitions: { combined: "Combined definition.", sea_ice: "Sea-ice definition.", iceberg: "Iceberg definition." } },
    departure: { recommended: rec ? "2023-11-14" : null, rule: "stand-in rule", explanation: "x",
      options: [option(14, rec, rec ? 0.0311 : 0.81, 30.25), option(15, false, 0.2, 31), option(16, false, rec ? 0.3 : 0.7911, 30.25)],
      depart_on_issue_date: option(14, rec, 0.0311, 30.25) },
    alternatives: [],
    daily: [
      { date: "2023-11-14", day_of_voyage: 1, scenario_layer: 0, layer_source: "observed", nominal_hours_since_departure: [0, 24],
        route_cell_index: [0, 2], position_end_of_day: [-60.6, -59.5], distance_km: 90.1,
        sea_ice: { mean_concentration: 0.01, max_concentration: 0.02, max_p_ge_vessel_limit: 0 }, iceberg: { max_p_presence: 0 },
        risk: { combined: { max_cell_breach_prob: 0 }, sea_ice: { max_cell_breach_prob: 0 }, iceberg: { max_cell_breach_prob: 0 } } },
      { date: "2023-11-15", day_of_voyage: 2, scenario_layer: 1, layer_source: "forecast", nominal_hours_since_departure: [24, 30.25],
        route_cell_index: [3, 3], position_end_of_day: [-61, -59], distance_km: 33.3,
        sea_ice: { mean_concentration: 0.2, max_concentration: 0.3, max_p_ge_vessel_limit: 0.07 }, iceberg: { max_p_presence: 0.4 },
        risk: { combined: { max_cell_breach_prob: 0.0777 }, sea_ice: { max_cell_breach_prob: 0.07 }, iceberg: { max_cell_breach_prob: 0.01 } } },
    ],
    daily_note: "Stand-in nominal timeline note.",
    layers: { grid, land, vessel_limit: 0.15, layers: [layer(0, 3), layer(1, 7)] },
    icebergs: { list_date: "2023-11-09", drifted: ["T1"] },
    iceberg_tracks: [{ id: "T1", daily_mean: [{ layer: 0, x_km: 62.5, y_km: 112.5, lat: -60.5, lon: -60.5 },
      { layer: 1, x_km: 87.5, y_km: 112.5, lat: -60.6, lon: -60.4 }] }],
  };
}

/* ------------------------------------------------------------- stand-in simulation */
// Shaped like POST /real/simulate's job result; the "replan" variant switches route on day 2.
function makeSim(replan = true) {
  const plan = makePlan();
  const g = { ...grid, land };
  const planned = plan.route.xy_km, cells = plan.route.cells;
  const alt = { cells: [[2, 3], [2, 4], [3, 4]], xy_km: [[87.5, 62.5], [112.5, 62.5], [112.5, 87.5]] };
  const pos = (k, xy, c) => ({ lat: -60 - k / 2, lon: -60 + k / 2, row: c[0], col: c[1], xy_km: xy });
  const risk = (u) => ({ combined: { breaches: 1, n_scenarios: 200, p_breach: 0.005, p_breach_upper: u, within_budget: u <= 0.05 },
    sea_ice: { p_breach_upper: u }, iceberg: { p_breach_upper: 0.0188 }, risk_budget: 0.05 });
  const frame = (k, phase, ts, xy, cell, track, route, version, extra = {}) => ({
    index: k, day_of_voyage: k, phase, timestamp_utc: ts, date: ts.slice(0, 10), position: pos(k, xy, cell),
    progress: { sailed_km: 40 * k, remaining_km: phase === "arrived" ? 0 : 120 - 40 * k, fraction: phase === "arrived" ? 1 : k / 3,
      sailed_hours: 20 * k, cells_sailed: k },
    segment: { cells: [], xy_km: [], latlon: [], sailed_on: null }, track,
    route: { ...route, version }, route_version: version,
    forecast: phase === "arrived" ? null : { issued: ts.slice(0, 10), route_ahead: { expected_hours: 30 - 10 * k,
      expected_fuel: 400 - 100 * k, distance_km: 120 - 40 * k, p_breach_upper: 0.0311 }, risk: risk(k === 1 && replan ? 0.081 : 0.0311),
      eta_utc: "2023-11-15T06:15Z" },
    observed: { sea_ice_on_segment: k ? { date: ts.slice(0, 10), mean_concentration: 0.01, max_concentration: 0.02 } : null,
      icebergs: { list_date: "2023-11-09", age_days: 5, in_grid: [{ id: "T1", lat: -60.5, lon: -60.5, xy_km: [62.5, 112.5] }],
        nearest_km: 55.5, nearest_id: "T1" } },
    map: { date: ts.slice(0, 10), source: "observed", concentration_pct: land.map((l, i) => (l ? null : (i * (k + 2)) % 100)) },
    decision: null, replanned: false, ...extra,
  });
  const keep = { action: "keep", triggers: [], explanation: "Keep current route: stand-in keep.", alert: false, deviation_km: 0, reasons: [] };
  const sw = { action: "switch", triggers: ["previous_route_exceeds_budget"], alert: true, deviation_km: 31.2,
    explanation: "Switch route (previous_route_exceeds_budget): stand-in switch.",
    reasons: ["The route ahead no longer met the risk budget under the new forecast."] };
  const frames = replan ? [
    frame(0, "departure", "2023-11-14T00:00Z", planned[0], cells[0], [planned[0]], { cells, xy_km: planned }, 0),
    frame(1, "at_sea", "2023-11-15T00:00Z", planned[2], cells[2], planned.slice(0, 3), { ...alt }, 1,
      { decision: sw, replanned: true }),
    frame(2, "at_sea", "2023-11-16T00:00Z", alt.xy_km[1], alt.cells[1], [...planned.slice(0, 3), alt.xy_km[1]],
      { cells: alt.cells.slice(1), xy_km: alt.xy_km.slice(1) }, 1, { decision: keep }),
    frame(3, "arrived", "2023-11-16T09:00Z", alt.xy_km[2], alt.cells[2], [...planned.slice(0, 3), ...alt.xy_km.slice(1)],
      { cells: [alt.cells[2]], xy_km: [alt.xy_km[2]] }, 1),
  ] : [
    frame(0, "departure", "2023-11-14T00:00Z", planned[0], cells[0], [planned[0]], { cells, xy_km: planned }, 0),
    frame(1, "at_sea", "2023-11-15T00:00Z", planned[2], cells[2], planned.slice(0, 3), { cells: cells.slice(2), xy_km: planned.slice(2) }, 0,
      { decision: keep }),
    frame(2, "arrived", "2023-11-15T06:15Z", planned[3], cells[3], planned, { cells: [cells[3]], xy_km: [planned[3]] }, 0),
  ];
  const last = frames.at(-1);
  const events = [{ type: "departed", frame: 0, timestamp_utc: frames[0].timestamp_utc, explanation: "Departed on the planned route." }];
  if (replan) {
    events.push({ type: "replan", frame: 1, timestamp_utc: frames[1].timestamp_utc, ...sw, issued: "2023-11-15",
      old_route: { cells: cells.slice(2), xy_km: planned.slice(2), p_breach_upper: 0.081, expected_hours: 10.5, expected_fuel: 210, distance_km: 50 },
      new_route: { ...alt, p_breach_upper: 0.0311, expected_hours: 12.0, expected_fuel: 230, distance_km: 61 },
      change: { p_breach_upper: -0.0499, expected_hours: 1.5, expected_fuel: 20, distance_km: 11 } });
  }
  events.push({ type: "arrived", frame: last.index, timestamp_utc: last.timestamp_utc, explanation: "Destination reached." });
  return {
    status: "simulated", label: "Real Historical Data",
    metadata: { ...plan.metadata, departure_date: "2023-11-14", simulation_note: "Stand-in simulation note." },
    locations: plan.locations, plan: { status: "recommended", explanation: "x", route: plan.route, risk: plan.risk },
    grid: g, frames, events, sailed_track: { cells: [], xy_km: last.track, latlon: [] },
    summary: { status: "arrived", reason: null, arrived: true, departure_utc: "2023-11-14T00:00Z", arrival_utc: last.timestamp_utc,
      days_at_sea: frames.length - 1, simulated_hours: replan ? 57 : 30.25, held_hours: 1.5, replans: replan ? 1 : 0,
      replan_note: replan ? null : "No replan was required during this voyage.", no_feasible_route_alerts: 0,
      sailed: { hours_through_observed_ice: replan ? 55.5 : 30.2, distance_km: replan ? 131 : 123.4, fuel_index: replan ? 470 : 456.7,
        observed_breach_cells: 0, hazard_hours: 0, berg_footprint_cells: 0, berg_min_distance_km: 55.5, berg_nearest: "T1" },
      planned: { distance_km: 123.4, expected_fuel: 456.7, expected_hours: 30.25, p_breach_upper: 0.0311 },
      final_forecast_risk: risk(0.0311), final_forecast_issued: replan ? "2023-11-16" : "2023-11-15",
      notes: { sailed: "Stand-in sailed note." } },
  };
}

/* ------------------------------------------------------------- harness */
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css" };

async function openDashboard(browser, { plan = () => ({ status: 200, body: makePlan() }), historical = "available",
  simulate = () => ({ status: 200, body: makeSim(true) }), dates = datesPayload, datesFor = null } = {}) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const calls = [];
  const errors = [];
  const jobs = new Map();
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/*", async (route) => {
    const req = route.request(), url = new URL(req.url());
    if (url.origin !== ORIGIN) return route.abort();
    const json = (body, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
    const p = url.pathname;
    if (p === "/" || p.startsWith("/static/")) {
      const file = p === "/" ? "index.html" : p.slice("/static/".length);
      return route.fulfill({ status: 200, contentType: TYPES[file.slice(file.lastIndexOf("."))], body: readFileSync(join(DASH, file)) });
    }
    calls.push({ method: req.method(), path: p, search: url.search, body: req.postData() });
    if (p === "/health") return json({ version: "test", disclaimer: "Test disclaimer." });
    if (p === "/config") return json({ config: { routing: { risk_budget: 0.05 } } });
    if (p === "/status") {
      return json({ real_data: { status: "unavailable", reason: "not used in this test" },
        historical: historical === "available" ? { status: "available", hindsight_forcing: DISCLOSURE }
          : { status: "unavailable", reason: "no archive configured" },
        figures: { execution_mode: "controlled_synthetic" } });
    }
    if (p === "/real/locations") return json({ label: "Real Historical Data", presets: PRESETS, max_snap_km: 60, map: pickMap });
    if (p === "/real/historical/dates") {
      if (datesFor) { const r = datesFor(url.searchParams); return json(r.body, r.status || 200); }
      return json(dates);
    }
    if (p === "/real/plan") {
      const r = await plan(JSON.parse(req.postData() || "{}"));
      if (r.abort) return route.abort("failed");
      if (r.delay) await new Promise((ok) => setTimeout(ok, r.delay));
      return route.fulfill({ status: r.status, contentType: "application/json", body: r.raw ?? JSON.stringify(r.body) });
    }
    if (p === "/real/simulate") {               // a background job: queued here, done on the first poll
      const r = await simulate(JSON.parse(req.postData() || "{}"));
      if (r.status !== 200) return json(r.body, r.status);
      jobs.set("sim1", r);
      return json({ job_id: "sim1", kind: "real_simulate", status: "queued", result: null, error: null });
    }
    if (p.startsWith("/jobs/")) {
      const r = jobs.get(p.slice(6));
      if (r.delay) await new Promise((ok) => setTimeout(ok, r.delay));
      return json(r.failed ? { job_id: "sim1", status: "failed", error: r.failed, result: null }
        : { job_id: "sim1", status: "done", result: r.body, error: null });
    }
    return json({ detail: "not mocked" }, 404);
  });
  await page.goto(ORIGIN + "/");
  return { page, calls, errors, planCalls: () => calls.filter((c) => c.path === "/real/plan") };
}

const ready = (page) => page.waitForFunction(() => !document.querySelector("#pr-submit").disabled);
const planned = (page) => page.waitForSelector("#pr-result:not([hidden])");
const text = (page, sel) => page.locator(sel).innerText();

let browser;
test.before(async () => { if (pw) browser = await pw.chromium.launch(); });
test.after(async () => { if (browser) await browser.close(); });

/* ------------------------------------------------------------- tests */
test("Plan Route is the default view and shows a clean landing, no precomputed result", { skip }, async () => {
  const { page, errors, planCalls } = await openDashboard(browser);
  await ready(page);
  assert.equal(await page.locator('.tabs button[aria-selected="true"]').getAttribute("data-tab"), "product");
  assert.equal(await text(page, '.primary-nav button[aria-selected="true"]'), "Plan Route");
  assert.ok(await page.locator("#pr-landing").isVisible());
  assert.ok(await page.locator("#pr-pick-map").isVisible());
  assert.ok(!(await page.locator("#pr-result").isVisible()));
  assert.match(await text(page, "#pr-landing"), /Start[\s\S]*Destination[\s\S]*Departure date[\s\S]*Plan Route/);
  assert.equal(planCalls().length, 0);
  assert.deepEqual(errors, []);
  await page.close();
});

test("the primary navigation has only the three product pages; research and synthetic views are tucked away", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await ready(page);
  assert.deepEqual(await page.locator(".primary-nav button").allInnerTexts(), ["Plan Route", "Voyage Simulation", "Data & Confidence"]);
  const primary = await text(page, ".primary-nav");
  for (const s of ["synthetic", "Validation", "Research", "Frozen demo", "Departure window", "Route plan"]) {
    assert.ok(!primary.includes(s), s);
  }
  assert.equal(await page.locator("#dev-area").getAttribute("open"), null);          // collapsed by default
  assert.ok(!(await page.locator('.dev-tabs button[data-tab="plan"]').isVisible()));
  await page.click("#dev-area summary");
  assert.match(await text(page, ".dev-tabs"), /Research & legacy views[\s\S]*Synthetic sandbox \(demo only\)/i);
  await page.click('.dev-tabs button[data-tab="plan"]');
  assert.ok(await page.locator("#dev-banner").isVisible());
  assert.ok(!(await page.locator("#tab-product").isVisible()));
  assert.equal(await page.locator('.primary-nav button[aria-selected="true"]').count(), 0);
  await page.click("#dev-back");
  assert.ok(await page.locator("#tab-product").isVisible());
  assert.ok(!(await page.locator("#dev-banner").isVisible()));
  await page.close();
});

test("Voyage Simulation and Data & Confidence before any plan point back to Plan Route", { skip }, async () => {
  const { page, calls } = await openDashboard(browser);
  await ready(page);
  await page.click('.primary-nav button[data-view="sim"]');
  assert.ok(await page.locator("#sim-empty").isVisible());
  assert.match(await text(page, "#sim-empty"), /No route planned yet\.[\s\S]*← Plan a route/);
  assert.ok(!(await page.locator("#pr-form").isVisible()));
  assert.equal(simCalls(calls).length, 0);
  await page.click('.primary-nav button[data-view="data"]');
  assert.ok(await page.locator("#data-empty").isVisible());
  const data = await text(page, "#pr-data");
  for (const s of ["OSI SAF sea ice", "Residual U-Net sea-ice forecast", "USNIC icebergs", "ERA5 winds",
    "Copernicus Marine (CMEMS) currents", "Sea-ice forecasting", "Iceberg drift", "Uncertainty", "Wilson risk",
    "Time-dependent routing", "Calibration", "Real Historical Data", "Forecast / hackathon estimate"]) assert.ok(data.includes(s), s);
  assert.ok(!(await page.locator("#pr-data-result").isVisible()));
  await page.click('#data-empty [data-goto="plan"]');
  assert.ok(await page.locator("#pr-form").isVisible());
  assert.equal(await text(page, '.primary-nav button[aria-selected="true"]'), "Plan Route");
  await page.close();
});

test("locations come from GET /real/locations and dates from the route's supported list", { skip }, async () => {
  const { page, calls } = await openDashboard(browser);
  await ready(page);
  const ids = await page.locator("#pr-origin option").evaluateAll((os) => os.map((o) => o.value));
  assert.deepEqual(ids, [...PRESETS.map((p) => p.id), "__map__"]);
  assert.equal(await page.inputValue("#pr-origin"), "alpha");
  assert.equal(await page.inputValue("#pr-destination"), "bravo");
  const d = calls.find((c) => c.path === "/real/historical/dates" && c.search);
  assert.equal(d.search, "?origin=alpha&destination=bravo");
  // The date has no min/max (any calendar date can be chosen); the supported list decides how it is planned.
  assert.equal(await page.getAttribute("#pr-issue", "min"), null);
  assert.equal(await page.getAttribute("#pr-issue", "max"), null);
  assert.equal(await page.inputValue("#pr-issue"), "2023-11-14");
  await page.close();
});

test("a snapped location shows a non-blocking note with requested and routing coordinates", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await ready(page);
  await page.selectOption("#pr-destination", "charlie");
  await ready(page);
  const notes = await text(page, "#pr-notes");
  assert.match(notes, /Destination moved 12\.4 km to open water/);
  assert.match(notes, /requested -62\.000°, -61\.000°, routing from -61\.900°, -60\.700°/);
  assert.equal(await page.isDisabled("#pr-submit"), false);
  await page.close();
});

test("unsupported dates and identical ends block Plan Route with a reason", { skip }, async () => {
  const { page, planCalls } = await openDashboard(browser);
  await ready(page);
  await page.fill("#pr-issue", "2023-06-01");
  assert.equal(await page.isDisabled("#pr-submit"), true);
  assert.match(await text(page, "#pr-notes"), /No Real Historical Data for 2023-06-01/);
  await page.fill("#pr-issue", "2023-11-14");
  await page.selectOption("#pr-destination", "alpha");
  assert.equal(await page.isDisabled("#pr-submit"), true);
  assert.match(await text(page, "#pr-notes"), /same place/);
  assert.equal(planCalls().length, 0);
  await page.close();
});

test("Plan Route sends exactly one POST /real/plan with the chosen inputs and shows a loading state", { skip }, async () => {
  const { page, planCalls } = await openDashboard(browser, { plan: () => ({ status: 200, body: makePlan(), delay: 1200 }) });
  await ready(page);
  await page.selectOption("#pr-destination", "charlie");
  await ready(page);
  await page.fill("#pr-issue", "2023-11-20");
  await page.click("#pr-submit");
  await page.waitForSelector("#pr-loading:not([hidden])");
  const loading = await text(page, "#pr-loading");
  assert.match(loading, /Building route estimate[\s\S]*\(real historical data\)\. Evaluating sea ice, iceberg scenarios, wind and current forcing, route alternatives and departure dates/);
  assert.doesNotMatch(loading, /\d+\s*%/);                 // elapsed seconds only, no fake progress percentage
  assert.equal(await page.isDisabled("#pr-submit"), true);
  await planned(page);
  const calls = planCalls();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].method, "POST");
  assert.deepEqual(JSON.parse(calls[0].body),
    { origin: { preset: "alpha" }, destination: { preset: "charlie" }, issue: "2023-11-20" });
  assert.ok(!(await page.locator("#pr-loading").isVisible()));
  await page.close();
});

test("a recommended result renders the verdict, metrics and three separate risk cards", { skip }, async () => {
  const { page, errors } = await openDashboard(browser);
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  assert.match(await text(page, "#pr-verdict-badge"), /Recommended/);
  assert.match(await text(page, "#pr-verdict-title"), /within the 5% risk budget/);
  const metrics = await text(page, "#pr-metrics");
  for (const s of ["14 Nov 2023, 00:00 UTC", "15 Nov 2023, 06:15 UTC", "30.3 h", "123 km", "457"]) assert.ok(metrics.includes(s), s);
  assert.match(await text(page, '[data-risk="combined"]'), /within budget[\s\S]*3\.1%[\s\S]*2 of 200 scenarios/);
  assert.match(await text(page, '[data-risk="sea_ice"]'), /2\.2%[\s\S]*1 of 200 scenarios/);
  assert.match(await text(page, '[data-risk="iceberg"]'), /1\.3%[\s\S]*1 of 200 scenarios/);
  assert.match(await text(page, '[data-risk="combined"]'), /95% upper bound \(Wilson\)/);
  assert.doesNotMatch(await text(page, "#pr-result"), /catastroph|probability of (sinking|loss)/i);
  assert.deepEqual(errors, []);
  await page.close();
});

test("no recommended departure: says so and shows the least-risky option as not recommended", { skip }, async () => {
  const { page } = await openDashboard(browser, { plan: () => ({ status: 200, body: makePlan("no_feasible_departure") }) });
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  assert.match(await text(page, "#pr-verdict-badge"), /Not recommended/);
  assert.equal(await text(page, "#pr-verdict-title"), "No departure in the selected window meets the 5% risk budget.");
  assert.match(await text(page, "#pr-verdict-text"), /least-risky option for reference only: depart 2023-11-16/);
  assert.match(await text(page, "#pr-metrics"), /Least-risky departure/i);
  assert.match(await text(page, '[data-risk="combined"]'), /exceeds budget[\s\S]*79\.1%/);
  assert.match(await text(page, "#pr-window-rec"), /◆ No date meets the budget; least-risky shown: 2023-11-16/);
  await page.click("#pr-options-more summary");                       // the full table is one click away
  assert.match(await text(page, "#pr-options tbody tr.rec"), /◆ 2023-11-16/);
  await page.close();
});

test("the map draws the returned layers and route, and the day strip changes the day without a new request", { skip }, async () => {
  const { page, planCalls } = await openDashboard(browser);
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  const plan = makePlan();
  const drawn = await page.evaluate(() => ({ ice: prMapState.plan.map.p_ice, land: prMapState.plan.map.land,
    route: prMapState.plan.candidates[0].xy_km, origin: prMapState.plan.origin_xy_km, k: prMapState.k }));
  assert.deepEqual(drawn.ice, plan.layers.layers[0].p_ice_ge_limit_pct);
  assert.deepEqual(drawn.land, plan.layers.land);
  assert.deepEqual(drawn.route, plan.route.xy_km);
  assert.deepEqual(drawn.origin, [grid.x_km[1], grid.y_km[1]]);
  assert.equal(drawn.k, 0);
  const before = await page.locator("#pr-map").screenshot();
  assert.match(await text(page, "#pr-day-title"), /Day 1 of 2/);
  await page.click('#pr-days [data-day="1"]');
  assert.match(await text(page, "#pr-day-title"), /Day 2 of 2/);
  const day = await text(page, "#pr-day-table");
  assert.match(day, /-61\.000°, -59\.000°/);
  assert.match(day, /7\.8%/);                               // max cell breach probability of that day
  assert.match(day, /40\.0%/);                              // max iceberg presence of that day
  assert.equal(await page.evaluate(() => prMapState.k), 1);
  assert.deepEqual(await page.evaluate(() => prMapState.plan.map.p_ice), plan.layers.layers[1].p_ice_ge_limit_pct);
  assert.notDeepEqual(await page.locator("#pr-map").screenshot(), before);
  await page.keyboard.press("ArrowLeft");                   // focus is on the strip
  assert.match(await text(page, "#pr-day-title"), /Day 1 of 2/);
  assert.equal(planCalls().length, 1);
  await page.close();
});

test("departure options come straight from the response, with the shown one marked", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  assert.match(await text(page, "#pr-window-rec"), /★ Recommended departure: 2023-11-14 · risk upper bound 3\.1%/);
  assert.equal(await page.locator("#pr-options-more").getAttribute("open"), null);   // concise: table collapsed
  await page.click("#pr-options-more summary");
  const rows = await page.locator("#pr-options tbody tr").allInnerTexts();
  assert.equal(rows.length, 3);
  assert.match(rows[0], /★ 2023-11-14[\s\S]*meets budget[\s\S]*3\.1%[\s\S]*shown/);
  assert.match(rows[1], /2023-11-15[\s\S]*exceeds budget[\s\S]*20\.0%[\s\S]*\+0\.8 h/);
  assert.match(await text(page, "#pr-window-verdict"), /1 of 3 departure dates meet the 5% budget/);
  await page.close();
});

test("the hindsight disclosure and Data & Confidence panel come from the response", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  assert.ok(await page.locator("#pr-disclosure").isVisible());
  assert.equal(await text(page, "#pr-disclosure"), DISCLOSURE);
  assert.match(await text(page, "#pr-banners"), /HISTORICAL MODE[\s\S]*ERA5\/CMEMS hindsight forcing[\s\S]*Research estimate, not certified navigation/i);
  await page.click('.primary-nav button[data-view="data"]');
  assert.ok(await page.locator("#pr-data").isVisible());
  assert.ok(!(await page.locator("#pr-result").isVisible()));
  const data = await text(page, "#pr-data");
  assert.ok(data.includes(DISCLOSURE));
  assert.match(data, /Stand-in limitation one\./);
  assert.match(data, /Research estimate, not certified navigation/);
  assert.doesNotMatch(data, /confidence[^\n]*\d+\s*%/i);   // no invented confidence percentages
  await page.close();
});

for (const [name, reply, expect] of [
  ["out of coverage (422)", { status: 422, body: { detail: { status: "out_of_coverage", reason: "no USNIC list within 14 days" } } },
    /archive does not cover this date[\s\S]*no USNIC list within 14 days/],
  ["invalid location (422)", { status: 422, body: { detail: { status: "invalid_location", reason: "outside the routing grid" } } },
    /cannot be routed[\s\S]*outside the routing grid/],
  ["model unavailable (503)", { status: 503, body: { detail: { status: "blocked", reason: "model checkpoint missing" } } },
    /unavailable right now[\s\S]*model checkpoint missing[\s\S]*Nothing synthetic/],
  ["malformed response", { status: 200, body: { status: "recommended", metadata: { mode: "historical", execution_mode: "real" } } },
    /could not be read[\s\S]*Malformed/],
  ["non-JSON response", { status: 200, raw: "<html>proxy error</html>" }, /could not be read/],
  ["unlabelled synthetic response", { status: 200, body: { ...makePlan(), metadata: { ...makePlan().metadata, execution_mode: "controlled_synthetic" } } },
    /not labelled as real historical data/],
  ["network failure", { abort: true }, /Could not reach the planning server/],
]) {
  test(`errors are shown in words, inputs kept, nothing substituted: ${name}`, { skip }, async () => {
    const { page, planCalls } = await openDashboard(browser, { plan: () => reply });
    await ready(page);
    await page.selectOption("#pr-destination", "charlie");
    await ready(page);
    await page.fill("#pr-issue", "2023-11-21");
    await page.click("#pr-submit");
    await page.waitForSelector("#pr-error:not([hidden])");
    assert.match(await text(page, "#pr-error"), expect);
    assert.ok(!(await page.locator("#pr-result").isVisible()));
    assert.equal(await page.evaluate(() => prod.result), null);
    assert.equal(await page.inputValue("#pr-destination"), "charlie");
    assert.equal(await page.inputValue("#pr-issue"), "2023-11-21");
    assert.equal(await page.isDisabled("#pr-submit"), false);
    assert.equal(planCalls().length, 1);
    await page.close();
  });
}

test("a slow request can be cancelled, leaving no result", { skip }, async () => {
  const { page } = await openDashboard(browser, { plan: () => ({ status: 200, body: makePlan(), delay: 4000 }) });
  await ready(page);
  await page.click("#pr-submit");
  await page.waitForSelector("#pr-loading:not([hidden])");
  await page.waitForFunction(() => Number(document.querySelector("#pr-elapsed").textContent) >= 1);
  await page.click("#pr-cancel");
  await page.waitForSelector("#pr-error:not([hidden])");
  assert.match(await text(page, "#pr-error"), /cancelled/i);
  assert.ok(!(await page.locator("#pr-result").isVisible()));
  await page.close();
});

test("without the real archive the product says it is unavailable and never plans", { skip }, async () => {
  const { page, planCalls } = await openDashboard(browser, { historical: "unavailable" });
  await page.waitForSelector("#pr-unavailable:not([hidden])");
  assert.match(await text(page, "#pr-unavailable"), /no archive configured[\s\S]*Nothing synthetic/);
  assert.equal(await page.isDisabled("#pr-submit"), true);
  assert.ok(!(await page.locator("#pr-landing").isVisible()));
  assert.equal(planCalls().length, 0);
  await page.close();
});

test("the result fits a phone-width screen without horizontal page scroll", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await page.setViewportSize({ width: 390, height: 844 });
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  await page.close();
});

/* ------------------------------------------------------------- Simulate Voyage */
async function simulated(page) {
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  await page.click("#pr-simulate");
  await page.waitForSelector("#sim-body:not([hidden])");
}
const simCalls = (calls) => calls.filter((c) => c.path === "/real/simulate");
const vessel = (page) => page.evaluate(() => sim.data.frames[sim.i].position.xy_km);

test("Simulate Voyage starts from the planned route without planning again", { skip }, async () => {
  const { page, calls, planCalls, errors } = await openDashboard(browser);
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  assert.ok(await page.locator("#pr-simulate").isVisible());
  await page.click("#pr-simulate");
  await page.waitForSelector("#sim-body:not([hidden])");
  assert.equal(planCalls().length, 1);
  const sc = simCalls(calls);
  assert.equal(sc.length, 1);
  assert.deepEqual(JSON.parse(sc[0].body), { origin: { preset: "alpha" }, destination: { preset: "bravo" },
    issue: "2023-11-14", departure: "2023-11-14" });
  assert.equal(await page.locator('.primary-nav button[aria-selected="true"]').getAttribute("data-view"), "sim");
  // back to the result and again: the frames are reused, nothing is recomputed
  await page.click('.primary-nav button[data-view="plan"]');
  assert.ok(await page.locator("#pr-result").isVisible());
  await page.click('.primary-nav button[data-view="sim"]');
  assert.equal(simCalls(calls).length, 1);
  assert.deepEqual(errors, []);
  await page.close();
});

test("the simulation shows a loading state while the server computes, without fake progress", { skip }, async () => {
  const { page } = await openDashboard(browser, { simulate: () => ({ status: 200, body: makeSim(true), delay: 1500 }) });
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  await page.click("#pr-simulate");
  await page.waitForSelector("#sim-loading:not([hidden])");
  const t = await text(page, "#sim-loading");
  assert.match(t, /Simulating the voyage on real historical data/);
  assert.doesNotMatch(t, /\d+\s*%/);
  await page.waitForSelector("#sim-body:not([hidden])");
  assert.ok(!(await page.locator("#sim-loading").isVisible()));
  await page.close();
});

test("next, previous and the slider change the frame and move the vessel", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await simulated(page);
  assert.match(await text(page, "#sim-slider-out"), /2023-11-14 \(0 of 3\)/);
  const p0 = await vessel(page), shot0 = await page.locator("#sim-map").screenshot();
  assert.equal(await page.isDisabled("#sim-prev"), true);
  await page.click("#sim-next");
  assert.match(await text(page, "#sim-slider-out"), /2023-11-15 \(1 of 3\)/);
  assert.notDeepEqual(await vessel(page), p0);
  assert.notDeepEqual(await page.locator("#sim-map").screenshot(), shot0);
  await page.click("#sim-prev");
  assert.match(await text(page, "#sim-slider-out"), /\(0 of 3\)/);
  await page.locator("#sim-slider").fill("2");
  assert.match(await text(page, "#sim-slider-out"), /2023-11-16 \(2 of 3\)/);
  assert.match(await text(page, "#sim-status"), /Route replanned \(1×\)/);
  await page.click('#sim-events [data-frame="0"]');
  assert.match(await text(page, "#sim-status"), /On planned route/);
  await page.close();
});

test("Play advances through the precomputed frames and Pause stops it", { skip }, async () => {
  const { page, calls } = await openDashboard(browser);
  await simulated(page);
  await page.click("#sim-play");
  assert.match(await text(page, "#sim-play"), /Pause/);
  await page.waitForFunction(() => sim.i >= 1, null, { timeout: 5000 });
  await page.click("#sim-play");                                   // pause
  const at = await page.evaluate(() => sim.i);
  await page.waitForTimeout(1800);
  assert.equal(await page.evaluate(() => sim.i), at);
  assert.match(await text(page, "#sim-play"), /Play/);
  await page.click("#sim-play");
  await page.waitForFunction(() => sim.i === sim.data.frames.length - 1, null, { timeout: 8000 });
  await page.waitForFunction(() => sim.timer === null);
  assert.match(await text(page, "#sim-play"), /Replay/);
  assert.equal(simCalls(calls).length, 1);                         // playback never calls the server
  await page.close();
});

test("a real replan event is visible: old vs new route, reason and changes", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await simulated(page);
  const chip = page.locator('#sim-events [data-frame="1"]');
  assert.match(await chip.innerText(), /REPLAN/);
  assert.match(await chip.getAttribute("class"), /replan/);
  const before = await page.evaluate(() => sim.data.frames[0].route.xy_km);
  await chip.click();
  assert.match(await text(page, "#sim-decision-title"), /REPLAN: new route selected/);
  const box = await text(page, "#sim-decision-card");
  assert.match(box, /stand-in switch/);
  assert.match(box, /no longer met the risk budget/);
  assert.match(box, /8\.1%[\s\S]*3\.1%[\s\S]*-5\.0 pp/);
  assert.match(box, /\+1\.5 h/);
  assert.match(box, /\+11 km/);
  assert.match(await text(page, "#sim-map-legend"), /Original planned route[\s\S]*New route ahead \(after replan\)/);
  const after = await page.evaluate(() => sim.data.frames[sim.i].route.xy_km);
  assert.notDeepEqual(after, before);
  assert.match(await text(page, "#sim-subtitle"), /route replanned 1×/);
  await page.close();
});

test("a voyage without a replan says so", { skip }, async () => {
  const { page } = await openDashboard(browser, { simulate: () => ({ status: 200, body: makeSim(false) }) });
  await simulated(page);
  assert.match(await text(page, "#sim-subtitle"), /No replan was required during this voyage\./);
  assert.doesNotMatch(await text(page, "#sim-events"), /REPLAN/);
  await page.click('#sim-events [data-frame="2"]');
  assert.match(await text(page, "#sim-summary"), /No replan was required during this voyage\./);
  await page.close();
});

test("the final summary appears at the last frame with the server's numbers", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await simulated(page);
  assert.ok(!(await page.locator("#sim-summary").isVisible()));
  await page.click('#sim-events [data-frame="3"]');
  assert.ok(await page.locator("#sim-summary").isVisible());
  const s = await text(page, "#sim-summary");
  assert.match(s, /Voyage completed: arrived 16 Nov 2023, 09:00 UTC\. Route replanned 1 time/);
  for (const v of ["57.0 h", "131 km", "470", "3.1%", "Arrived"]) assert.ok(s.includes(v), v);
  assert.match(s, /REPLAN[\s\S]*stand-in switch/);
  await page.close();
});

test("historical disclosure stays visible during the simulation", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await simulated(page);
  assert.match(await text(page, "#sim-banners"), /HISTORICAL MODE[\s\S]*ERA5\/CMEMS hindsight forcing[\s\S]*Research estimate, not certified navigation/i);
  assert.ok(await page.locator("#sim-disclosure").isVisible());
  assert.match(await text(page, "#sim-disclosure"), /not live vessel tracking/);
  assert.equal(await text(page, "#mode-badge"), "REAL HISTORICAL DATA · HINDSIGHT FORCING");
  await page.close();
});

for (const [name, opts, expect] of [
  ["not simulated", { simulate: () => ({ status: 200, body: { status: "not_simulated", reason: "the archive cannot issue the daily forecasts" } }) },
    /cannot be simulated[\s\S]*cannot issue the daily forecasts/],
  ["job failed", { simulate: () => ({ status: 200, body: null, failed: "HTTPException: model missing" }) }, /failed on the server[\s\S]*model missing/],
  ["synthetic-labelled result", { simulate: () => ({ status: 200, body: { ...makeSim(true), metadata: { ...makeSim(true).metadata, execution_mode: "controlled_synthetic" } } }) },
    /not labelled as real historical data/],
  ["frames out of order", { simulate: () => { const s = makeSim(true); s.frames.reverse(); return { status: 200, body: s }; } }, /out of order/],
  ["503 on submit", { simulate: () => ({ status: 503, body: { detail: { status: "blocked", reason: "no archive" } } }) }, /unavailable[\s\S]*no archive/],
]) {
  test(`simulation errors show without playback or substitution: ${name}`, { skip }, async () => {
    const { page } = await openDashboard(browser, opts);
    await ready(page);
    await page.click("#pr-submit");
    await planned(page);
    await page.click("#pr-simulate");
    await page.waitForSelector("#sim-error:not([hidden])");
    assert.match(await text(page, "#sim-error"), expect);
    assert.ok(!(await page.locator("#sim-body").isVisible()));
    assert.equal(await page.evaluate(() => sim.data), null);
    await page.close();
  });
}

test("a new plan clears the previous simulation", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await simulated(page);
  await page.click('.primary-nav button[data-view="plan"]');
  await page.click("#pr-submit");
  await planned(page);
  assert.equal(await page.evaluate(() => sim.data), null);
  await page.close();
});

test("the simulation fits a phone-width screen", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await page.setViewportSize({ width: 390, height: 844 });
  await simulated(page);
  await page.click('#sim-events [data-frame="3"]');
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  await page.close();
});

test("an additional season from the archive is selectable with its evaluation label, no special path", { skip }, async () => {
  const NEW = ["2024-11-14", "2024-11-15", "2024-11-16"];
  const s25 = { season: "2024-25", first: NEW[0], last: NEW.at(-1), n_dates: NEW.length, out_of_sample: true,
    in_sample_notes: [], independent_evaluation: true,
    evaluation_notes: ["the frozen sea-ice U-Net and its calibration were scored on this season after freezing (nothing fitted)"] };
  const dates = { ...datesPayload, route_dates: [...DATES, ...NEW], window_dates: [...DATES, ...NEW],
    seasons: { route: [...datesPayload.seasons.route, s25], window: [...datesPayload.seasons.window, s25] } };
  const bodies = [];
  const { page } = await openDashboard(browser, { dates, plan: (b) => { bodies.push(b); return { status: 200, body: makePlan() }; } });
  await ready(page);
  // No season list: the date alone picks the season, and the new season's dates are accepted as historical.
  assert.equal(await page.locator("#pr-season").count(), 0);
  await page.fill("#pr-issue", "2024-11-14");
  await page.dispatchEvent("#pr-issue", "change");
  assert.match(await text(page, "#pr-mode"), /REAL HISTORICAL DATA/);
  assert.equal(await page.getAttribute("#pr-issue", "max"), null);
  await page.click("#pr-submit");
  await page.waitForSelector("#pr-result:not([hidden])");
  assert.equal(bodies.length, 1);
  assert.equal(bodies[0].issue, "2024-11-14");
  await page.close();
});

/* ------------------------------------------------------------- Pick on map / coordinates */
// The dates endpoint resolves the chosen ends on the server (snapping included); this stand-in echoes a point.
function datesForPoints(snap = {}) {
  return (q) => {
    const end = (role) => {
      if (q.get(role)) { const p = PRESETS.find((x) => x.id === q.get(role)); return { ...p, cell: [p.resolved.row, p.resolved.col] }; }
      const lat = Number(q.get(`${role}_lat`)), lon = Number(q.get(`${role}_lon`));
      const s = snap[role];
      return { id: null, name: `${lat.toFixed(4)}, ${lon.toFixed(4)}`, requested: { lat, lon }, cell: [2, 2],
        resolved: { lat: s ? lat + 0.1 : lat, lon, row: 2, col: 2 }, snapped: Boolean(s), distance_km: s ? s : 3.2,
        reason: s ? "the requested point is on land; moved to the nearest navigable routing cell" : "the requested point lies in a navigable routing cell" };
    };
    return { body: { ...datesPayload, route: { origin: end("origin"), destination: end("destination") } } };
  };
}
const planBodies = (calls) => calls.filter((c) => c.path === "/real/plan").map((c) => JSON.parse(c.body));
const datesQueries = (calls) => calls.filter((c) => c.path === "/real/historical/dates").map((c) => c.search);

async function pickOnMap(page, role, fx, fy) {
  await page.click(`#pr-pick-${role}`);
  assert.ok(await page.locator("#pr-pick.picking").isVisible());
  await page.locator("#pr-pick-map").scrollIntoViewIfNeeded();
  const box = await page.locator("#pr-pick-map").boundingBox();
  await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
  await ready(page);
}

test("a map-picked start and a preset destination plan through the coordinate API", { skip }, async () => {
  const { page, calls, errors } = await openDashboard(browser, { datesFor: datesForPoints() });
  await ready(page);
  await pickOnMap(page, "origin", 0.4, 0.5);
  assert.equal(await page.inputValue("#pr-origin"), "__map__");
  const lat = Number(await page.inputValue("#pr-origin-lat")), lon = Number(await page.inputValue("#pr-origin-lon"));
  assert.ok(lat <= -60 && lat >= -61.1 && lon >= -62.3 && lon <= -59.7, `${lat}, ${lon}`);
  assert.ok(datesQueries(calls).at(-1).includes(`origin_lat=${lat}&origin_lon=${lon}&destination=bravo`));
  assert.match(await text(page, "#pr-notes"), /Start is in open water: requested[\s\S]*not moved/);
  await page.click("#pr-submit");
  await planned(page);
  assert.deepEqual(planBodies(calls), [{ origin: { lat, lon }, destination: { preset: "bravo" }, issue: "2023-11-14" }]);
  assert.deepEqual(errors, []);
  await page.close();
});

test("a preset start and a map-picked destination", { skip }, async () => {
  const { page, calls } = await openDashboard(browser, { datesFor: datesForPoints() });
  await ready(page);
  await pickOnMap(page, "destination", 0.6, 0.4);
  await page.click("#pr-submit");
  await planned(page);
  const b = planBodies(calls)[0];
  assert.deepEqual(b.origin, { preset: "alpha" });
  assert.ok(Number.isFinite(b.destination.lat) && Number.isFinite(b.destination.lon) && !b.destination.preset);
  await page.close();
});

test("both ends typed as coordinates; a point on land shows its move to open water", { skip }, async () => {
  const { page, calls } = await openDashboard(browser, { datesFor: datesForPoints({ destination: 18 }) });
  await ready(page);
  for (const [role, lat, lon] of [["origin", "-60.25", "-61.5"], ["destination", "-61", "-60.2"]]) {
    await page.selectOption(`#pr-${role}`, "__map__");
    await page.fill(`#pr-${role}-lat`, lat);
    await page.fill(`#pr-${role}-lon`, lon);
    await page.dispatchEvent(`#pr-${role}-lon`, "change");
  }
  await ready(page);
  const notes = await text(page, "#pr-notes");
  assert.match(notes, /Destination moved 18\.0 km to open water: requested -61\.000°, -60\.200°, routing from -60\.900°, -60\.200°/);
  assert.match(notes, /on land; moved to the nearest navigable routing cell/);
  await page.click("#pr-submit");
  await planned(page);
  assert.deepEqual(planBodies(calls), [{ origin: { lat: -60.25, lon: -61.5 }, destination: { lat: -61, lon: -60.2 }, issue: "2023-11-14" }]);
  await page.close();
});

test("a point outside the routing grid is refused with the server's reason, never planned", { skip }, async () => {
  const { page, planCalls } = await openDashboard(browser, { datesFor: (q) => (q.get("origin_lat") ?
    { status: 422, body: { detail: { status: "invalid_location", reason: "-40.0000, -60.0000 (-40.0, -60.0) is outside the routing grid" } } } :
    { body: datesPayload }) });
  await ready(page);
  await page.selectOption("#pr-origin", "__map__");
  await page.fill("#pr-origin-lat", "-40");
  await page.fill("#pr-origin-lon", "-60");
  await page.dispatchEvent("#pr-origin-lon", "change");
  await page.waitForFunction(() => /outside the routing grid/.test(document.querySelector("#pr-notes").innerText));
  assert.match(await text(page, "#pr-notes"), /cannot be routed[\s\S]*outside the routing grid/);
  assert.ok(await page.isDisabled("#pr-submit"));
  assert.equal(planCalls().length, 0);
  assert.equal(await page.evaluate(() => kmToLatLon(prod.locations.map, 1e5, 0)), null);   // clicks off the grid are ignored
  await page.close();
});

/* ------------------------------------------------------------- forecast mode (dates after the archive) */
const FC_DISCLOSURE = "Forecast estimate: stand-in disclosure for dates after the archive.";
const FC_DATES = ["2026-11-14", "2026-11-15", "2026-11-19"];
const fcDates = { ...datesPayload, forecast: { mode: "forecast", label: "Forecast / hackathon estimate", disclosure: FC_DISCLOSURE,
  banners: ["FORECAST / HACKATHON ESTIMATE", "Proxy sea ice and forcing from an analogue season", "Research estimate, not certified navigation"],
  dates: FC_DATES, first: FC_DATES[0], last: FC_DATES.at(-1), available: true, reason: null } };

function makeForecastPlan() {
  const p = makePlan();
  const files = (f) => [{ file: f, sha256: "ab".repeat(32), first: "2024-11-01", last: "2025-02-28" }];
  p.metadata = { mode: "forecast", label: "Forecast / hackathon estimate", execution_mode: "modelled", data_status: "forecast_estimate",
    issue_date: "2026-11-19", requested_date: "2026-11-19", hindsight_forcing: false, proxy_forcing: true,
    forecast_disclosure: FC_DISCLOSURE, banners: fcDates.forecast.banners, disclaimer: "Stand-in disclaimer.", window_days: 14,
    horizon_days: 6, scenario_days: 20, n_scenarios: 200, layer_source: ["proxy_analogue_observed", "forecast"],
    forecast: { forecast_mode: true, requested_date: "2026-11-19", archive_last_date: "2025-02-28",
      observations_for_requested_dates: "none: stand-in",
      date_mapping: { analogue_start_date: "2024-11-19", analogue_season: "2024-25", offset_days: 730,
        engine_days: ["2024-11-19", "2024-12-08"], shown_as: ["2026-11-19", "2026-12-08"] },
      sea_ice: { status: "proxy_analogue", observed_window_used: ["2024-11-06", "2024-11-19"], source: "OSI SAF stand-in",
        file: "sea_ice_stand_in.nc", sha256: "cd".repeat(32) },
      sea_ice_forecast: { status: "forecast", model: { id: "unet_stand_in", sha256: "ef".repeat(32), lead_days: 21 } },
      forcing: { status: "proxy_analogue_reanalysis", winds: { product: "ERA5 daily reanalysis", dates_used: ["2024-11-19", "2024-12-08"],
        files: files("winds.nc") }, currents: { product: "CMEMS GLOBAL reanalysis daily", dates_used: ["2024-11-19", "2024-12-08"],
        files: files("currents.nc") } },
      icebergs: { status: "observed_snapshot_held", source: "USNIC stand-in", snapshot_date: "2026-10-01", file: "AntarcticIcebergs_20261001.csv",
        sha256: "12".repeat(32), age_days_at_requested_date: 49, max_age_days: 120, n_source_bergs: 33, n_in_grid: 1, ids_in_grid: ["T1"],
        drift_model: { model: "calibrated ensemble drift", beta: 0.1, alpha_scale: 0.1, spread_factor: 0.6053, members: 200 } },
      limitations: ["Research / decision-support prototype; not certified for navigation.",
        "The iceberg snapshot may be up to 120 days old; its actual age is shown in the result."],
      engine_provenance: { limitations: ["Iceberg positions come from the latest weekly USNIC list (up to 14 days old)."] } } };
  p.daily[0].layer_source = "proxy_analogue_observed";
  p.layers.layers[0].source = "proxy_analogue_observed";
  return p;
}

const anDates = { ...fcDates, estimate: { any_date_after: "2025-02-28", label: "Forecast / hackathon estimate",
  pathways: { proxy_forecast: "listed", seasonal_analogue: "every other date" },
  analogue: { available: true, reason: null, label: "Historical seasonal analogue" } } };

function makeAnaloguePlan() {
  const p = makeForecastPlan();
  const years = [2023, 2022, 2021, 2020, 2019, 2018, 2017];
  const fc = p.metadata.forecast;
  p.metadata.requested_date = p.metadata.issue_date = "2027-08-14";
  p.metadata.banners = ["FORECAST / HACKATHON ESTIMATE", "Historical seasonal analogue: real sea ice, winds and currents of earlier years",
    "Research estimate, not certified navigation"];
  p.metadata.layer_source = ["analogue_observed", "analogue_observed"];
  p.metadata.forecast = { ...fc, pathway: "seasonal_analogue", pathway_label: "Historical seasonal analogue",
    pathway_reason: "stand-in reason", requested_date: "2027-08-14", analogue_date: "2024-08-14",
    date_mapping: { analogue_start_date: "2024-08-14", analogue_season: "2024 (outside the Nov-Feb archive seasons)",
      offset_days: 1095, engine_days: ["2024-08-14", "2024-09-02"], shown_as: ["2027-08-14", "2027-09-02"] },
    sea_ice: { status: "historical_analogue_ensemble", member_years: years, day_shifts: [-7, 7], distinct_sequences: 105,
      members: 200, source: "OSI SAF OSI-450-a / OSI-430-a daily sea-ice concentration, 25 km",
      member_windows: years.map((y) => ({ year: y, shift_days: 0, first: `${y}-08-14`, last: `${y}-09-02` })),
      off_season_files: [{ file: "sea_ice_25km_offseason_2023_mar_oct.nc", sha256: "aa".repeat(32) }] },
    forcing: { status: "historical_reanalysis_analogue", winds: { product: "ERA5 daily reanalysis", dates_used: ["2024-08-14", "2024-09-02"],
      files: [] }, currents: { product: "CMEMS GLOBAL reanalysis daily", dates_used: ["2024-08-14", "2024-09-02"], files: [] } },
    icebergs: { ...fc.icebergs, kind: "analogue_year", status: "analogue_year_snapshot", snapshot_date: "2024-08-09",
      age_days_at_requested_date: 1100, age_days_at_analogue_date: 5, file: "AntarcticIcebergs_20240809.csv" },
    confidence: { level: "low", member_years: years, caveats: ["Not a forecast: stand-in.", "Risk percentages: stand-in.",
      "No official iceberg list falls within 120 days of the requested date: stand-in."],
      sea_ice_spread: { mean_concentration_by_year: { 2023: 0.21, 2022: 0.24 }, min: 0.21, max: 0.24, range: 0.03 } },
    limitations: ["Seasonal analogue: stand-in limitation."] };
  delete p.metadata.forecast.sea_ice_forecast;
  delete p.metadata.forecast.engine_provenance;
  p.daily[0].layer_source = "analogue_observed";
  p.layers.layers[0].source = "analogue_observed";
  return p;
}

test("a future date is offered as a labelled forecast estimate and planned through the same call", { skip }, async () => {
  const bodies = [];
  const { page, errors } = await openDashboard(browser, { dates: fcDates,
    plan: (b) => { bodies.push(b); return { status: 200, body: makeForecastPlan() }; } });
  await ready(page);
  // 2026-11-19 is in the server's forecast range, so it is the default demonstration date.
  assert.equal(await page.inputValue("#pr-issue"), "2026-11-19");
  assert.match(await text(page, "#pr-mode"), /FORECAST \/ HACKATHON ESTIMATE[\s\S]*proxy/);
  // No season list in the product: the date is the only control; the examples come from the server's lists.
  assert.equal(await page.locator("#pr-season").count(), 0);
  // No min/max on the date: any calendar date can be chosen; the server decides how it is planned.
  assert.equal(await page.getAttribute("#pr-issue", "max"), null);
  assert.equal(await page.getAttribute("#pr-issue", "min"), null);
  assert.match(await text(page, "#pr-examples"), /Try: 19 Nov 2026 \(forecast\) · 14 Nov 2023 \(historical\)/);
  await page.click('#pr-examples [data-date="2023-11-14"]');
  assert.equal(await page.inputValue("#pr-issue"), "2023-11-14");
  assert.match(await text(page, "#pr-mode"), /REAL HISTORICAL DATA/);
  await page.fill("#pr-issue", "2026-11-19");
  await page.dispatchEvent("#pr-issue", "change");
  assert.match(await text(page, "#pr-mode"), /FORECAST \/ HACKATHON ESTIMATE[\s\S]*proxy/);
  assert.match(await text(page, "#mode-badge"), /FORECAST \/ HACKATHON ESTIMATE/);
  await page.click("#pr-submit");
  await planned(page);
  assert.equal(bodies.length, 1);
  assert.equal(bodies[0].issue, "2026-11-19");
  assert.equal(await text(page, "#pr-disclosure"), FC_DISCLOSURE);
  assert.match(await text(page, "#pr-banners"), /FORECAST \/ HACKATHON ESTIMATE/);
  assert.ok(!(await text(page, "#pr-banners")).includes("HISTORICAL MODE"));
  assert.match(await text(page, "#pr-verdict"), /forecast estimate/i);
  const how = await text(page, "#pr-howto");
  assert.match(how, /How this estimate was generated/);
  assert.match(how, /Sea ice historical analogue start[\s\S]*U-Net/);
  assert.match(how, /Wind ERA5 daily reanalysis[\s\S]*not a 2026 forecast/);
  assert.match(how, /Currents CMEMS GLOBAL reanalysis daily[\s\S]*not a 2026 forecast/);
  assert.match(how, /Icebergs latest available official USNIC list \(2026-10-01[\s\S]*calibrated drift/);
  assert.match(how, /Result hackathon research estimate, not certified navigation/);
  assert.match(await text(page, '[data-risk="combined"]'), /0 of 200|of 200 scenarios/);
  const data = await page.locator("#pr-data-modes").innerText();
  assert.match(data, /Sea ice \(proxy\)[\s\S]*2024-11-06[\s\S]*NOT observations of[\s\S]*2026/);
  assert.match(data, /Winds & currents \(proxy\)[\s\S]*not a forecast/);
  assert.match(data, /list of 2026-10-01 \(49 d before/);
  // Each input says what it is; proxy forcing is never called a forecast.
  assert.match(await text(page, "#pr-data-kinds"), /Real observation[\s\S]*Hindsight reanalysis[\s\S]*Proxy \/ analogue[\s\S]*Model output[\s\S]*Derived estimate/);
  const sources = await text(page, "#pr-data-sources");
  assert.match(sources, /Proxy \/ analogue\s*ERA5 winds[^.]*proxy, not a forecast/);
  assert.match(sources, /Proxy \/ analogue\s*Copernicus Marine \(CMEMS\) currents[^.]*proxy, not a forecast/);
  assert.match(sources, /Model output\s*Residual U-Net sea-ice forecast/);
  assert.match(await text(page, "#pr-data-method"), /Calibration[\s\S]*residual bank[\s\S]*calibrated on hindcasts/);
  // The forecast's own limitations: the 120-day snapshot rule, not the historical 14-day line.
  const limits = await page.locator("#pr-data-limits").textContent();
  assert.match(limits, /may be up to 120 days old; its actual age is shown in the result/);
  assert.doesNotMatch(limits, /up to 14 days old/);
  assert.deepEqual(errors, []);
  await page.close();
});

test("a later date is refused with the server's reason when no seasonal analogue is available", { skip }, async () => {
  const blocked = { ...anDates, estimate: { ...anDates.estimate, analogue: { available: false,
    reason: "blocked: no off-season inputs are listed for the seasonal analogue" } } };
  const { page, planCalls } = await openDashboard(browser, { dates: blocked });
  await ready(page);
  await page.fill("#pr-issue", "2026-12-25");
  await page.dispatchEvent("#pr-issue", "change");
  assert.match(await text(page, "#pr-notes"), /No estimate is available for 2026-12-25 on this route: blocked: no off-season inputs/);
  assert.ok(await page.locator("#pr-submit").isDisabled());
  assert.equal(planCalls().length, 0);
  await page.close();
});

test("a chosen future date is never overwritten by a route change or the season list", { skip }, async () => {
  const { page, calls } = await openDashboard(browser, { dates: fcDates });
  await ready(page);
  await page.fill("#pr-issue", "2026-11-15");
  await page.dispatchEvent("#pr-issue", "change");
  await page.selectOption("#pr-destination", "charlie");
  await ready(page);
  assert.equal(datesQueries(calls).at(-1), "?origin=alpha&destination=charlie");
  assert.equal(await page.inputValue("#pr-issue"), "2026-11-15");
  await page.click("#pr-swap");
  await ready(page);
  assert.equal(await page.inputValue("#pr-issue"), "2026-11-15");
  assert.match(await text(page, "#pr-mode"), /FORECAST \/ HACKATHON ESTIMATE/);
  await page.close();
});

test("without 2026-11-19 in the forecast range the default stays the historical demo date", { skip }, async () => {
  const fc = { ...fcDates, forecast: { ...fcDates.forecast, dates: ["2026-11-14"], first: "2026-11-14", last: "2026-11-14" } };
  const { page } = await openDashboard(browser, { dates: fc });
  await ready(page);
  assert.equal(await page.inputValue("#pr-issue"), "2023-11-14");
  assert.match(await text(page, "#pr-mode"), /REAL HISTORICAL DATA/);
  await page.close();
});

for (const day of ["2026-12-15", "2027-01-20", "2027-02-15", "2027-08-14", "2028-03-03", "2030-08-14", "2025-11-14", "2026-06-01"]) {
  test(`any future date is accepted as an estimate, never clamped or replaced: ${day}`, { skip }, async () => {
    const { page, planCalls } = await openDashboard(browser, { dates: anDates });
    await ready(page);
    await page.fill("#pr-issue", day);
    await page.dispatchEvent("#pr-issue", "change");
    assert.equal(await page.inputValue("#pr-issue"), day);
    assert.match(await text(page, "#pr-mode"), /FORECAST \/ HACKATHON ESTIMATE[\s\S]*Historical seasonal analogue/);
    assert.doesNotMatch(await text(page, "#pr-notes"), /outside|No estimate|No Real Historical Data/);
    assert.ok(await page.isEnabled("#pr-submit"));
    assert.equal(planCalls().length, 0);
    await page.close();
  });
}

test("a past date outside the archive is still explained and not planned", { skip }, async () => {
  const { page, planCalls } = await openDashboard(browser, { dates: anDates });
  await ready(page);
  await page.fill("#pr-issue", "2024-06-01");
  await page.dispatchEvent("#pr-issue", "change");
  assert.match(await text(page, "#pr-notes"), /No Real Historical Data for 2024-06-01[\s\S]*Any date after 2025-02-28 is accepted/);
  assert.ok(await page.isDisabled("#pr-submit"));
  assert.equal(planCalls().length, 0);
  await page.close();
});

test("a seasonal-analogue estimate discloses the method, analogue date, forcing, icebergs and confidence", { skip }, async () => {
  const bodies = [];
  const { page, errors } = await openDashboard(browser, { dates: anDates,
    plan: (b) => { bodies.push(b); return { status: 200, body: makeAnaloguePlan() }; } });
  await ready(page);
  assert.match(await text(page, "#pr-examples"), /14 Aug 2027 \(seasonal analogue\)/);
  await page.click('#pr-examples [data-date="2027-08-14"]');
  assert.equal(await page.inputValue("#pr-issue"), "2027-08-14");
  assert.match(await text(page, "#pr-notes"), /Any other date after 2025-02-28: historical seasonal analogue estimate/);
  assert.match(await text(page, "#mode-badge"), /FORECAST \/ HACKATHON ESTIMATE · SEASONAL ANALOGUE/);
  await page.click("#pr-submit");
  await page.waitForSelector("#pr-result:not([hidden])");
  assert.equal(bodies[0].issue, "2027-08-14");
  assert.match(await text(page, "#mode-badge"), /SEASONAL ANALOGUE/);
  assert.match(await text(page, "#pr-banners"), /FORECAST \/ HACKATHON ESTIMATE[\s\S]*Historical seasonal analogue/i);
  const how = await text(page, "#pr-howto");
  assert.match(how, /Method Historical seasonal analogue/);
  assert.match(how, /Analogue date \w+, 14 Aug 2024/);
  assert.match(how, /Wind\/current forcing historical reanalysis analogue/);
  assert.match(how, /Iceberg source official USNIC list of [\s\S]*2024 \(the analogue year's list/);
  assert.match(how, /Sea ice real OSI SAF observations of the same calendar days in 2017–2023/);
  assert.match(how, /Confidence LOW/);
  assert.doesNotMatch(how, /U-Net/);
  await page.click("#pr-howto-more");
  const modes = await page.locator("#pr-data-modes").innerText();
  assert.match(modes, /historical seasonal analogue/);
  assert.match(modes, /Sea ice \(analogue\)[\s\S]*2017–2023[\s\S]*No model/);
  assert.match(modes, /Confidence[\s\S]*LOW/);
  assert.match(modes, /list of 2024-08-09 \(5 d before the analogue date\)/);
  assert.doesNotMatch(modes, /undefined/);
  const inputs = await page.locator("#pr-data-inputs").innerText();
  assert.match(inputs, /Analogue date\s*2024-08-14/);
  assert.match(inputs, /Sea-ice members[\s\S]*2023-08-14 – 2023-09-02/);
  assert.match(inputs, /Why an analogue\s*stand-in reason/);
  assert.match(await text(page, "#pr-data-badge"), /Seasonal analogue estimate/);
  assert.match(await text(page, "#pr-data-modes-general"), /historical seasonal analogue \(this result\)/);
  assert.match(await text(page, "#pr-data-sources"), /same calendar days in 2017–2023[\s\S]*No sea-ice model is run/);
  assert.deepEqual(errors, []);
  await page.close();
});

test("a forecast_unavailable refusal from the server is shown in words", { skip }, async () => {
  const { page } = await openDashboard(browser, { dates: fcDates, plan: () => ({ status: 422,
    body: { detail: { status: "forecast_unavailable", reason: "no official USNIC iceberg list within 120 days" } } }) });
  await ready(page);
  await page.click("#pr-submit");
  await page.waitForSelector("#pr-error:not([hidden])");
  assert.match(await text(page, "#pr-error"), /No forecast estimate for this date[\s\S]*no official USNIC iceberg list within 120 days/);
  assert.ok(!(await page.locator("#pr-result").isVisible()));
  await page.close();
});

test("typed coordinates out of range are explained and never sent", { skip }, async () => {
  const { page, calls } = await openDashboard(browser, { datesFor: datesForPoints() });
  await ready(page);
  const before = datesQueries(calls).length;
  await page.selectOption("#pr-origin", "__map__");
  await page.fill("#pr-origin-lat", "-95");
  await page.fill("#pr-origin-lon", "-60");
  await page.dispatchEvent("#pr-origin-lon", "change");
  assert.match(await text(page, "#pr-notes"), /Start: -95\.000°, -60\.000° is not a valid position/);
  assert.ok(await page.isDisabled("#pr-submit"));
  assert.ok(!datesQueries(calls).slice(before).some((q) => q.includes("origin_lat=-95")));
  await page.close();
});

test("the progress strip follows the page and the plan, and is not a second navigation", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await ready(page);
  const state = () => page.$$eval("#pr-flow li", (ls) => ls.map((l) => `${l.dataset.step}:${l.className}`));
  assert.deepEqual(await state(), ["plan:current", "result:", "sim:", "data:"]);
  assert.equal(await page.locator("#pr-flow button, #pr-flow a").count(), 0);
  await page.click("#pr-submit");
  await planned(page);
  assert.deepEqual(await state(), ["plan:done", "result:current", "sim:", "data:"]);
  await page.click('.primary-nav button[data-view="data"]');
  assert.deepEqual(await state(), ["plan:done", "result:done", "sim:", "data:current"]);
  await page.click('.primary-nav button[data-view="sim"]');
  await page.waitForSelector("#sim-body:not([hidden])");
  assert.deepEqual(await state(), ["plan:done", "result:done", "sim:current", "data:"]);
  await page.close();
});

test("the Research / Developer banner uses the agreed wording", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await ready(page);
  await page.click("#dev-area summary");
  await page.click('.dev-tabs button[data-tab="validation"]');
  assert.match(await text(page, "#dev-banner"), /Research \/ Developer view — not part of the main hackathon product\.[\s\S]*← Back to Plan Route/);
  await page.close();
});

/* ------------------------------------------------------------- optional real smoke */
const SMOKE = process.env.ANTROUTE_SMOKE_URL;
test("real smoke: Drake Passage to Bransfield Strait, 2023-11-14, reproduces the frozen plan", {
  skip: skip || (SMOKE ? false : "set ANTROUTE_SMOKE_URL to run against a live API with the real archive"), timeout: 240000,
}, async () => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const posts = [];
  page.on("request", (r) => { if (new URL(r.url()).pathname === "/real/plan") posts.push(r.postData()); });
  await page.goto(SMOKE);
  await ready(page);
  await page.selectOption("#pr-origin", "drake_passage");
  await page.selectOption("#pr-destination", "bransfield_strait");
  await ready(page);
  await page.fill("#pr-issue", "2023-11-14");
  await page.click("#pr-submit");
  await page.waitForSelector("#pr-result:not([hidden])", { timeout: 220000 });
  assert.equal(posts.length, 1);
  assert.match(await text(page, "#pr-verdict-badge"), /Recommended/);
  const metrics = await text(page, "#pr-metrics");
  for (const s of ["14 Nov 2023, 00:00 UTC", "15 Nov 2023, 13:30 UTC", "37.5 h", "838 km", "838"]) assert.ok(metrics.includes(s), s);
  assert.match(await text(page, '[data-risk="combined"]'), /1\.9%[\s\S]*0 of 200 scenarios/);
  assert.equal((await page.locator("#pr-options tbody tr").allInnerTexts()).length, 14);
  assert.match(await text(page, "#pr-disclosure"), /Hindsight forcing/);
  await page.close();
});

test("real smoke: Simulate Voyage sails the frozen 2023-11-14 plan with no replan", {
  skip: skip || (SMOKE ? false : "set ANTROUTE_SMOKE_URL to run against a live API with the real archive"), timeout: 480000,
}, async () => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const calls = { plan: 0, simulate: 0 };
  page.on("request", (r) => {
    const p = new URL(r.url()).pathname;
    if (r.method() === "POST" && p === "/real/plan") calls.plan += 1;
    if (r.method() === "POST" && p === "/real/simulate") calls.simulate += 1;
  });
  await page.goto(SMOKE);
  await ready(page);
  await page.selectOption("#pr-origin", "drake_passage");
  await page.selectOption("#pr-destination", "bransfield_strait");
  await ready(page);
  await page.fill("#pr-issue", "2023-11-14");
  await page.click("#pr-submit");
  await page.waitForSelector("#pr-result:not([hidden])", { timeout: 220000 });
  await page.click("#pr-simulate");
  await page.waitForSelector("#sim-body:not([hidden])", { timeout: 220000 });
  assert.deepEqual(calls, { plan: 1, simulate: 1 });
  assert.match(await text(page, "#sim-slider-out"), /2023-11-14 \(0 of 2\)/);
  assert.match(await text(page, "#sim-subtitle"), /No replan was required during this voyage\./);
  assert.match(await text(page, "#sim-banners"), /HISTORICAL MODE[\s\S]*hindsight[\s\S]*not certified navigation/i);
  await page.click('#sim-events [data-frame="2"]');
  assert.ok(await page.locator("#sim-summary").isVisible());
  assert.match(await text(page, "#sim-summary"), /No replan was required during this voyage\./);
  await page.close();
});

test("real smoke: a 2024-25 issue date plans and simulates through the normal flow", {
  skip: skip || (SMOKE ? false : "set ANTROUTE_SMOKE_URL to run against a live API with the real archive"), timeout: 480000,
}, async () => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await page.goto(SMOKE);
  await ready(page);
  await page.selectOption("#pr-origin", "drake_passage");
  await page.selectOption("#pr-destination", "bransfield_strait");
  await ready(page);
  await page.fill("#pr-issue", "2024-11-14");
  await page.dispatchEvent("#pr-issue", "change");
  assert.match(await text(page, "#pr-mode"), /REAL HISTORICAL DATA/);
  await page.click("#pr-submit");
  await page.waitForSelector("#pr-result:not([hidden])", { timeout: 220000 });
  assert.match(await text(page, "#pr-verdict-badge"), /Recommended/);
  assert.match(await text(page, "#pr-metrics"), /14 Nov 2024, 00:00 UTC/);
  await page.click("#pr-simulate");
  await page.waitForSelector("#sim-body:not([hidden])", { timeout: 220000 });
  assert.match(await text(page, "#sim-slider-out"), /2024-11-14 \(0 of 2\)/);
  assert.match(await text(page, "#sim-subtitle"), /No replan was required during this voyage\./);
  assert.match(await text(page, "#sim-banners"), /HISTORICAL MODE[\s\S]*hindsight[\s\S]*not certified navigation/i);
  await page.close();
});

test("real smoke: a future date (2026-11-19) plans and simulates as a labelled forecast estimate", {
  skip: skip || (SMOKE ? false : "set ANTROUTE_SMOKE_URL to run against a live API with the real archive"), timeout: 480000,
}, async () => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await page.goto(SMOKE);
  await ready(page);
  await page.selectOption("#pr-origin", "drake_passage");
  await page.selectOption("#pr-destination", "bransfield_strait");
  await ready(page);
  await page.fill("#pr-issue", "2026-11-19");
  await page.dispatchEvent("#pr-issue", "change");
  assert.match(await text(page, "#pr-mode"), /FORECAST \/ HACKATHON ESTIMATE/);
  await page.click("#pr-submit");
  await page.waitForSelector("#pr-result:not([hidden])", { timeout: 220000 });
  assert.match(await text(page, "#pr-verdict-badge"), /Recommended/);
  const metrics = await text(page, "#pr-metrics");
  for (const s of ["19 Nov 2026, 00:00 UTC", "20 Nov 2026, 13:09 UTC", "37.1 h", "838 km"]) assert.ok(metrics.includes(s), s);
  assert.match(await text(page, "#pr-disclosure"), /^Forecast estimate: future-year Antarctic/);
  assert.match(await text(page, "#pr-banners"), /FORECAST \/ HACKATHON ESTIMATE/);
  await page.click("#pr-simulate");
  await page.waitForSelector("#sim-body:not([hidden])", { timeout: 220000 });
  assert.match(await text(page, "#sim-slider-out"), /2026-11-19 \(0 of 2\)/);
  assert.match(await text(page, "#sim-banners"), /FORECAST \/ HACKATHON ESTIMATE/);
  assert.ok(!(await text(page, "#sim-banners")).includes("HISTORICAL MODE"));
  await page.close();
});

test("real smoke: 14 Aug 2027 is accepted as a historical seasonal analogue and says why it is not recommended", {
  skip: skip || (SMOKE ? false : "set ANTROUTE_SMOKE_URL to run against a live API with the real archive"), timeout: 480000,
}, async () => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await page.goto(SMOKE);
  await ready(page);
  await page.selectOption("#pr-origin", "drake_passage");
  await page.selectOption("#pr-destination", "bransfield_strait");
  await ready(page);
  assert.equal(await page.getAttribute("#pr-issue", "max"), null);
  assert.equal(await page.getAttribute("#pr-issue", "min"), null);
  await page.fill("#pr-issue", "2027-08-14");
  await page.dispatchEvent("#pr-issue", "change");
  assert.match(await text(page, "#pr-mode"), /FORECAST \/ HACKATHON ESTIMATE[\s\S]*seasonal analogue/i);
  await page.click("#pr-submit");
  await page.waitForSelector("#pr-result:not([hidden])", { timeout: 220000 });
  assert.match(await text(page, "#pr-verdict-badge"), /Not recommended/);
  assert.match(await text(page, "#pr-banners"), /FORECAST \/ HACKATHON ESTIMATE[\s\S]*Historical seasonal analogue/i);
  const how = await text(page, "#pr-howto");
  assert.match(how, /Analogue date \w+, 14 Aug 2024/);
  assert.match(how, /Wind\/current forcing historical reanalysis analogue/);
  assert.match(how, /Iceberg source official USNIC list of \w+, 8 Aug 2024/);
  assert.match(how, /Confidence VERY LOW/);
  await page.close();
});

test("real smoke: 3 Mar 2028 plans and simulates from a historical seasonal analogue", {
  skip: skip || (SMOKE ? false : "set ANTROUTE_SMOKE_URL to run against a live API with the real archive"), timeout: 480000,
}, async () => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await page.goto(SMOKE);
  await ready(page);
  await page.selectOption("#pr-origin", "drake_passage");
  await page.selectOption("#pr-destination", "bransfield_strait");
  await ready(page);
  await page.fill("#pr-issue", "2028-03-03");
  await page.dispatchEvent("#pr-issue", "change");
  await page.click("#pr-submit");
  await page.waitForSelector("#pr-result:not([hidden])", { timeout: 220000 });
  assert.match(await text(page, "#pr-verdict-badge"), /Recommended/);
  assert.match(await text(page, "#pr-metrics"), /3 Mar 2028, 00:00 UTC/);
  assert.match(await text(page, "#pr-howto"), /Analogue date \w+, 3 Mar 2024/);
  await page.click("#pr-simulate");
  await page.waitForSelector("#sim-body:not([hidden])", { timeout: 220000 });
  assert.match(await text(page, "#sim-slider-out"), /2028-03-03 \(0 of 2\)/);
  assert.match(await text(page, "#sim-banners"), /FORECAST \/ HACKATHON ESTIMATE/);
  assert.ok(!(await text(page, "#sim-banners")).includes("HISTORICAL MODE"));
  await page.close();
});
