/* Antarctic Ice-Risk Routing dashboard - no external dependencies. */
"use strict";

const $ = (sel) => document.querySelector(sel);
// API location: same origin unless <meta name="antroute-api-base"> names another (separately hosted dashboard).
const API_BASE = (document.querySelector('meta[name="antroute-api-base"]')?.content || "").replace(/\/$/, "");
const apiUrl = (path) => API_BASE + path;
const css = (name) => getComputedStyle(document.querySelector(".viz-root")).getPropertyValue(name).trim();
const SERIES = ["--series-1", "--series-2", "--series-3", "--series-4", "--series-5", "--series-6", "--series-7", "--series-8"];
const ICE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
const pct = (v, d = 1) => (v == null || !isFinite(v) ? "–" : (100 * v).toFixed(d) + "%");
const num = (v, d = 0) => (v == null || !isFinite(v) ? "–" : Number(v).toFixed(d));

async function api(path, opts = {}) {
  const res = await fetch(apiUrl(path), { headers: { "Content-Type": "application/json" }, ...opts });
  const text = await res.text();
  let body;
  try { body = JSON.parse(text); } catch { body = text; }
  if (!res.ok) throw new Error(typeof body === "object" && body.detail ? JSON.stringify(body.detail) : String(body));
  return body;
}

function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v; else if (k === "style") n.style.cssText = v; else n.setAttribute(k, v);
  }
  for (const c of children) n.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return n;
}

function hexToRgb(h) { const v = parseInt(h.slice(1), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; }
function rampColor(p) {
  const t = Math.min(1, Math.max(0, p / 100)) * (ICE_RAMP.length - 1);
  const i = Math.min(ICE_RAMP.length - 2, Math.floor(t));
  const a = hexToRgb(ICE_RAMP[i]), b = hexToRgb(ICE_RAMP[i + 1]), f = t - i;
  return a.map((x, k) => Math.round(x + (b[k] - x) * f));
}

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.width;
  const h = Math.round(w * (canvas.height / canvas.width));
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

/* ------------------------------------------------------- data status */
const STATUS_TEXT = { real: "Real", historical: "Historical", forecast: "Forecast", schematic: "Schematic",
  proxy: "Proxy", unavailable: "Unavailable" };
function statusBadge(status) {
  const s = STATUS_TEXT[status] ? status : "unavailable";
  return el("span", { class: "badge status-" + s }, STATUS_TEXT[s]);
}
// Data & Confidence names what each input is, not just where it came from.
const KIND = { observation: ["real", "Real observation"], reanalysis: ["historical", "Hindsight reanalysis"],
  proxy: ["proxy", "Proxy / analogue"], model: ["forecast", "Model output"], derived: ["derived", "Derived estimate"],
  historicalMode: ["historical", "Historical"], forecastMode: ["forecast", "Forecast estimate"],
  research: ["schematic", "Research use"] };
function kindBadge(kind) {
  const [cls, text] = KIND[kind] || KIND.derived;
  return el("span", { class: "badge status-" + cls }, text);
}
const currentTab = () => document.querySelector('.tabs button[aria-selected="true"]').dataset.tab;

function setModeBadge(tab) {
  const b = $("#mode-badge"), st = window.__status;
  let status = "schematic", text = "CONTROLLED-SYNTHETIC DATA";
  if (!st) { status = "unavailable"; text = "DATA STATUS UNKNOWN"; }
  else if (tab === "product") {
    const ok = st.historical && st.historical.status === "available";
    const fc = ok && typeof prodForecastShown === "function" && prodForecastShown();
    const an = fc && prodAnalogueShown();
    status = !ok ? "unavailable" : fc ? "forecast" : "historical";
    text = !ok ? "REAL HISTORICAL DATA UNAVAILABLE" : an ? "FORECAST / HACKATHON ESTIMATE · SEASONAL ANALOGUE"
      : fc ? "FORECAST / HACKATHON ESTIMATE · PROXY INPUTS" : "REAL HISTORICAL DATA · HINDSIGHT FORCING";
  } else if (tab === "real") {
    const ok = st.real_data.status === "available";
    status = ok ? "historical" : "unavailable";
    text = ok ? "REAL DATA · HISTORICAL REPLAY" : "REAL DATA UNAVAILABLE";
  } else if (isHist(tab)) {
    const ok = st.historical && st.historical.status === "available";
    status = ok ? "historical" : "unavailable";
    text = ok ? "REAL HISTORICAL DATA · HINDSIGHT FORCING" : "REAL HISTORICAL DATA UNAVAILABLE";
  } else if (tab === "validation") {
    const real = (window.__figsMode || st.figures.execution_mode) === "real";
    status = real ? "real" : "schematic";
    text = real ? "REAL-DATA FIGURES" : "CONTROLLED-SYNTHETIC FIGURES";
  }
  b.className = "badge status-" + status;
  b.textContent = text;
}

/* ------------------------------------------------------------- tabs */
const HIST_TABS = ["hroute", "hwindow", "hvoyage"];
const isHist = (tab) => HIST_TABS.includes(tab);
// The primary nav holds the three product pages (all data-tab="product", one data-view each); the research
// and synthetic views sit in the collapsed "Research / Developer" area and never in the primary nav.
function openTab(b) {
  const tab = b.dataset.tab;
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
  document.querySelectorAll(".tab").forEach((s) => { s.hidden = s.id !== "tab-" + tab; });
  $("#dev-banner").hidden = tab === "product";
  if (tab === "validation") loadFigures();
  if (tab === "real") redrawReal();
  if (tab !== "product") simPause();
  if (tab === "plan" && mapState.plan) drawMap(mapState.plan);
  if (tab === "window" && winState.result) drawWindow(winState.result, budget());
  if (isHist(tab)) redrawHist(tab);
  if (tab === "product") { if (b.dataset.view === "sim") openSimulation(); else setView(b.dataset.view); }
  setModeBadge(tab);
}
const primaryButton = (view) => $(`.primary-nav button[data-view="${view}"]`);
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => openTab(b)));
$("#dev-back").addEventListener("click", () => openTab(primaryButton("plan")));
document.querySelectorAll("[data-goto]").forEach((b) => b.addEventListener("click", () => openTab(primaryButton(b.dataset.goto))));
const budget = () => (window.__config ? window.__config.routing.risk_budget : 0.05);

/* ------------------------------------------------------------- map */
const mapState = { plan: null, toScreen: null, screenRoutes: [] };

function drawMap(plan, canvasSel = "#map", st = mapState, legendSel = "#map-legend") {
  const canvas = $(canvasSel);
  const { ctx, w, h } = fitCanvas(canvas);
  const m = plan.map;
  const c = Math.cos(m.rotation_rad), s = Math.sin(m.rotation_rad);
  const rot = (x, y) => [x * c - y * s, x * s + y * c];
  const X1 = m.x0_km + m.nx * m.res_km, Y1 = m.y0_km + m.ny * m.res_km;
  // plan.focus_xy_km (optional): zoom to these points plus a margin instead of the whole grid
  const corners = plan.focus_xy_km && plan.focus_xy_km.length ? focusBox(plan.focus_xy_km.map(([x, y]) => rot(x, y)))
    : [[m.x0_km, m.y0_km], [X1, m.y0_km], [m.x0_km, Y1], [X1, Y1]].map(([x, y]) => rot(x, y));
  const minx = Math.min(...corners.map((p) => p[0])), maxx = Math.max(...corners.map((p) => p[0]));
  const miny = Math.min(...corners.map((p) => p[1])), maxy = Math.max(...corners.map((p) => p[1]));
  const pad = 8;
  const scale = Math.min((w - 2 * pad) / (maxx - minx), (h - 2 * pad) / (maxy - miny));
  const e0 = pad - scale * minx + ((w - 2 * pad) - scale * (maxx - minx)) / 2;
  const f0 = pad + scale * maxy + ((h - 2 * pad) - scale * (maxy - miny)) / 2;
  const toScreen = (x, y) => { const [rx, ry] = rot(x, y); return [e0 + scale * rx, f0 - scale * ry]; };
  st.toScreen = toScreen;
  st.fromScreen = (px, py) => {           // screen pixel -> grid km (inverse of toScreen)
    const rx = (px - e0) / scale, ry = (f0 - py) / scale;
    return [rx * c + ry * s, -rx * s + ry * c];
  };

  ctx.fillStyle = css("--surface-1");
  ctx.fillRect(0, 0, w, h);

  // raster: one pixel per grid cell, row 0 of the grid is the bottom of the image
  const img = new ImageData(m.nx, m.ny);
  const land = hexToRgb(css("--land")), berg = hexToRgb(css("--berg")), surf = hexToRgb(css("--surface-1"));
  const missing = hexToRgb(css("--missing"));
  let anyMissing = false;
  for (let j = 0; j < m.ny; j++) {
    for (let i = 0; i < m.nx; i++) {
      const k = j * m.nx + i, o = ((m.ny - 1 - j) * m.nx + i) * 4;
      let rgb = surf, a = 255;
      if (m.land[k]) rgb = land;
      else if (m.p_ice[k] == null) { rgb = missing; anyMissing = true; }   // no data is never open water
      else if (m.p_ice[k] > 0) rgb = rampColor(m.p_ice[k]);
      if (!m.land[k] && m.p_berg && m.p_berg[k] >= 5) {
        const f = 0.35 + 0.5 * Math.min(1, m.p_berg[k] / 50);
        rgb = rgb.map((v, q) => Math.round(v * (1 - f) + berg[q] * f));
      }
      img.data.set([...rgb, a], o);
    }
  }
  const off = document.createElement("canvas");
  off.width = m.nx; off.height = m.ny;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  const r = m.res_km;
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr * scale * c * r, dpr * -scale * s * r, dpr * scale * s * r, dpr * scale * c * r,
    dpr * (e0 + scale * (c * m.x0_km - s * Y1)), dpr * (f0 - scale * (s * m.x0_km + c * Y1)));
  ctx.drawImage(off, 0, 0);
  ctx.restore();

  // routes: non-recommended first, recommended last and thicker
  st.screenRoutes = [];
  const order = plan.candidates.map((_, i) => i).sort((a, b) => (a === plan.recommended_index) - (b === plan.recommended_index));
  for (const i of order) {
    const cand = plan.candidates[i];
    const pts = cand.xy_km.map(([x, y]) => toScreen(x, y));
    const rec = i === plan.recommended_index;
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    if (rec) { ctx.strokeStyle = css("--surface-1"); ctx.lineWidth = 6; strokePath(ctx, pts); }
    ctx.strokeStyle = css(SERIES[i % SERIES.length]);
    ctx.lineWidth = rec ? 3.5 : 2;
    ctx.setLineDash(cand.feasible ? [] : [6, 4]);
    strokePath(ctx, pts);
    ctx.setLineDash([]);
    st.screenRoutes.push({ i, pts });
  }
  if (plan.origin_xy_km) marker(ctx, toScreen(...plan.origin_xy_km), css("--series-3"), "Origin");
  if (plan.destination_xy_km) marker(ctx, toScreen(...plan.destination_xy_km), css("--series-4"), "Destination");

  const legend = $(legendSel);
  legend.replaceChildren(
    el("span", {}, el("i", { class: "box", style: `background:${css("--land")}` }), plan.land_label || "Land (schematic)"),
    ...(m.p_berg ? [el("span", {}, el("i", { class: "box", style: `background:${css("--berg")}` }), "Iceberg presence")] : []),
    ...(anyMissing ? [el("span", {}, el("i", { class: "box", style: `background:${css("--missing")}` }), "No data")] : []),
    el("span", {}, el("i", { style: "border-top-style:dashed;border-color:" + css("--muted") }), "Exceeds budget"),
  );
}

function focusBox(pts) {
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const pad = Math.max(150, 0.25 * Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)));
  return [[Math.min(...xs) - pad, Math.min(...ys) - pad], [Math.max(...xs) + pad, Math.max(...ys) + pad]];
}

function strokePath(ctx, pts) {
  ctx.beginPath();
  pts.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
}

function marker(ctx, [x, y], colour, label) {
  ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI);
  ctx.fillStyle = colour; ctx.fill();
  ctx.lineWidth = 2; ctx.strokeStyle = css("--surface-1"); ctx.stroke();
  ctx.fillStyle = css("--text-primary"); ctx.font = "12px system-ui, sans-serif";
  ctx.fillText(label, x + 9, y - 8);
}

function distToSegment(px, py, [x1, y1], [x2, y2]) {
  const dx = x2 - x1, dy = y2 - y1, L = dx * dx + dy * dy;
  const t = L ? Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / L)) : 0;
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
}

function attachMapTip(canvasSel, tipSel, st) {
$(canvasSel).addEventListener("pointermove", (ev) => {
  const tip = $(tipSel);
  if (!st.plan) return;
  const rect = ev.target.getBoundingClientRect();
  const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
  let best = null;
  for (const r of st.screenRoutes) {
    for (let k = 1; k < r.pts.length; k++) {
      const d = distToSegment(px, py, r.pts[k - 1], r.pts[k]);
      if (d < 10 && (!best || d < best.d)) best = { d, i: r.i };
    }
  }
  if (!best) { tip.hidden = true; return; }
  const cand = st.plan.candidates[best.i];
  tip.replaceChildren(
    el("strong", {}, cand.labels.join(", ")),
    row("P(breach)", pct(cand.p_breach)), row("95% upper bound", pct(cand.p_breach_upper)),
    row("Expected time", num(cand.expected_hours, 1) + " h"), row("Fuel index", num(cand.expected_fuel)),
    row("Distance", num(cand.distance_km) + " km"),
  );
  tip.style.left = Math.min(px + 14, rect.width - 200) + "px";
  tip.style.top = (py + 40) + "px";
  tip.hidden = false;
});
$(canvasSel).addEventListener("pointerleave", () => { $(tipSel).hidden = true; });
}
attachMapTip("#map", "#map-tip", mapState);

function row(label, value) { return el("div", { class: "row" }, el("span", {}, label), el("b", {}, value)); }

function statusPill(ok) {
  return el("span", { class: "pill " + (ok ? "ok" : "bad") }, ok ? "✓ meets budget" : "✕ exceeds budget");
}

function renderPlan(plan) {
  mapState.plan = plan;
  drawMap(plan);
  $("#verdict").textContent = plan.explanation;
  const tbody = $("#cand-table tbody");
  tbody.replaceChildren(...plan.candidates.map((c, i) => {
    const tr = el("tr", { class: i === plan.recommended_index ? "rec" : "" },
      el("td", {}, el("span", { class: "swatch", style: `background:${css(SERIES[i % SERIES.length])}` })),
      el("td", {}, (i === plan.recommended_index ? "★ " : "") + c.labels.join(", ")),
      el("td", {}, statusPill(c.feasible)),
      el("td", { class: "num" }, num(c.distance_km)),
      el("td", { class: "num" }, num(c.expected_hours, 1)),
      el("td", { class: "num" }, num(c.expected_fuel)),
      el("td", { class: "num" }, pct(c.p_breach)),
      el("td", { class: "num" }, pct(c.p_breach_upper)));
    return tr;
  }));
}

$("#plan-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  const body = {
    departure: f.get("departure"), scenarios: Number(f.get("scenarios")),
    resolution_km: Number(f.get("resolution_km")), scenario_routes: 2,
    icebergs: f.get("berg") ? [{ id: "BERG-1", lat: Number(f.get("berg_lat")), lon: Number(f.get("berg_lon")) }] : [],
  };
  const status = $("#plan-status");
  status.textContent = "Planning across joint scenarios…";
  $("#plan-btn").disabled = true;
  try {
    const job = await api("/routes?wait=true", { method: "POST", body: JSON.stringify(body) });
    if (job.status !== "done") throw new Error(job.error || "planning failed");
    renderPlan(job.result);
    status.textContent = `${job.result.n_scenarios} joint scenarios · ${job.result.execution_mode}`;
  } catch (e) {
    status.textContent = "Error: " + e.message;
  } finally {
    $("#plan-btn").disabled = false;
  }
});

/* ------------------------------------------------------------- window */
const winState = { result: null, bars: [] };

function drawWindow(result, budget, canvasSel = "#window-chart", st = winState) {
  const canvas = $(canvasSel);
  const { ctx, w, h } = fitCanvas(canvas);
  const opts = result.options;
  const left = 52, right = 12, top = 16, bottom = 46;
  const peak = Math.max(budget, ...opts.map((o) => Math.min(1, o.p_breach_upper)));
  const step = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25].find((st) => (1.15 * peak) / st <= 5) || 0.25;
  const ymax = Math.min(1, Math.max(step * 2, Math.ceil((1.15 * peak) / step) * step));
  const y = (v) => top + (h - top - bottom) * (1 - v / ymax);
  ctx.fillStyle = css("--surface-1"); ctx.fillRect(0, 0, w, h);
  ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = css("--muted"); ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  for (let v = 0; v <= ymax + 1e-9; v += step) {
    const yy = y(v);
    ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(w - right, yy); ctx.stroke();
    ctx.fillText(pct(v, 0), 8, yy + 4);
  }
  const bw = (w - left - right) / opts.length;
  st.bars = [];
  opts.forEach((o, k) => {
    const x = left + k * bw + bw * 0.15, bwid = bw * 0.7, v = Math.min(1, o.p_breach_upper);
    ctx.fillStyle = o.feasible ? css("--good") : css("--critical");
    const yy = y(v), r = Math.min(4, bwid / 2);
    ctx.beginPath();
    ctx.moveTo(x, y(0)); ctx.lineTo(x, yy + r); ctx.quadraticCurveTo(x, yy, x + r, yy);
    ctx.lineTo(x + bwid - r, yy); ctx.quadraticCurveTo(x + bwid, yy, x + bwid, yy + r); ctx.lineTo(x + bwid, y(0));
    ctx.fill();
    ctx.fillStyle = css("--text-primary");
    ctx.fillText(o.feasible ? "✓" : "✕", x + bwid / 2 - 4, yy - 4);
    ctx.fillStyle = css("--muted");
    ctx.save(); ctx.translate(x + bwid / 2, h - bottom + 12); ctx.rotate(-0.6);
    ctx.fillText(o.departure.slice(5), -18, 8); ctx.restore();
    st.bars.push({ x, w: bwid, o });
  });
  ctx.strokeStyle = css("--text-primary"); ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(left, y(budget)); ctx.lineTo(w - right, y(budget)); ctx.stroke(); ctx.setLineDash([]);
  const label = `risk budget ${pct(budget, 0)}`;
  ctx.fillStyle = css("--surface-1");
  ctx.fillRect(left + 2, y(budget) - 17, ctx.measureText(label).width + 6, 14);
  ctx.fillStyle = css("--text-primary"); ctx.fillText(label, left + 5, y(budget) - 6);
  if (result.selected) {
    const b = st.bars.find((q) => q.o.departure === result.selected);
    if (b) {
      ctx.fillStyle = css("--text-primary");
      ctx.fillText("▼ selected", b.x + b.w / 2 - 26, Math.max(top + 10, y(Math.min(1, b.o.p_breach_upper)) - 18));
    }
  }
}

function attachWindowTip(canvasSel, tipSel, st) {
$(canvasSel).addEventListener("pointermove", (ev) => {
  const tip = $(tipSel);
  const rect = ev.target.getBoundingClientRect();
  const px = ev.clientX - rect.left;
  const b = st.bars.find((q) => px >= q.x - 4 && px <= q.x + q.w + 4);
  if (!b) { tip.hidden = true; return; }
  tip.replaceChildren(el("strong", {}, b.o.departure), row("95% upper bound", pct(b.o.p_breach_upper)),
    row("P(breach)", pct(b.o.p_breach)), row("Expected time", num(b.o.expected_hours, 1) + " h"),
    row("Fuel index", num(b.o.expected_fuel)), row("Status", b.o.feasible ? "✓ meets budget" : "✕ exceeds budget"));
  tip.style.left = Math.min(px + 14, rect.width - 200) + "px"; tip.style.top = "60px"; tip.hidden = false;
});
$(canvasSel).addEventListener("pointerleave", () => { $(tipSel).hidden = true; });
}
attachWindowTip("#window-chart", "#window-tip", winState);

$("#window-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  const status = $("#window-status");
  status.textContent = "Sweeping departure dates…";
  try {
    const job = await api("/departures?wait=true", { method: "POST", body: JSON.stringify({
      start: f.get("start"), end: f.get("end"), step_days: Number(f.get("step_days")),
      scenarios: Number(f.get("scenarios")), resolution_km: 25 }) });
    if (job.status !== "done") throw new Error(job.error || "sweep failed");
    winState.result = job.result;
    drawWindow(job.result, budget());
    $("#window-verdict").textContent = job.result.explanation + " Rule: " + job.result.rule + ".";
    $("#window-table tbody").replaceChildren(...job.result.options.map((o) => el("tr", {},
      el("td", {}, o.departure), el("td", {}, statusPill(o.feasible)), el("td", { class: "num" }, pct(o.p_breach_upper)),
      el("td", { class: "num" }, num(o.expected_hours, 1)), el("td", { class: "num" }, num(o.expected_fuel)))));
    status.textContent = "";
  } catch (e) { status.textContent = "Error: " + e.message; }
});

/* ------------------------------------------------------------- voyage */
const voyage = { id: null, latlon: [] };

function setRoute(route) {
  voyage.latlon = route.latlon;
  const slider = $("#wp-slider");
  slider.max = String(route.latlon.length - 2);
  slider.value = String(Math.min(Number(slider.value), route.latlon.length - 2));
  updateWp();
}
function updateWp() {
  const k = Number($("#wp-slider").value), p = voyage.latlon[k];
  $("#wp-out").textContent = p ? `#${k} (${p[0].toFixed(2)}, ${p[1].toFixed(2)})` : "–";
}
$("#wp-slider").addEventListener("input", updateWp);

async function refreshLog() {
  const hist = await api(`/voyages/${voyage.id}/history`);
  $("#voyage-log").replaceChildren(...hist.events.map((e) => el("li", {},
    el("div", {}, el("strong", {}, e.event === "planned" ? "Planned" : (e.action || "").toUpperCase()),
      e.alert ? " ⚠ alert" : "", e.triggers && e.triggers.length ? " · " + e.triggers.join(", ") : ""),
    el("div", {}, e.explanation || ""),
    el("div", { class: "meta" }, (e.logged_at || e.at || "") + (e.issued ? " · forecast " + e.issued : "")))));
}

$("#voyage-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  const status = $("#voyage-status");
  status.textContent = "Creating voyage…";
  try {
    const v = await api("/voyages", { method: "POST", body: JSON.stringify({
      departure: f.get("departure"), scenarios: 120, resolution_km: 25, scenario_routes: 0 }) });
    voyage.id = v.voyage_id;
    setRoute(v.route);
    $("#replan-btn").disabled = false;
    for (const [id, fmt] of [["#export-geojson", "geojson"], ["#export-csv", "csv"]]) {
      const a = $(id); a.href = apiUrl(`/voyages/${voyage.id}/export?format=${fmt}`); a.hidden = false;
    }
    status.textContent = "Voyage " + voyage.id + " · " + v.explanation;
    await refreshLog();
  } catch (e) { status.textContent = "Error: " + e.message; }
});

$("#replan-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  const p = voyage.latlon[Number($("#wp-slider").value)];
  const status = $("#voyage-status");
  status.textContent = "Replanning…";
  try {
    const d = await api(`/voyages/${voyage.id}/replan`, { method: "POST", body: JSON.stringify({
      lat: p[0], lon: p[1], issued: f.get("issued"), scenarios: 120, data_age_hours: Number(f.get("age")) }) });
    setRoute(d.route);
    $("#wp-slider").value = "0"; updateWp();
    status.textContent = d.explanation;
    await refreshLog();
  } catch (e) { status.textContent = "Error: " + e.message; }
});

/* ------------------------------------------------------------- validation */
async function loadFigures() {
  const box = $("#figs");
  if (box.dataset.loaded) return;
  try {
    const figs = await api("/figures");
    window.__figsMode = figs.execution_mode;
    const real = figs.execution_mode === "real";
    $("#figs-badge").replaceWith(Object.assign(statusBadge(real ? "real" : "schematic"), { id: "figs-badge" }));
    $("#figs-note").textContent = real ? "Figures from real-data runs." :
      "These figures come from controlled-synthetic runs, not real conditions.";
    setModeBadge(currentTab());
    box.replaceChildren(...figs.figures.map((fig) => el("figure", {},
      el("img", { src: apiUrl(fig.url), alt: fig.caption, loading: "lazy" }), el("figcaption", {}, fig.caption))));
    if (!figs.figures.length) box.replaceChildren(el("p", { class: "sub" }, "No figures yet - run the CLI reports."));
    box.dataset.loaded = "1";
  } catch (e) { box.replaceChildren(el("p", {}, "Could not load figures: " + e.message)); }
}

/* ------------------------------------------------------------- real data */
// Everything here is read from the backend's verified real-data bundle; the page computes nothing.
const realState = { plan: null, toScreen: null, screenRoutes: [], layers: {}, route: null, dates: null, k: 0,
  tracks: null };
const realWinState = { result: null, bars: [] };
attachMapTip("#real-map", "#real-map-tip", realState);
attachWindowTip("#real-window-chart", "#real-window-tip", realWinState);
const short = (h) => (h ? String(h).slice(0, 12) + "…" : "–");

function showRealUnavailable(reason) {
  $("#real-unavailable").hidden = false;
  $("#real-content").hidden = true;
  $("#real-reason").textContent = reason;
}

function kv(rows) {
  return rows.map(([k, v]) => el("tr", {}, el("th", {}, k), el("td", {}, v)));
}

async function showRealLayer(k) {
  const info = realState.dates.layers[k];
  const observed = k === 0 && info.source === "observed";
  if (!realState.layers[k]) realState.layers[k] = await api(observed ? "/real/sea-ice" : `/real/forecast-map?layer=${k}`);
  const L = realState.layers[k], g = L.grid, route = realState.route, xy = route.xy_km || [];
  const ev = route.evaluation || {};
  realState.k = k;
  realState.plan = {
    map: { nx: g.nx, ny: g.ny, res_km: g.res_km, x0_km: g.x0_km, y0_km: g.y0_km, rotation_rad: g.rotation_rad,
      land: L.land, p_ice: observed ? L.concentration_pct : L.p_ice_ge_limit_pct, p_berg: observed ? null : L.p_berg_pct },
    candidates: xy.length ? [{ ...ev, xy_km: xy, labels: [`selected route, departing ${route.departure}`], feasible: true }] : [],
    recommended_index: xy.length ? 0 : null,
    origin_xy_km: xy[0], destination_xy_km: xy[xy.length - 1],
    land_label: "Land (sea-ice product mask; not a navigational coastline)",
  };
  if (!$("#tab-real").hidden) { drawMap(realState.plan, "#real-map", realState, "#real-map-legend"); drawTracks(); }
  $("#real-layer-out").textContent = `${info.date} · ${observed ? "observed (Historical)" : info.source + " (Forecast)"}`;
  $("#real-ramp-label").textContent = observed ? "100% observed concentration"
    : `100% P(ice ≥ ${pct(L.tau, 0)}) across ${L.n_members} members`;
}

// Ensemble-mean iceberg drift (forecast) up to the shown day; a hollow ring marks the position on that day.
function drawTracks() {
  const tracks = realState.tracks;
  if (!tracks || !tracks.length || !realState.toScreen) return;
  const ctx = $("#real-map").getContext("2d"), col = css("--berg");
  ctx.save();
  ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
  for (const t of tracks) {
    const pts = t.daily_mean.filter((d) => d.layer <= realState.k && d.x_km != null)
      .map((d) => realState.toScreen(d.x_km, d.y_km));
    if (!pts.length) continue;
    strokePath(ctx, pts);
    const [x, y] = pts[pts.length - 1];
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.stroke();
    ctx.font = "10px system-ui, sans-serif"; ctx.fillText(t.id, x + 6, y + 3);
    ctx.setLineDash([4, 3]);
  }
  ctx.restore();
  const legend = $("#real-map-legend");
  if (!legend.querySelector(".track-key")) {
    legend.append(el("span", { class: "track-key" }, el("i", { style: `border-top-style:dashed;border-color:${col}` }),
      "Iceberg mean drift (forecast)"));
  }
}

function redrawReal() {
  if (realState.plan) { drawMap(realState.plan, "#real-map", realState, "#real-map-legend"); drawTracks(); }
  if (realWinState.result) drawWindow(realWinState.result, realWinState.result.risk_budget, "#real-window-chart", realWinState);
}

async function loadReal(status) {
  const real = status.real_data;
  if (real.status !== "available") { showRealUnavailable(real.reason || "The real-data bundle is unavailable."); return; }
  const [route, win, bergs, dates, prov] = await Promise.all([api("/real/route"), api("/real/departure-window"),
    api("/real/icebergs"), api("/real/forecast-dates"), api("/provenance")]);
  $("#real-unavailable").hidden = true;
  $("#real-content").hidden = false;
  $("#real-bundle").textContent = `Bundle ${real.bundle_id} · forecast issued ${real.issue} · execution mode ` +
    `"${real.execution_mode}". Historical replay with real inputs; not a live forecast.`;
  $("#real-sources").replaceChildren(...real.sources.map((s) => el("li", {}, statusBadge(s.status),
    el("span", { class: "src-name" }, s.name.replace(/_/g, " ")), el("span", {}, s.label))));
  $("#real-limits").replaceChildren(...(real.limitations || []).map((t) => el("li", {}, t)));

  realState.route = route; realState.dates = dates; realState.tracks = bergs.tracks;
  const ev = route.evaluation || {}, pres = route.iceberg_presence_on_route || {};
  $("#real-verdict").textContent = route.explanation || "";
  $("#real-summary tbody").replaceChildren(...kv(route.selected ? [
    ["Selected departure", route.selected],
    ["Expected time", `${num(ev.expected_hours, 1)} h (p10–p90 ${num(ev.hours_p10, 2)}–${num(ev.hours_p90, 2)})`],
    ["Expected fuel index", num(ev.expected_fuel, 1)],
    ["Distance", `${num(ev.distance_km, 1)} km`],
    ["Breaches", `${ev.breaches} of ${ev.n_scenarios} joint scenarios`],
    ["P(breach) 95% upper bound", `${pct(ev.p_breach_upper, 2)} (budget ${pct(win.risk_budget, 0)})`],
    ["Route cells with iceberg presence", `${pres.route_cells_with_any_presence ?? "–"} of ${(route.cells_row_col || []).length}`],
  ] : [["Selected departure", "none met the risk budget"]]));

  realWinState.result = win;
  $("#real-window-verdict").textContent = `${win.explanation} Rule: ${win.rule}.`;

  const src = bergs.source || {};
  $("#real-berg-badge").replaceWith(Object.assign(statusBadge(bergs.status), { id: "real-berg-badge" }));
  $("#real-berg-source").textContent = `${src.source || "–"}, list of ${(src.update_dates || []).join(", ")} ` +
    `(${src.filename || "–"}). Drift: beta ${bergs.drift?.beta}, spread factor ${bergs.drift?.spread_factor}, ` +
    `radius ${num((bergs.drift?.radius_m || 0) / 1000)} km.`;
  $("#real-bergs tbody").replaceChildren(
    ...bergs.drifted.map((b) => {
      const tr = (bergs.tracks || []).find((t) => t.id === b.id), end = tr && tr.daily_mean[tr.daily_mean.length - 1];
      const drift = end && end.lat != null ? ` → day ${end.layer}: ${num(end.lat, 2)}, ${num(end.lon, 2)} ` +
        `(±${num(end.spread_p90_km)} km p90)` : "";
      return el("tr", {}, el("td", {}, b.id), el("td", { class: "num" }, num(b.lat, 2)),
        el("td", { class: "num" }, num(b.lon, 2)), el("td", {}, "drifted" + drift));
    }),
    el("tr", {}, el("td", { colspan: "4" }, `${bergs.outside_grid.length} more outside the routing grid (not drifted)`)));

  const man = prov.manifest || {}, env = man.environment || {}, b = prov.bundle || {};
  $("#real-prov tbody").replaceChildren(...kv([
    ["Bundle", `${b.bundle_id} (created ${b.created_utc})`],
    ["Plan output (canonical sha256)", short(b.plan_window_canonical_sha256)],
    ["Forecast model", `${man.sea_ice?.forecast_model?.id || "–"} · ${short(man.sea_ice?.forecast_model?.sha256)}`],
    ...Object.entries(man.inputs || {}).map(([k, v]) => [`Input: ${k}`, `${short(v.sha256)} ${v.verified ? "✓ verified" : ""}`]),
    ["Code", `${short(env.git_commit)}${env.worktree_dirty ? " + uncommitted changes" : ""}`],
    ["Python / torch", `${env.python || "–"} / ${(env.packages || {}).torch || "–"}`],
  ]));
  $("#real-prov-link").href = apiUrl("/provenance");

  const slider = $("#real-layer");
  slider.max = String(dates.layers.length - 1);
  slider.value = "0";
  slider.addEventListener("input", () => showRealLayer(Number(slider.value)).catch((e) => {
    $("#real-layer-out").textContent = "Error: " + e.message; }));
  await showRealLayer(0);
  redrawReal();
}

/* ------------------------------------------------------------- real historical data */
// The backend runs the frozen real-data planner on past issue dates; the page only draws what comes back.
const hist = { status: null, dates: null };
const hRouteState = { plan: null, toScreen: null, screenRoutes: [], tracks: null, k: 0 };
const hWinState = { result: null, bars: [] };
const hWinMapState = { plan: null, toScreen: null, screenRoutes: [], tracks: null, k: 0 };
const hVoyState = { plan: null, toScreen: null, screenRoutes: [], tracks: null, k: 0 };
const hReplayState = { plan: null, toScreen: null, screenRoutes: [], bergs: null };
const hvoy = { id: null, latlon: [], lastIssue: null };
attachMapTip("#hroute-map", "#hroute-map-tip", hRouteState);
attachMapTip("#hwindow-map", "#hwindow-map-tip", hWinMapState);
attachMapTip("#hvoyage-map", "#hvoyage-map-tip", hVoyState);
attachWindowTip("#hwindow-chart", "#hwindow-tip", hWinState);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function errText(e) {
  try { const d = JSON.parse(e.message); return d.reason || e.message; } catch { return e.message; }
}

// Background job + polling, so long real-data runs never hold a request open.
async function runJob(path, body, statusEl, msg) {
  const first = await api(path, { method: "POST", body: JSON.stringify(body) });
  let job = first;
  const t0 = Date.now();
  while (job.status === "queued" || job.status === "running") {
    statusEl.textContent = `${msg} (${job.status}, ${Math.round((Date.now() - t0) / 1000)} s)`;
    await sleep(1500);
    job = await api(`/jobs/${first.job_id}`);
  }
  if (job.status !== "done") throw new Error(job.error || "the computation failed");
  return job.result;
}

function setupHistorical(st) {
  hist.status = st;
  for (const tab of HIST_TABS) {
    const sec = $("#tab-" + tab), slot = sec.querySelector(".hist-slot"), body = sec.querySelector(".hist-body");
    if (!st || st.status !== "available") {
      const card = $("#hist-unavailable-template").content.cloneNode(true);
      card.querySelector('[data-field="reason"]').textContent =
        st ? `${st.status}: ${st.reason || "no reason given"}` : "The data status could not be read.";
      slot.replaceChildren(card);
      body.hidden = true;
      continue;
    }
    const note = $("#hist-note-template").content.cloneNode(true);
    note.querySelector('[data-field="disclosure"]').textContent = st.hindsight_forcing;
    slot.replaceChildren(note);
    body.hidden = false;
  }
}

function seasonOf(kind, day) {
  return ((hist.dates && hist.dates.seasons[kind]) || []).find((s) => s.first <= day && day <= s.last) || null;
}

function seasonText(s) {
  if (!s) return { text: "", warn: false };
  if (s.out_of_sample) {
    return { text: `${s.season} season: out-of-sample for the sea-ice U-Net and the iceberg drift calibration.` +
      (s.independent_evaluation ? ` Independent evaluation season: ${s.evaluation_notes.join("; ")}.` : ""),
      warn: false };
  }
  return { text: `${s.season} season is in-sample: ${s.in_sample_notes.join("; ")}. Results here are not ` +
    "independent evidence of skill.", warn: true };
}

function showSeasonFlag(sel, s) {
  const p = $(sel);
  if (!p) return;
  const t = seasonText(s);
  p.textContent = t.text;
  p.classList.toggle("in-sample", t.warn);
}

function flagForForm(form, kind) {
  const day = form.querySelector('input[type="date"]').value;
  const s = seasonOf(kind, day);
  const target = form.dataset.flag;
  if (!target) return;
  if (!s) {
    const ranges = hist.dates.seasons[kind].map((x) => `${x.first} to ${x.last}`).join(", ");
    $(target).textContent = day ? `No Real Historical Data for ${day}. Issue dates run ${ranges} ` +
      "(some single days inside these ranges are missing upstream)." : "";
    $(target).classList.add("in-sample");
  } else showSeasonFlag(target, s);
}

async function loadHistoricalDates() {
  const d = await api("/real/historical/dates");
  hist.dates = d;
  $("#hroute-form").dataset.flag = "#hroute-season";
  $("#hwindow-form").dataset.flag = "#hwindow-season";
  document.querySelectorAll("[data-season-pick]").forEach((sel) => {
    const kind = sel.dataset.seasonPick, seasons = d.seasons[kind], form = sel.form;
    const input = form.querySelector('input[type="date"]');
    const dates = kind === "route" ? d.route_dates : d.window_dates;
    sel.replaceChildren(...seasons.map((s) => el("option", { value: s.first },
      `${s.season} · ${s.out_of_sample ? "out-of-sample" : "in-sample"}${s.independent_evaluation ?
        " · independent evaluation" : ""} · ${s.n_dates} dates`)));
    input.min = dates[0];
    input.max = dates[dates.length - 1];
    const pick = seasons.find((s) => s.season === "2023-24") || seasons[seasons.length - 1];
    sel.value = pick.first;
    input.value = pick.first;
    sel.addEventListener("change", () => { input.value = sel.value; flagForForm(form, kind); });
    input.addEventListener("change", () => flagForForm(form, kind));
    flagForForm(form, kind);
  });
}

// Ensemble-mean iceberg drift up to layer k (forecast); a ring marks the position on that layer.
function drawTrackLines(canvasSel, st, tracks, k, legendSel) {
  if (!tracks || !tracks.length || !st.toScreen) return;
  const ctx = $(canvasSel).getContext("2d"), col = css("--berg");
  ctx.save();
  ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.5;
  for (const t of tracks) {
    const pts = t.daily_mean.filter((d) => d.layer <= k && d.x_km != null).map((d) => st.toScreen(d.x_km, d.y_km));
    if (!pts.length) continue;
    ctx.setLineDash([4, 3]); strokePath(ctx, pts); ctx.setLineDash([]);
    const [x, y] = pts[pts.length - 1];
    ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.stroke();
    ctx.font = "10px system-ui, sans-serif"; ctx.fillText(t.id, x + 6, y + 3);
  }
  ctx.restore();
  const legend = $(legendSel);
  legend.append(el("span", {}, el("i", { style: `border-top-style:dashed;border-color:${col}` }),
    "Iceberg mean drift (forecast)"));
}

function drawHistMap(canvasSel, st, legendSel) {
  if (!st.plan) return;
  drawMap(st.plan, canvasSel, st, legendSel);
  drawTrackLines(canvasSel, st, st.tracks, st.k, legendSel);
}

function candRows(plan) {
  return plan.candidates.map((c, i) => el("tr", { class: i === plan.recommended_index ? "rec" : "" },
    el("td", {}, el("span", { class: "swatch", style: `background:${css(SERIES[i % SERIES.length])}` })),
    el("td", {}, (i === plan.recommended_index ? "★ " : "") + c.labels.join(", ")),
    el("td", {}, statusPill(c.feasible)),
    el("td", { class: "num" }, num(c.distance_km)), el("td", { class: "num" }, num(c.expected_hours, 1)),
    el("td", { class: "num" }, num(c.expected_fuel)), el("td", { class: "num" }, pct(c.p_breach)),
    el("td", { class: "num" }, pct(c.p_breach_upper))));
}

function inputRows(h) {
  const files = (f) => (f.files || []).map((x) => x.file).join(", ");
  const b = h.icebergs, p = h.parameters, s = h.season;
  return kv([
    ["Forecast issued", `${h.issue} (sea ice observed up to this day; scenario days ${h.scenario_days.join(" to ")})`],
    ["Season", s ? `${s.season} · ${s.out_of_sample ? "out-of-sample" : "in-sample: " + s.in_sample_notes.join("; ")}` : "–"],
    ["Sea ice", `${h.sea_ice.source} · ${h.sea_ice.file} (${short(h.sea_ice.sha256)})`],
    ["Forecast model", `${h.forecast_model.id} (${short(h.forecast_model.sha256)}), ${h.forecast_model.lead_days ?? "–"}-day leads`],
    ["Winds (hindsight)", `${h.forcing.winds.product}: ${files(h.forcing.winds)}`],
    ["Currents (hindsight)", `${h.forcing.currents.product}: ${files(h.forcing.currents)}`],
    ["Icebergs", b ? `USNIC list of ${b.list_date} (${b.age_days} d old): ${b.drifted.length} drifted` +
      `${b.drifted.length ? " (" + b.drifted.join(", ") + ")" : ""}; ${b.outside_grid.length} outside the grid` : "–"],
    ["Iceberg drift", `beta ${p.drift_beta}, alpha scale ${p.drift_alpha_scale}, spread factor ` +
      `${p.drift_spread_factor}, radius ${p.berg_radius_km} km`],
    ["Scenarios", `${p.members} joint scenarios, seed ${p.seed}`],
    ["Probability calibration", h.probability_calibration.applied_in_route_risk ? "applied" : "not applied to route risk"],
    ["Vessel", `${h.vessel.name}, ice class ${h.vessel.ice_class}, limit ${pct(h.vessel.max_ice_concentration, 0)} (${h.vessel.note})`],
  ]);
}

$("#hroute-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const issue = new FormData(ev.target).get("issue");
  const status = $("#hroute-status");
  $("#hroute-btn").disabled = true;
  try {
    const res = await runJob("/real/historical/routes", { issue }, status,
      "Running the real-data planner (the first run also loads the model)");
    hRouteState.plan = res;
    hRouteState.tracks = res.iceberg_tracks;
    hRouteState.k = res.map.layer_day;
    drawHistMap("#hroute-map", hRouteState, "#hroute-map-legend");
    $("#hroute-verdict").textContent = res.explanation;
    $("#hroute-table tbody").replaceChildren(...candRows(res));
    $("#hroute-inputs tbody").replaceChildren(...inputRows(res.historical));
    $("#hroute-ramp").textContent = `100% P(ice ≥ limit), day ${res.map.layer_day} after issue`;
    showSeasonFlag("#hroute-season", res.historical.season);
    status.textContent = `${res.n_scenarios} joint scenarios · ${res.data_label}`;
  } catch (e) {
    status.textContent = "Error: " + errText(e);
  } finally {
    $("#hroute-btn").disabled = false;
  }
});

$("#hwindow-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  const status = $("#hwindow-status");
  $("#hwindow-btn").disabled = true;
  try {
    const res = await runJob("/real/historical/departures",
      { issue: f.get("issue"), window_days: Number(f.get("window_days")) }, status,
      "Planning every departure date from one real forecast");
    hWinState.result = res;
    drawWindow(res, res.risk_budget, "#hwindow-chart", hWinState);
    $("#hwindow-verdict").textContent = `${res.explanation} Rule: ${res.rule}.`;
    $("#hwindow-table tbody").replaceChildren(...res.options.map((o) => el("tr", {},
      el("td", {}, o.departure), el("td", { class: "num" }, o.lead_days ?? "–"), el("td", {}, o.support || "–"),
      el("td", {}, statusPill(o.feasible)), el("td", { class: "num" }, pct(o.p_breach_upper)),
      el("td", { class: "num" }, num(o.expected_hours, 1)), el("td", { class: "num" }, num(o.expected_fuel)))));
    const sel = res.selected_route;
    hWinMapState.plan = { map: res.map, candidates: sel ? [sel] : [], recommended_index: sel ? 0 : null,
      origin_xy_km: res.origin_xy_km, destination_xy_km: res.destination_xy_km, land_label: res.land_label };
    hWinMapState.tracks = res.iceberg_tracks;
    hWinMapState.k = res.map.layer_day;
    drawHistMap("#hwindow-map", hWinMapState, "#hwindow-map-legend");
    $("#hwindow-map-title").textContent = res.selected ?
      `Selected departure ${res.selected} (map: day ${res.map.layer_day} after issue)` : "No departure meets the budget";
    $("#hwindow-inputs tbody").replaceChildren(...inputRows(res.historical));
    showSeasonFlag("#hwindow-season", res.historical.season);
    status.textContent = `${res.n_scenarios} joint scenarios · ${res.data_label}`;
  } catch (e) {
    status.textContent = "Error: " + errText(e);
  } finally {
    $("#hwindow-btn").disabled = false;
  }
});

/* voyage */
function setHRoute(route) {
  hvoy.latlon = route.latlon;
  const slider = $("#hwp-slider");
  slider.max = String(route.latlon.length - 2);
  slider.value = String(Math.min(Number(slider.value), route.latlon.length - 2));
  updateHWp();
}
function updateHWp() {
  const k = Number($("#hwp-slider").value), p = hvoy.latlon[k];
  $("#hwp-out").textContent = p ? `#${k} (${p[0].toFixed(2)}, ${p[1].toFixed(2)})` : "–";
}
$("#hwp-slider").addEventListener("input", updateHWp);

function showVoyageMap(res, route) {
  hVoyState.plan = { map: res.map, candidates: [route], recommended_index: 0, origin_xy_km: route.xy_km[0],
    destination_xy_km: route.xy_km[route.xy_km.length - 1], land_label: res.land_label };
  hVoyState.tracks = res.iceberg_tracks;
  hVoyState.k = res.map.layer_day;
  if (!$("#tab-hvoyage").hidden) drawHistMap("#hvoyage-map", hVoyState, "#hvoyage-map-legend");
}

function renderLog(sel, events) {
  $(sel).replaceChildren(...events.map((e) => el("li", {},
    el("div", {}, el("strong", {}, e.event === "planned" ? "Planned" : (e.action || e.event || "").toUpperCase()),
      e.alert ? " ⚠ alert" : "", e.triggers && e.triggers.length ? " · " + e.triggers.join(", ") : ""),
    el("div", {}, e.explanation || ""),
    el("div", { class: "meta" }, [e.logged_at || e.at || e.day || "", e.issued ? "forecast " + e.issued : "",
      e.usnic_list ? "USNIC list " + e.usnic_list : ""].filter(Boolean).join(" · ")))));
}

$("#hvoyage-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const issue = new FormData(ev.target).get("issue");
  const status = $("#hvoyage-status");
  status.textContent = "Planning the voyage with the real-data planner…";
  try {
    const v = await api("/real/historical/voyages", { method: "POST", body: JSON.stringify({ issue }) });
    hvoy.id = v.voyage_id;
    hvoy.lastIssue = issue;
    setHRoute(v.route);
    showVoyageMap(v, v.route);
    const input = $("#hreplan-form").querySelector('input[name="issued"]');
    input.min = issue;
    input.max = hist.dates.route_dates[hist.dates.route_dates.length - 1];
    const next = new Date(issue + "T00:00:00Z");
    next.setUTCDate(next.getUTCDate() + 1);
    input.value = next.toISOString().slice(0, 10);
    $("#hreplan-btn").disabled = false;
    for (const [id, fmt] of [["#hexport-geojson", "geojson"], ["#hexport-csv", "csv"]]) {
      const a = $(id); a.href = apiUrl(`/voyages/${hvoy.id}/export?format=${fmt}`); a.hidden = false;
    }
    const s = seasonText(v.historical.season);
    status.textContent = `Voyage ${hvoy.id} · ${v.explanation}`;
    $("#hvoyage-age").textContent = `Icebergs: USNIC list of ${v.icebergs.list_date} (${v.icebergs.age_days} d old). ${s.text}`;
    renderLog("#hvoyage-log", (await api(`/voyages/${hvoy.id}/history`)).events);
  } catch (e) { status.textContent = "Error: " + errText(e); }
});

$("#hreplan-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const issued = new FormData(ev.target).get("issued");
  const p = hvoy.latlon[Number($("#hwp-slider").value)];
  const status = $("#hvoyage-status");
  status.textContent = "Replanning with the forecast issued " + issued + "…";
  $("#hreplan-btn").disabled = true;
  try {
    const d = await api(`/real/historical/voyages/${hvoy.id}/replan`, { method: "POST",
      body: JSON.stringify({ lat: p[0], lon: p[1], issued }) });
    hvoy.lastIssue = issued;
    $("#hreplan-form").querySelector('input[name="issued"]').min = issued;
    setHRoute(d.route);
    $("#hwp-slider").value = "0"; updateHWp();
    showVoyageMap(d, d.route);
    status.textContent = d.explanation;
    $("#hvoyage-age").textContent = `${d.data_age_basis} USNIC list of ${d.usnic_list} (${d.usnic_age_days} d old).`;
    renderLog("#hvoyage-log", (await api(`/voyages/${hvoy.id}/history`)).events);
  } catch (e) { status.textContent = "Error: " + errText(e); }
  finally { $("#hreplan-btn").disabled = false; }
});

/* replay */
function drawReplay() {
  const st = hReplayState;
  if (!st.plan) return;
  drawMap(st.plan, "#hreplay-map", st, "#hreplay-map-legend");
  const ctx = $("#hreplay-map").getContext("2d"), col = css("--berg");
  ctx.save(); ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.5;
  for (const b of st.bergs || []) {
    const [x, y] = st.toScreen(b.xy_km[0], b.xy_km[1]);
    ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.stroke();
    ctx.font = "10px system-ui, sans-serif"; ctx.fillText(b.id, x + 6, y + 3);
  }
  ctx.restore();
  $("#hreplay-map-legend").replaceChildren(
    el("span", {}, el("i", { class: "box", style: `background:${css("--land")}` }), st.plan.land_label),
    el("span", {}, el("i", { style: `border-color:${css(SERIES[0])}` }), "Sailed (planner)"),
    el("span", {}, el("i", { style: `border-color:${css(SERIES[1])}` }), "Naive (shortest, left on day 1)"),
    el("span", {}, el("i", { class: "box", style: `background:${css("--berg")}` }), "USNIC-reported berg"));
}

$("#hreplay-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  const status = $("#hreplay-status");
  $("#hreplay-btn").disabled = true;
  try {
    const res = await runJob("/real/historical/replay",
      { start: f.get("start"), max_wait_days: Number(f.get("max_wait_days")) }, status,
      "Replaying day by day (each day in port plans a 14-day window)");
    $("#hreplay-result").hidden = false;
    const cands = [];
    if (res.sailed_xy_km) cands.push({ xy_km: res.sailed_xy_km, labels: ["sailed (planner)"], feasible: true });
    if (res.naive_xy_km) cands.push({ xy_km: res.naive_xy_km, labels: ["naive"], feasible: true });
    hReplayState.plan = { map: res.map, candidates: cands, recommended_index: cands.length ? 0 : null,
      origin_xy_km: res.origin_xy_km, destination_xy_km: res.destination_xy_km, land_label: res.land_label };
    hReplayState.bergs = res.observed_bergs;
    $("#hreplay-map-title").textContent = `Observed sea ice on ${res.map.observed_date}`;
    drawReplay();
    $("#hreplay-verdict").textContent = res.departure ?
      `Departed ${res.departure} after waiting ${res.days_waited} day(s); ` +
      (res.arrived ? `arrived ${res.arrival_day}.` : "did not arrive within the replay.") :
      "No departure met the risk budget within the wait limit.";
    const t = res.truth;
    $("#hreplay-table tbody").replaceChildren(...(t ? [
      ["Left on", t.planner.departure, t.naive.departure],
      ["Hours (observed ice)", num(t.planner.hours, 1), num(t.naive.hours, 1)],
      ["Fuel index", num(t.planner.fuel_index), num(t.naive.fuel_index)],
      ["Distance (km)", num(t.planner.distance_km), num(t.naive.distance_km)],
      ["Cells in observed ice ≥ limit", t.planner.observed_breach_cells, t.naive.observed_breach_cells],
      ["Hours in observed ice ≥ limit", num(t.planner.hazard_hours, 1), num(t.naive.hazard_hours, 1)],
      ["Cells in a reported berg's footprint", t.planner.berg_footprint_cells, t.naive.berg_footprint_cells],
      ["Nearest reported berg (km)", `${num(t.planner.berg_min_distance_km)} (${t.planner.berg_nearest || "–"})`,
        `${num(t.naive.berg_min_distance_km)} (${t.naive.berg_nearest || "–"})`],
    ] : []).map(([k, a, b]) => el("tr", {}, el("td", {}, k), el("td", { class: "num" }, a), el("td", { class: "num" }, b))));
    renderLog("#hreplay-log", res.days.map((d) => ({ ...d, event: d.phase })));
    const s = seasonText(res.historical.season);
    status.textContent = `${res.data_label} · icebergs included · ${s.text}`;
  } catch (e) {
    status.textContent = "Error: " + errText(e);
  } finally {
    $("#hreplay-btn").disabled = false;
  }
});

function redrawHist(tab) {
  if (tab === "hroute") drawHistMap("#hroute-map", hRouteState, "#hroute-map-legend");
  if (tab === "hwindow") {
    if (hWinState.result) drawWindow(hWinState.result, hWinState.result.risk_budget, "#hwindow-chart", hWinState);
    drawHistMap("#hwindow-map", hWinMapState, "#hwindow-map-legend");
  }
  if (tab === "hvoyage") { drawHistMap("#hvoyage-map", hVoyState, "#hvoyage-map-legend"); drawReplay(); }
}

/* ------------------------------------------------------------- product: Plan Route */
// One judge-facing flow: origin, destination and date from the API's own lists, one POST /real/plan, one result.
// The page draws what /real/plan returns and nothing else: no risk, route or forecast is computed here, and no
// synthetic data is ever shown in place of a missing or failed result.
const prod = { locations: null, dates: null, datesKey: null, result: null, day: 0, view: "plan", busy: false,
  picking: null, picked: { origin: null, destination: null }, datesError: null, dateChosen: false,
  controller: null, timer: null, lastBody: null };
const prMapState = { plan: null, toScreen: null, screenRoutes: [], tracks: null, k: 0 };
const prWinState = { result: null, bars: [] };
attachMapTip("#pr-map", "#pr-map-tip", prMapState);
attachWindowTip("#pr-window-chart", "#pr-window-tip", prWinState);
const PLAN_TIMEOUT_MS = 180000;
const PLAN_STATUSES = ["recommended", "no_feasible_departure", "no_route"];

const fmtHours = (h) => {
  if (h == null || !isFinite(h)) return "–";
  const d = Math.floor(h / 24), r = h - 24 * d;
  return d ? `${num(h, 1)} h (${d} d ${num(r, 1)} h)` : `${num(h, 1)} h`;
};
const fmtUtc = (iso) => {
  if (!iso) return "–";
  const t = new Date(iso.length === 17 ? iso.replace("Z", ":00Z") : iso);   // "2023-11-15T13:30Z"
  if (isNaN(t)) return iso;
  return t.toLocaleString("en-GB", { timeZone: "UTC", day: "numeric", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit" }) + " UTC";
};
const fmtDay = (d) => {
  const t = new Date(d + "T00:00:00Z");
  return isNaN(t) ? d : t.toLocaleDateString("en-GB", { timeZone: "UTC", weekday: "short", day: "numeric", month: "short" });
};
const ll = (p) => (p && p.lat != null ? `${num(p.lat, 3)}°, ${num(p.lon, 3)}°` : "–");
const pctOf = (count, n) => (n ? `${count} of ${n} scenarios` : "–");

/* ---- API errors, in words a person can act on */
class PlanError extends Error {
  constructor(title, text, hint = "") { super(text); this.title = title; this.hint = hint; }
}

function describeFailure(status, body) {
  const d = body && typeof body === "object" ? body.detail : null;
  const reason = d && typeof d === "object" && !Array.isArray(d) ? d.reason : typeof d === "string" ? d : "";
  if (status === 422 && d && d.status === "invalid_location") {
    return new PlanError("This start or destination cannot be routed", reason,
      "Pick another location or point. Points must be inside the routing area and within reach of open water.");
  }
  if (status === 422 && d && d.status === "out_of_coverage") {
    return new PlanError("The archive does not cover this date for this route", reason,
      "Pick another date from the supported range shown under the form. Longer routes need more covered days.");
  }
  if (status === 422 && d && d.status === "forecast_unavailable") {
    return new PlanError("No forecast estimate for this date", reason,
      "Forecast estimates exist only for the in-season future dates listed under the form. Nothing synthetic is shown.");
  }
  if (status === 422 && Array.isArray(d)) {
    return new PlanError("The request was not accepted", d.map((e) => `${(e.loc || []).slice(1).join(".")}: ${e.msg}`)
      .join("; "), "Check the form fields and try again.");
  }
  if (status === 503) {
    return new PlanError("The real historical model is unavailable right now",
      reason || (typeof body === "string" ? body : "The server could not run the model."),
      "Nothing synthetic is shown in its place. If the server is busy, try again shortly.");
  }
  return new PlanError(`The server answered with an error (HTTP ${status})`,
    reason || (typeof body === "string" ? body.slice(0, 300) : "No reason was given."), "Try again.");
}

async function planFetch(path, opts = {}) {
  let res;
  try {
    res = await fetch(apiUrl(path), { headers: { "Content-Type": "application/json" }, ...opts });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new PlanError("Could not reach the planning server", "The request did not reach the API (network error).",
      "Check that the API is running and reachable, then try again.");
  }
  const text = await res.text();
  let body = null;
  try { body = JSON.parse(text); } catch { body = text; }
  if (!res.ok) throw describeFailure(res.status, body);
  if (!body || typeof body !== "object") {
    throw new PlanError("The server's answer could not be read", "The response was not JSON.",
      "No result is shown. Try again; if it persists, the API may be misconfigured.");
  }
  return body;
}

// Only a complete, labelled real-historical result is drawn; anything else is reported, not patched.
function checkPlan(p) {
  const bad = (why) => new PlanError("The server's answer could not be read", `Malformed /real/plan response: ${why}.`,
    "No result is shown and nothing is substituted. Try again.");
  if (!p || typeof p !== "object") throw bad("not an object");
  if (!PLAN_STATUSES.includes(p.status)) throw bad(`unknown status "${p.status}"`);
  const m = p.metadata;
  if (!labelledMode(m)) throw bad("not labelled as real historical data or as a forecast estimate");
  if (!p.locations || !p.locations.origin || !p.locations.destination) throw bad("locations missing");
  if (!p.departure || !Array.isArray(p.departure.options)) throw bad("departure options missing");
  if (p.status !== "no_route") {
    const r = p.route, k = p.risk;
    if (!r || !Array.isArray(r.xy_km) || r.xy_km.length < 2) throw bad("route geometry missing");
    if (!k || !k.combined || !k.sea_ice || !k.iceberg) throw bad("risk breakdown missing");
    if (!Array.isArray(p.daily)) throw bad("daily timeline missing");
  }
  return p;
}

// Two labelled modes: real historical data, or a forecast estimate for dates after the archive (proxy inputs).
function labelledMode(m) {
  if (!m) return false;
  if (m.mode === "historical") return m.execution_mode === "real";
  return m.mode === "forecast" && m.execution_mode === "modelled" && Boolean(m.forecast && m.forecast.forecast_mode);
}

function srcLabel(src) {
  return src === "proxy_analogue_observed" ? "proxy: analogue-season observation" :
    src === "analogue_observed" ? "historical analogue: observed in earlier years" : (src || "–");
}

// A forecast-mode result that came from a historical seasonal analogue (not the proxy forecast pathway).
function isAnalogue(m) { return Boolean(m && m.mode === "forecast" && m.forecast && m.forecast.pathway === "seasonal_analogue"); }

function obsWord(src) {
  return src === "proxy_analogue_observed" ? "proxy sea ice (analogue-season observation)" : "observed sea ice";
}

/* ---- form */
function presetById(id) { return (prod.locations?.presets || []).find((p) => p.id === id) || null; }

const MAP_VALUE = "__map__";            // the "point on the map" choice in the origin/destination lists
const ROLE_LABEL = { origin: "Start", destination: "Destination" };

// What the form sends for one end: {preset} or a {lat, lon} point; null while a point is incomplete.
function endOf(role) {
  const v = $(`#pr-${role}`).value;
  if (v !== MAP_VALUE) return v ? { preset: v } : null;
  const lat = parseFloat($(`#pr-${role}-lat`).value), lon = parseFloat($(`#pr-${role}-lon`).value);
  return Number.isFinite(lat) && Number.isFinite(lon) ? { lat, lon } : null;
}
// A typed point outside the valid latitude/longitude range is explained here and never sent.
function coordProblem(role) {
  if ($(`#pr-${role}`).value !== MAP_VALUE) return null;
  const e = endOf(role);
  if (!e) return null;
  if (e.lat < -90 || e.lat > 90 || e.lon < -180 || e.lon > 180) {
    return `${ROLE_LABEL[role]}: ${num(e.lat, 3)}°, ${num(e.lon, 3)}° is not a valid position. Latitude must be ` +
      "between -90 and 90 and longitude between -180 and 180.";
  }
  return null;
}
const endName = (e) => (!e ? "–" : e.preset ? (presetById(e.preset)?.name || e.preset) : `${num(e.lat, 3)}°, ${num(e.lon, 3)}°`);

// The server's resolution of an end (from the dates response for the chosen ends), else the preset's.
function resolvedEnd(role) {
  const r = prod.dates && prod.dates.route && prod.dates.route[role];
  if (r) return r;
  const v = $(`#pr-${role}`).value;
  return v === MAP_VALUE ? null : presetById(v);
}

function snapNote(role, loc) {
  if (!loc) return null;
  const who = ROLE_LABEL[role], point = $(`#pr-${role}`).value === MAP_VALUE;
  if (loc.available === false) return el("p", { class: "pr-note bad" }, `${who}: ${loc.reason}`);
  const r = loc.resolved || {};
  if (loc.snapped) {
    return el("p", { class: "pr-note", "data-snap": role }, el("b", {}, `${who} moved ${num(loc.distance_km, 1)} km to open water`),
      `: requested ${ll(loc.requested)}, routing from ${ll(r)}. ${loc.reason || ""}`);
  }
  if (!point) return null;
  return el("p", { class: "pr-note muted", "data-snap": role }, el("b", {}, `${who} is in open water`),
    `: requested ${ll(loc.requested)}, routing from the cell centre ${ll(r)} (${num(loc.distance_km, 1)} km away; not moved).`);
}

function forecastInfo() {
  const f = prod.dates && prod.dates.forecast;
  return f && f.available && Array.isArray(f.dates) && f.dates.length ? f : null;
}

// "historical" when the real archive supports the date, "forecast" for a date the forecast pathway lists,
// "analogue" for any other date after the archive (a historical seasonal analogue estimate), else null.
function dateMode(day) {
  if (!prod.dates || !day) return null;
  if ((prod.dates.window_dates || []).includes(day)) return "historical";
  const f = forecastInfo();
  if (f && f.dates.includes(day)) return "forecast";
  const e = prod.dates.estimate;
  return e && e.analogue && e.analogue.available && /^\d{4}-\d{2}-\d{2}$/.test(day) && day > e.any_date_after ?
    "analogue" : null;
}

// The product shows a forecast estimate: the plan on screen is one, or (before planning) the chosen date is one.
function prodForecastShown() {
  try {
    if (prod.result) return prod.result.metadata.mode === "forecast";
    return ["forecast", "analogue"].includes(dateMode($("#pr-issue").value));
  } catch { return false; }      // called by the header badge before the product module has initialised
}

function prodAnalogueShown() {
  try {
    if (prod.result) return isAnalogue(prod.result.metadata);
    return dateMode($("#pr-issue").value) === "analogue";
  } catch { return false; }
}

function modeNote() {
  const mode = dateMode($("#pr-issue").value);
  if (!mode) return null;
  if (mode === "historical") {
    return el("p", { class: "pr-note pr-mode historical", id: "pr-mode" }, el("span", { class: "pr-mode-badge" }, "REAL HISTORICAL DATA"),
      " Past season, real archive with hindsight forcing.");
  }
  return el("p", { class: "pr-note pr-mode forecast", id: "pr-mode" }, el("span", { class: "pr-mode-badge" }, "FORECAST / HACKATHON ESTIMATE"),
    mode === "forecast" ? " No observations exist for this date: proxy sea ice and forcing from an analogue season, latest official icebergs." :
      " Historical seasonal analogue: no forecast exists for this date, so real sea ice, winds and currents of the same " +
      "calendar period in earlier years stand in. Not a prediction; the confidence is shown with the result.");
}

function dateNote() {
  const d = prod.dates;
  if (!d) return null;
  const dates = d.window_dates || [], f = forecastInfo();
  const e = d.estimate, anyLater = e && e.analogue && e.analogue.available;
  if (!dates.length && !f && !anyLater) return el("p", { class: "pr-note bad" }, "No supported dates for this route.");
  return el("p", { class: "pr-note muted" }, (dates.length ? `Real historical dates for this route: ${dates.length} days ` +
    `between ${dates[0]} and ${dates[dates.length - 1]} (Nov–Feb seasons; some single days are missing upstream). ` : "") +
    (f ? `Forecast estimates: ${f.first} to ${f.last}. ` : "") +
    (anyLater ? `Any other date after ${e.any_date_after}: historical seasonal analogue estimate. ` : "") +
    `Route horizon ${d.horizon_days} days, ${d.window_days_max}-day departure window.`);
}

// Why a date cannot be planned, from the server's lists only (the forecast range is forecast_mode's, not ours).
function unsupportedDate(day) {
  const d = prod.dates, dates = d.window_dates || [], f = forecastInfo(), fr = d.forecast || {}, e = d.estimate || {};
  const hist = dates.length ? ` Real historical dates: ${dates[0]} to ${dates[dates.length - 1]}.` : "";
  const fcRange = f ? ` Forecast estimates: ${f.first} to ${f.last}.` : "";
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) return "Enter the date as YYYY-MM-DD.";
  if (!e.any_date_after || day <= e.any_date_after) {
    return `No Real Historical Data for ${day} on this route (past seasons run Nov–Feb, and some single days are ` +
      `missing upstream).${hist} Any date after ${e.any_date_after || "the archive"} is accepted as an estimate.`;
  }
  const why = (e.analogue && e.analogue.reason) || fr.reason;
  return `No estimate is available for ${day} on this route${why ? `: ${why}` : "."}${fcRange}${hist}`;
}

function formProblem() {
  const f = $("#pr-form"), o = f.origin.value, dst = f.destination.value, day = f.issue.value;
  if (!prod.locations) return "Locations are still loading.";
  for (const role of ["origin", "destination"]) {
    if ($(`#pr-${role}`).value === MAP_VALUE && !endOf(role)) {
      return `${ROLE_LABEL[role]}: press 📍 Pick on map and click the sea, or type a latitude and longitude.`;
    }
  }
  for (const role of ["origin", "destination"]) if (coordProblem(role)) return coordProblem(role);
  if (!o || !dst) return "Choose a start and a destination.";
  if (o === dst && o !== MAP_VALUE) return "Start and destination are the same place. Choose two different locations.";
  if (presetById(o)?.available === false || presetById(dst)?.available === false) return "This location cannot be routed.";
  if (prod.datesError) return prod.datesError;
  if (!prod.dates) return "Supported dates are still loading.";
  if (!day) return "Choose an issue date.";
  if (!dateMode(day)) return unsupportedDate(day);
  return null;
}

function refreshForm() {
  const f = $("#pr-form"), problem = formProblem();
  const notes = [snapNote("origin", resolvedEnd("origin")), snapNote("destination", resolvedEnd("destination")),
    problem && prod.locations ? el("p", { class: "pr-note bad", id: "pr-form-problem" }, problem) : null, modeNote(),
    dateNote()];
  $("#pr-notes").replaceChildren(...notes.filter(Boolean));
  $("#pr-submit").disabled = Boolean(problem) || prod.busy;
  if (currentTab() === "product") setModeBadge("product");
  drawPickMap();
  updateFlow();
}

// The default demonstration date: the 2026-11-19 forecast estimate when forecast_mode supports it on this
// route, else the frozen 2023-11-14 replay, else the first supported date. A UI default only.
const DEMO_FORECAST_DATE = "2026-11-19", DEMO_HISTORICAL_DATE = "2023-11-14", DEMO_ANALOGUE_DATE = "2027-08-14";
function defaultDate(dates, seasons, fc) {
  if (fc && fc.dates.includes(DEMO_FORECAST_DATE)) return DEMO_FORECAST_DATE;
  if (dates.includes(DEMO_HISTORICAL_DATE)) return DEMO_HISTORICAL_DATE;
  const pick = seasons.find((s) => s.season === "2023-24") || seasons[seasons.length - 1];
  return pick ? pick.first : dates[0] || (fc ? fc.first : null);
}

// Two one-click examples under the date, each shown only when the server lists that date for this route.
function renderExamples() {
  const box = $("#pr-examples"), opts = [[DEMO_FORECAST_DATE, "19 Nov 2026 (forecast)"],
    [DEMO_ANALOGUE_DATE, "14 Aug 2027 (seasonal analogue)"],
    [DEMO_HISTORICAL_DATE, "14 Nov 2023 (historical)"]].filter(([d]) => dateMode(d));
  box.hidden = !opts.length;
  box.replaceChildren(...(opts.length ? ["Try: ", ...opts.flatMap(([d, label], k) => [k ? " · " : "",
    el("button", { type: "button", class: "pr-link pr-example", "data-date": d }, label)])] : []));
  box.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    $("#pr-issue").value = b.dataset.date; prod.dateChosen = true; refreshForm();
  }));
}

async function loadProductDates() {
  const o = endOf("origin"), dst = endOf("destination");
  prod.datesError = null;
  if (!o || !dst || (o.preset && o.preset === dst.preset) || coordProblem("origin") || coordProblem("destination")) {
    prod.dates = null; prod.datesKey = null; refreshForm(); return;
  }
  const key = JSON.stringify([o, dst]);
  if (prod.datesKey === key && prod.dates) { refreshForm(); return; }
  prod.datesKey = key;
  prod.dates = null;
  $("#pr-issue").disabled = true;
  refreshForm();
  const q = (role, e) => (e.preset ? `${role}=${encodeURIComponent(e.preset)}` : `${role}_lat=${e.lat}&${role}_lon=${e.lon}`);
  try {
    const d = await planFetch(`/real/historical/dates?${q("origin", o)}&${q("destination", dst)}`);
    if (prod.datesKey !== key) return;          // a newer selection won
    prod.dates = d;
    const input = $("#pr-issue"), dates = d.window_dates || [];
    const seasons = (d.seasons && d.seasons.window) || [];
    const fc = forecastInfo();
    // No min/max: any calendar date can be typed or picked; the server decides how it is planned (historical,
    // forecast pathway or historical seasonal analogue) and an unsupported date is explained, never clamped.
    input.removeAttribute("min");
    input.removeAttribute("max");
    // A date the person typed or picked stays as it is (an unsupported one is explained, not replaced);
    // until then the page shows the default demonstration date.
    if (!prod.dateChosen || !input.value) {
      const def = defaultDate(dates, seasons, fc);
      if (def && (!input.value || !dateMode(input.value))) input.value = def;
    }
    renderExamples();
    input.disabled = false;
  } catch (e) {
    if (prod.datesKey !== key) return;
    prod.datesKey = null;
    // A point the server cannot route (outside the grid, too far from open water) is explained, not guessed at.
    prod.datesError = e.title && e.title.startsWith("This start or destination") ?
      `This start or destination cannot be routed: ${e.message}` : e.title === "The request was not accepted" ?
      "The start or destination coordinates were not accepted. Latitude must be between -90 and 90 and longitude " +
      "between -180 and 180." : `Could not load the supported dates: ${e.message}`;
    refreshForm();
    return;
  }
  refreshForm();
}

function fillLocationSelects(locs) {
  const usable = locs.presets;
  const option = (p) => el("option", p.available === false ? { value: p.id, disabled: "" } : { value: p.id },
    p.name + (p.snapped ? ` (snaps ${num(p.distance_km, 1)} km)` : "") + (p.available === false ? " (unavailable)" : ""));
  const byRegion = (sel) => {
    const groups = new Map();
    for (const p of usable) {
      if (!groups.has(p.region)) groups.set(p.region, el("optgroup", { label: p.region || "Other" }));
      groups.get(p.region).append(option(p));
    }
    sel.replaceChildren(...groups.values());
  };
  const o = $("#pr-origin"), dst = $("#pr-destination");
  byRegion(o); byRegion(dst);
  const ok = usable.filter((p) => p.available !== false);
  // The API's own order: the configured origin comes first and the configured destination second.
  for (const sel of [o, dst]) sel.append(el("option", { value: MAP_VALUE }, "📍 Point on the map…"));
  if (ok[0]) o.value = ok[0].id;
  if (ok[1]) dst.value = ok[1].id;
  o.disabled = dst.disabled = $("#pr-swap").disabled = $("#pr-pick-origin").disabled = $("#pr-pick-destination").disabled = false;
}

/* ---- Pick on map: the click becomes a WGS84 point; the server snaps it (never onto land) */
const pickState = { plan: null, toScreen: null, fromScreen: null, screenRoutes: [] };

function showCoords(role) {
  const point = $(`#pr-${role}`).value === MAP_VALUE;
  $(`#pr-${role}-coords`).hidden = !point;
  if (!point && prod.picking === role) prod.picking = null;
}

function armPick(role) {
  $(`#pr-${role}`).value = MAP_VALUE;
  showCoords(role);
  prod.picking = role;
  if (prod.view !== "plan") openTab(primaryButton("plan")); else setView("plan");
  $("#pr-pick").scrollIntoView({ block: "nearest", behavior: "smooth" });
  refreshForm();
}

// Grid km -> WGS84, interpolated between the cell centres' positions the server sent (display only).
function kmToLatLon(g, x, y) {
  const fc = (x - g.x_km[0]) / g.res_km, fr = (y - g.y_km[0]) / g.res_km;
  if (fc < -0.5 || fc > g.nx - 0.5 || fr < -0.5 || fr > g.ny - 0.5) return null;
  const c0 = Math.max(0, Math.min(g.nx - 2, Math.floor(fc))), r0 = Math.max(0, Math.min(g.ny - 2, Math.floor(fr)));
  const tc = fc - c0, tr = fr - r0, at = (a, r, c) => a[r * g.nx + c];
  const bil = (a) => (1 - tr) * ((1 - tc) * at(a, r0, c0) + tc * at(a, r0, c0 + 1)) +
    tr * ((1 - tc) * at(a, r0 + 1, c0) + tc * at(a, r0 + 1, c0 + 1));
  return { lat: Math.round(bil(g.lat) * 1e4) / 1e4, lon: Math.round(bil(g.lon) * 1e4) / 1e4 };
}

function endXY(role) {
  const g = prod.locations && prod.locations.map, r = resolvedEnd(role);
  if (!g || !r || !r.resolved || r.resolved.row == null) return null;
  return [g.x_km[r.resolved.col], g.y_km[r.resolved.row]];
}

function drawPickMap() {
  const g = prod.locations && prod.locations.map, card = $("#pr-pick");
  if (!g || card.closest("[hidden]") || $("#tab-product").hidden) return;
  const plan = {
    map: { nx: g.nx, ny: g.ny, res_km: g.res_km, x0_km: g.x0_km, y0_km: g.y0_km, rotation_rad: g.rotation_rad,
      land: g.land, p_ice: g.land.map(() => 0), p_berg: null },
    candidates: [], recommended_index: null,
    origin_xy_km: endXY("origin"), destination_xy_km: endXY("destination"),
  };
  drawMap(plan, "#pr-pick-map", pickState, "#pr-pick-legend");
  pickState.plan = plan;
  const ctx = $("#pr-pick-map").getContext("2d");
  ctx.save();
  ctx.font = "11px system-ui, sans-serif";
  const chosen = [$("#pr-origin").value, $("#pr-destination").value];
  for (const p of prod.locations.presets) {           // named locations, for orientation
    if (!p.resolved || p.resolved.row == null || chosen.includes(p.id)) continue;
    const [x, y] = pickState.toScreen(g.x_km[p.resolved.col], g.y_km[p.resolved.row]);
    ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.fillStyle = css("--muted"); ctx.fill();
    ctx.fillStyle = css("--text-secondary"); ctx.fillText(p.name.replace(/ \(.*\)$/, ""), x + 5, y + 12);
  }
  for (const role of ["origin", "destination"]) {     // a picked point, and the dashed move to its routing cell
    const pk = prod.picked[role], cell = endXY(role);
    if (!pk || $(`#pr-${role}`).value !== MAP_VALUE) continue;
    const [px, py] = pickState.toScreen(...pk.xy);
    if (cell) {
      const [cx, cy] = pickState.toScreen(...cell);
      ctx.setLineDash([4, 3]); ctx.strokeStyle = css("--text-secondary"); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(cx, cy); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.beginPath(); ctx.arc(px, py, 6, 0, 2 * Math.PI); ctx.lineWidth = 2;
    ctx.strokeStyle = css(role === "origin" ? "--series-3" : "--series-4"); ctx.stroke();
  }
  ctx.restore();
  $("#pr-pick-legend").replaceChildren(
    el("span", {}, el("i", { class: "box", style: `background:${css("--land")}` }), "Land (routing land mask)"),
    el("span", {}, el("i", { class: "box", style: `background:${css("--muted")}` }), "Named locations"),
    el("span", {}, el("i", { style: `border-top-style:dashed;border-color:${css("--text-secondary")}` }), "Moved to open water"));
  const who = prod.picking ? ROLE_LABEL[prod.picking].toLowerCase() : null;
  $("#pr-pick-title").textContent = who ? `Click the sea to set the ${who}` : "Routing area";
  card.classList.toggle("picking", Boolean(who));
}

$("#pr-pick-map").addEventListener("click", (ev) => {
  const g = prod.locations && prod.locations.map, role = prod.picking;
  if (!g || !pickState.fromScreen) return;
  const note = $("#pr-pick-note");
  if (!role) {
    note.hidden = false;
    note.textContent = "Press 📍 Pick on map next to the start or the destination first.";
    return;
  }
  const rect = ev.target.getBoundingClientRect();
  const xy = pickState.fromScreen(ev.clientX - rect.left, ev.clientY - rect.top);
  const p = kmToLatLon(g, ...xy);
  if (!p) {
    note.hidden = false;
    note.className = "pr-note bad";
    note.textContent = "That point is outside the routing area. Click inside the map.";
    return;
  }
  note.hidden = true; note.className = "pr-note";
  prod.picked[role] = { ...p, xy };
  $(`#pr-${role}-lat`).value = String(p.lat);
  $(`#pr-${role}-lon`).value = String(p.lon);
  prod.picking = null;
  loadProductDates();
});

for (const role of ["origin", "destination"]) {
  $(`#pr-pick-${role}`).addEventListener("click", () => armPick(role));
  for (const k of ["lat", "lon"]) {
    $(`#pr-${role}-${k}`).addEventListener("change", () => { prod.picked[role] = null; loadProductDates(); });
  }
}

function showProductUnavailable(reason) {
  $("#pr-unavailable").hidden = false;
  $("#pr-unavailable-reason").textContent = reason;
  $("#pr-landing").hidden = true;
  for (const id of ["#pr-origin", "#pr-destination", "#pr-issue", "#pr-submit", "#pr-swap",
    "#pr-pick-origin", "#pr-pick-destination"]) $(id).disabled = true;
  $("#pr-notes").replaceChildren();
}

async function setupProduct(status) {
  const hs = status && status.historical;
  if (!status) { showProductUnavailable("The API could not be reached, so no route can be planned."); return; }
  if (!hs || hs.status !== "available") {
    showProductUnavailable(hs ? `${hs.status}: ${hs.reason || "no reason given"}` : "The server reports no real historical archive.");
    return;
  }
  try {
    const locs = await planFetch("/real/locations");
    if (!Array.isArray(locs.presets) || locs.presets.length < 2) throw new Error("fewer than two locations returned");
    prod.locations = locs;
    fillLocationSelects(locs);
  } catch (e) {
    showProductUnavailable(`Could not load the locations: ${e.message}`);
    return;
  }
  await loadProductDates();
}

for (const role of ["origin", "destination"]) {
  $(`#pr-${role}`).addEventListener("change", () => {
    showCoords(role);
    if ($(`#pr-${role}`).value === MAP_VALUE && !endOf(role)) { armPick(role); return; }
    loadProductDates();
  });
}
$("#pr-swap").addEventListener("click", () => {
  const o = $("#pr-origin"), dst = $("#pr-destination"), a = o.value;
  o.value = dst.value; dst.value = a;
  for (const k of ["lat", "lon"]) {
    const x = $(`#pr-origin-${k}`), y = $(`#pr-destination-${k}`), t = x.value;
    x.value = y.value; y.value = t;
  }
  [prod.picked.origin, prod.picked.destination] = [prod.picked.destination, prod.picked.origin];
  showCoords("origin"); showCoords("destination");
  loadProductDates();
});
$("#pr-issue").addEventListener("change", () => { prod.dateChosen = true; refreshForm(); });
$("#pr-issue").addEventListener("input", () => { prod.dateChosen = true; refreshForm(); });

/* ---- views: Plan / Result / Data & Confidence */
/* ---- the three product pages: Plan Route (form, map, result), Voyage Simulation, Data & Confidence */
function setView(view) {
  if (view === "result") view = "plan";                    // the result lives on the Plan Route page
  prod.view = view;
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected",
    String(b.dataset.tab === "product" && b.dataset.view === view)));
  $("#tab-product").dataset.view = view;
  const has = Boolean(prod.result), plan = view === "plan", unavailable = !$("#pr-unavailable").hidden;
  const idle = !has && !prod.busy && $("#pr-error").hidden;
  $("#pr-landing").hidden = !plan || unavailable || !(idle || prod.picking);
  $("#pr-landing").classList.toggle("single", has || prod.busy);
  $("#pr-landing .pr-landing").hidden = has || prod.busy;
  $("#pr-result").hidden = !plan || !has;
  $("#pr-data").hidden = view !== "data";
  $("#data-empty").hidden = has;
  $("#pr-data-result").hidden = !has;
  const sailable = has && Boolean(prod.result.route);
  $("#pr-sim").hidden = view !== "sim" || !sailable;
  $("#sim-empty").hidden = view !== "sim" || sailable;
  if (view !== "sim") simPause();
  if (plan && has) drawProduct();
  if (plan) drawPickMap();
  if (view === "sim" && sailable) drawSimMap();
  if (view === "data" && !has) renderDataGeneral(null);
  updateFlow();
}

// The subordinate progress strip: which step the person is on, and which they have reached. Not a nav.
function updateFlow() {
  const has = Boolean(prod.result);
  const cur = prod.view === "plan" ? (has ? "result" : "plan") : prod.view;
  let simulated = false;
  try { simulated = Boolean(sim.data); } catch { /* the simulation module is not initialised yet */ }
  const reached = { plan: has, result: has && prod.view !== "plan", sim: simulated, data: false };
  document.querySelectorAll("#pr-flow li").forEach((li) => {
    const step = li.dataset.step;
    li.className = step === cur ? "current" : reached[step] ? "done" : "";
    if (step === cur) li.setAttribute("aria-current", "step"); else li.removeAttribute("aria-current");
  });
}

/* ---- Plan Route: exactly one POST /real/plan */
function setBusy(on, body) {
  prod.busy = on;
  $("#pr-loading").hidden = !on;
  for (const id of ["#pr-origin", "#pr-destination", "#pr-issue", "#pr-swap", "#pr-pick-origin",
    "#pr-pick-destination", "#pr-origin-lat", "#pr-origin-lon", "#pr-destination-lat", "#pr-destination-lon"]) $(id).disabled = on;
  $("#pr-submit").replaceChildren(...(on ? ["Planning…"] : [el("b", { class: "pr-step-no" }, "4"), " Plan Route"]));
  clearInterval(prod.timer);
  if (on) {
    const t0 = Date.now(), fc = dateMode(body.issue) === "forecast";
    $("#pr-loading-text").textContent = `${endName(body.origin)} → ${endName(body.destination)}, departing ` +
      `${body.issue} (${fc ? "forecast / hackathon estimate" : "real historical data"}). Evaluating sea ice, ` +
      "iceberg scenarios, wind and current forcing, route alternatives and departure dates.";
    $("#pr-elapsed").textContent = "0";
    prod.timer = setInterval(() => { $("#pr-elapsed").textContent = String(Math.round((Date.now() - t0) / 1000)); }, 1000);
  }
  refreshForm();
}

function showError(err) {
  $("#pr-error").hidden = false;
  $("#pr-error-title").textContent = err.title || "Could not plan this route";
  $("#pr-error-text").textContent = err.message;
  $("#pr-error-hint").textContent = err.hint || "";
}

async function planRoute() {
  const problem = formProblem();
  if (problem || prod.busy) { refreshForm(); return; }
  const f = $("#pr-form");
  const body = { origin: endOf("origin"), destination: endOf("destination"), issue: f.issue.value };
  prod.picking = null;
  prod.lastBody = body;
  $("#pr-error").hidden = true;
  // A new plan replaces the old one: never leave a previous result on screen next to new inputs or an error.
  prod.result = null;
  resetSimulation();
  const ctrl = new AbortController();
  prod.controller = ctrl;
  let timedOut = false;
  const kill = setTimeout(() => { timedOut = true; ctrl.abort(); }, PLAN_TIMEOUT_MS);
  setBusy(true, body);
  setView("plan");
  try {
    const res = checkPlan(await planFetch("/real/plan", { method: "POST", body: JSON.stringify(body), signal: ctrl.signal }));
    prod.result = res;
    renderProduct(res);
    setView("plan");
  } catch (e) {
    if (e.name === "AbortError") {
      showError(timedOut ? new PlanError("The plan took too long", `No answer from the server after ` +
        `${PLAN_TIMEOUT_MS / 1000} s, so the request was stopped.`, "The server may be overloaded. Try again.")
        : new PlanError("Planning cancelled", "The request was cancelled; no result is shown.", ""));
    } else showError(e instanceof PlanError ? e : new PlanError("Could not plan this route", e.message));
    setView("plan");
  } finally {
    clearTimeout(kill);
    prod.controller = null;
    setBusy(false);
  }
}

$("#pr-form").addEventListener("submit", (ev) => { ev.preventDefault(); planRoute(); });
$("#pr-retry").addEventListener("click", () => planRoute());
$("#pr-cancel").addEventListener("click", () => { if (prod.controller) prod.controller.abort(); });

/* ---- result */
function metric(label, value, sub = "") {
  return el("div", { class: "pr-metric" }, el("span", { class: "pr-metric-label" }, label),
    el("b", { class: "pr-metric-value" }, value), sub ? el("span", { class: "pr-metric-sub" }, sub) : "");
}

function budgetBar(upper, budget) {
  // Scale so the budget sits at 50% of the bar; values beyond twice the budget fill the bar.
  const w = Math.min(100, (100 * upper) / (2 * budget));
  return el("div", { class: "pr-budget" },
    el("div", { class: "pr-budget-fill " + (upper <= budget ? "ok" : "bad"), style: `width:${w.toFixed(1)}%` }),
    el("div", { class: "pr-budget-mark" }), el("span", { class: "pr-budget-label" }, `budget ${pct(budget, 0)}`));
}

function riskCard(kind, title, r, risk, extra = []) {
  const main = kind === "combined", budget = risk.risk_budget;
  const head = el("div", { class: "pr-risk-head" }, el("h3", {}, title),
    main ? el("span", { class: "pill " + (r.within_budget ? "ok" : "bad") },
      r.within_budget ? "✓ within budget" : "✕ exceeds budget") : el("span", { class: "pr-risk-tag" }, "component"));
  return el("div", { class: "card pr-risk " + kind, "data-risk": kind }, head,
    el("div", { class: "pr-risk-value" }, pct(r.p_breach_upper, 1)),
    el("div", { class: "pr-risk-caption" }, `P(breach) ${Math.round(100 * risk.confidence)}% upper bound (Wilson)`),
    main ? budgetBar(r.p_breach_upper, budget) : "",
    el("div", { class: "pr-risk-rows" },
      row("Scenarios breaching", pctOf(r.breaches, r.n_scenarios)),
      row("Observed share", pct(r.p_breach, 1)), ...extra),
    el("p", { class: "pr-risk-def" }, (risk.definitions || {})[kind] || ""));
}

function renderVerdict(p) {
  const r = p.route, o = p.locations.origin, dst = p.locations.destination, risk = p.risk;
  const bud = risk ? risk.risk_budget : budget();
  const card = $("#pr-verdict");
  card.classList.remove("ok", "bad", "none");
  $("#pr-route-name").textContent = `${o.name || "Origin"} → ${dst.name || "Destination"} · ` +
    (p.metadata.mode === "forecast" ? `forecast estimate for ${p.metadata.requested_date}` : `forecast issued ${p.metadata.issue_date}`);
  const badge = $("#pr-verdict-badge");
  if (p.status === "recommended") {
    card.classList.add("ok");
    badge.textContent = "✓ Recommended";
    $("#pr-verdict-title").textContent = `Depart ${fmtDay(r.departure_date)}: within the ${pct(bud, 0)} risk budget`;
  } else if (p.status === "no_feasible_departure") {
    card.classList.add("bad");
    badge.textContent = "✕ Not recommended";
    $("#pr-verdict-title").textContent =
      `No departure in the selected window meets the ${pct(bud, 0)} risk budget.`;
  } else {
    card.classList.add("none");
    badge.textContent = "✕ No route";
    $("#pr-verdict-title").textContent = "No usable route was found for any departure in the window.";
  }
  $("#pr-verdict-text").textContent = p.status === "no_feasible_departure" ?
    `Showing the least-risky option for reference only: depart ${r.departure_date}. ${p.explanation}` : p.explanation;
  $("#pr-metrics").replaceChildren(...(r ? [
    metric(p.status === "recommended" ? "Recommended departure" : "Least-risky departure", fmtUtc(r.departure_utc),
      r.lead_days ? `+${r.lead_days} d after issue` : "on the issue date"),
    metric("ETA (expected)", fmtUtc(r.eta_utc)),
    metric("Voyage duration", fmtHours(r.expected_hours), `p10–p90 ${num(r.hours_p10, 1)}–${num(r.hours_p90, 1)} h`),
    metric("Distance", `${num(r.distance_km)} km`),
    metric("Fuel index", num(r.fuel_index?.expected), "relative index, not tonnes"),
  ] : []));
  $("#pr-metrics").hidden = !r;
  $("#pr-sim-cta").hidden = !r;
  $("#pr-sim-cta-note").textContent = p.metadata.mode === "forecast" ?
    "Sail this route day by day with the same daily forecasts and replanning rules. Forecast estimate: the ice it " +
    "sails through is the analogue season's real observation (proxy), not an observation of the requested dates." :
    p.status === "recommended" ?
    "Sail this route day by day through the observed sea ice, with a new real forecast and the replanning rules " +
    "applied each day." : "This route was not recommended. The simulation shows what sailing the least-risky " +
    "option would have met, day by day, with the replanning rules applied.";
}

function renderRisks(p) {
  const risk = p.risk, box = $("#pr-risks");
  if (!risk) { box.replaceChildren(); box.hidden = true; return; }
  box.hidden = false;
  const inc = risk.iceberg_only_increment || {};
  box.replaceChildren(
    riskCard("combined", "Combined risk (authoritative)", risk.combined, risk,
      [row("Estimator", `${risk.estimator} at ${Math.round(100 * risk.confidence)}%`)]),
    riskCard("sea_ice", "Sea-ice risk", risk.sea_ice, risk),
    riskCard("iceberg", "Iceberg risk", risk.iceberg, risk, [
      row("Breaching only because of icebergs", pctOf(inc.breaches ?? 0, inc.n_scenarios ?? risk.combined.n_scenarios)),
      ...(risk.iceberg.note ? [row("Note", risk.iceberg.note)] : [])]),
  );
}

function prPlanForLayer(p, k) {
  const L = p.layers, g = L.grid, layer = L.layers.find((x) => x.scenario_layer === k) || L.layers[0];
  const r = p.route, cell = (loc) => {
    const q = loc.resolved;
    return q && q.row != null ? [g.x_km[q.col], g.y_km[q.row]] : null;
  };
  return {
    layer,
    plan: {
      map: { nx: g.nx, ny: g.ny, res_km: g.res_km, x0_km: g.x0_km, y0_km: g.y0_km, rotation_rad: g.rotation_rad,
        land: L.land, p_ice: layer.p_ice_ge_limit_pct, p_berg: layer.p_berg_pct },
      candidates: r ? [{ xy_km: r.xy_km, labels: [p.status === "recommended" ? "recommended route" : "least-risky route (not recommended)"],
        feasible: p.status === "recommended", p_breach: p.risk.combined.p_breach,
        p_breach_upper: p.risk.combined.p_breach_upper, expected_hours: r.expected_hours,
        expected_fuel: r.fuel_index?.expected, distance_km: r.distance_km }] : [],
      recommended_index: r ? 0 : null,
      origin_xy_km: cell(p.locations.origin), destination_xy_km: cell(p.locations.destination),
      land_label: "Land (sea-ice product mask; not a navigational coastline)",
    },
  };
}

function drawProduct() {
  const p = prod.result;
  if (!p || $("#tab-product").hidden || $("#pr-result").hidden) return;
  if (p.layers && p.daily.length) {
    const day = p.daily[prod.day], { plan, layer } = prPlanForLayer(p, day.scenario_layer);
    prMapState.plan = plan; prMapState.tracks = p.iceberg_tracks; prMapState.k = day.scenario_layer;
    drawMap(plan, "#pr-map", prMapState, "#pr-map-legend");
    drawTrackLines("#pr-map", prMapState, p.iceberg_tracks, day.scenario_layer, "#pr-map-legend");
    drawDaySegment(p, day);
    $("#pr-map-title").textContent = `Day ${day.day_of_voyage}: ${fmtDay(day.date)} (${srcLabel(layer.source)} layer)`;
    $("#pr-ramp-label").textContent = `100% P(ice ≥ ${pct(p.layers.vessel_limit, 0)}) across ${p.metadata.n_scenarios} scenarios`;
  }
  if (prWinState.result) drawWindow(prWinState.result, prWinState.budget, "#pr-window-chart", prWinState);
}

// The selected day's stretch of route, and the nominal end-of-day position, on top of the full route.
function drawDaySegment(p, day) {
  if (!day.route_cell_index || !prMapState.toScreen) return;
  const [a, b] = day.route_cell_index, xy = p.route.xy_km.slice(Math.max(0, a - 1), b + 1);
  const pts = xy.map(([x, y]) => prMapState.toScreen(x, y));
  const ctx = $("#pr-map").getContext("2d");
  ctx.save();
  ctx.lineJoin = "round"; ctx.lineCap = "round";
  ctx.strokeStyle = css("--surface-1"); ctx.lineWidth = 9; strokePath(ctx, pts);
  ctx.strokeStyle = css("--series-2"); ctx.lineWidth = 5; strokePath(ctx, pts);
  const [x, y] = pts[pts.length - 1];
  ctx.beginPath(); ctx.arc(x, y, 7, 0, 2 * Math.PI);
  ctx.fillStyle = css("--series-2"); ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = css("--surface-1"); ctx.stroke();
  ctx.restore();
  const legend = $("#pr-map-legend");
  legend.append(el("span", {}, el("i", { style: `border-top-width:4px;border-color:${css("--series-2")}` }),
    `Day ${day.day_of_voyage} (nominal)`));
}

function renderDays(p) {
  const strip = $("#pr-days");
  strip.replaceChildren(...p.daily.map((d, i) => {
    const r = d.risk?.combined?.max_cell_breach_prob;
    const b = el("button", { type: "button", role: "tab", "aria-selected": String(i === prod.day), "data-day": String(i),
      class: "pr-day" + (r > 0 ? " warn" : "") },
      el("b", {}, `Day ${d.day_of_voyage}`), el("span", {}, fmtDay(d.date)), el("span", { class: "pr-day-risk" },
        `risk ${pct(r, 1)}`));
    b.addEventListener("click", () => selectDay(i));
    return b;
  }));
  strip.hidden = !p.daily.length;
}

function selectDay(i) {
  const p = prod.result;
  if (!p || !p.daily.length) return;
  prod.day = Math.max(0, Math.min(p.daily.length - 1, i));
  document.querySelectorAll("#pr-days .pr-day").forEach((b) => b.setAttribute("aria-selected", String(Number(b.dataset.day) === prod.day)));
  renderDayTable(p);
  drawProduct();
}

$("#pr-days").addEventListener("keydown", (ev) => {
  if (ev.key === "ArrowRight") { selectDay(prod.day + 1); ev.preventDefault(); }
  if (ev.key === "ArrowLeft") { selectDay(prod.day - 1); ev.preventDefault(); }
});

function renderDayTable(p) {
  const d = p.daily[prod.day];
  if (!d) {
    $("#pr-day-title").textContent = "Daily timeline";
    $("#pr-day-table tbody").replaceChildren(...kv([["Timeline", "No route, so no daily timeline."]]));
    return;
  }
  const pos = d.position_end_of_day, h = d.nominal_hours_since_departure || [];
  const bergs = (p.iceberg_tracks || []).map((t) => ({ id: t.id, m: (t.daily_mean || []).find((q) => q.layer === d.scenario_layer) }))
    .filter((t) => t.m && t.m.lat != null);
  $("#pr-day-title").textContent = `Day ${d.day_of_voyage} of ${p.daily.length}: ${fmtDay(d.date)}`;
  $("#pr-day-table tbody").replaceChildren(...kv([
    ["Date", `${d.date} (scenario day ${d.scenario_layer}, ${srcLabel(d.layer_source)} layer)`],
    ["Hours since departure (nominal)", `${num(h[0], 1)}–${num(h[1], 1)} h`],
    ["Position at end of day (nominal)", pos ? `${num(pos[0], 3)}°, ${num(pos[1], 3)}°` : "–"],
    ["Distance this day", `${num(d.distance_km, 1)} km`],
    ["Sea ice on this stretch", `mean ${pct(d.sea_ice?.mean_concentration, 1)}, max ${pct(d.sea_ice?.max_concentration, 1)} concentration`],
    ["Max P(ice ≥ vessel limit)", pct(d.sea_ice?.max_p_ge_vessel_limit, 1)],
    ["Max P(iceberg presence)", pct(d.iceberg?.max_p_presence, 1)],
    ["Max cell breach probability: combined", pct(d.risk?.combined?.max_cell_breach_prob, 1)],
    ["… sea ice / icebergs", `${pct(d.risk?.sea_ice?.max_cell_breach_prob, 1)} / ${pct(d.risk?.iceberg?.max_cell_breach_prob, 1)}`],
    ["Icebergs tracked this day", bergs.length ? bergs.map((t) => `${t.id} (${num(t.m.lat, 2)}°, ${num(t.m.lon, 2)}°)`).join(", ")
      : "none in the routing grid"],
  ]));
}

function renderWindow(p) {
  const opts = p.departure.options, shown = p.route ? p.route.departure_date : null;
  const budgetV = p.risk ? p.risk.risk_budget : budget();
  // The chart's "selected" marker is reserved for a recommendation; a least-risky fallback is marked in the table.
  prWinState.result = { options: opts, selected: p.status === "recommended" ? shown : null };
  prWinState.budget = budgetV;
  const nOk = opts.filter((o) => o.feasible).length;
  const recOpt = opts.find((o) => o.departure === shown);
  $("#pr-window-rec").replaceChildren(...(p.status === "recommended" ? [el("b", {}, `★ Recommended departure: ${shown}`),
    recOpt ? ` · risk upper bound ${pct(recOpt.p_breach_upper, 1)} · ${num(recOpt.expected_hours, 1)} h` : ""] :
    shown ? [el("b", {}, `◆ No date meets the budget; least-risky shown: ${shown}`)] : ["No departure could be planned."]));
  $("#pr-window-rec").className = "pr-window-rec" + (p.status === "recommended" ? "" : " bad");
  $("#pr-window-verdict").textContent = `${nOk} of ${opts.length} departure dates meet the ${pct(budgetV, 0)} budget. ` +
    (p.status === "recommended" ? `Recommended: ${shown}. ` : shown ? `None meets it; least-risky shown: ${shown}. ` : "") +
    `Rule: ${p.departure.rule}.`;
  const ref = opts.find((o) => o.departure === shown);
  $("#pr-options tbody").replaceChildren(...opts.map((o) => {
    const arr = o.expected_hours != null && isFinite(o.expected_hours) ?
      fmtUtc(new Date(Math.round((Date.parse(o.departure + "T00:00:00Z") + o.expected_hours * 3600e3) / 60e3) * 60e3)
        .toISOString()) : "–";
    const dt = ref && o.expected_hours != null && ref.expected_hours != null ? o.expected_hours - ref.expected_hours : null;
    const isShown = o.departure === shown;
    return el("tr", { class: isShown ? "rec" : "" },
      el("td", {}, (isShown ? (p.status === "recommended" ? "★ " : "◆ ") : "") + o.departure),
      el("td", {}, statusPill(o.feasible)),
      el("td", { class: "num" }, pct(o.p_breach_upper, 1)),
      el("td", { class: "num" }, num(o.expected_hours, 1)),
      el("td", { class: "num" }, isShown ? "shown" : dt == null ? "–" : (dt >= 0 ? "+" : "") + num(dt, 1) + " h"),
      el("td", {}, arr),
      el("td", { class: "num" }, num(o.expected_fuel)),
      el("td", {}, o.support || "–"));
  }));
}

// Seasonal-analogue result: every input with its real dates (Data & Confidence).
function renderAnalogueData(p) {
  const m = p.metadata, fc = m.forecast, map = fc.date_mapping, sea = fc.sea_ice, f = fc.forcing, b = fc.icebergs;
  const c = fc.confidence || {}, yrs = sea.member_years || [];
  $("#pr-data-modes").replaceChildren(...[
    ["forecastMode", "Mode", `${m.label} for ${m.requested_date}: historical seasonal analogue. No observation or ` +
      "forecast exists for this date; not live and not a prediction."],
    ["proxy", "Sea ice (analogue)", `Real ${sea.source} observations of the same calendar days in ${yrs.at(-1)}–${yrs[0]} ` +
      `(${yrs.length} years, ±${sea.day_shifts[1]} days: ${sea.distinct_sequences} sequences, ${sea.members} members). No model.`],
    ["proxy", "Winds & currents (analogue)", `${f.winds.product} and ${f.currents.product} of ${f.winds.dates_used[0]} to ` +
      `${f.winds.dates_used[1]}: historical reanalysis analogue, not a forecast.`],
    [b.kind === "recent_official" ? "observation" : "proxy", "Icebergs", `${b.source} list of ${b.snapshot_date} ` +
      (b.kind === "recent_official" ? `(${b.age_days_at_requested_date} d before the requested date), ` :
        `(${b.age_days_at_analogue_date} d before the analogue date), `) + `${b.n_in_grid} of ${b.n_source_bergs} in the grid, ` +
      (b.kind === "recent_official" ? "held at their last reported position, then calibrated drift." :
        "the analogue year's official positions, then calibrated drift.")],
    ["derived", "Confidence", `${(c.level || "low").toUpperCase()}: ${(c.caveats || []).join(" ")}`],
    ["research", "Use", "Research / hackathon estimate, not certified navigation."],
  ].map(([kind, k, v]) => el("li", {}, kindBadge(kind), el("span", { class: "src-name" }, k), el("span", {}, v))));
  $("#pr-data-inputs tbody").replaceChildren(...kv([
    ["Requested date", `${m.requested_date} (after the archive, which ends ${fc.archive_last_date})`],
    ["Why an analogue", fc.pathway_reason || "the forecast pathway cannot serve this date"],
    ["Observations for the requested dates", fc.observations_for_requested_dates],
    ["Analogue date", `${fc.analogue_date}; engine days ${map.engine_days[0]} to ${map.engine_days[1]} shown as ` +
      `${map.shown_as[0]} to ${map.shown_as[1]} (offset ${map.offset_days} d)`],
    ["Sea-ice members", (sea.member_windows || []).filter((w) => w.shift_days === 0)
      .map((w) => `${w.first} – ${w.last}`).join("; ") + ` (each also shifted ${sea.day_shifts[0]}…+${sea.day_shifts[1]} d)`],
    ["Off-season sea-ice files", (sea.off_season_files || []).map((x) => `${x.file} (${short(x.sha256)})`).join(", ") || "–"],
    ["Winds", `${f.status}: ${(f.winds.files || []).map((x) => x.file).join(", ")}`],
    ["Currents", `${f.status}: ${(f.currents.files || []).map((x) => x.file).join(", ")}`],
    ["Icebergs", `${b.status}: ${b.file} (${short(b.sha256)}); ${(b.ids_in_grid || []).join(", ") || "none in the grid"}`],
    ["Drift", `${b.drift_model.model}: beta ${b.drift_model.beta}, alpha scale ${b.drift_model.alpha_scale}, ` +
      `spread factor ${b.drift_model.spread_factor}, ${b.drift_model.members} members`],
    ["Sea-ice spread across years", Object.entries((c.sea_ice_spread || {}).mean_concentration_by_year || {})
      .map(([y, v]) => `${y}: ${pct(v, 0)}`).join(", ") || "–"],
  ]));
}

function renderForecastData(p) {
  if (isAnalogue(p.metadata)) return renderAnalogueData(p);
  const m = p.metadata, fc = m.forecast, map = fc.date_mapping, sea = fc.sea_ice, f = fc.forcing, b = fc.icebergs;
  const model = (fc.sea_ice_forecast && fc.sea_ice_forecast.model) || {};
  $("#pr-data-modes").replaceChildren(...[
    ["forecastMode", "Mode", `${m.label} for ${m.requested_date}: no observations or operational forecasts exist for ` +
      "this date in the runtime data; not live."],
    ["proxy", "Sea ice (proxy)", `Real ${sea.source || "OSI SAF"} observations of ${sea.observed_window_used[0]} to ` +
      `${sea.observed_window_used[1]} (analogue season ${map.analogue_season}) as the starting state; NOT observations of ` +
      `${m.requested_date.slice(0, 4)}.`],
    ["model", "Forecast", `Frozen residual U-Net ${model.id || ""}, ${m.n_scenarios} joint scenarios from the proxy start.`],
    ["proxy", "Winds & currents (proxy)", `${f.winds.product} and ${f.currents.product} of ${f.winds.dates_used[0]} to ` +
      `${f.winds.dates_used[1]}: analogue reanalysis, not a forecast.`],
    ["observation", "Icebergs", `${b.source} list of ${b.snapshot_date} (${b.age_days_at_requested_date} d before the requested date), ` +
      `${b.n_in_grid} of ${b.n_source_bergs} in the grid, held at their last reported position, then calibrated drift.`],
    ["research", "Use", "Research / hackathon estimate, not certified navigation."],
  ].map(([kind, k, v]) => el("li", {}, kindBadge(kind), el("span", { class: "src-name" }, k), el("span", {}, v))));
  $("#pr-data-inputs tbody").replaceChildren(...kv([
    ["Requested date", `${m.requested_date} (after the archive, which ends ${fc.archive_last_date})`],
    ["Observations for the requested dates", fc.observations_for_requested_dates],
    ["Analogue start", `${map.analogue_start_date} (season ${map.analogue_season}); engine days ${map.engine_days[0]} to ` +
      `${map.engine_days[1]} shown as ${map.shown_as[0]} to ${map.shown_as[1]} (offset ${map.offset_days} d)`],
    ["Sea ice", `${sea.status}: ${sea.file} (${short(sea.sha256)})`],
    ["Forecast model", `${model.id || "–"} (${short(model.sha256)}), lead ${model.lead_days ?? "–"} d`],
    ["Winds", `${f.status}: ${(f.winds.files || []).map((x) => x.file).join(", ")}`],
    ["Currents", `${f.status}: ${(f.currents.files || []).map((x) => x.file).join(", ")}`],
    ["Icebergs", `${b.status}: ${b.file} (${short(b.sha256)}); ${(b.ids_in_grid || []).join(", ") || "none in the grid"}`],
    ["Drift", `${b.drift_model.model}: beta ${b.drift_model.beta}, alpha scale ${b.drift_model.alpha_scale}, ` +
      `spread factor ${b.drift_model.spread_factor}, ${b.drift_model.members} members`],
  ]));
}

function renderData(p) {
  const m = p.metadata, forecast = m.mode === "forecast";
  const h = forecast ? m.forecast.engine_provenance || {} : m.provenance || {};
  for (const id of ["#pr-disclosure", "#pr-data-disclosure"]) {
    $(id).textContent = forecast ? m.forecast_disclosure : m.hindsight_disclosure;
  }
  $("#pr-disclosure").classList.toggle("forecast", forecast);
  $("#pr-data-badge").textContent = forecast ? (isAnalogue(m) ? "Seasonal analogue estimate" : "Forecast estimate") : "Historical";
  $("#pr-data-badge").className = "badge " + (forecast ? "status-forecast" : "status-historical");
  $("#pr-banners").replaceChildren(...(m.banners || []).map((b) => el("span", { class: "pr-banner" }, b)),
    el("button", { type: "button", class: "pr-link", id: "pr-open-data" }, "Data & Confidence →"));
  $("#pr-open-data").addEventListener("click", () => openTab(primaryButton("data")));
  const sea = h.sea_ice || {}, model = h.forecast_model || {}, f = h.forcing || {}, b = h.icebergs || {}, s = h.season;
  if (!forecast) $("#pr-data-modes").replaceChildren(...[
    ["historicalMode", "Mode", `${m.label}: a past forecast issued ${m.issue_date}, replayed with the real archive; not live.`],
    ["observation", "Sea ice", `${sea.source || "OSI SAF"} observations up to the issue date.`],
    ["model", "Forecast", `Frozen residual U-Net ${model.id || ""}, ${m.n_scenarios} joint scenarios.`],
    ["reanalysis", "Winds & currents", `${f.winds?.product || "ERA5"} and ${f.currents?.product || "CMEMS"}: hindsight forcing.`],
    ["observation", "Icebergs", `${b.source || "USNIC"} list of ${b.list_date || "–"} (${b.age_days ?? "–"} d old), ${(b.drifted || []).length} in the grid, calibrated drift.`],
    ["research", "Use", "Research estimate, not certified navigation."],
  ].map(([kind, k, v]) => el("li", {}, kindBadge(kind), el("span", { class: "src-name" }, k), el("span", {}, v))));
  if (forecast) renderForecastData(p);
  else $("#pr-data-inputs tbody").replaceChildren(...(h.sea_ice ? inputRows(h) : kv([["Inputs", "not reported"]])));
  const locRows = (role, loc) => [
    [`${role}: requested`, `${loc.name || loc.id || "–"} at ${ll(loc.requested)}`],
    [`${role}: used for routing`, `${ll(loc.resolved)} (cell ${(loc.cell || []).join(", ")}), ${loc.snapped ?
      `snapped ${num(loc.distance_km, 1)} km` : `${num(loc.distance_km, 1)} km from the requested point`}`],
  ];
  $("#pr-data-locs tbody").replaceChildren(...kv([
    ...locRows("Origin", p.locations.origin), ...locRows("Destination", p.locations.destination),
    ["Route horizon", `${p.horizon?.horizon_days ?? m.horizon_days} days (great circle ${num(p.horizon?.great_circle_km)} km)`],
    ["Scenario days", `${m.scenario_days} (${m.window_days}-day window + horizon)`],
    ["Season", isAnalogue(m) ? `historical seasonal analogue of ${m.forecast.analogue_date} (see inputs)` :
      forecast ? `analogue ${m.forecast.date_mapping.analogue_season} (proxy start; see inputs)` :
      s ? `${s.season} · ${s.out_of_sample ? "out-of-sample for the U-Net and drift calibration" :
      "in-sample: " + (s.in_sample_notes || []).join("; ")}` +
      `${s.independent_evaluation ? " · independent evaluation season: " + (s.evaluation_notes || []).join("; ") : ""}`
      : "–"],
  ]));
  const defs = (p.risk && p.risk.definitions) || {};
  $("#pr-data-defs tbody").replaceChildren(...kv([
    ...Object.entries(defs).map(([k, v]) => [k.replace(/_/g, " "), v]),
    ["Daily timeline", p.daily_note || "–"],
    ["Time resolution", p.route?.time_resolution || "–"],
  ]));
  // A forecast result lists its own limitations (the iceberg-age rule of a held snapshot); else the engine's.
  const limits = (forecast && m.forecast.limitations) || h.limitations || [];
  $("#pr-data-limits").replaceChildren(...limits.map((t) => el("li", {}, t)));
  $("#pr-data-disclaimer").textContent = m.disclaimer || "";
}

// Forecast estimates: a compact "how this estimate was generated", from the response's own metadata.
function renderHowto(p) {
  const m = p.metadata, card = $("#pr-howto");
  card.hidden = m.mode !== "forecast";
  if (card.hidden) return;
  const fc = m.forecast, map = fc.date_mapping, sea = fc.sea_ice, f = fc.forcing, b = fc.icebergs;
  const model = (fc.sea_ice_forecast && fc.sea_ice_forecast.model) || {}, year = m.requested_date.slice(0, 4);
  const span = (d) => `${fmtDay(d[0])} – ${fmtDay(d[1])} ${d[1].slice(0, 4)}`;
  card.classList.toggle("analogue", isAnalogue(m));
  if (isAnalogue(m)) {
    const c = fc.confidence || {}, yrs = sea.member_years || [];
    $("#pr-howto-list").replaceChildren(...[
      ["Method", "Historical seasonal analogue (no forecast exists for this date)."],
      ["Analogue date", `${fmtDay(fc.analogue_date)} ${fc.analogue_date.slice(0, 4)}`],
      ["Wind/current forcing", `historical reanalysis analogue (${f.winds.product}, ${f.currents.product} of ` +
        `${span(f.winds.dates_used)})`],
      ["Iceberg source", `official USNIC list of ${fmtDay(b.snapshot_date)} ${b.snapshot_date.slice(0, 4)} ` +
        (b.kind === "recent_official" ? `(latest list, ${b.age_days_at_requested_date} days before departure)` :
          "(the analogue year's list; no list within 120 days of departure)") +
        `; ${b.n_in_grid} of ${b.n_source_bergs} bergs in the area, calibrated drift.`],
      ["Sea ice", `real OSI SAF observations of the same calendar days in ${yrs.at(-1)}–${yrs[0]} ` +
        `(${yrs.length} years, ±${sea.day_shifts[1]} days); no model. Not ${year} observations.`],
      ["Confidence", `${(c.level || "low").toUpperCase()}. ${(c.caveats || []).slice(2).join(" ") ||
        "Risk is how often the route met hazards across the analogue years, not a calibrated probability."}`],
      ["Result", "hackathon research estimate, not certified navigation."],
    ].map(([k, v]) => el("li", { "data-k": k }, el("b", {}, k), " ", v)));
    return;
  }
  $("#pr-howto-list").replaceChildren(...[
    ["Sea ice", `historical analogue start (real OSI SAF observations of ${span(sea.observed_window_used)}, ` +
      `season ${map.analogue_season}) + frozen U-Net forecast ${model.id || ""}. Proxy, not ${year} observations.`],
    ["Wind", `${f.winds.product} of ${span(f.winds.dates_used)}: proxy reanalysis, not a ${year} forecast.`],
    ["Currents", `${f.currents.product} of ${span(f.currents.dates_used)}: proxy reanalysis, not a ${year} forecast.`],
    ["Icebergs", `latest available official USNIC list (${b.snapshot_date}, ${b.age_days_at_requested_date} days ` +
      `before departure; ${b.n_in_grid} of ${b.n_source_bergs} bergs in the area) + calibrated drift.`],
    ["Result", "hackathon research estimate, not certified navigation."],
  ].map(([k, v]) => el("li", {}, el("b", {}, k), " ", v)));
}
$("#pr-howto-more").addEventListener("click", () => openTab(primaryButton("data")));

// Data & Confidence: sources and method, filled from the result's metadata when there is one.
function renderDataGeneral(p) {
  const m = p && p.metadata;
  const an = isAnalogue(m) ? m.forecast : null, fc = m && m.mode === "forecast" && !an ? m.forecast : null;
  const h = m ? (fc ? fc.engine_provenance || {} : an ? {} : m.provenance || {}) : {};
  const sea = h.sea_ice || {}, model = h.forecast_model || {}, f = h.forcing || {}, b = h.icebergs || {};
  const par = h.parameters || {}, risk = p && p.risk;
  const item = (kind, k, v) => el("li", {}, kindBadge(kind), el("span", { class: "src-name" }, k), el("span", {}, v));
  $("#pr-data-kinds").replaceChildren(...["observation", "reanalysis", "proxy", "model", "derived"].map(kindBadge));
  const files = (x) => (x && x.files ? ` (${x.files.map((q) => q.file).join(", ")})` : "");
  $("#pr-data-sources").replaceChildren(
    item(fc ? "proxy" : "observation", "OSI SAF sea ice", fc ?
      `Real daily 25 km observations of ${fc.sea_ice.observed_window_used.join(" – ")} (analogue season ` +
      `${fc.date_mapping.analogue_season}), used as a proxy starting state; not observations of ${m.requested_date.slice(0, 4)}.` :
      `${sea.source || "Daily 25 km sea-ice concentration (OSI-450-a / OSI-430-a)"}${sea.observed_through ?
        `, observed up to ${sea.observed_through}` : ""}.`),
    item("model", "Residual U-Net sea-ice forecast", model.id ? `Frozen ${model.id}, ${model.lead_days} days ahead, ` +
      `trained on ${(model.train_seasons || []).length} seasons (${(model.train_seasons || [])[0]}–` +
      `${(model.train_seasons || []).at(-1)}).` : "Frozen residual U-Net trained on past seasons; forecasts daily sea " +
      "ice from the last 14 observed days."),
    item("observation", "USNIC icebergs", fc ? `Official list of ${fc.icebergs.snapshot_date}, held at the last reported ` +
      `positions until departure, then drifted.` : b.list_date ? `Official weekly list of ${b.list_date} ` +
      `(${b.age_days} days old), ${(b.drifted || []).length} bergs in the area.` : "Official weekly U.S. National Ice Center iceberg lists."),
    item(fc ? "proxy" : "reanalysis", "ERA5 winds", fc ? `${fc.forcing.winds.product} of ` +
      `${fc.forcing.winds.dates_used.join(" – ")}: proxy, not a forecast.` : `${f.winds?.product || "ERA5 daily reanalysis"}` +
      `${files(f.winds)}: hindsight reanalysis of the voyage days.`),
    item(fc ? "proxy" : "reanalysis", "Copernicus Marine (CMEMS) currents", fc ? `${fc.forcing.currents.product} of ` +
      `${fc.forcing.currents.dates_used.join(" – ")}: proxy, not a forecast.` : `${f.currents?.product || "CMEMS daily reanalysis"}` +
      `${files(f.currents)}: hindsight reanalysis of the voyage days.`),
  );
  if (an) {
    const yrs = an.sea_ice.member_years || [], ib = an.icebergs;
    $("#pr-data-sources").replaceChildren(
      item("proxy", "OSI SAF sea ice", `Real daily 25 km observations of the same calendar days in ${yrs.at(-1)}–${yrs[0]} ` +
        `(±${an.sea_ice.day_shifts[1]} days), used as a historical seasonal analogue; not observations of ` +
        `${m.requested_date.slice(0, 4)}. No sea-ice model is run.`),
      item(ib.kind === "recent_official" ? "observation" : "proxy", "USNIC icebergs", `Official list of ${ib.snapshot_date}` +
        (ib.kind === "recent_official" ? ", held at the last reported positions until departure, then drifted." :
          " (the analogue year's positions; no list within 120 days of departure), then drifted.")),
      item("proxy", "ERA5 winds", `${an.forcing.winds.product} of ${an.forcing.winds.dates_used.join(" – ")}: historical ` +
        "reanalysis analogue, not a forecast."),
      item("proxy", "Copernicus Marine (CMEMS) currents", `${an.forcing.currents.product} of ` +
        `${an.forcing.currents.dates_used.join(" – ")}: historical reanalysis analogue, not a forecast.`),
    );
  }
  const n = m ? m.n_scenarios : par.members, bud = risk ? risk.risk_budget : budget();
  $("#pr-data-method").replaceChildren(
    item("model", "Sea-ice forecasting", "The residual U-Net forecasts the change in daily sea-ice concentration " +
      "from the last 14 days of observed ice; a bank of its past errors turns one forecast into many plausible futures."),
    item("model", "Iceberg drift", "Each reported berg drifts with the winds and currents in a calibrated " +
      "ensemble" + (par.drift_beta != null ? ` (beta ${par.drift_beta}, alpha scale ${par.drift_alpha_scale}, ` +
      `spread factor ${par.drift_spread_factor})` : "") + "; its footprint is a hazard for the route."),
    item("derived", "Uncertainty", `${n || 200} joint sea-ice and iceberg scenarios` + (par.seed != null ?
      ` (seed ${par.seed})` : "") + ", each scored along the whole route at its own arrival times."),
    item("derived", "Wilson risk", `P(breach) is the share of scenarios where the route meets ice above the ` +
      `vessel limit or an iceberg footprint. Its Wilson ${risk ? Math.round(100 * risk.confidence) : 95}% upper ` +
      `bound must stay within the ${pct(bud, 0)} budget.`),
    item("derived", "Time-dependent routing", "A* finds the route through the daily forecast layers for each " +
      "departure day in the window; the lowest expected fuel within the budget is recommended."),
    item("model", "Calibration", "Sea-ice scenarios add whole forecast-error fields from the training seasons " +
      "(residual bank). The iceberg drift parameters" + (par.drift_beta != null ? ` (beta ${par.drift_beta}, alpha ` +
      `scale ${par.drift_alpha_scale}, spread factor ${par.drift_spread_factor})` : "") + " were calibrated on " +
      "hindcasts of observed iceberg tracks." + (h.probability_calibration?.applied_in_route_risk === false ?
      " Route risk uses the scenario shares as they are; no extra probability recalibration is applied." : "")),
    ...(fc ? [item("proxy", "Forecast mode", `Dates after the archive (${fc.archive_last_date}) start from the ` +
      `same calendar day of ${fc.date_mapping.analogue_season} (${fc.date_mapping.offset_days} days earlier) and are ` +
      "shown on the requested dates. Research/hackathon estimate, not certified navigation.")] : []),
    ...(an ? [item("proxy", "Historical seasonal analogue", `Dates after the archive (${an.archive_last_date}) that the ` +
      "forecast pathway cannot serve are estimated from earlier years: each of the scenarios takes the real observed sea " +
      `ice of the same calendar days in one of ${an.sea_ice.member_years.length} earlier years (±${an.sea_ice.day_shifts[1]} ` +
      `days), with the winds and currents of ${an.analogue_date}. The U-Net is not used. Risk is how often the route met ` +
      "hazards across those years, not a calibrated probability.")] : []),
  );
  const shown = m ? (an ? "analogue" : fc ? "forecast" : "historical") : null;
  const mark = (k) => (shown === k ? " (this result)" : "");
  $("#pr-data-modes-general").replaceChildren(
    item("historicalMode", "Real Historical Data" + mark("historical"), "A past departure date inside the archive " +
      "(Nov–Feb seasons). Real observed sea ice, the frozen U-Net forecast, ERA5/CMEMS reanalysis and the USNIC lists " +
      "of that time. The forcing is hindsight reanalysis, so this is a replay, not a live forecast."),
    item("forecastMode", "Forecast / hackathon estimate" + mark("forecast"), "A future in-season date in the range the " +
      "server lists for the route. The same engine runs from an analogue season: proxy sea ice and ERA5/CMEMS forcing " +
      "of the same calendar days, plus the latest official USNIC iceberg list. No observation of the future year is " +
      "used. Not a real forecast and not certified navigation."),
    item("forecastMode", "Forecast / hackathon estimate: historical seasonal analogue" + mark("analogue"), "Any other " +
      "date after the archive, in any month or year. Real sea ice of the same calendar days in earlier years, the " +
      "ERA5/CMEMS reanalysis of the analogue date, and the latest official USNIC list (or the analogue year's list when " +
      "none is recent). Every date used is listed; a confidence level and its caveats come with the result."),
  );
  // General limitations without a plan; with one, the result's own limitations (from the API) are shown below.
  $("#pr-data-limits-general-wrap").hidden = Boolean(m);
  $("#pr-data-limits-general").replaceChildren(...[
    "Research and hackathon decision support only: not certified for navigation.",
    "The vessel and fuel index are generic placeholders, not a specific ship.",
    "Risk covers sea ice above the vessel limit and iceberg footprints only; weather, sea state and other hazards " +
      "are not scored.",
    "Historical replays use hindsight reanalysis forcing; forecast estimates use proxy forcing from another year.",
    "Forecast-mode estimates have not been validated against what actually happened.",
  ].map((t) => el("li", {}, t)));
}

function renderProduct(p) {
  prod.day = 0;
  renderData(p);
  renderDataGeneral(p);
  renderHowto(p);
  renderVerdict(p);
  renderRisks(p);
  renderDays(p);
  renderDayTable(p);
  renderWindow(p);
  $("#pr-daily-note").textContent = p.daily_note || "";
  const mapCard = $("#pr-map").closest(".grid2");
  mapCard.hidden = !(p.route && p.layers);
}

/* ------------------------------------------------------------- product: Simulate Voyage */
// Plays back the frames POST /real/simulate returns for the route just planned. The server sails the route
// through the observed ice and applies the existing daily replanning; the page only steps through the frames.
const sim = { data: null, key: null, i: 0, timer: null, busy: false, clock: null };
const simMapState = { plan: null, toScreen: null, screenRoutes: [] };
const SIM_STEP_MS = 1400;
const SIM_TIMEOUT_MS = 600000;

function simKey() {
  const b = prod.lastBody, r = prod.result && prod.result.route;
  return b && r ? JSON.stringify([b, r.departure_date]) : null;
}

function resetSimulation() {
  simPause();
  sim.data = null; sim.key = null; sim.i = 0;
  $("#sim-body").hidden = true; $("#sim-error").hidden = true; $("#sim-loading").hidden = true;
}

function checkSim(s) {
  const bad = (why) => new PlanError("The simulation could not be read", `Malformed /real/simulate result: ${why}.`,
    "No playback is shown and nothing is substituted.");
  if (!s || typeof s !== "object") throw bad("not an object");
  if (s.status === "not_simulated") throw new PlanError("This voyage cannot be simulated", s.reason || "No reason given.", "");
  if (s.status !== "simulated") throw bad(`unknown status "${s.status}"`);
  const m = s.metadata;
  if (!labelledMode(m)) throw bad("not labelled as real historical data or as a forecast estimate");
  if (!Array.isArray(s.frames) || s.frames.length < 2) throw bad("fewer than two frames");
  if (s.frames.some((f, k) => f.index !== k || !f.position || !f.route || !Array.isArray(f.track))) throw bad("frames out of order or incomplete");
  if (!s.summary || !Array.isArray(s.events) || !s.grid) throw bad("summary, events or grid missing");
  return s;
}

async function runSimulation() {
  if (sim.busy || !prod.result || !prod.result.route) return;
  const body = { ...prod.lastBody, departure: prod.result.route.departure_date };
  sim.busy = true;
  $("#sim-error").hidden = true; $("#sim-body").hidden = true; $("#sim-loading").hidden = false;
  const o = prod.result.locations.origin, d = prod.result.locations.destination;
  $("#sim-loading-text").textContent = `${o.name} → ${d.name}, departing ${prod.result.route.departure_date}. ` +
    (prod.result.metadata.mode === "forecast" ? "Forecast estimate: for each day at sea the server sails the route " +
    "through the analogue season's observed ice (proxy), issues that day's forecast from the same proxy inputs and " +
    "applies the existing replanning rules." :
    "For each day at sea the server sails the route through the observed sea ice, issues that day's real forecast " +
    "(frozen U-Net, ERA5/CMEMS, USNIC icebergs with calibrated drift) and applies the existing replanning rules.");
  const t0 = Date.now();
  $("#sim-elapsed").textContent = "0";
  clearInterval(sim.clock);
  sim.clock = setInterval(() => { $("#sim-elapsed").textContent = String(Math.round((Date.now() - t0) / 1000)); }, 1000);
  try {
    let job = await planFetch("/real/simulate", { method: "POST", body: JSON.stringify(body) });
    while (job.status === "queued" || job.status === "running") {
      if (Date.now() - t0 > SIM_TIMEOUT_MS) throw new PlanError("The simulation took too long", "No result after 10 minutes.", "Try again.");
      await sleep(1500);
      job = await planFetch(`/jobs/${encodeURIComponent(job.job_id)}`);
    }
    if (job.status !== "done") {
      throw new PlanError("The simulation failed on the server", job.error || "No reason was given.",
        "Nothing synthetic is shown in its place.");
    }
    sim.data = checkSim(job.result);
    updateFlow();
    sim.key = simKey();
    sim.i = 0;
    renderSimulation();
  } catch (e) {
    $("#sim-error").hidden = false;
    $("#sim-error-title").textContent = e.title || "Could not simulate this voyage";
    $("#sim-error-text").textContent = e.message + (e.hint ? " " + e.hint : "");
  } finally {
    clearInterval(sim.clock);
    sim.busy = false;
    $("#sim-loading").hidden = true;
  }
}

function openSimulation() {
  setView("sim");
  if (sim.data && sim.key === simKey()) { renderSimulation(); return; }
  if (!sim.busy) runSimulation();
}

function renderSimulation() {
  const s = sim.data;
  $("#sim-body").hidden = false;
  const m = s.metadata, o = s.locations.origin, d = s.locations.destination;
  $("#sim-banners").replaceChildren(...(m.banners || []).map((b) => el("span", { class: "pr-banner" }, b)));
  $("#sim-disclosure").textContent = m.mode === "forecast" ? m.forecast_disclosure + " The ice sailed through is the " +
    (isAnalogue(m) ? `real observation of the analogue date's year (${m.forecast.analogue_date.slice(0, 4)}; proxy)` :
      "analogue season's real observation (proxy)") + "; this is not live vessel tracking." :
    m.hindsight_disclosure + " This is a replay of a past season, not live vessel tracking.";
  $("#sim-disclosure").classList.toggle("forecast", m.mode === "forecast");
  $("#sim-ramp-label").textContent = m.mode === "forecast" ? "100% proxy (analogue-season) sea-ice concentration" :
    "100% observed sea-ice concentration";
  $("#sim-title").textContent = `${o.name} → ${d.name}`;
  $("#sim-subtitle").textContent = `Departed ${fmtUtc(s.frames[0].timestamp_utc)} · forecast issued ${m.issue_date} · ` +
    `${s.frames.length - 1} day(s) at sea · ` + (s.plan.status === "recommended" ? "recommended route" :
    "least-risky route (not recommended)") + " · " + (s.summary.replans ? `route replanned ${s.summary.replans}×` :
    "No replan was required during this voyage.");
  $("#sim-note").textContent = m.simulation_note || "";
  const slider = $("#sim-slider");
  slider.max = String(s.frames.length - 1);
  $("#sim-events").replaceChildren(...s.frames.map((f, k) => {
    const label = f.phase === "departure" ? "Depart" : f.phase === "arrived" ? "Arrive" : f.phase === "stopped" ? "Stopped"
      : f.replanned ? "REPLAN" : f.decision && f.decision.action === "no_feasible_route" ? "No feasible route" : "Keep route";
    const cls = "sim-event" + (f.replanned ? " replan" : "") +
      (f.decision && f.decision.action === "no_feasible_route" || f.phase === "stopped" ? " alert" : "");
    const b = el("button", { type: "button", role: "tab", class: cls, "data-frame": String(k), "aria-selected": "false" },
      el("b", {}, k === 0 ? "Day 0" : `Day ${f.day_of_voyage}`), el("span", {}, fmtDay(f.date)), el("span", { class: "sim-event-label" }, label));
    b.addEventListener("click", () => { simPause(); showFrame(k); });
    return b;
  }));
  renderSimSummary(s);
  showFrame(sim.i);
}

function simRouteLine(ctx, xy, colour, width, dash = []) {
  if (!xy || xy.length < 2) return;
  ctx.save(); ctx.lineJoin = "round"; ctx.lineCap = "round";
  ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.setLineDash(dash);
  strokePath(ctx, xy.map(([x, y]) => simMapState.toScreen(x, y)));
  ctx.restore();
}

function drawSimMap() {
  const s = sim.data;
  if (!s || $("#pr-sim").hidden || $("#sim-body").hidden) return;
  const f = s.frames[sim.i], g = s.grid, first = s.frames[0];
  const plan = {
    map: { nx: g.nx, ny: g.ny, res_km: g.res_km, x0_km: g.x0_km, y0_km: g.y0_km, rotation_rad: g.rotation_rad,
      land: g.land, p_ice: f.map ? f.map.concentration_pct : g.land.map(() => null), p_berg: null },
    candidates: [], recommended_index: null,
    origin_xy_km: first.route.xy_km[0], destination_xy_km: first.route.xy_km[first.route.xy_km.length - 1],
    land_label: "Land (sea-ice product mask)",
    focus_xy_km: [...first.route.xy_km, ...s.frames.flatMap((x) => x.route.xy_km)],
  };
  simMapState.plan = plan;
  drawMap(plan, "#sim-map", simMapState, "#sim-map-legend");
  const ctx = $("#sim-map").getContext("2d");
  const replanned = f.route_version > 0;
  simRouteLine(ctx, first.route.xy_km, css("--muted"), 2, replanned ? [6, 5] : []);           // as planned
  if (replanned) simRouteLine(ctx, f.route.xy_km, css("--series-2"), 3);                        // new route ahead
  else simRouteLine(ctx, f.route.xy_km, css("--series-1"), 3);                                  // route ahead
  simRouteLine(ctx, f.track, css("--surface-1"), 7);
  simRouteLine(ctx, f.track, css("--series-3"), 4);                                              // sailed so far
  ctx.save(); ctx.strokeStyle = css("--berg"); ctx.fillStyle = css("--berg"); ctx.font = "10px system-ui, sans-serif";
  for (const b of (f.observed && f.observed.icebergs && f.observed.icebergs.in_grid) || []) {
    const [x, y] = simMapState.toScreen(b.xy_km[0], b.xy_km[1]);
    ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fill(); ctx.fillText(b.id, x + 6, y + 3);
  }
  ctx.restore();
  const [vx, vy] = simMapState.toScreen(...f.position.xy_km);
  ctx.save();
  ctx.beginPath(); ctx.arc(vx, vy, 9, 0, 2 * Math.PI); ctx.fillStyle = css("--series-8"); ctx.fill();
  ctx.lineWidth = 3; ctx.strokeStyle = css("--surface-1"); ctx.stroke();
  ctx.fillStyle = css("--text-primary"); ctx.font = "bold 12px system-ui, sans-serif"; ctx.fillText("Vessel", vx + 12, vy + 4);
  ctx.restore();
  const key = (colour, text, dashed = false, box = false) => el("span", {}, el("i", box ? { class: "box", style: `background:${colour}` }
    : { style: `border-color:${colour}${dashed ? ";border-top-style:dashed" : ""}` }), text);
  $("#sim-map-legend").replaceChildren(
    key(css("--land"), "Land (sea-ice product mask)", false, true),
    key(css("--muted"), replanned ? "Original planned route" : "Planned route", replanned),
    replanned ? key(css("--series-2"), "New route ahead (after replan)") : key(css("--series-1"), "Route ahead"),
    key(css("--series-3"), "Sailed so far"), key(css("--series-8"), "Vessel", false, true),
    key(css("--berg"), "USNIC-reported iceberg", false, true));
  $("#sim-map-title").textContent = `${fmtUtc(f.timestamp_utc)}: ` + (f.map ? `${obsWord(f.map.source)} on ${f.map.date}` : "no sea-ice observation for this day");
}

function frameRisk(f) {
  return f.forecast && f.forecast.risk ? f.forecast.risk.combined : null;
}

function showFrame(k) {
  const s = sim.data;
  if (!s) return;
  sim.i = Math.max(0, Math.min(s.frames.length - 1, k));
  const f = s.frames[sim.i], last = sim.i === s.frames.length - 1;
  $("#sim-slider").value = String(sim.i);
  $("#sim-slider-out").textContent = `${f.date} (${sim.i} of ${s.frames.length - 1})`;
  document.querySelectorAll("#sim-events .sim-event").forEach((b) => b.setAttribute("aria-selected", String(Number(b.dataset.frame) === sim.i)));
  $("#sim-prev").disabled = sim.i === 0;
  $("#sim-next").disabled = last;
  const nReplans = s.frames.slice(0, sim.i + 1).filter((x) => x.replanned).length;
  const r = frameRisk(f), fc = f.forecast, ahead = fc && fc.route_ahead;
  const seg = f.observed && f.observed.sea_ice_on_segment, bergs = f.observed && f.observed.icebergs;
  $("#sim-progress-fill").style.width = `${(100 * f.progress.fraction).toFixed(1)}%`;
  $("#sim-status-title").textContent = f.phase === "arrived" ? "Voyage status: arrived" : f.phase === "stopped" ?
    "Voyage status: stopped" : `Voyage status: day ${f.day_of_voyage}`;
  $("#sim-status").replaceChildren(
    metric(f.phase === "arrived" ? "Arrived" : "Date and time", fmtUtc(f.timestamp_utc), f.phase === "departure" ? "departure" : ""),
    metric("Progress", `${Math.round(100 * f.progress.fraction)}%`, `${num(f.progress.sailed_km)} km sailed`),
    metric("Position", `${num(f.position.lat, 2)}°, ${num(f.position.lon, 2)}°`, "last route cell reached (25 km)"),
    metric("Route status", nReplans ? `Route replanned (${nReplans}×)` : f.phase === "arrived" ? "Completed" : "On planned route",
      f.decision ? `today: ${f.decision.action.replace(/_/g, " ")}` : ""),
    metric("Risk ahead (combined)", r ? pct(r.p_breach_upper, 1) : f.phase === "arrived" ? "–" : "unavailable",
      r ? `${r.within_budget ? "within" : "exceeds"} the ${pct(fc.risk.risk_budget, 0)} budget · forecast ${fc.issued || m0(s)}` : ""),
    metric("ETA", fc && fc.eta_utc ? fmtUtc(fc.eta_utc) : f.phase === "arrived" ? fmtUtc(f.timestamp_utc) : "–"),
    metric("Distance remaining", `${num(f.progress.remaining_km)} km`),
    metric("Fuel index ahead", ahead && ahead.expected_fuel != null ? num(ahead.expected_fuel) : "–", "relative index"),
    metric("Sea ice sailed today", seg && seg.max_concentration != null ? `max ${pct(seg.max_concentration, 0)}` : "–",
      seg && seg.date ? (seg.analogue_date ? `proxy: observed ${seg.analogue_date} (analogue)`
        : `${f.observed.sea_ice_source === "proxy_analogue_observed" ? "proxy" : "observed"} ${seg.date}`) : ""),
    metric("Nearest reported berg", bergs && bergs.nearest_km != null ? `${num(bergs.nearest_km)} km` : "none in grid",
      bergs ? `${bergs.nearest_id || ""} · USNIC list ${bergs.list_date}` : ""),
  );
  renderDecision(f);
  $("#sim-summary").hidden = !last;
  $("#sim-play").textContent = sim.timer ? "⏸ Pause" : last ? "↺ Replay" : "▶ Play";
  $("#sim-play").setAttribute("aria-label", sim.timer ? "Pause" : "Play");
  drawSimMap();
}
const m0 = (s) => s.metadata.issue_date;

function renderDecision(f) {
  const box = $("#sim-replan");
  const replanEvent = sim.data.events.find((e) => e.type === "replan" && e.frame === f.index);
  $("#sim-decision-card").classList.toggle("replan", Boolean(replanEvent));
  if (f.phase === "departure") {
    $("#sim-decision-title").textContent = "Departure";
    $("#sim-decision").textContent = `Sailing the planned route (${sim.data.plan.status === "recommended" ? "recommended" :
      "least-risky, not recommended"}). Planned risk: ${pct(frameRisk(f)?.p_breach_upper, 1)} upper bound.`;
  } else if (f.phase === "arrived") {
    $("#sim-decision-title").textContent = "Arrived";
    $("#sim-decision").textContent = "Destination reached.";
  } else if (f.phase === "stopped") {
    $("#sim-decision-title").textContent = "Simulation stopped";
    $("#sim-decision").textContent = sim.data.summary.reason || "";
  } else {
    $("#sim-decision-title").textContent = replanEvent ? "REPLAN: new route selected" :
      f.decision.action === "no_feasible_route" ? "No feasible route: human review" : "Today's decision: keep route";
    $("#sim-decision").textContent = f.decision.explanation;
  }
  if (!replanEvent) { box.replaceChildren(); return; }
  const o = replanEvent.old_route, n = replanEvent.new_route, c = replanEvent.change;
  const sign = (v, d, u) => (v == null ? "–" : `${v >= 0 ? "+" : ""}${num(v, d)}${u}`);
  box.replaceChildren(
    el("p", { class: "sim-reason" }, el("b", {}, "Why: "), (replanEvent.reasons || []).join(" ") || replanEvent.triggers.join(", ")),
    el("div", { class: "table-wrap" }, el("table", { class: "data sim-compare" },
      el("thead", {}, el("tr", {}, el("th", {}, ""), el("th", { class: "num" }, "Old route"), el("th", { class: "num" }, "New route"),
        el("th", { class: "num" }, "Change"))),
      el("tbody", {},
        el("tr", {}, el("td", {}, "P(breach) 95% upper"), el("td", { class: "num" }, pct(o.p_breach_upper, 1)),
          el("td", { class: "num" }, pct(n.p_breach_upper, 1)), el("td", { class: "num" }, c.p_breach_upper == null ? "–" : sign(100 * c.p_breach_upper, 1, " pp"))),
        el("tr", {}, el("td", {}, "Expected time ahead"), el("td", { class: "num" }, `${num(o.expected_hours, 1)} h`),
          el("td", { class: "num" }, `${num(n.expected_hours, 1)} h`), el("td", { class: "num" }, sign(c.expected_hours, 1, " h"))),
        el("tr", {}, el("td", {}, "Distance ahead"), el("td", { class: "num" }, `${num(o.distance_km)} km`),
          el("td", { class: "num" }, `${num(n.distance_km)} km`), el("td", { class: "num" }, sign(c.distance_km, 0, " km"))),
        el("tr", {}, el("td", {}, "Fuel index ahead"), el("td", { class: "num" }, num(o.expected_fuel)),
          el("td", { class: "num" }, num(n.expected_fuel)), el("td", { class: "num" }, sign(c.expected_fuel, 0, ""))),
      ))),
    el("p", { class: "sub pr-small" }, `Path moves up to ${num(replanEvent.deviation_km)} km. Forecast issued ${replanEvent.issued}.`));
}

function renderSimSummary(s) {
  const u = s.summary, sl = u.sailed || {}, fr = u.final_forecast_risk && u.final_forecast_risk.combined;
  $("#sim-summary-verdict").textContent = (u.arrived ? `Voyage completed: arrived ${fmtUtc(u.arrival_utc)}. ` :
    `Voyage not completed: ${u.reason}. `) + (u.replans ? `Route replanned ${u.replans} time(s).` : u.replan_note);
  $("#sim-summary-metrics").replaceChildren(
    metric("Status", u.arrived ? "Arrived" : "Incomplete", `${u.days_at_sea} day(s) at sea`),
    metric("Voyage duration", u.arrived ? fmtHours(u.simulated_hours) : "–",
      u.arrived ? `${num(u.held_hours, 1)} h held at day boundaries; continuous ${num(sl.hours_through_observed_ice, 1)} h` : ""),
    metric("Distance sailed", `${num(sl.distance_km)} km`, `planned ${num(u.planned.distance_km)} km`),
    metric("Fuel index", num(sl.fuel_index), `planned ${num(u.planned.expected_fuel)} · relative index`),
    metric("Replans", String(u.replans), u.replans ? "" : "No replan was required"),
    metric("Final route risk", fr ? pct(fr.p_breach_upper, 1) : "unavailable", fr ? `forecast issued ${u.final_forecast_issued}` : ""),
    metric("Observed ice ≥ limit", `${sl.observed_breach_cells ?? "–"} cells`, `${num(sl.hazard_hours, 1)} h on the sailed track`),
    metric("Reported-berg footprint", `${sl.berg_footprint_cells ?? "–"} cells`,
      sl.berg_min_distance_km != null ? `nearest ${sl.berg_nearest} at ${num(sl.berg_min_distance_km)} km` : ""),
  );
  $("#sim-summary-events").replaceChildren(...s.events.map((e) => el("li", {},
    el("div", {}, el("strong", {}, e.type === "replan" ? "REPLAN" : e.type.replace(/_/g, " ")), ` · ${fmtUtc(e.timestamp_utc)}`),
    el("div", {}, e.explanation || ""))));
  $("#sim-summary-notes").textContent = Object.values(u.notes || {}).join(" ");
}

function simPause() {
  clearInterval(sim.timer); sim.timer = null;
  if (sim.data) { $("#sim-play").textContent = sim.i === sim.data.frames.length - 1 ? "↺ Replay" : "▶ Play"; $("#sim-play").setAttribute("aria-label", "Play"); }
}

function simPlay() {
  if (!sim.data) return;
  if (sim.timer) { simPause(); return; }
  if (sim.i >= sim.data.frames.length - 1) showFrame(0);
  sim.timer = setInterval(() => {
    if (sim.i >= sim.data.frames.length - 1) { simPause(); return; }
    showFrame(sim.i + 1);
    if (sim.i >= sim.data.frames.length - 1) simPause();
  }, SIM_STEP_MS);
  $("#sim-play").textContent = "⏸ Pause";
  $("#sim-play").setAttribute("aria-label", "Pause");
}

$("#sim-play").addEventListener("click", simPlay);
$("#sim-prev").addEventListener("click", () => { simPause(); showFrame(sim.i - 1); });
$("#sim-next").addEventListener("click", () => { simPause(); showFrame(sim.i + 1); });
$("#sim-slider").addEventListener("input", () => { simPause(); showFrame(Number($("#sim-slider").value)); });
$("#sim-retry").addEventListener("click", () => runSimulation());
$("#pr-simulate").addEventListener("click", openSimulation);

/* ------------------------------------------------------------- boot */
(async () => {
  try {
    const h = await api("/health");
    $("#disclaimer").textContent = h.disclaimer;
    $("#version-badge").textContent = "v" + h.version;
    const c = await api("/config");
    window.__config = c.config;
    window.__status = await api("/status");
  } catch (e) {
    $("#disclaimer").textContent = "API unavailable: " + e.message;
    showRealUnavailable("API unavailable: " + e.message);
  }
  const productReady = setupProduct(window.__status || null);
  setModeBadge(currentTab());
  setupHistorical(window.__status ? window.__status.historical : null);
  if (window.__status) {
    try { await loadReal(window.__status); } catch (e) { showRealUnavailable("Could not load real data: " + e.message); }
    if (window.__status.historical && window.__status.historical.status === "available") {
      try { await loadHistoricalDates(); } catch (e) {
        setupHistorical({ status: "failed", reason: "could not load the available dates: " + errText(e) });
      }
    }
  }
  await productReady;
  window.addEventListener("resize", () => {
    if (!$("#tab-product").hidden) { drawProduct(); drawSimMap(); }
    if (mapState.plan && !$("#tab-plan").hidden) drawMap(mapState.plan);
    if (winState.result && !$("#tab-window").hidden) drawWindow(winState.result, budget());
    if (!$("#tab-real").hidden) redrawReal();
    const tab = currentTab();
    if (isHist(tab)) redrawHist(tab);
  });
})();
