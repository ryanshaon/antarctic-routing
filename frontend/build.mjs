// Static production build of the dashboard for separate hosting (e.g. Vercel).
//
//   ANTROUTE_API_BASE=https://<api-host> node frontend/build.mjs [out-dir]   (default: frontend/dist)
//
// Copies src/antarctic_routing/dashboard (the same files the API serves at "/") and writes the
// API origin into the page, with a Content-Security-Policy that allows only that origin.
// It reads exactly one environment variable, ANTROUTE_API_BASE; nothing else reaches the output.
// No dependencies, no bundler: the dashboard is plain HTML, CSS and JavaScript.
import { copyFileSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "..", "src", "antarctic_routing", "dashboard");
const out = resolve(process.argv[2] || join(here, "dist"));
const META = '<meta name="antroute-api-base" content="">';

function fail(message) {
  console.error(`frontend build failed: ${message}`);
  process.exit(1);
}

function apiOrigin(raw) {
  if (!raw) fail("set ANTROUTE_API_BASE to the API's public origin, e.g. https://antarctic-routing-api.onrender.com");
  let u;
  try {
    u = new URL(raw);
  } catch {
    fail(`ANTROUTE_API_BASE is not a URL: ${raw}`);
  }
  const local = u.hostname === "localhost" || u.hostname === "127.0.0.1";
  if (u.protocol !== "https:" && !(u.protocol === "http:" && local)) {
    fail("ANTROUTE_API_BASE must use https (http only for localhost)");
  }
  if (u.username || u.password || u.search || u.hash || u.pathname !== "/") {
    fail("ANTROUTE_API_BASE must be an origin only (https://host[:port]), without path, query or credentials");
  }
  return u.origin;
}

const origin = apiOrigin((process.env.ANTROUTE_API_BASE || "").trim());
const html = readFileSync(join(src, "index.html"), "utf8");
if (html.split(META).length !== 2) fail(`index.html must contain exactly one ${META}`);

// Same policy as the API-served page (DASHBOARD_CSP), plus the API origin for data and figures.
// frame-ancestors cannot be set in a <meta> policy; vercel.json sends it as a header.
const csp = [
  "default-src 'self'",
  `connect-src 'self' ${origin}`,
  `img-src 'self' data: ${origin}`,
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'self'",
].join("; ");

rmSync(out, { recursive: true, force: true });
mkdirSync(join(out, "static"), { recursive: true });
writeFileSync(join(out, "index.html"), html.replace(META,
  `<meta name="antroute-api-base" content="${origin}">\n  <meta http-equiv="Content-Security-Policy" content="${csp}">`));
for (const file of ["app.js", "style.css"]) copyFileSync(join(src, file), join(out, "static", file));
console.log(`dashboard built in ${out} for the API at ${origin}`);
