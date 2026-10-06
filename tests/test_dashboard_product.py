"""Dashboard Plan Route product view (S3).

Static checks on the served dashboard files, plus the browser tests in ``frontend/tests`` (Playwright with
Chromium, against a mocked API) when Node and Playwright are installed; otherwise those are skipped.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from antarctic_routing.locations import PRESETS

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "src" / "antarctic_routing" / "dashboard"
BROWSER_TESTS = ROOT / "frontend" / "tests" / "dashboard_product.test.mjs"
NODE = shutil.which("node")


def _html() -> str:
    return (DASH / "index.html").read_text()


def _js() -> str:
    return (DASH / "app.js").read_text()


def test_plan_route_is_the_default_page_and_the_primary_nav_has_only_the_product():
    html = _html()
    nav = html[html.index('<nav class="tabs primary-nav"'):html.index("</nav>")]
    pages = re.findall(r'<button role="tab" aria-selected="(true|false)" data-tab="product" data-view="([a-z]+)"[^>]*>'
                       r"([^<]+)</button>", nav)
    assert pages == [("true", "plan", "Plan Route"), ("false", "sim", "Voyage Simulation"),
                     ("false", "data", "Data &amp; Confidence")]
    assert len(re.findall(r"<button", nav)) == 3
    for word in ("synthetic", "Synthetic", "Validation", "Research", "legacy"):
        assert word not in nav, word
    assert re.search(r'<section class="tab" id="tab-product" data-view="plan">', html)
    # research and synthetic views stay available, collapsed under "Research / Developer"
    start = html.index('<details id="dev-area">')
    dev = html[start:html.index("</details>", start)]
    assert "<summary>Research / Developer</summary>" in dev and "open" not in dev.split(">")[0]
    tabs = re.findall(r'<button role="tab" aria-selected="false" data-tab="([a-z]+)"', dev)
    assert tabs == ["real", "hroute", "hwindow", "hvoyage", "plan", "window", "voyage", "validation"]
    for tab in tabs:
        assert re.search(rf'<section class="tab" id="tab-{tab}" hidden>', html), tab
    assert "Research &amp; legacy views" in dev and "Synthetic sandbox (demo only)" in dev
    assert "Antarctic Ice-Risk Routing" in html


def test_dashboard_has_no_inline_styles_or_scripts():
    html = _html()
    assert " style=" not in html                 # blocked by the CSP (no 'unsafe-inline')
    assert "<style" not in html
    assert re.findall(r"<script[^>]*>", html) == ['<script src="static/app.js">']


def test_locations_are_not_hard_coded_in_the_page():
    js, html = _js(), _html()
    for p in PRESETS:
        assert p.id not in js and p.id not in html, p.id
        assert p.name not in js and p.name not in html, p.name
    assert '"/real/locations"' in js


def test_the_product_makes_one_plan_call_and_never_a_synthetic_one():
    js = _js()
    boot = js.index("/* ------------------------------------------------------------- boot */")
    product = js[js.index("product: Plan Route"):boot]
    assert product.count('"/real/plan"') == 1
    assert product.count('"/real/simulate"') == 1      # Simulate Voyage: one job, then local playback
    for synthetic in ('"/routes', '"/departures', '"/voyages"', "controlled_synthetic"):
        assert synthetic not in product, synthetic


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_plan_route_flow():
    probe = subprocess.run([NODE, "-e", "const {execSync}=require('child_process');"
                            "try{require('playwright')}catch{require(execSync('npm root -g').toString().trim()"
                            "+'/playwright')}"], capture_output=True, text=True)
    if probe.returncode:
        pytest.skip("Playwright is not installed for node")
    res = subprocess.run([NODE, "--test", str(BROWSER_TESTS)], capture_output=True, text=True, timeout=600)
    summary = "\n".join(line for line in res.stdout.splitlines() if line.startswith(("# pass", "# fail", "not ok")))
    assert res.returncode == 0, res.stdout[-6000:] + res.stderr[-2000:]
    assert "# fail 0" in summary
    assert int(re.search(r"# pass (\d+)", summary).group(1)) >= 53
