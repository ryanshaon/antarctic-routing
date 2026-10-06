"""Deployment: static dashboard build (Vercel), API image and Render settings keep secrets out."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from antarctic_routing.api.main import create_app

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "config.yaml"
DASHBOARD = ROOT / "src" / "antarctic_routing" / "dashboard"
CREDENTIALS = ("CDSAPI_KEY", "CDSAPI_URL", "COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD")
NODE = shutil.which("node")


def build(out: Path, api_base: str | None, **extra_env) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "ANTROUTE_API_BASE"} | extra_env
    if api_base is not None:
        env["ANTROUTE_API_BASE"] = api_base
    return subprocess.run([NODE, str(ROOT / "frontend" / "build.mjs"), str(out)], env=env, capture_output=True,
                          text=True, timeout=60)


@pytest.mark.skipif(NODE is None, reason="Node.js is needed for the dashboard build")
def test_frontend_build_points_the_dashboard_at_the_api_and_nothing_else(tmp_path):
    secrets = {k: f"sentinel-{k.lower()}" for k in CREDENTIALS}
    r = build(tmp_path / "dist", "https://api.example.org/", **secrets)
    assert r.returncode == 0, r.stderr
    html = (tmp_path / "dist" / "index.html").read_text(encoding="utf-8")
    assert '<meta name="antroute-api-base" content="https://api.example.org">' in html
    csp = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', html).group(1)
    assert "connect-src 'self' https://api.example.org" in csp and "img-src 'self' data: https://api.example.org" in csp
    assert "unsafe" not in csp
    for name in ("app.js", "style.css"):
        assert (tmp_path / "dist" / "static" / name).read_bytes() == (DASHBOARD / name).read_bytes()
    built = "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "dist").rglob("*") if p.is_file())
    assert not [v for v in secrets.values() if v in built]


@pytest.mark.skipif(NODE is None, reason="Node.js is needed for the dashboard build")
@pytest.mark.parametrize("api_base, reason", [
    (None, "set ANTROUTE_API_BASE"),
    ("api.example.org", "not a URL"),
    ("http://api.example.org", "must use https"),
    ("https://api.example.org/v1", "origin only"),
    ("https://user:pw@api.example.org", "origin only"),
])
def test_frontend_build_refuses_a_missing_or_unsafe_api_location(tmp_path, api_base, reason):
    r = build(tmp_path / "dist", api_base)
    assert r.returncode != 0 and reason in r.stderr
    assert not (tmp_path / "dist").exists()


def test_api_reads_and_returns_no_data_service_credentials(monkeypatch):
    secrets = {k: f"sentinel-{k.lower()}" for k in CREDENTIALS}
    for k, v in secrets.items():
        monkeypatch.setenv(k, v)
    with TestClient(create_app(CONFIG)) as c:
        body = "".join(c.get(p).text for p in ("/health", "/ready", "/status", "/versions", "/config"))
    assert not [v for v in secrets.values() if v in body]
    api_code = "".join(p.read_text(encoding="utf-8") for p in (ROOT / "src" / "antarctic_routing" / "api").glob("*.py"))
    assert not [k for k in CREDENTIALS if k in api_code]


def test_api_starts_without_pytorch():
    """The API image has no PyTorch ('ml' extra): the app must still import and serve /health."""
    code = (
        "import sys\n"
        "class NoTorch:\n"  # any `import torch...` now fails as if PyTorch were not installed
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'torch': raise ImportError('No module named torch')\n"
        "sys.meta_path.insert(0, NoTorch())\n"
        "from fastapi.testclient import TestClient\n"
        "from antarctic_routing.api.main import create_app\n"
        f"r = TestClient(create_app({str(CONFIG)!r})).get('/health')\n"
        "assert r.status_code == 200, r.text\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr


def test_deployment_files_keep_credentials_and_data_out():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    assert {".env", ".env.*", ".cdsapirc", "*.nc", "*.pt", "data", "models"} <= set(ignored)
    render = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "healthCheckPath: /ready" in render and not [k for k in CREDENTIALS if k in render]
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    api_stage = dockerfile[dockerfile.index("FROM python AS api"):]
    assert "torch" not in api_stage and "cdsapi" not in api_stage and "${PORT:-8000}" in api_stage
    assert not re.search(r"^\s*(ENV|ARG)\s+\S*(KEY|PASSWORD|TOKEN|SECRET)", dockerfile, re.M | re.I)
