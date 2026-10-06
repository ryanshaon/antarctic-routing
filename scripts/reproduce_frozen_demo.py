#!/usr/bin/env python
"""Reproduce the frozen real-data demo and, optionally, publish it as an API bundle.

    ANTROUTE_DATA_ROOT=/path/to/real-data \
        python scripts/reproduce_frozen_demo.py --out reports/frozen-demo --bundle artifacts/frozen-demo

Nothing is fitted, trained or tuned. Steps:

1. every input listed in docs/frozen_demo/frozen_demo_2023-11-14.json must exist under the data root
   and match its SHA-256 (missing -> exit 2 "blocked", mismatch -> exit 3);
2. the frozen drift parameters are read back from the calibration files and must equal the command's;
3. ``antroute plan-window`` runs with ``--require-real-forcing --export-maps`` (failure -> exit 4);
4. the output is compared with the expected results (any difference -> exit 5);
5. with ``--bundle``, a verified read-only bundle is written for ``ANTROUTE_ARTIFACTS_DIR``.

The large inputs are never copied into the repository or the bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "docs" / "frozen_demo" / "frozen_demo_2023-11-14.json"
BLOCKED, MISMATCH, RUN_FAILED, RESULT_DIFFERS = 2, 3, 4, 5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(code: int, msg: str) -> None:
    label = {BLOCKED: "BLOCKED", MISMATCH: "CHECKSUM MISMATCH", RUN_FAILED: "FAILED", RESULT_DIFFERS: "RESULT DIFFERS"}
    print(f"{label[code]}: {msg}", file=sys.stderr)
    raise SystemExit(code)


def verify_inputs(spec: dict, root: Path) -> dict:
    out = {}
    for name, rec in spec["inputs"].items():
        path = root / rec["path"]
        if not path.is_file():
            fail(BLOCKED, f"input {name} not found at $ANTROUTE_DATA_ROOT/{rec['path']}")
        got = sha256(path)
        if got != rec["sha256"]:
            fail(MISMATCH, f"{name}: expected {rec['sha256']}, found {got}")
        out[name] = {"path": rec["path"], "sha256": got, "bytes": path.stat().st_size, "verified": True,
                     "description": rec.get("description")}
    cfg = REPO / spec["config"]["path"]
    got = sha256(cfg)
    if got != spec["config"]["sha256"]:
        fail(MISMATCH, f"config {spec['config']['path']}: expected {spec['config']['sha256']}, found {got}")
    out["config"] = {"path": spec["config"]["path"], "sha256": got, "verified": True}
    return out


def verify_parameters(spec: dict, root: Path) -> dict:
    p = spec["parameters"]
    sel = json.loads((root / spec["inputs"]["drift_selection"]["path"]).read_text())["selected_params"]
    spr = json.loads((root / spec["inputs"]["spread_calibration"]["path"]).read_text())["selected_model"]
    if (sel["beta"], sel["alpha_scale"]) != (p["drift_beta"], p["drift_alpha_scale"]):
        fail(MISMATCH, f"drift selection {sel} differs from the frozen parameters")
    if (spr["c"], spr["q"]) != (p["drift_spread_factor"], 0.0):
        fail(MISMATCH, f"spread calibration {spr} differs from the frozen parameters")
    return {"drift_beta": sel["beta"], "drift_alpha_scale": sel["alpha_scale"], "spread_c": spr["c"],
            "spread_q": spr["q"]}


def command(spec: dict, root: Path, out: Path) -> list[str]:
    i, p = spec["inputs"], spec["parameters"]
    return [sys.executable, "-m", "antarctic_routing.cli", "plan-window",
            "--config", str(REPO / spec["config"]["path"]), "--data", str(root / i["sea_ice"]["path"]),
            "--weights", str(root / i["checkpoint"]["path"]),
            "--wind-forcing", str(root / i["wind_forcing"]["path"]),
            "--current-forcing", str(root / i["current_forcing"]["path"]),
            "--issue", spec["issue"], "--window-days", str(p["window_days"]), "--members", str(p["members"]),
            "--seed", str(p["seed"]), "--icebergs", str(root / i["icebergs"]["path"]),
            "--berg-radius-km", str(p["berg_radius_km"]), "--drift-beta", str(p["drift_beta"]),
            "--drift-alpha-scale", str(p["drift_alpha_scale"]), "--drift-spread-factor", str(p["drift_spread_factor"]),
            "--require-real-forcing", "--export-maps", "--out", str(out)]


def compare(spec: dict, plan: dict, raw_sha: str, root: Path) -> tuple[dict, list[str]]:
    from antarctic_routing.publish import canonical_sha256

    e = spec["expected"]
    sel = plan.get("selected_route") or {}
    ev = sel.get("evaluation") or {}
    got = {
        "execution_mode": plan.get("execution_mode"),
        "forcing_status": (plan.get("forcing_provenance") or {}).get("status"),
        "forcing_label": (plan.get("forcing_provenance") or {}).get("label"),
        "selected": plan.get("selected"), "n_scenarios": ev.get("n_scenarios"), "breaches": ev.get("breaches"),
        "expected_hours": ev.get("expected_hours"), "expected_fuel": ev.get("expected_fuel"),
        "distance_km": ev.get("distance_km"), "p_breach_upper": ev.get("p_breach_upper"),
        "route_cells": len(sel.get("cells_row_col") or []),
        "route_cells_with_iceberg_presence": (sel.get("iceberg_presence_on_route") or {}).get(
            "route_cells_with_any_presence"),
        "plan_window_canonical_sha256": canonical_sha256(plan),
    }
    diffs = [f"{k}: expected {e[k]!r}, got {v!r}" for k, v in got.items() if e.get(k) != v]
    same_paths = root.resolve() == Path(e["original_data_root"]).resolve()
    got["plan_window_sha256"] = raw_sha
    got["plan_window_byte_identical_to_original"] = (raw_sha == e["plan_window_sha256_at_original_paths"]
                                                     if same_paths else None)
    if same_paths and raw_sha != e["plan_window_sha256_at_original_paths"]:
        diffs.append(f"plan_window.json bytes differ from the original run ({raw_sha})")
    return got, diffs


def environment() -> dict:
    pkgs = {}
    for name in ("numpy", "scipy", "xarray", "netCDF4", "pyproj", "torch", "scikit-learn", "matplotlib"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    def git(*a, text=True):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=text, check=True).stdout

    try:   # read-only git commands; the index and working tree are not touched
        commit = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain").strip())
        h = hashlib.sha256(git("diff", "HEAD", "--binary", text=False))
        untracked = sorted(git("ls-files", "--others", "--exclude-standard").split())
        for name in untracked:
            h.update(name.encode() + b"\0" + (REPO / name).read_bytes())
        worktree = h.hexdigest()
    except (OSError, subprocess.CalledProcessError):
        commit, dirty, worktree, untracked = None, None, None, []
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": pkgs,
            "git_commit": commit, "worktree_dirty": dirty, "worktree_sha256": worktree,
            "untracked_files": len(untracked)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=os.environ.get("ANTROUTE_DATA_ROOT"),
                    help="directory holding the inputs (default: $ANTROUTE_DATA_ROOT)")
    ap.add_argument("--out", required=True, help="output directory for the plan-window run")
    ap.add_argument("--bundle", default=None, help="also write a verified API bundle here")
    args = ap.parse_args(argv)
    if not args.data_root:
        fail(BLOCKED, "no data root: set ANTROUTE_DATA_ROOT or pass --data-root")
    root, out = Path(args.data_root), Path(args.out)
    spec = json.loads(SPEC.read_text())

    inputs = verify_inputs(spec, root)
    params = verify_parameters(spec, root)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run(command(spec, root, out), capture_output=True, text=True)
    seconds = time.perf_counter() - t0
    (out / "stdout.txt").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        fail(RUN_FAILED, f"plan-window exited {proc.returncode}: {proc.stderr.strip()[-1500:]}")
    plan_path = out / "plan_window.json"
    plan = json.loads(plan_path.read_text())
    got, diffs = compare(spec, plan, sha256(plan_path), root)
    report = {"kind": "frozen_demo_reproduction", "issue": spec["issue"], "spec_sha256": sha256(SPEC),
              "inputs": inputs, "frozen_parameters": params, "environment": environment(),
              "seconds": round(seconds, 1), "result": got, "differences": diffs, "reproduced": not diffs}
    (out / "reproduction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if diffs:
        fail(RESULT_DIFFERS, "; ".join(diffs))
    print(f"REPRODUCED: selected {got['selected']}, E[t] {got['expected_hours']:.1f} h, "
          f"fuel {got['expected_fuel']:.1f}, {got['breaches']}/{got['n_scenarios']} breaches, "
          f"Wilson UB {got['p_breach_upper']:.2%}; canonical sha256 {got['plan_window_canonical_sha256'][:16]}")

    if args.bundle:
        import xarray as xr

        from antarctic_routing.config import load_config
        from antarctic_routing.publish import write_bundle

        cfg = load_config(REPO / spec["config"]["path"])
        with xr.open_dataset(root / spec["inputs"]["sea_ice"]["path"]) as ds:
            attrs = {k: ds.attrs.get(k) for k in ("source_product", "crs", "resolution_m", "execution_mode")}
        attrs = {k: v.item() if hasattr(v, "item") else v for k, v in attrs.items()}   # numpy scalars -> JSON
        manifest = json.loads(json.dumps({**report, "spec": {k: spec[k] for k in ("description", "parameters")}}))
        manifest["sea_ice"] = {"dataset": attrs,
                               "forecast_model": {"id": spec["inputs"]["checkpoint"]["path"],
                                                  "sha256": inputs["checkpoint"]["sha256"],
                                                  "description": spec["inputs"]["checkpoint"]["description"]},
                               "probability_calibration": {"applied_in_route_risk": False}}
        manifest["routing"] = {"risk_budget": cfg.routing.risk_budget, "risk_estimator": cfg.routing.risk_estimator,
                               "confidence": cfg.routing.confidence, "vessel": cfg.vessel.name,
                               "ice_class": cfg.vessel.ice_class}
        idx = write_bundle(Path(args.bundle), plan_path, out / "forecast_maps.json", manifest,
                           figure=out / "departure_window.png", kind="frozen_demo",
                           limitations=spec["limitations"])
        print(f"Bundle written: {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
