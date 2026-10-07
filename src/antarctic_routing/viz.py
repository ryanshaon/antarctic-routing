"""Static figures: route map on the polar grid and departure-window chart."""

from __future__ import annotations

import functools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

from antarctic_routing import DISCLAIMER  # noqa: E402
from antarctic_routing.routing.candidates import PlanResult  # noqa: E402
from antarctic_routing.routing.departure import DepartureSweep  # noqa: E402
from antarctic_routing.routing.hazard import exceedance_probability  # noqa: E402
from antarctic_routing.synthetic import ScenarioSet  # noqa: E402

# The dashboard's chart palette (dashboard/style.css), so figures and screens read as one product.
_INK, _MUTED, _GRID = "#0f2231", "#5b6d7b", "#e3e9ed"
_ACCENT, _GOOD, _CRITICAL, _BAR = "#a8195f", "#17694a", "#b2301c", "#5b8aa6"
_BLUE, _OCHRE, _BERG = "#1f5f99", "#a85a12", "#5a45a8"
_SEA, _LAND, _COAST, _GRATICULE = "#f3f8fb", "#ecdcab", "#a48c4a", "#c3d1da"
_ICE = LinearSegmentedColormap.from_list("chart_ice", [_SEA, "#d5e6f2", "#aecfe6", "#86b5d8", "#5e98c6", "#3f79ad",
                                                       "#275a8f", "#143d6b"])
_ROUTE_COLOURS = ["#5b6d7b", _OCHRE, "#0f6b5c", "#8a6100", "#2f6fb0", "#3c7d3f", "#5b4bb5"]

_FONTS = [f for f in ("Bahnschrift", "DIN Alternate", "Avenir Next", "Segoe UI")
          if f in set(font_manager.get_font_names())]
_STYLE = {
    "font.family": [*_FONTS, "DejaVu Sans"], "text.color": _INK, "axes.labelcolor": _INK, "axes.edgecolor": "#b9c5cd",
    "axes.titlecolor": _INK, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "xtick.color": _MUTED, "ytick.color": _MUTED, "grid.color": _GRID, "grid.alpha": 1.0, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
}


def _styled(plot):
    """Draw one figure in the chart style without changing matplotlib's settings for anything else."""
    @functools.wraps(plot)
    def wrapper(*args, **kwargs):
        with plt.rc_context(_STYLE):
            return plot(*args, **kwargs)
    return wrapper


def _rotator(grid):
    """Display rotation turning the domain's central meridian to point up.

    EPSG:3031 puts Greenwich at +y, so for this sector north would point to the
    upper left. Analysis stays in EPSG:3031; only the figure is rotated.
    """
    lam = np.deg2rad(float(np.median(grid.lon2d)))
    c, s = np.cos(lam), np.sin(lam)

    def rot(x, y):
        x, y = np.asarray(x, float) / 1e3, np.asarray(y, float) / 1e3
        return x * c - y * s, x * s + y * c

    return rot


def _graticule(ax, grid, rot) -> None:
    from antarctic_routing.preprocessing.grid import PolarGrid

    lats = np.arange(np.floor(grid.lat2d.min()), np.ceil(grid.lat2d.max()) + 1, 2)
    lons = np.arange(np.floor(grid.lon2d.min() / 5) * 5, np.ceil(grid.lon2d.max() / 5) * 5 + 1, 5)
    for lat in lats:
        lo = np.linspace(lons.min(), lons.max(), 200)
        x, y = rot(*PolarGrid.to_xy(np.full_like(lo, lat), lo))
        ax.plot(x, y, color=_GRATICULE, lw=0.5, zorder=1)
        ax.annotate(f"{abs(lat):.0f}\u00b0S", (x[0], y[0]), fontsize=6, color=_MUTED)
    for lon in lons:
        la = np.linspace(lats.min(), lats.max(), 200)
        x, y = rot(*PolarGrid.to_xy(la, np.full_like(la, lon)))
        ax.plot(x, y, color=_GRATICULE, lw=0.5, zorder=1)
        ax.annotate(f"{abs(lon):.0f}\u00b0W", (x[-1], y[-1]), fontsize=6, color=_MUTED)


@_styled
def plot_plan(world: ScenarioSet, plan: PlanResult, tau: float, path: str | Path, title: str) -> Path:
    grid = world.grid
    rot = _rotator(grid)
    half = grid.resolution_m / 2
    xe = np.r_[grid.x - half, grid.x[-1] + half]
    ye = np.r_[grid.y - half, grid.y[-1] + half]
    XE, YE = rot(*np.meshgrid(xe, ye))
    hours = plan.recommended.evaluation.expected_hours if plan.recommended else 24.0
    t_idx = world.time_index(hours / 2 if np.isfinite(hours) else 0.0)
    p_ice = exceedance_probability(world.conc[:, t_idx], tau)

    fig, ax = plt.subplots(figsize=(9, 8.6), dpi=130)
    im = ax.pcolormesh(XE, YE, np.ma.masked_invalid(p_ice), cmap=_ICE, vmin=0, vmax=1, zorder=0)
    ax.pcolormesh(XE, YE, np.ma.masked_where(~world.land, world.land.astype(float)),
                  cmap=ListedColormap([_LAND]), zorder=2)
    _graticule(ax, grid, rot)
    if world.berg is not None:
        XC, YC = rot(*np.meshgrid(grid.x, grid.y))
        p_berg = world.berg[:, t_idx].mean(axis=0)
        if p_berg.max() > 0:
            ax.contourf(XC, YC, p_berg, levels=[0.05, 0.25, 0.5, 1.01], colors=["#c9c1ea", "#8f7fd0", _BERG],
                        alpha=0.75, zorder=2.5)
            ax.contour(XC, YC, p_berg, levels=[0.05], colors=[_BERG], linewidths=0.8, zorder=2.6)
            ax.plot([], [], color="#8f7fd0", lw=6, label="iceberg presence P >= 5/25/50%")

    def xy(cells):
        r = np.array([c[0] for c in cells])
        c = np.array([c[1] for c in cells])
        return rot(grid.x[c], grid.y[r])

    for i, cand in enumerate(plan.candidates):
        if cand is plan.recommended:
            continue
        x, y = xy(cand.route.cells)
        ax.plot(x, y, color=_ROUTE_COLOURS[i % len(_ROUTE_COLOURS)], lw=1.2, alpha=0.85, zorder=3,
                label=f"{cand.labels[0]} - P(breach) {cand.evaluation.p_breach:.0%}")
    if plan.recommended:
        x, y = xy(plan.recommended.route.cells)
        ev = plan.recommended.evaluation
        ax.plot(x, y, color=_ACCENT, lw=3.0, zorder=4,
                label=f"RECOMMENDED - {ev.expected_hours:.0f} h, fuel {ev.expected_fuel:.0f}, "
                      f"P(breach) {ev.p_breach:.1%} (UB {ev.p_breach_upper:.1%})")
    first = plan.candidates[0].route.cells if plan.candidates else None
    if first:
        ox, oy = xy([first[0]])
        dx, dy = xy([first[-1]])
        ax.scatter(ox, oy, marker="o", s=70, color=_INK, edgecolor="white", zorder=5, label="Origin")
        ax.scatter(dx, dy, marker="o", s=90, color="white", edgecolor=_INK, linewidth=2, zorder=5,
                   label="Destination")

    ax.set_xlim(XE.min(), XE.max())
    ax.set_ylim(YE.min(), YE.max())
    ax.set_aspect("equal")
    ax.set_xlabel("km (EPSG:3031, display rotated: local north up)")
    ax.set_ylabel("km")
    ax.set_title(f"{title}\nstatus: {plan.status.upper()}  |  background: P(C >= {tau:.2f}) at day {t_idx}",
                 fontsize=10)
    ax.set_facecolor("#e2e9ed")
    ax.legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.95, edgecolor="#cfd8de")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="P(ice concentration >= vessel limit)")
    fig.text(0.01, 0.005, f"{world.execution_mode.upper()} DATA - {DISCLAIMER}", fontsize=6, color="#555", wrap=True)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


@_styled
def plot_departures(sweep: DepartureSweep, risk_budget: float, path: str | Path, title: str,
                    issue=None, forecast_days: int | None = None, trust_horizon_days: int | None = None) -> Path:
    opts = sweep.options
    dates = [o.departure for o in opts]
    ub = [min(o.p_breach_upper, 1.0) for o in opts]
    hours = [o.expected_hours if np.isfinite(o.expected_hours) else np.nan for o in opts]
    fuel = [o.expected_fuel if np.isfinite(o.expected_fuel) else np.nan for o in opts]
    ok = [o.feasible for o in opts]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), dpi=130, sharex=True)
    bars = axes[0].bar(dates, ub, color=[_BAR if f else _CRITICAL for f in ok], width=0.8)
    for bar, opt in zip(bars, opts, strict=True):
        if opt.support == "climatology-dominated":
            bar.set_hatch("///")
            bar.set_edgecolor("white")
    top = axes[0].get_xaxis_transform()  # x in data, y in axes fraction
    if issue is not None:
        from datetime import timedelta

        marks: dict = {}
        if forecast_days is not None:
            marks.setdefault(issue + timedelta(days=forecast_days), []).append(("forecast", forecast_days))
        if trust_horizon_days is not None:
            marks.setdefault(issue + timedelta(days=trust_horizon_days), []).append(("trust", trust_horizon_days))
        for when, kinds in marks.items():
            colour = _BERG if any(k == "trust" for k, _ in kinds) else _BLUE
            label = " & ".join(k for k, _ in kinds) + f" horizon ({kinds[0][1]} d)"
            for ax in axes:
                ax.axvline(when, color=colour, ls="--", lw=1.2)
            axes[0].text(when, 1.01, label, transform=top, color=colour, fontsize=7, ha="center", va="bottom")
    handles = []
    if any(o.support == "climatology-dominated" for o in opts):
        from matplotlib.patches import Patch

        handles.append(Patch(facecolor="#888", hatch="///", edgecolor="white", label="climatology-dominated voyage"))
    budget_line = axes[0].axhline(risk_budget, color=_INK, ls="--", lw=1, label=f"risk budget {risk_budget:.0%}")
    axes[0].set_ylabel("P(breach)\nupper bound")
    axes[0].set_ylim(0, min(1.05, max(0.12, 1.3 * max([*ub, risk_budget]))))
    axes[0].legend(handles=[budget_line, *handles], fontsize=8)
    axes[1].plot(dates, hours, marker="o", ms=4, color=_BLUE)
    axes[1].set_ylabel("E[voyage] (h)")
    axes[2].plot(dates, fuel, marker="o", ms=4, color=_OCHRE)
    axes[2].set_ylabel("E[fuel index]")
    if sweep.selected:
        for ax in axes:
            ax.axvline(sweep.selected.departure, color=_ACCENT, lw=2, alpha=0.7)
        axes[0].text(sweep.selected.departure, 0.95, " selected", transform=top, color=_ACCENT, fontsize=8,
                     va="top")
    axes[0].set_title(f"{title}\n{sweep.rule}", fontsize=9, pad=16)
    for ax in axes:
        ax.grid(axis="y")
    fig.autofmt_xdate()
    fig.text(0.01, 0.005, DISCLAIMER, fontsize=6, color="#555", wrap=True)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


_METHOD_STYLE = {
    "persistence": ("#7b8794", "--"),
    "climatology": (_OCHRE, ":"),
    "damped_persistence": (_BLUE, "-."),
}


@_styled
def plot_forecast_skill(report: dict, path: str | Path, title: str) -> Path:
    """MAE and IIEE against lead time for the model and every baseline."""
    leads = report["leads"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130)
    for method in report["methods"]:
        colour, ls = _METHOD_STYLE.get(method, (_ACCENT, "-"))
        lw = 2.6 if method not in _METHOD_STYLE else 1.5
        axes[0].plot(leads, report["mae"][method], ls, color=colour, lw=lw, marker="o", ms=3, label=method)
        axes[1].plot(leads, np.asarray(report["iiee_km2"][method]) / 1e3, ls, color=colour, lw=lw,
                     marker="o", ms=3, label=method)
    axes[0].set_ylabel("MAE (concentration fraction)")
    axes[1].set_ylabel(f"IIEE (10^3 km^2, edge at C >= {report['edge_threshold']})")
    for ax in axes:
        ax.set_xlabel("lead time (days)")
        ax.grid()
        ax.set_xticks(leads)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{title}\n{report['execution_mode'].upper()} - test seasons {report['test_seasons']}, "
                 f"{report['n_samples']} forecasts, rho={report['rho']:.3f}", fontsize=10)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


@_styled
def plot_reliability(report: dict, path: str | Path, title: str) -> Path:
    """Reliability diagram (raw vs calibrated vs climatology) and Brier score per lead."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=130)
    styles = {"raw": ("#7b8794", "o--"), "calibrated": (_ACCENT, "o-"), "climatology": (_OCHRE, "s:")}
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=_INK, lw=0.8, alpha=0.6, label="perfect reliability")
    for name, (colour, fmt) in styles.items():
        rows = [r for r in report["reliability"][name] if r["count"]]
        ax.plot([r["mean_predicted"] for r in rows], [r["observed_frequency"] for r in rows], fmt,
                color=colour, ms=4, label=name)
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title(f"Reliability, P(C >= {report['tau']}) pooled over leads", fontsize=9)
    ax = axes[1]
    for name, (colour, fmt) in styles.items():
        ax.plot(report["leads"], report["brier"][name], fmt, color=colour, ms=4, label=name)
    ax.set_xlabel("lead time (days)")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_xticks(report["leads"])
    ax.grid()
    ax.legend(fontsize=8)
    fig.suptitle(f"{title}\n{report['execution_mode'].upper()} - calibrated on {report['val_seasons']}, "
                 f"tested on {report['test_seasons']}, {report['n_members']} members", fontsize=10)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


@_styled
def plot_trust_horizon(result: dict, path: str | Path, title: str) -> Path:
    """Delta(h) = E_baseline - E_model with bootstrap bands and the trust-horizon marker."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=130)
    colours = {"damped_persistence": _BLUE, "climatology": _OCHRE, "persistence": "#7b8794"}
    for name, th in result["by_baseline"].items():
        c = colours.get(name, "#555")
        ax.fill_between(th["leads"], th["ci_low"], th["ci_high"], color=c, alpha=0.18)
        ax.plot(th["leads"], th["delta"], "o-", color=c, ms=4,
                label=f"vs {name}: trust horizon {th['trust_horizon_days']} d")
    ax.axhline(result["min_delta"], color=_INK, lw=0.8, ls="--", label="required improvement")
    h = result["trust_horizon_days"]
    if h:
        ax.axvspan(0.5, h + 0.5, color=_GOOD, alpha=0.08)
        limited = any(t["limited_by_max_lead"] for t in result["by_baseline"].values())
        label = f"trusted >= {h} d (limit of evaluated leads)" if limited else f"trusted up to {h} d"
        ax.text(h + 0.45, ax.get_ylim()[1] * 0.92, label, ha="right", color=_GOOD, fontsize=9)
    ax.set_xlabel("lead time (days)")
    ax.set_ylabel("MAE improvement over baseline")
    ax.set_xticks(result["leads"])
    ax.grid()
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"{title}\n{result['execution_mode'].upper()} - {result['method']}, "
                 f"{result['n_seasons']} seasons, {int(100 * (1 - result['alpha']))}% lower bound", fontsize=9)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


@_styled
def plot_replay(result: dict, ctx, tau: float, path: str | Path, title: str) -> Path:
    """Sailed vs naive track over observed ice, and how the departure advice evolved."""
    from datetime import date as _date

    from antarctic_routing.forecasting.scenarios import grid_of

    grid = grid_of(ctx.ds)
    rot = _rotator(grid)
    half = grid.resolution_m / 2
    XE, YE = rot(*np.meshgrid(np.r_[grid.x - half, grid.x[-1] + half], np.r_[grid.y - half, grid.y[-1] + half]))
    land = ctx.ds["land_mask"].values.astype(bool)
    dep = result.get("departure") or result["start"]
    obs = ctx.ds["ice_concentration"].values[ctx.index_of(_date.fromisoformat(dep))]

    fig, (ax, tx) = plt.subplots(1, 2, figsize=(14, 6.4), dpi=120, gridspec_kw={"width_ratios": [1.15, 1]})
    ax.pcolormesh(XE, YE, np.ma.masked_where(land | ~np.isfinite(obs), np.nan_to_num(obs) >= tau),
                  cmap=ListedColormap([_SEA, "#275a8f"]), vmin=0, vmax=1, zorder=0)
    ax.pcolormesh(XE, YE, np.ma.masked_where(~land, land.astype(float)), cmap=ListedColormap([_LAND]), zorder=1)

    def xy(cells):
        r = np.array([c[0] for c in cells])
        c = np.array([c[1] for c in cells])
        return rot(grid.x[c], grid.y[r])

    if result.get("naive_cells"):
        nv = result["truth"]["naive"]
        ax.plot(*xy(result["naive_cells"]), "--", color=_INK, lw=1.6,
                label=f"naive: leave {nv['departure']}, shortest route - "
                      f"{nv['observed_breach_cells']} cells in observed ice")
    if result.get("sailed_cells"):
        pl = result["truth"]["planner"]
        ax.plot(*xy(result["sailed_cells"]), color=_ACCENT, lw=2.6,
                label=f"planner: leave {pl['departure']} - {pl['observed_breach_cells']} cells in observed ice")
        for rec in result["days"]:
            if rec.get("position"):
                px, py = rot(*grid.to_xy(*rec["position"]))
                ax.scatter(px, py, s=28, color="white", edgecolor=_ACCENT, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(XE.min(), XE.max())
    ax.set_ylim(YE.min(), YE.max())
    ax.set_facecolor("#e2e9ed")
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    ax.legend(loc="lower left", fontsize=7.5, frameon=True, framealpha=0.95, edgecolor="#cfd8de")
    ax.set_title(f"Observed ice >= {tau} on departure day ({dep}); dots = daily positions", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    port = [r for r in result["days"] if r["phase"] == "port"]
    issue = [_date.fromisoformat(r["day"]) for r in port]
    rec = [_date.fromisoformat(r["recommended_departure"]) if r["recommended_departure"] else None for r in port]
    ok = [(i, d) for i, d in zip(issue, rec, strict=True) if d is not None]
    none = [i for i, d in zip(issue, rec, strict=True) if d is None]
    if ok:
        tx.scatter([i for i, _ in ok], [d for _, d in ok], color=_GOOD, s=40,
                   label="recommended departure", zorder=3)
    if none:
        tx.scatter(none, none, marker="x", color=_CRITICAL, s=40, label="no date in window meets the budget", zorder=3)
    if issue:
        tx.plot([issue[0], issue[-1]], [issue[0], issue[-1]], color=_INK, lw=0.8, alpha=0.5, label="today")
    if result.get("departure"):
        tx.axhline(_date.fromisoformat(result["departure"]), color=_ACCENT, lw=1.2, ls="--", label="actual departure")
    tx.set_xlabel("forecast issue day (in port)")
    tx.set_ylabel("recommended departure date")
    tx.grid()
    tx.legend(fontsize=8, loc="lower right")
    at_sea = [r for r in result["days"] if r["phase"] == "at_sea"]
    switches = sum(r["action"] == "switch" for r in at_sea)
    tx.set_title(f"Departure advice as information arrived; at sea: {len(at_sea)} replans, {switches} route switch(es)",
                 fontsize=9)
    fig.autofmt_xdate()
    fig.suptitle(f"{title}\n{result['execution_mode'].upper()} - only information available each day was used",
                 fontsize=10)
    fig.text(0.01, 0.005, DISCLAIMER, fontsize=6, color="#555")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


@_styled
def plot_backtest(result: dict, path: str | Path, title: str) -> Path:
    """Per-method comparison: breach rate, hazard hours, elapsed time, fuel."""
    methods = ["planner", "naive", "ice_edge_buffer"]
    labels = {"planner": "this system", "naive": "naive (leave day 1,\nshortest route)",
              "ice_edge_buffer": f"ice-edge buffer\n({result['buffer_km']:g} km, no replan)"}
    colours = {"planner": _ACCENT, "naive": "#7b8794", "ice_edge_buffer": _BAR}
    panels = [("breach_rate", "voyages meeting observed ice >= tau", 100.0, "%"),
              ("mean_hazard_hours", "hours in observed hazardous ice", 1.0, "h"),
              ("mean_elapsed_hours", "start-to-arrival time (wait + sail)", 1 / 24.0, "days"),
              ("mean_fuel_index", "fuel index (sailing only)", 1.0, "")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), dpi=120)
    for ax, (key, name, scale, unit) in zip(axes, panels, strict=True):
        vals = [result["summary"][m][key] for m in methods]
        vals = [v * scale if v is not None else 0.0 for v in vals]
        bars = ax.bar([labels[m] for m in methods], vals, color=[colours[m] for m in methods])
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}{unit}", ha="center", va="bottom",
                    fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylim(0, max(vals + [1e-9]) * 1.2 or 1)
    rc = result["planner_risk_check"]
    n = result["summary"]["planner"]["n"]
    note = ""
    if rc["n_departures"]:
        note = (f"planner: mean predicted P(breach) {rc['mean_predicted_p_breach']:.1%} "
                f"(upper bound {rc['mean_predicted_upper_bound']:.1%}) vs realised {rc['realised_breach_rate']:.0%} "
                f"over {rc['n_departures']} departures")
    fig.suptitle(f"{title}\n{result['execution_mode'].upper()} - {n} start dates in held-out seasons. {note}",
                 fontsize=10)
    fig.tight_layout()
    fig.text(0.01, -0.02, DISCLAIMER, fontsize=6, color="#555")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


@_styled
def plot_sensitivity(result: dict, path: str | Path, title: str) -> Path:
    """Heatmaps over (lambda, k): route deviation from the baseline and expected fuel."""
    lams = sorted({r["lambda"] for r in result["grid"]})
    ks = sorted({r["k"] for r in result["grid"]})
    dev = np.full((len(lams), len(ks)), np.nan)
    fuel = np.full_like(dev, np.nan)
    for r in result["grid"]:
        i, j = lams.index(r["lambda"]), ks.index(r["k"])
        dev[i, j] = r["deviation_km"] if r["deviation_km"] is not None else np.nan
        fuel[i, j] = r["expected_fuel"] if r["expected_fuel"] is not None else np.nan
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=120)
    for ax, data, name, cmap in ((axes[0], dev, "route deviation from baseline (km)", _ICE),
                                 (axes[1], fuel, "expected fuel index", _ICE)):
        im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower")
        for i in range(len(lams)):
            for j in range(len(ks)):
                if np.isfinite(data[i, j]):
                    dark = np.nanmax(data) > np.nanmin(data) and \
                        (data[i, j] - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data)) > 0.55
                    ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if dark else _INK)
        ax.set_xticks(range(len(ks)), [f"{k:g}" for k in ks])
        ax.set_yticks(range(len(lams)), [f"{v:g}" for v in lams])
        ax.set_xlabel("speed-in-ice factor k")
        ax.set_ylabel("fuel penalty lambda")
        ax.set_title(name, fontsize=9)
        b = result["baseline"]
        if b["lambda"] in lams and b["k"] in ks:
            ax.add_patch(plt.Rectangle((ks.index(b["k"]) - 0.5, lams.index(b["lambda"]) - 0.5), 1, 1,
                                       fill=False, ec=_ACCENT, lw=2))
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{title}\n{result['execution_mode'].upper()} - recommendation unchanged in "
                 f"{result['stable_fraction']:.0%} of settings (route moves > {result['change_threshold_km']:g} km "
                 "counts as changed); outlined box = configured assumption", fontsize=10)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
