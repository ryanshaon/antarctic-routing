"""Stage 11.3 - PDF voyage brief.

Page 1 summarises the decision (departure, recommended route, expected time,
fuel index, risk-budget status, data mode, key assumptions, limitations);
page 2 is the route map; page 3 compares every candidate evaluated on the same
joint scenarios.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from antarctic_routing import DISCLAIMER, __version__  # noqa: E402
from antarctic_routing.common.provenance import utc_now  # noqa: E402
from antarctic_routing.config import ProjectConfig  # noqa: E402
from antarctic_routing.routing.candidates import PlanResult  # noqa: E402
from antarctic_routing.synthetic import ScenarioSet  # noqa: E402

_A4 = (8.27, 11.69)


def _text_page(pdf: PdfPages, title: str, lines: list[tuple[str, str]], notes: list[str],
               row_gap: float = 0.01) -> None:
    fig = plt.figure(figsize=_A4)
    fig.text(0.08, 0.95, title, fontsize=16, weight="bold")
    fig.text(0.08, 0.925, f"Generated {utc_now()} - antarctic-routing {__version__}", fontsize=8, color="#555")
    y = 0.89
    for key, value in lines:
        wrapped = textwrap.fill(str(value), width=62)
        fig.text(0.08, y, key, fontsize=10, color="#52514e", va="top")
        fig.text(0.38, y, wrapped, fontsize=10, va="top")
        y -= 0.026 * (wrapped.count("\n") + 1) + row_gap
    y -= 0.02
    fig.text(0.08, y, "Assumptions and limitations", fontsize=11, weight="bold", va="top")
    y -= 0.03
    for note in notes:
        wrapped = textwrap.fill("- " + note, width=105, subsequent_indent="  ")
        height = 0.02 * (wrapped.count("\n") + 1) + 0.008
        if y - height < 0.07:      # never run into the disclaimer
            fig.text(0.09, y, "- (further notes are listed with the plan's data sources)", fontsize=8.5, va="top")
            break
        fig.text(0.09, y, wrapped, fontsize=8.5, va="top")
        y -= height
    fig.text(0.08, 0.06, textwrap.fill(DISCLAIMER, width=110), fontsize=8, color="#d03b3b", va="top")
    pdf.savefig(fig)
    plt.close(fig)


def write_brief(
    path: str | Path,
    cfg: ProjectConfig,
    world: ScenarioSet,
    plan: PlanResult,
    map_png: str | Path,
    trust_horizon_days: int | None = None,
    trust_limited: bool = False,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = plan.recommended
    ev = rec.evaluation if rec else None
    lines = [
        ("Route", f"({cfg.route.origin.lat}, {cfg.route.origin.lon}) -> "
                  f"({cfg.route.destination.lat}, {cfg.route.destination.lon})"),
        ("Departure (UTC)", world.start.isoformat()),
        ("Decision", "RECOMMENDED ROUTE" if rec else "NO ROUTE MEETS THE RISK BUDGET"),
        ("Risk budget", f"{cfg.routing.risk_budget:.1%} via {cfg.routing.risk_estimator} at "
                        f"{cfg.routing.confidence:.0%} confidence"),
        ("Joint scenarios", str(world.n_scenarios)),
        ("Data mode", world.execution_mode.upper()),
    ]
    if ev:
        lines += [
            ("Route labels", ", ".join(rec.labels)),
            ("P(breach)", f"{ev.p_breach:.1%} (95% upper bound {ev.p_breach_upper:.1%})"),
            ("Expected time", f"{ev.expected_hours:.1f} h (P10-P90 {ev.hours_p10:.1f}-{ev.hours_p90:.1f} h)"),
            ("Fuel index", f"{ev.expected_fuel:.0f} (relative, not litres)"),
            ("Distance", f"{ev.distance_km:.0f} km"),
            ("Beyond scenario horizon", f"{ev.beyond_horizon_fraction:.0%} of scenarios"),
        ]
    if trust_horizon_days is not None:
        lines.append(("Forecast trust horizon", f">= {trust_horizon_days} days (limited by evaluated leads)"
                      if trust_limited else f"{trust_horizon_days} days"))
    notes = [
        f"Vessel ice limit tau_v = {cfg.vessel.max_ice_concentration} and ice class "
        f"'{cfg.vessel.ice_class}' are placeholders; derive them from verified vessel capability.",
        "Fuel is a relative index d[1 + lambda C^2]; speed in ice v(1 - kC) is an assumption (see sensitivity).",
        "Route risk is evaluated jointly across scenarios; it is a model estimate, not a guarantee.",
        "Tracked icebergs (USNIC) cover giant bergs only; smaller bergs and growlers are not represented.",
        "Geography in controlled-synthetic runs is schematic and must not be used for navigation.",
        plan.explanation,
    ]
    with PdfPages(out) as pdf:
        _text_page(pdf, "Voyage brief - Antarctic ice-risk routing", lines, notes)
        fig = plt.figure(figsize=_A4)
        img = plt.imread(str(map_png))
        ax = fig.add_axes([0.04, 0.1, 0.92, 0.82])
        ax.imshow(img)
        ax.axis("off")
        fig.text(0.08, 0.95, "Route map", fontsize=14, weight="bold")
        pdf.savefig(fig)
        plt.close(fig)
        fig = plt.figure(figsize=_A4)
        fig.text(0.08, 0.95, "Candidate routes (same joint scenarios)", fontsize=14, weight="bold")
        ax = fig.add_axes([0.05, 0.3, 0.9, 0.6])
        ax.axis("off")
        rows = [[", ".join(c.labels)[:38], "yes" if c.feasible else "no", f"{c.evaluation.distance_km:.0f}",
                 f"{c.evaluation.expected_hours:.1f}", f"{c.evaluation.expected_fuel:.0f}",
                 f"{c.evaluation.p_breach:.1%}", f"{c.evaluation.p_breach_upper:.1%}"] for c in plan.candidates]
        table = ax.table(cellText=rows, colLabels=["route", "meets budget", "km", "E[h]", "E[fuel]", "P(breach)",
                                                   "95% UB"], loc="upper center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)
        pdf.savefig(fig)
        plt.close(fig)
        info = pdf.infodict()
        info["Title"] = "Antarctic voyage brief"
        info["Subject"] = DISCLAIMER
    return out


# --------------------------------------------------------------------------- brief of a /real/plan result
_STATUS = {"recommended": "RECOMMENDED DEPARTURE AND ROUTE",
           "no_feasible_departure": "NO DEPARTURE MEETS THE RISK BUDGET (least risky option shown)",
           "no_route": "NO ROUTE"}
_INK, _ROUTE, _MUTED = "#1f3a5f", "#b5179e", "#8a94a0"
_SOURCE = {"observed": "observed", "forecast": "forecast", "climatology": "climatology",
           "proxy_analogue_observed": "the analogue season's observation, a proxy"}


def _strategies(labels: list[str]) -> str:
    """The first search strategy that found a route, and how many others found the same one."""
    first = labels[0].replace("_", " ") if labels else "-"
    return first if len(labels) < 2 else f"{first} (+{len(labels) - 1} more)"


def _pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


def _risk_line(r: dict) -> str:
    return (f"{r['breaches']} of {r['n_scenarios']} scenarios ({_pct(r['p_breach'])}, 95% upper bound "
            f"{_pct(r['p_breach_upper'])})")


def _plan_lines(plan: dict) -> list[tuple[str, str]]:
    meta, loc, route, risk = plan["metadata"], plan["locations"], plan.get("route"), plan.get("risk")
    options = plan["departure"]["options"]
    lines = [
        ("Route", f"{loc['origin'].get('name') or '-'} -> {loc['destination'].get('name') or '-'}"),
        ("Data", f"{meta.get('label') or meta['mode']} ({meta.get('n_scenarios', '-')} joint scenarios)"),
        ("Forecast issue date", str(meta.get("issue_date"))),
        ("Decision", _STATUS.get(plan["status"], plan["status"])),
        ("Departure window", f"{sum(bool(o['feasible']) for o in options)} of {len(options)} departure dates "
                             "meet the risk budget"),
    ]
    if route:
        fuel, hours = route["fuel_index"], route["expected_hours"]
        lines += [
            ("Departure (UTC)", route["departure_utc"]),
            ("Expected arrival (UTC)", route.get("eta_utc") or "-"),
            ("Expected time", "-" if hours is None else
             f"{hours:.1f} h (P10-P90 {route['hours_p10']:.1f}-{route['hours_p90']:.1f} h)"),
            ("Distance", f"{route['distance_km']:.0f} km"),
            ("Fuel index", "-" if fuel["expected"] is None else f"{fuel['expected']:.0f} (relative, not litres)"),
        ]
    if risk:
        within = "within" if risk["combined"].get("within_budget") else "EXCEEDS"
        lines += [
            ("Combined risk (decision)", f"{_risk_line(risk['combined'])}: {within} the "
                                         f"{_pct(risk['risk_budget'])} budget"),
            ("Sea-ice risk", _risk_line(risk["sea_ice"])),
            ("Iceberg risk", _risk_line(risk["iceberg"])),
        ]
    return lines


def _plan_notes(plan: dict) -> list[str]:
    meta = plan["metadata"]
    notes = [plan.get("explanation"), meta.get("hindsight_disclosure"), meta.get("forecast_disclosure"),
             *((meta.get("provenance") or {}).get("limitations") or [])]
    return [str(n) for n in notes if n][:7]


def _map_page(pdf: PdfPages, plan: dict) -> None:
    import numpy as np
    from matplotlib.colors import ListedColormap

    layers, route = plan["layers"], plan["route"]
    g, layer = layers["grid"], layers["layers"][0]
    x, y = np.asarray(g["x_km"], float), np.asarray(g["y_km"], float)
    half = g["res_km"] / 2
    c, s = np.cos(g["rotation_rad"]), np.sin(g["rotation_rad"])

    def rot(px, py):       # display only: the domain's central meridian points up, as in the dashboard
        px, py = np.asarray(px, float), np.asarray(py, float)
        return px * c - py * s, px * s + py * c

    XE, YE = rot(*np.meshgrid(np.r_[x - half, x[-1] + half], np.r_[y - half, y[-1] + half]))
    shape = (g["ny"], g["nx"])
    p_ice = np.array([np.nan if v is None else v for v in layer["p_ice_ge_limit_pct"]], float).reshape(shape)
    land = np.array(layers["land"], bool).reshape(shape)
    others = [a["xy_km"] for a in plan.get("alternatives") or [] if a.get("xy_km") and a["xy_km"] != route["xy_km"]]

    fig = plt.figure(figsize=_A4)
    fig.text(0.08, 0.95, "Route map", fontsize=14, weight="bold")
    fig.text(0.08, 0.935, f"Background: probability that sea ice reaches the vessel limit on {layer['date']} "
                          f"({_SOURCE.get(layer.get('source'), 'scenario layer')}).\n25 km cells; not a "
                          "navigational chart.", fontsize=8, color="#555", va="top")
    ax = fig.add_axes([0.08, 0.2, 0.84, 0.7])
    im = ax.pcolormesh(XE, YE, np.ma.masked_invalid(p_ice), cmap="Blues", vmin=0, vmax=100, zorder=0)
    ax.pcolormesh(XE, YE, np.ma.masked_where(~land, land.astype(float)), cmap=ListedColormap(["#e4d9bd"]), zorder=1)
    for i, xy in enumerate(others):
        ax.plot(*rot(*zip(*xy, strict=True)), color=_MUTED, lw=1.0, zorder=2,
                label="Other candidate routes" if i == 0 else None)
    rx, ry = rot(*zip(*route["xy_km"], strict=True))
    ax.plot(rx, ry, color=_ROUTE, lw=2.4, zorder=3, label="Selected route")
    ax.scatter(rx[:1], ry[:1], marker="o", s=60, color="white", edgecolor=_INK, linewidth=1.5, zorder=4, label="Start")
    ax.scatter(rx[-1:], ry[-1:], marker="s", s=60, color=_INK, zorder=4, label="Destination")
    ax.set_xlim(XE.min(), XE.max())
    ax.set_ylim(YE.min(), YE.max())
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.035, pad=0.03, label="P(ice >= vessel limit), %")
    fig.text(0.08, 0.06, textwrap.fill(DISCLAIMER, width=110), fontsize=8, color="#d03b3b", va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _comparison_page(pdf: PdfPages, plan: dict) -> None:
    options, risk = plan["departure"]["options"], plan.get("risk") or {}
    selected = (plan.get("route") or {}).get("departure_date")
    fig = plt.figure(figsize=_A4)
    fig.text(0.08, 0.95, "Departure window", fontsize=14, weight="bold")
    fig.text(0.08, 0.935, "95% upper bound of the breach probability for each departure date, from one scenario "
                          "set.\nMagenta: the selected date. Hatched: exceeds the budget.",
             fontsize=8, color="#555", va="top")
    ax = fig.add_axes([0.1, 0.62, 0.82, 0.27])
    for i, o in enumerate(options):
        chosen, ok = o["departure"] == selected, bool(o["feasible"])
        ax.bar(i, 100 * (o["p_breach_upper"] or 0), width=0.7, color=_ROUTE if chosen else _INK if ok else "white",
               edgecolor=_ROUTE if chosen else _INK if ok else _MUTED, hatch=None if ok or chosen else "////")
    if risk.get("risk_budget") is not None:
        ax.axhline(100 * risk["risk_budget"], color="#d03b3b", lw=1, ls="--")
        ax.set_title(f"dashed line: risk budget {_pct(risk['risk_budget'])}", loc="right", fontsize=7,
                     color="#d03b3b")
    ax.set_xticks(range(len(options)), [o["departure"][5:] for o in options], fontsize=7, rotation=45)
    ax.set_ylabel("upper bound, %", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    alts = plan.get("alternatives") or []
    if alts:
        fig.text(0.08, 0.54, "Candidate routes for the selected departure (same joint scenarios)",
                 fontsize=11, weight="bold")
        tax = fig.add_axes([0.06, 0.1, 0.88, 0.42])
        tax.axis("off")
        rows = [[_strategies(a["labels"]), "yes" if a["feasible"] else "no", f"{a['distance_km']:.0f}",
                 "-" if a["expected_hours"] is None else f"{a['expected_hours']:.1f}",
                 "-" if a["expected_fuel"] is None else f"{a['expected_fuel']:.0f}",
                 f"{a['breaches']} of {a['n_scenarios']}", _pct(a["p_breach_upper"])] for a in alts]
        table = tax.table(cellText=rows, colLabels=["search strategy", "meets budget", "km", "hours", "fuel index",
                                                    "breaches", "95% bound"],
                          loc="upper center", cellLoc="left", colWidths=[0.36, 0.12, 0.08, 0.09, 0.11, 0.13, 0.11])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)
    fig.text(0.08, 0.06, textwrap.fill(DISCLAIMER, width=110), fontsize=8, color="#d03b3b", va="top")
    pdf.savefig(fig)
    plt.close(fig)


def plan_brief(plan: dict) -> bytes:
    """A PDF brief of one ``POST /real/plan`` response: decision, route map, departure window and candidates.

    Every number is read from the response; nothing is recomputed."""
    import io

    buf = io.BytesIO()
    with PdfPages(buf, metadata={"Title": "Antarctic voyage brief", "Subject": DISCLAIMER,
                                 "Creator": f"antarctic-routing {__version__}"}) as pdf:
        _text_page(pdf, "Voyage brief - Antarctic ice-risk routing", _plan_lines(plan), _plan_notes(plan),
                   row_gap=0.003)
        if plan.get("route") and plan.get("layers"):
            _map_page(pdf, plan)
        _comparison_page(pdf, plan)
    return buf.getvalue()
