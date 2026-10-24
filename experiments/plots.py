"""Every figure in results: experiment 1's validation and ranking, experiment 2's scale."""

from __future__ import annotations

import csv
import os
from typing import Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from auction.game import REIMBURSEMENT_POLICIES, VALIDATED_POLICIES, Auction, closed_form_revenue, region
from auction.mccfr import REVENUE_BAR, STAY_BAR, Verdict, closed_forms, stay_schedule
from auction.exact import CERTIFIED
from experiments.scale import COUNTS, claims
from experiments.two_bidder import CELLS, RESULTS, SIGNAL_PROBABILITIES, margin

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CORNER_COLOR = "#9a9993"  # the unnamed corners share one recessive grey
RULE_COLOR = {"zero": "#2a78d6", "full": "#eb6834", "war_all": "#1baf7a", "backloaded": "#c2185b"}
RULE_COLOR |= {rule: CORNER_COLOR for rule in REIMBURSEMENT_POLICIES if rule not in RULE_COLOR}
RULE_LABEL = {"zero": "$r=0$", "full": "$r=c$", "war_all": "$(0,c,c,c)$", "backloaded": "back-loaded"}
Q_COLOR = {0.7: "#6da7ec", 0.75: "#2a78d6", 0.9: "#104281"}
SCHEDULE_MARKER = {"open_high": "^", "open_low": "v", "sigma1": "o", "sigma2": "s", "sigma3": "D"}
SCHEDULE_LABEL = {
    "open_high": "open, high",
    "open_low": "open, low",
    "sigma1": r"$\sigma_1$",
    "sigma2": r"$\sigma_2$",
    "sigma3": r"$\sigma_3$",
}


def span_of(q: float) -> float:
    """Each q's x range, wide enough to hold its outside placement."""
    return 1.6 * region(q)


def theory_schedule(q: float, c: float) -> Dict[str, float]:
    """The closed-form stay probabilities under no reimbursement at one cell."""
    auction = Auction(q=q, c=c, reimbursement_policy="zero")
    return stay_schedule(auction, closed_forms(auction)[0])


def style(axis: plt.Axes, title: str, xlabel: str, ylabel: str, bar: float) -> None:
    """Square axes around y = x, with the band the verdicts tolerate shaded about it."""
    low, high = axis.get_xlim()
    axis.fill_between([low, high], [low - bar, high - bar], [low + bar, high + bar],
                      color=GRID, alpha=0.6, linewidth=0, zorder=0)
    axis.plot([low, high], [low, high], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=0)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal")
    chrome(axis, title, xlabel, ylabel)


def chrome(axis: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Recessive grid, spines and ticks, shared by every panel."""
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    axis.set_title(title, loc="left", fontsize=10, color=INK)
    axis.set_xlabel(xlabel, fontsize=9, color=INK)
    axis.set_ylabel(ylabel, fontsize=9, color=INK)


def revenue_panel(axis: plt.Axes, runs: List[Dict[str, object]]) -> None:
    """Learned seller net revenue against the closed form, verdict carried by the marker fill."""
    for rule in VALIDATED_POLICIES:
        color = RULE_COLOR[rule]
        for row in (row for row in runs if row["reimbursement_policy"] == rule):
            verdict = Verdict(row["verdict"])
            face = {Verdict.DERIVED: color, Verdict.DIFFERENT: "white", Verdict.UNSOLVED: "none"}[verdict]
            marker = "x" if verdict is Verdict.UNSOLVED else "o"
            edge = color if verdict is not Verdict.UNSOLVED else None
            axis.scatter(
                row["Pi_closed_form"], row["Pi"], s=22, marker=marker, facecolors=face if marker == "o" else color,
                edgecolors=edge, linewidths=1.0, alpha=0.8,
            )
    axis.set_xlim(0.0, 1.5)
    style(axis, "A. Seller net revenue", r"$\Pi_{\mathrm{theory}}$", r"$\Pi_{\mathrm{MCCFR}}$", REVENUE_BAR)
    handles = [Line2D([], [], marker="o", linestyle="", color=RULE_COLOR[rule], label=RULE_LABEL[rule])
               for rule in VALIDATED_POLICIES]
    handles += [
        Line2D([], [], marker="o", linestyle="", color=MUTED, label="matches theory"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=MUTED, label="different equilibrium"),
        Line2D([], [], marker="x", linestyle="", color=MUTED, label="no equilibrium"),
    ]
    axis.legend(handles=handles, loc="upper left", fontsize=7.5, frameon=False, labelcolor=INK)


def schedule_panel(axis: plt.Axes, runs: List[Dict[str, object]]) -> None:
    """Learned stay probabilities under no reimbursement, the one rule whose closed form mixes."""
    zero = [row for row in runs if row["reimbursement_policy"] == "zero"]
    targets = {(row["q"], row["c"]): theory_schedule(row["q"], row["c"]) for row in zero}
    for row in zero:
        target = targets[(row["q"], row["c"])]
        for name, marker in SCHEDULE_MARKER.items():
            axis.scatter(target[name], row[name], s=22, marker=marker, color=Q_COLOR[row["q"]],
                         edgecolors="white", linewidths=0.5, alpha=0.85)
    axis.set_xlim(-0.05, 1.05)
    style(axis, "B. Stay probabilities, $r=0$", "closed form", "MCCFR", STAY_BAR)
    handles = [Line2D([], [], marker="s", linestyle="", color=color, label=f"$q={q:g}$")
               for q, color in Q_COLOR.items()]
    handles += [Line2D([], [], marker=marker, linestyle="", color=MUTED, label=SCHEDULE_LABEL[name])
                for name, marker in SCHEDULE_MARKER.items()]
    axis.legend(handles=handles, loc="upper left", fontsize=7.5, frameon=False, labelcolor=INK)


# Per-placement panels show only these, since neighbouring placements crowd each other.
PLACEMENT_COLOR = {"inside": "#6da7ec", "bound": "#2a78d6", "outside": "#104281"}


def trace_panels() -> None:
    """Utility loss along every run, a row per rule, then the r=0 stay schedule along the same runs."""
    points = read("two_bidder_trace.csv", ("q", "iteration", "utility_loss", "sigma1", "sigma2", "sigma3"))
    cells = [cell for cell in CELLS if cell[1] in PLACEMENT_COLOR]
    rows = len(VALIDATED_POLICIES) + 1
    fig, axes = plt.subplots(rows, len(cells), figsize=(7.0, 7.6), sharex=True, sharey="row", constrained_layout=True)
    for column, (q, placement, c) in enumerate(cells):
        at = [p for p in points if (p["q"], p["placement"]) == (q, placement)]
        for row, rule in enumerate(VALIDATED_POLICIES):
            axis = axes[row, column]
            runs = [[p for p in at if p["reimbursement_policy"] == rule and p["seed"] == seed]
                    for seed in sorted({p["seed"] for p in at})]
            iterations = [p["iteration"] for p in runs[0]]
            losses = np.maximum([[p["utility_loss"] for p in run] for run in runs], 1e-12)  # below is round-off
            for loss in losses:
                axis.plot(iterations, loss, color=RULE_COLOR[rule], linewidth=0.6, alpha=0.35)
            axis.plot(iterations, np.median(losses, axis=0), color=RULE_COLOR[rule], linewidth=2)
            axis.axhline(CERTIFIED, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
            axis.set_xscale("log")
            axis.set_yscale("log")
            chrome(axis, f"$q={q:g}$\n{placement}" if row == 0 else "", "", f"{RULE_LABEL[rule]} utility loss" if column == 0 else "")
        axis = axes[rows - 1, column]
        zero = [p for p in at if p["reimbursement_policy"] == "zero"]
        target = theory_schedule(q, c)
        for name, color in zip(("sigma1", "sigma2", "sigma3"), PLACEMENT_COLOR.values()):
            series = np.array([[p[name] for p in zero if p["seed"] == seed] for seed in sorted({p["seed"] for p in zero})])
            axis.plot(iterations, np.median(series, axis=0), color=color, linewidth=2)
            axis.axhline(target[name], color=color, linewidth=0.8, linestyle=(0, (4, 3)))
        chrome(axis, "", "iteration", "$r=0$ stay probability" if column == 0 else "")
    for axis in axes.flat:
        axis.tick_params(which="both", labelsize=6.5)
        axis.title.set_fontsize(8)
        axis.set_xticks([1_000, 10_000, 100_000], ["1k", "10k", "100k"], minor=False)
        axis.set_xticks([], minor=True)
    handles = [Line2D([], [], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), label=r"$\varepsilon$, or closed form")]
    handles += [Line2D([], [], color=color, linewidth=2, label=SCHEDULE_LABEL[name])
                for name, color in zip(("sigma1", "sigma2", "sigma3"), PLACEMENT_COLOR.values())]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=7.5, frameon=False, labelcolor=INK)
    save(fig, "validation_trace")


def save(fig: plt.Figure, name: str) -> None:
    """Write one figure as a PNG."""
    path = os.path.join(RESULTS, f"{name}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {path}")


def read(name: str, numeric: tuple) -> List[Dict[str, object]]:
    """One results CSV, with the named columns parsed as floats."""
    with open(os.path.join(RESULTS, name)) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def bounds(axis: plt.Axes, q: float) -> None:
    """The incentive bound, with the stretch past it where the theorem is silent greyed out."""
    bound = region(q)
    axis.axvspan(bound, span_of(q), color=GRID, alpha=0.6, linewidth=0, zorder=0)
    axis.axvline(bound, color=MUTED, linewidth=0.8, linestyle=(0, (1, 2)))
    axis.text(bound, 1.0, " past IC: theorem\n silent", transform=axis.get_xaxis_transform(),
              fontsize=7, color=MUTED, va="top")


def ranking_figure() -> None:
    """Back-loading's learned margin over the best other policy per seed, against the closed-form margin."""
    rows = read("learned_ranking.csv", ("q", "c", "margin"))
    jitter = np.random.default_rng(0)
    color = RULE_COLOR["backloaded"]
    fig, axes = plt.subplots(1, len(SIGNAL_PROBABILITIES), figsize=(7.0, 2.9), constrained_layout=True)
    for axis, q in zip(axes, SIGNAL_PROBABILITIES):
        span = np.linspace(0.0, region(q), 200)
        axis.axhline(0.0, color=INK, linewidth=0.8)
        theory = [margin({rule: closed_form_revenue(rule, c, q) for rule in REIMBURSEMENT_POLICIES}) for c in span]
        axis.plot(span, theory, color=color, linewidth=2)
        for row in (row for row in rows if row["q"] == q):
            counted = row["counted"] == "True"
            axis.scatter(row["c"] + jitter.uniform(-0.006, 0.006) * span_of(q), row["margin"], s=16,
                         facecolors=color if counted else "white", edgecolors=color, linewidths=0.9, zorder=3)
        bounds(axis, q)
        axis.set_xlim(0.0, span_of(q))
        chrome(axis, f"$q={q:g}$", "cost $c$", r"$\widehat\Pi - $ best other $\Pi$" if axis is axes[0] else "")
    handles = [
        Line2D([], [], color=color, linewidth=2, label="closed form"),
        Line2D([], [], color=color, linewidth=0, marker="o", label="counted seed"),
        Line2D([], [], color=color, linewidth=0, marker="o", markerfacecolor="white", label="seed, a policy reached no equilibrium"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7.5, frameon=False, labelcolor=INK)
    save(fig, "robustness_ranking")


def gap_figure() -> None:
    """Learned minus closed-form revenue per policy and seed, the per-policy gaps the ranking margin is built from."""
    rows = read("two_bidder.csv", ("q", "c", "Pi", "Pi_closed_form"))
    dodge = dict(zip(REIMBURSEMENT_POLICIES, np.linspace(-0.02, 0.02, len(REIMBURSEMENT_POLICIES))))
    fig, axes = plt.subplots(1, len(SIGNAL_PROBABILITIES), figsize=(7.0, 2.9), sharey=True, constrained_layout=True)
    for axis, q in zip(axes, SIGNAL_PROBABILITIES):
        axis.axhspan(-REVENUE_BAR, REVENUE_BAR, color=GRID, alpha=0.6, linewidth=0, zorder=0)
        axis.axhline(0.0, color=INK, linewidth=0.8)
        for rule in REIMBURSEMENT_POLICIES:
            at = [row for row in rows if row["q"] == q and row["reimbursement_policy"] == rule]
            costs = sorted({row["c"] for row in at})
            shift = dodge[rule] * span_of(q)
            gaps = {c: [row["Pi"] - row["Pi_closed_form"] for row in at if row["c"] == c] for c in costs}
            axis.plot([c + shift for c in costs], [np.median(gaps[c]) for c in costs], color=RULE_COLOR[rule],
                      linewidth=1.2)
            for c in costs:
                axis.scatter([c + shift] * len(gaps[c]), gaps[c], s=12, color=RULE_COLOR[rule],
                             edgecolors="white", linewidths=0.5, zorder=3)
        axis.axvline(region(q), color=MUTED, linewidth=0.8, linestyle=(0, (1, 2)))
        axis.text(region(q), 1.0, " IC", transform=axis.get_xaxis_transform(), fontsize=8, color=INK, va="top")
        axis.set_xlim(0.0, span_of(q))
        chrome(axis, f"$q={q:g}$", "cost $c$",
               r"$\Pi_{\mathrm{MCCFR}} - \Pi_{\mathrm{theory}}$" if axis is axes[0] else "")
    handles = [Line2D([], [], color=RULE_COLOR[rule], linewidth=1.2, marker="o", label=RULE_LABEL[rule])
               for rule in VALIDATED_POLICIES]
    handles.append(Line2D([], [], color=CORNER_COLOR, linewidth=1.2, marker="o", label="other corners"))
    handles.append(Line2D([], [], color=GRID, linewidth=6, label=r"$\pm$ revenue tolerance"))
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=7.5, frameon=False, labelcolor=INK)
    save(fig, "robustness_gaps")


N_COLOR = {2: "#6da7ec", 4: "#2a78d6", 6: "#104281"}
WAR_MARKER = {"open_high": "^", "open_low": "v", "war_t1": "o", "war_t2": "s", "war_t3": "D"}
WAR_LABEL = {"open_high": "open, high", "open_low": "open, low",
             "war_t1": "war, $a/2$", "war_t2": "war, $a$", "war_t3": "war, $3a/2$"}
CLAIM_LABEL = {"trigger_minus_none": "Trigger $-$ None", "preserve_minus_full": "Preserve $-$ Full",
               "preserve_minus_full_high": "Preserve $-$ Full, high $c$"}
CLAIM_COLOR = {"trigger_minus_none": "#c2185b", "preserve_minus_full": "#1baf7a",
               "preserve_minus_full_high": "#0d6b4a"}


def scale_figure() -> None:
    """Learned stay probabilities against theory, and learned revenue differences against the formulas."""
    rows = read("scale.csv", ("n", "Pi", *WAR_MARKER, *(f"war_t{t}_theory" for t in (1, 2, 3))))
    fig, (stays, revenue) = plt.subplots(1, 2, figsize=(7.0, 3.4), constrained_layout=True)
    theory = {"open_high": lambda row: 1.0, "open_low": lambda row: 0.0}
    theory |= {f"war_t{t}": (lambda row, t=t: row[f"war_t{t}_theory"]) for t in (1, 2, 3)}
    for row in rows:
        for name, marker in WAR_MARKER.items():
            stays.scatter(theory[name](row), row[name], s=20, marker=marker, color=N_COLOR[int(row["n"])],
                          edgecolors="white", linewidths=0.5, alpha=0.8)
    stays.set_xlim(-0.05, 1.05)
    style(stays, "A. Stay probabilities", "theory", "MCCFR", STAY_BAR)
    handles = [Line2D([], [], marker="s", linestyle="", color=color, label=f"$n={n}$") for n, color in N_COLOR.items()]
    handles += [Line2D([], [], marker=marker, linestyle="", color=MUTED, label=WAR_LABEL[name])
                for name, marker in WAR_MARKER.items()]
    stays.legend(handles=handles, loc="upper left", fontsize=7, frameon=False, labelcolor=INK)

    found = claims(rows)
    for offset, name in zip((-0.15, 0.0, 0.15), CLAIM_LABEL):
        at = [claim for claim in found if claim["claim"] == name]
        x = [COUNTS.index(claim["n"]) + offset for claim in at]
        revenue.scatter(x, [claim["predicted"] for claim in at], s=60, marker="_", color=CLAIM_COLOR[name], zorder=3)
        revenue.scatter(x, [claim["learned"] for claim in at], s=18, color=CLAIM_COLOR[name], zorder=4,
                        label=CLAIM_LABEL[name])
    revenue.axhline(0.0, color=INK, linewidth=0.8)
    revenue.set_xticks(range(len(COUNTS)), [str(n) for n in COUNTS])
    revenue.set_xlim(-0.6, len(COUNTS) - 0.4)
    chrome(revenue, "B. Revenue differences", "bidders $n$", "difference in $\\Pi$")
    handles = [Line2D([], [], marker="o", linestyle="", color=CLAIM_COLOR[name], label=label)
               for name, label in CLAIM_LABEL.items()]
    handles.append(Line2D([], [], marker="_", markersize=9, linestyle="", color=MUTED, label="formula"))
    revenue.legend(handles=handles, loc="upper left", fontsize=7, frameon=False, labelcolor=INK)
    save(fig, "scale")


def validation_figure() -> None:
    """Both validation panels side by side."""
    runs = read("two_bidder.csv", ("c", "q", "Pi", "Pi_closed_form", *SCHEDULE_MARKER))
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.6), constrained_layout=True)
    revenue_panel(left, runs)
    schedule_panel(right, runs)
    save(fig, "validation")
    trace_panels()


if __name__ == "__main__":
    validation_figure()
    ranking_figure()
    gap_figure()
    scale_figure()
