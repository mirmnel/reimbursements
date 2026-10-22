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

from auction.game import (
    REIMBURSEMENT_POLICIES, VALIDATED_POLICIES, Auction, closed_form_revenue, payoff_scale, region,
)
from auction.mccfr import REVENUE_BAR, STAY_BAR, Verdict, closed_forms, stay_schedule
from auction.exact import CERTIFIED
from experiments.scale import COUNTS, ROUNDS
from experiments.two_bidder import CELLS, RESULTS, SIGNAL_PROBABILITIES, margin

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CORNER_COLOR = "#9a9993"  # the unnamed corners share one recessive grey
RULE_COLOR = {"zero": "#2a78d6", "full": "#eb6834", "least_cost": "#1baf7a", "backloaded": "#c2185b"}
RULE_COLOR |= {rule: CORNER_COLOR for rule in REIMBURSEMENT_POLICIES if rule not in RULE_COLOR}
RULE_LABEL = {"zero": "$r=0$", "full": "$r=c$", "least_cost": "$r^*$", "backloaded": "back-loaded"}
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
        Line2D([], [], marker="o", linestyle="", color=MUTED, label="derived"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=MUTED, label="different"),
        Line2D([], [], marker="x", linestyle="", color=MUTED, label="unsolved"),
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


ARM_COLOR = {True: "#3d3d3a", False: "#9a9993"}  # symmetry-restricted, unrestricted
ARM_LABEL = {True: "symmetry-restricted", False: "unrestricted"}
N_COLOR = {3: "#6da7ec", 4: "#2a78d6", 6: "#104281"}


def scatter(axis: plt.Axes, rows: list, value) -> None:
    """One dot per seed at its bidder count, the two arms dodged apart."""
    jitter = np.random.default_rng(0)
    for index, n in enumerate(COUNTS):
        for restricted, shift in ((True, -0.17), (False, 0.17)):
            color = ARM_COLOR[restricted]
            for row in rows:
                if row["n"] != n or (row["symmetry_restricted"] == "True") != restricted:
                    continue
                x = index + shift + jitter.uniform(-0.07, 0.07)
                axis.scatter(x, value(row), s=16, color=color, linewidths=0.9, zorder=3)
    axis.set_xticks(range(len(COUNTS)), [str(n) for n in COUNTS])
    axis.set_xlim(-0.6, len(COUNTS) - 0.4)


def recovery_figure() -> None:
    """Per-seed utility loss, revenue gap and distance to the reference, each on the two-bidder scale."""
    rows = read("scale.csv", ("n", "utility_loss", "Pi", "Pi_target", "arrival_weighted_distance"))
    for row in rows:
        row["n"] = int(row["n"])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), constrained_layout=True)

    loss, gap, distance = axes
    loss.axhline(CERTIFIED, color=MUTED, linewidth=0.8, linestyle=(0, (4, 2)))
    loss.text(-0.55, CERTIFIED, r"$\varepsilon$", fontsize=8, color=INK, va="bottom")
    scatter(loss, rows, lambda row: row["utility_loss"] / payoff_scale(row["n"]))
    chrome(loss, "A. Utility loss", "bidders $n$", r"loss $/\ (n/2)$")

    gap.axhspan(-REVENUE_BAR, REVENUE_BAR, color=GRID, alpha=0.6, linewidth=0, zorder=0)
    gap.axhline(0.0, color=INK, linewidth=0.8)
    scatter(gap, rows, lambda row: (row["Pi"] - row["Pi_target"]) / payoff_scale(row["n"]))
    chrome(gap, "B. Net revenue gap", "bidders $n$", r"$(\Pi_{\mathrm{MCCFR}} - \Pi_{\mathrm{ref}})\ /\ (n/2)$")

    distance.axhline(STAY_BAR, color=MUTED, linewidth=0.8, linestyle=(0, (4, 2)))
    distance.text(-0.55, STAY_BAR, " stay bar", fontsize=7, color=MUTED, va="bottom")
    scatter(distance, rows, lambda row: row["arrival_weighted_distance"])
    distance.set_ylim(bottom=0.0)
    chrome(distance, "C. Distance to ref.", "bidders $n$", "worst bidder's weighted distance")

    handles = [Line2D([], [], color=ARM_COLOR[arm], linewidth=0, marker="o", label=ARM_LABEL[arm])
               for arm in (True, False)]
    handles.append(Line2D([], [], color=GRID, linewidth=6, label=r"$\pm$ revenue tolerance"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7.5, frameon=False, labelcolor=INK)
    save(fig, "scale")


def rule_figure() -> None:
    """The minimum-cost rule's reimbursement by round as a share of the cost, one line per bidder count."""
    schedule = tuple(f"r_t{t}" for t in range(ROUNDS))
    rows = read("scale_rules.csv", ("n", "c", *schedule))
    rounds = np.arange(ROUNDS)
    fig, axis = plt.subplots(figsize=(4.2, 2.7), constrained_layout=True)
    for row in rows:
        n = int(row["n"])
        axis.plot(rounds, [row[name] / row["c"] for name in schedule], color=N_COLOR[n], linewidth=1.6,
                  marker="o", markersize=4, label=f"$n={n}$")
    axis.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 2)))
    axis.set_xticks(rounds)
    axis.set_ylim(-0.03, 1.08)
    chrome(axis, "Reimbursement by round", "round $t$", "$r_t / c$")
    axis.legend(loc="lower right", fontsize=7.5, frameon=False, labelcolor=INK)
    save(fig, "scale_rules")


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
    recovery_figure()
    rule_figure()
