"""Every figure in results: experiment 1's scale figure, experiment 2's search."""

from __future__ import annotations

import csv
import os
from typing import Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from experiments.scale import COSTS, COUNTS, RESULTS, preserve_saving, trigger_gain
from experiments.schedule_search import LEVELS, ranking

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


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


CLAIM_LABEL = {"trigger_minus_none": "Trigger $-$ None", "preserve_minus_full": "Preserve $-$ Full"}
CLAIM_COLOR = {"trigger_minus_none": "#c2185b", "preserve_minus_full": "#1baf7a"}
CLAIM_POLICIES = {"trigger_minus_none": ("trigger", "zero"), "preserve_minus_full": ("preserve", "full")}
CLAIM_FORMULA = {"trigger_minus_none": trigger_gain, "preserve_minus_full": preserve_saving}


def scale_figure() -> None:
    """Learned revenue differences, one dot per seed, against the formulas, a row per difference and a column per n."""
    rows = read("scale.csv", ("n", "c", "seed", "Pi"))
    earned = {(int(row["n"]), row["policy"], row["c"], int(row["seed"])): row["Pi"] for row in rows}
    fig, axes = plt.subplots(len(CLAIM_POLICIES), len(COUNTS), figsize=(7.0, 4.6), constrained_layout=True,
                             sharex=True, sharey="row")
    grid = np.linspace(0.0, 1.1 * max(COSTS), 100)
    for row, (name, (first, second)) in enumerate(CLAIM_POLICIES.items()):
        for axis, n in zip(axes[row], COUNTS):
            axis.plot(grid, [CLAIM_FORMULA[name](n, c) for c in grid], color=CLAIM_COLOR[name], linewidth=1.2)
            dots = [(c, earned[(n, first, c, seed)] - earned[(n, second, c, seed)])
                    for (count, policy, c, seed) in earned if (count, policy) == (n, first)
                    and (n, second, c, seed) in earned]
            axis.scatter(*zip(*dots), s=12, color=CLAIM_COLOR[name], edgecolors="white", linewidths=0.4, zorder=3)
            chrome(axis, f"$n={n}$" if row == 0 else "", "cost per stay $\\kappa$" if row == 1 else "",
                   CLAIM_LABEL[name] if n == COUNTS[0] else "")
    save(fig, "scale")


SHARE_LABEL = dict(zip(LEVELS, ("0", "c/4", "c/2", "3c/4", "c")))
NAMED = {"None": (0, 0, 0, 0), "Full": (1, 1, 1, 1), "back-loaded": (0, 0, 0, 1), "war-all": (0, 1, 1, 1)}
NAMED_COLOR = {"None": "#2a78d6", "Full": "#eb6834", "war-all": "#1baf7a", "back-loaded": "#c2185b",
               "least-cost preservation": "#eda100"}
REVENUE_MAP = LinearSegmentedColormap.from_list("revenue", ["#cde2fb", "#6da7ec", "#2a78d6", "#104281", "#0d366b"])
FLAGGED = "#c3c2b7"


def search_results() -> tuple:
    """Experiment 2's grid ranked best first, each with rho as shares of c, and the trigger and Preserve entries."""
    rows = read("schedule_search.csv", ("c",))
    c = rows[0]["c"]
    ranked = ranking(rows)
    for entry in ranked:
        entry["shares"] = tuple(round(rho / c * 4) / 4 for rho in entry["rho"])
    grid = [entry for entry in ranked if entry["policy"] == "schedule"]
    reference = {entry["policy"]: entry for entry in ranked if entry["policy"] != "schedule"}
    return grid, reference["trigger"], reference["preserve"]


def ranked_figure() -> None:
    """Every schedule's mean revenue in rank order, the best and named ones marked and the trigger as a line."""
    grid, trigger, preserve = search_results()
    ranked = [entry for entry in grid if not entry["flagged"]]
    rank = np.arange(1, len(ranked) + 1)
    revenue = np.array([entry["Pi"] for entry in ranked])
    fig, axis = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)
    axis.plot(rank, revenue, color=INK, linewidth=1.2)
    axis.axhline(trigger["Pi"], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    axis.text(len(ranked), trigger["Pi"] + 0.004, "two-survivor trigger", fontsize=7.5, color=MUTED, ha="right",
              va="bottom")
    at = {entry["shares"]: (position, entry["Pi"]) for position, entry in zip(rank, ranked)}
    best = "best $(0,c/2,c/4,0)$"
    marks = {name: at[shares] for name, shares in NAMED.items()} | {best: (1, revenue[0])}
    marks["least-cost preservation"] = (1 + np.sum(revenue > preserve["Pi"]), preserve["Pi"])
    offsets = {"None": (10, -14), "back-loaded": (-10, 12), "war-all": (-12, -12),
               "least-cost preservation": (12, 12), "Full": (-10, 10), best: (8, 10)}
    for name, (x, y) in marks.items():
        dx, dy = offsets[name]
        axis.scatter(x, y, s=40, color=NAMED_COLOR.get(name, INK), edgecolors="white", linewidths=1.5, zorder=3)
        axis.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=INK,
                      ha="right" if dx < 0 else "left", va="center",
                      arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6, "shrinkB": 3})
    axis.set_xscale("log")
    axis.set_xticks([1, 10, 100, len(ranked)], ["1", "10", "100", str(len(ranked))])
    chrome(axis, "", "rank (log scale)", r"seller net revenue $\Pi$")
    save(fig, "schedule_ranked")


def grid_figure() -> None:
    """The whole grid: panels over (rho0, rho1), each a heatmap over (rho2, rho3), flagged schedules grey."""
    grid, _, _ = search_results()
    by_shares = {entry["shares"]: entry for entry in grid}
    shown = [entry["Pi"] for entry in grid if not entry["flagged"]]
    norm = plt.Normalize(min(shown), max(shown))
    best = max((entry for entry in grid if not entry["flagged"]), key=lambda entry: entry["Pi"])["shares"]
    size = len(LEVELS)
    fig, axes = plt.subplots(size, size, figsize=(7.0, 7.4), sharex=True, sharey=True, constrained_layout=True)
    for i, first in enumerate(LEVELS):
        for j, second in enumerate(LEVELS):
            axis = axes[size - 1 - i, j]
            colors = np.ones((size, size, 4))
            for k, third in enumerate(LEVELS):
                for m, fourth in enumerate(LEVELS):
                    entry = by_shares[(first, second, third, fourth)]
                    colors[k, m] = matplotlib.colors.to_rgba(FLAGGED) if entry["flagged"] else REVENUE_MAP(norm(entry["Pi"]))
                    if entry["shares"] == best:
                        axis.add_patch(plt.Rectangle((m - 0.5, k - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.5))
            axis.imshow(colors, origin="lower")
            axis.set_xticks(range(size), [SHARE_LABEL[level] for level in LEVELS], fontsize=5.5, rotation=90)
            axis.set_yticks(range(size), [SHARE_LABEL[level] for level in LEVELS], fontsize=5.5)
            axis.tick_params(length=0, colors=MUTED, labelcolor=INK)
            for side in axis.spines.values():
                side.set_visible(False)
            if i == size - 1:
                axis.set_title(rf"$\rho_1={SHARE_LABEL[second]}$", fontsize=8, color=INK)
            if j == 0:
                axis.set_ylabel(rf"$\rho_0={SHARE_LABEL[first]}$" + "\n" + r"$\rho_2$", fontsize=8, color=INK)
            if i == 0:
                axis.set_xlabel(r"$\rho_3$", fontsize=8, color=INK)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm, REVENUE_MAP), ax=axes, location="right", shrink=0.5)
    bar.set_label(r"seller net revenue $\Pi$ (grey: flagged, outlined: best)", fontsize=8, color=INK)
    bar.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK)
    bar.outline.set_visible(False)
    save(fig, "schedule_grid")


def top_figure() -> None:
    """The ten best schedules, each one's rho as bars beside its revenue."""
    grid, _, _ = search_results()
    top = [entry for entry in grid if not entry["flagged"]][:10]
    fig, axes = plt.subplots(len(top), 2, figsize=(7.0, 5.0), sharex="col", constrained_layout=True,
                             gridspec_kw={"width_ratios": (1, 3), "hspace": 0.02})
    for place, (entry, (steps, revenue)) in enumerate(zip(top, axes)):
        steps.bar(range(len(entry["shares"])), entry["shares"], width=0.7, color=INK)
        steps.set_ylim(0, 1)
        steps.set_yticks([])
        rho = ", ".join(SHARE_LABEL[share] for share in entry["shares"])
        chrome(steps, "", "", "")
        steps.set_ylabel(f"{place + 1}. ({rho})", rotation=0, ha="right", va="center", fontsize=7.5, color=INK)
        revenue.errorbar(entry["Pi"], 0, xerr=entry["se"], fmt="o", color=INK, markersize=4, capsize=2, linewidth=1)
        revenue.set_yticks([])
        chrome(revenue, "", "", "")
        for axis in (steps, revenue):
            axis.tick_params(labelsize=7)
            axis.spines["bottom"].set_visible(axis is axes[-1, 0] or axis is axes[-1, 1])
            if axis not in (axes[-1, 0], axes[-1, 1]):
                axis.tick_params(bottom=False)
    axes[0, 0].set_title(r"$\rho_t$, as a share of $c$", loc="left", fontsize=9, color=INK)
    axes[0, 1].set_title(r"seller net revenue $\Pi$ $\pm$ s.e. over seeds", loc="left", fontsize=9, color=INK)
    axes[-1, 0].set_xticks(range(4), [f"{t}" for t in range(4)])
    axes[-1, 0].set_xlabel("round $t$", fontsize=8, color=INK)
    save(fig, "schedule_top")


if __name__ == "__main__":
    scale_figure()
    ranked_figure()
    grid_figure()
    top_figure()
