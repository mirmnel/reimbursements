"""The paper's one figure: experiment 2's ranked schedules."""

from __future__ import annotations

import csv
import os
from typing import Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.scale import RESULTS
from experiments.schedule_search import LEVELS, NAMED, ranking

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read(name: str, numeric: tuple) -> List[Dict[str, object]]:
    """One results CSV, with the named columns parsed as floats."""
    with open(os.path.join(RESULTS, name)) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


SHARE_LABEL = dict(zip(LEVELS, ("0", "c/4", "c/2", "3c/4", "c")))
NAMED_COLOR = {"None": "#2a78d6", "Full": "#eb6834", "war-all": "#1baf7a", "back-loaded": "#c2185b",
               "least-cost preservation": "#eda100"}


def ranked_figure() -> None:
    """Every schedule's mean revenue in rank order, the best and named ones marked."""
    rows = read("schedule_search.csv", ("c",))
    c = rows[0]["c"]
    entries = ranking(rows)
    for entry in entries:
        entry["shares"] = tuple(round(rho / c * 4) / 4 for rho in entry["rho"])
    preserve = next(entry for entry in entries if entry["policy"] == "preserve")
    ranked = [entry for entry in entries if entry["policy"] == "schedule" and not entry["flagged"]]
    rank = np.arange(1, len(ranked) + 1)
    revenue = np.array([entry["Pi"] for entry in ranked])
    fig, axis = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)
    axis.plot(rank, revenue, color=INK, linewidth=1.2)
    at = {entry["shares"]: (position, entry["Pi"]) for position, entry in zip(rank, ranked)}
    best = "best"
    marks = {name: at[shares] for name, shares in NAMED.items()} | {best: (1, revenue[0])}
    marks["least-cost preservation"] = (1 + np.sum(revenue > preserve["Pi"]), preserve["Pi"])
    arrays = {name: shares for name, shares in NAMED.items()} | {best: ranked[0]["shares"]}
    labels = {name: "$(" + ",".join(SHARE_LABEL[share] for share in shares) + ")$" for name, shares in arrays.items()}
    labels["least-cost preservation"] = "$(c-w_0,c,c,c)$"
    offsets = {"None": (10, -14), "back-loaded": (-10, 12), "war-all": (-12, -12),
               "least-cost preservation": (12, 12), "Full": (-10, 10), best: (8, 10)}
    for name, (x, y) in marks.items():
        dx, dy = offsets[name]
        axis.scatter(x, y, s=40, color=NAMED_COLOR.get(name, INK), edgecolors="white", linewidths=1.5, zorder=3)
        axis.annotate(labels[name], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=INK,
                      ha="right" if dx < 0 else "left", va="center",
                      arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6, "shrinkB": 3})
    axis.set_xscale("log")
    axis.set_xticks([1, 10, 100, len(ranked)], ["1", "10", "100", str(len(ranked))])
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    axis.set_xlabel("rank (log scale)", fontsize=9, color=INK)
    axis.set_ylabel(r"seller net revenue $\Pi$", fontsize=9, color=INK)
    path = os.path.join(RESULTS, "schedule_ranked.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == "__main__":
    ranked_figure()
