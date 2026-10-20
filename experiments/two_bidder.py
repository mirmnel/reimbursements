"""Experiments 1 and 2: whether MCCFR recovers the two-bidder closed forms, and whether back-loading still wins."""

from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Sequence, Tuple

from auction.game import REIMBURSEMENT_POLICIES, VALIDATED_POLICIES, Auction, ic_bound, region
from auction.mccfr import BUDGET, REVENUE_BAR, Verdict, closed_forms, score, solve, stay_schedule
from auction.exact import CERTIFIED, utility_loss

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

SEEDS = 10

# Costs are placed as multiples of region(q). Every q is above one half, so the incentive bound binds.
PLACEMENTS = {"inside": 0.5, "under": 0.9, "bound": 1.0, "over": 1.1, "outside": 1.5}

# Experiment 1 samples the (c, q) plane rather than crossing it.
VALIDATION = {0.7: ("inside", "bound", "outside"), 0.75: ("bound", "outside"), 0.9: ("bound", "outside")}

# Experiment 2 runs every placement at two signal probabilities.
SIGNAL_PROBABILITIES = (0.7, 0.9)
ROBUSTNESS = {q: tuple(PLACEMENTS) for q in SIGNAL_PROBABILITIES}

TRACE = 1_000  # iterations between the snapshots a run is scored at along the way

Row = Dict[str, object]
Cell = Tuple[float, str, float]


def write(name: str, rows: Sequence[Row]) -> None:
    """Write ``rows`` as a CSV file of the given name under ``results``."""
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as handle:
        fields = [name for name in rows[0] if name != "trace"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def grid(cells: Dict[float, Sequence[str]]) -> List[Cell]:
    """A grid's cells, each cost placed as a multiple of the region bound at its signal probability."""
    return [(q, placement, PLACEMENTS[placement] * region(q)) for q, named in cells.items() for placement in named]


def run(auction: Auction, placement: str, seed: int) -> Row:
    """One symmetry-restricted run, scored against the closed forms."""
    solved = solve(auction, seed=seed, iterations=BUDGET, trace_every=TRACE, symmetry_restricted=True)
    scored = score(auction, solved.profile)
    derived, closed_form_revenue = closed_forms(auction)
    return {
        "reimbursement_policy": auction.reimbursement_policy,
        "placement": placement,
        "c": auction.c,
        "q": auction.q,
        "seed": seed,
        "iterations": solved.iterations,
        "verdict": scored.verdict.value,
        "utility_loss": scored.utility_loss,
        # Past the incentive bound the target is no equilibrium, which puts derived out of reach.
        "closed_form_loss": float(utility_loss(auction, derived)),
        "Pi": scored.revenue,
        "Pi_closed_form": closed_form_revenue,
        "trace": [
            {"reimbursement_policy": auction.reimbursement_policy, "placement": placement, "q": auction.q,
             "seed": seed, "iteration": TRACE * (k + 1), "utility_loss": float(utility_loss(auction, profile))}
            | {name: p for name, p in stay_schedule(auction, profile).items() if name.startswith("sigma")}
            for k, profile in enumerate(solved.averages)
        ],
    } | stay_schedule(auction, solved.profile)


def run_job(job: Tuple[str, float, str, float, int]) -> Row:
    """``run`` on one job, built in the worker process."""
    rule, q, placement, c, seed = job
    return run(Auction(q=q, c=c, reimbursement_policy=rule), placement, seed)


def sweep(cells: Sequence[Cell], policies: Sequence[str]) -> List[Row]:
    """Every policy at every cell at every seed, with verdict counts printed per cell."""
    jobs = [(rule, q, placement, c, seed) for rule in policies for (q, placement, c) in cells for seed in range(SEEDS)]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(run_job, jobs))
    for rule in policies:
        for q, placement, c in cells:
            at = [r for r in rows if (r["reimbursement_policy"], r["q"], r["placement"]) == (rule, q, placement)]
            counts = " ".join(f"{v.value} {sum(r['verdict'] == v.value for r in at)}" for v in Verdict)
            print(f"  {rule:10s} q={q:.2f} {placement:8s} c={c:.4f}  {counts}")
    return rows


def check(rows: Sequence[Row]) -> None:
    """What both experiments assert of every run they record."""
    derived = [r for r in rows if r["verdict"] == Verdict.DERIVED.value]
    assert all(abs(float(r["Pi"]) - float(r["Pi_closed_form"])) <= REVENUE_BAR for r in derived), derived
    beyond = [r for r in rows if float(r["closed_form_loss"]) > CERTIFIED]
    assert all(float(r["c"]) > ic_bound(float(r["q"])) for r in beyond), beyond


def margin(earned: Dict[str, float]) -> float:
    """What back-loading earns above the best other policy."""
    return earned["backloaded"] - max(value for rule, value in earned.items() if rule != "backloaded")


def ranked(rows: Sequence[Row]) -> List[Row]:
    """Back-loading against every other policy at learned play, one row per cell and seed, counted or not."""
    paired: Dict[tuple, Dict[str, Row]] = {}
    for row in rows:
        paired.setdefault((row["q"], row["placement"], row["seed"]), {})[str(row["reimbursement_policy"])] = row
    ranking = []
    for (q, placement, seed), runs in paired.items():
        budgets = {int(runs[rule]["iterations"]) for rule in REIMBURSEMENT_POLICIES}
        assert len(budgets) == 1, (q, placement, seed, budgets)
        earned = lambda column: {rule: float(runs[rule][column]) for rule in REIMBURSEMENT_POLICIES}
        counted = all(runs[rule]["verdict"] != Verdict.UNSOLVED.value for rule in REIMBURSEMENT_POLICIES)
        ranking.append({
            "q": float(q),
            "placement": placement,
            "c": float(runs["backloaded"]["c"]),
            "seed": int(seed),
            "iterations": budgets.pop(),
            "counted": counted,
            "margin": margin(earned("Pi")),
            "backloaded_wins": counted and margin(earned("Pi")) > 0.0,
            "margin_closed_form": margin(earned("Pi_closed_form")),
        })
    return ranking


def report(ranking: Sequence[Row]) -> None:
    """Print, per cell, how many seeds counted and how many of those back-loading won."""
    for q, placement in dict.fromkeys((r["q"], r["placement"]) for r in ranking):
        at = [r for r in ranking if (r["q"], r["placement"]) == (q, placement)]
        margins = [float(r["margin"]) for r in at if r["counted"]]
        spread = f"{min(margins):+.4f} to {max(margins):+.4f}" if margins else "none counted"
        print(f"  q={q:.2f} {placement:8s} counted {len(margins)}/{len(at)}"
              f"  back-loaded wins {sum(bool(r['backloaded_wins']) for r in at)}"
              f"  learned margin {spread}  closed form {float(at[0]['margin_closed_form']):+.4f}")


def validation() -> List[Row]:
    """Experiment 1: ten seeds at each sampled cell under each policy the paper prints closed forms for."""
    rows = sweep(grid(VALIDATION), VALIDATED_POLICIES)
    write("validation_trace.csv", [point for row in rows for point in row["trace"]])
    write("validation.csv", rows)  # before the claims are checked, so a failed claim keeps its runs
    check(rows)
    inside = [r for r in rows if (r["reimbursement_policy"], r["placement"]) == ("zero", "inside")]
    assert all(float(r["utility_loss"]) <= CERTIFIED for r in inside), inside
    return rows


def robustness() -> List[Row]:
    """Experiment 2: every price-only corner and full reimbursement across the grid, ranked at learned play."""
    rows = sweep(grid(ROBUSTNESS), REIMBURSEMENT_POLICIES)
    write("robustness.csv", rows)
    check(rows)
    ranking = ranked(rows)
    report(ranking)
    write("learned_ranking.csv", ranking)
    return ranking


if __name__ == "__main__":
    experiments = {"validation": validation, "robustness": robustness}
    for name in sys.argv[1:] or experiments:
        print(f"== {name} ==")
        experiments[name]()
