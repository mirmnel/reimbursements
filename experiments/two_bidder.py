"""Experiments 1 and 2: whether MCCFR recovers the two-bidder closed forms, and whether the ranking survives."""

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Sequence, Tuple

from auction.game import PI, REIMBURSEMENT_POLICIES, Auction, ic_bound, region, region_claim
from auction.mccfr import BUDGET, STAY_BAR, Verdict, closed_forms, score, solve, stay_schedule
from auction.exact import CERTIFIED, utility_loss

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

SEEDS = 10

# Costs are placed as multiples of region(q), since which of its two bounds binds moves with q.
PLACEMENTS = {"inside": 0.5, "under": 0.9, "bound": 1.0, "over": 1.1, "outside": 1.5}

# Experiment 1 samples the (c, q) plane rather than crossing it. At 0.5 the cost root binds, at 0.9
# the incentive bound does, and 0.75 sits between.
VALIDATION = {0.5: ("inside", "bound", "outside"), 0.75: ("bound", "outside"), 0.9: ("bound", "outside")}

# Experiment 2 refines both bounds, one signal probability per bound.
SIGNAL_PROBABILITIES = (0.5, 0.9)
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
    started = time.perf_counter()
    solved = solve(auction, seed=seed, iterations=BUDGET, trace_every=TRACE, symmetry_restricted=True)
    seconds = time.perf_counter() - started
    scored = score(auction, solved.profile)
    derived, _ = closed_forms(auction)
    schedule, target = stay_schedule(auction, solved.profile), stay_schedule(auction, derived)
    row: Row = {
        "reimbursement_policy": auction.reimbursement_policy,
        "placement": placement,
        "c": auction.c,
        "q": auction.q,
        "seed": seed,
        "iterations": solved.iterations,
        "seconds": seconds,
        "verdict": scored.verdict.value,
        "utility_loss": scored.utility_loss,
        # Past the incentive bound the target is no equilibrium, which puts derived out of reach.
        "closed_form_loss": float(utility_loss(auction, derived)),
        "on_path_distance": scored.on_path_distance,
        "Pi": scored.revenue,
        "Pi_closed_form": PI[auction.reimbursement_policy](auction.c, auction.q),
        "revenue_distance": scored.revenue_distance,
        "info_sets": len(auction.info_sets),
        "trace": [
            {"reimbursement_policy": auction.reimbursement_policy, "placement": placement, "q": auction.q,
             "seed": seed, "iteration": TRACE * (k + 1), "utility_loss": float(utility_loss(auction, profile))}
            | {name: p for name, p in stay_schedule(auction, profile).items() if name.startswith("sigma")}
            for k, profile in enumerate(solved.averages)
        ],
    } | schedule
    assert float(row["open_high"]) > float(row["open_low"]), row
    if scored.verdict is Verdict.DERIVED:
        assert all(abs(schedule[name] - target[name]) <= STAY_BAR for name in target), (row, target)
    return row


def run_job(job: Tuple[str, float, str, float, int]) -> Row:
    """``run`` on one job, built in the worker process."""
    rule, q, placement, c, seed = job
    return run(Auction(q=q, c=c, reimbursement_policy=rule), placement, seed)


def sweep(cells: Sequence[Cell]) -> List[Row]:
    """Every published rule at every cell at every seed, with verdict counts printed per cell."""
    jobs = [(rule, q, placement, c, seed) for rule in REIMBURSEMENT_POLICIES for (q, placement, c) in cells for seed in range(SEEDS)]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(run_job, jobs))
    for rule in REIMBURSEMENT_POLICIES:
        for q, placement, c in cells:
            at = [r for r in rows if (r["reimbursement_policy"], r["q"], r["placement"]) == (rule, q, placement)]
            counts = " ".join(f"{v.value} {sum(r['verdict'] == v.value for r in at)}" for v in Verdict)
            print(f"  {rule:9s} q={q:.2f} {placement:8s} c={c:.4f}  {counts}")
    return rows


def check(rows: Sequence[Row]) -> None:
    """What both experiments assert of every run they record."""
    # Another equilibrium is a result, so it is counted. What is asserted is that it is certified.
    assert all(float(r["utility_loss"]) <= CERTIFIED for r in rows if r["verdict"] == Verdict.DIFFERENT.value)
    beyond = [r for r in rows if float(r["closed_form_loss"]) > CERTIFIED]
    assert {r["reimbursement_policy"] for r in beyond} <= {"targeted"}, beyond
    assert all(float(r["c"]) > ic_bound(float(r["q"])) for r in beyond), beyond
    assert all(r["verdict"] != Verdict.DERIVED.value for r in beyond), beyond


def ranked(rows: Sequence[Row]) -> List[Row]:
    """The revenue ranking at learned play, one row per cell and seed, with the closed-form one beside it."""
    paired: Dict[tuple, Dict[str, Row]] = {}
    for row in rows:
        paired.setdefault((row["q"], row["placement"], row["seed"]), {})[str(row["reimbursement_policy"])] = row
    ranking = []
    for (q, placement, seed), runs in paired.items():
        if set(runs) != set(REIMBURSEMENT_POLICIES):
            raise ValueError(f"cell q={q} {placement} seed {seed} ran {sorted(runs)}, not the three rules")
        budgets = {int(runs[rule]["iterations"]) for rule in REIMBURSEMENT_POLICIES}
        assert len(budgets) == 1, (q, placement, seed, budgets)
        q, c = float(q), float(runs["targeted"]["c"])
        learned = {rule: float(runs[rule]["Pi"]) for rule in REIMBURSEMENT_POLICIES}
        derived = {rule: float(PI[rule](c, q)) for rule in REIMBURSEMENT_POLICIES}
        margin = lambda earned: earned["targeted"] - max(earned["zero"], earned["full"])
        row: Row = {
            "q": q,
            "placement": placement,
            "c": c,
            "seed": int(seed),
            "iterations": budgets.pop(),
            **{f"Pi_{rule}": learned[rule] for rule in REIMBURSEMENT_POLICIES},
            "argmax": max(learned, key=learned.get),
            "margin": margin(learned),
            "targeted_wins": margin(learned) > 0.0,
            "margin_closed_form": margin(derived),
            "theory_wins": region_claim(c, q),
            **{f"verdict_{rule}": runs[rule]["verdict"] for rule in REIMBURSEMENT_POLICIES},
            "certified": all(runs[rule]["verdict"] != Verdict.UNSOLVED.value for rule in REIMBURSEMENT_POLICIES),
            "separation_survives": float(runs["targeted"]["closed_form_loss"]) <= CERTIFIED,
        }
        row["agrees"] = row["targeted_wins"] == row["theory_wins"]
        ranking.append(row)
    return ranking


def report(ranking: Sequence[Row]) -> None:
    """Print how often the targeted rule won at each cell, as counts rather than an average over seeds."""
    for q, placement in dict.fromkeys((r["q"], r["placement"]) for r in ranking):
        at = [r for r in ranking if (r["q"], r["placement"]) == (q, placement)]
        margins = [float(r["margin"]) for r in at]
        print(f"  q={q:.2f} {placement:8s} targeted wins {sum(bool(r['targeted_wins']) for r in at)}/{len(at)}"
              f"  learned margin {min(margins):+.4f} to {max(margins):+.4f}"
              f"  closed form {float(at[0]['margin_closed_form']):+.4f}")


def validation() -> List[Row]:
    """Experiment 1: ten seeds at each sampled cell under each published rule."""
    rows = sweep(grid(VALIDATION))
    write("validation_trace.csv", [point for row in rows for point in row["trace"]])
    write("validation.csv", rows)  # before the claims are checked, so a failed claim keeps its runs
    check(rows)
    inside = [r for r in rows if (r["reimbursement_policy"], r["q"], r["placement"]) == ("zero", 0.5, "inside")]
    assert all(float(r["utility_loss"]) <= CERTIFIED for r in inside), inside
    return rows


def robustness() -> List[Row]:
    """Experiment 2: the refined grid, and the ranking at learned play."""
    rows = sweep(grid(ROBUSTNESS))
    write("robustness.csv", rows)
    check(rows)
    ranking = ranked(rows)
    report(ranking)
    write("learned_ranking.csv", ranking)
    past = [r for r in ranking if float(r["c"]) > ic_bound(float(r["q"]))]
    assert not any(r["separation_survives"] for r in past), past
    # Well inside the region the ranking survives being learned, and outside it the targeted rule loses.
    inside = [r for r in ranking if r["placement"] == "inside" and r["q"] == 0.5]
    assert len(inside) == SEEDS and all(r["targeted_wins"] for r in inside), inside
    assert not any(r["targeted_wins"] for r in ranking if r["placement"] == "outside")
    return ranking


if __name__ == "__main__":
    experiments = {"validation": validation, "robustness": robustness}
    for name in sys.argv[1:] or experiments:
        print(f"== {name} ==")
        experiments[name]()
