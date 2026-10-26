"""The experiment: whether learning finds the paper's sorting equilibria and revenue gaps at two, three and four bidders."""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from auction import symmetric
from auction.exact import RESOLVED, certified, evaluate, utility_loss
from auction.game import Auction, flat, opening_margin, preserve, sorting_profile, trigger
from auction.mccfr import STAY_BAR, Verdict, solve
COUNTS = (2, 3, 4)
Q = 0.3
COSTS = (0.025, 0.05, 0.1)
SEEDS = 5
ITERATIONS = 2_000_000

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

Row = Dict[str, object]


def write(name: str, rows: Sequence[Row]) -> None:
    """Write ``rows`` as a CSV file of the given name under ``results``."""
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def auction_at(n: int, policy: str, c: float) -> Auction:
    """The auction on the theorem's clock: step a/2 from zero to the cap n a, signals independent."""
    rounds = 2 * n
    rule = {
        "zero": lambda: flat(c, rounds, 0.0),
        "frictionless": lambda: flat(c, rounds, 0.0),
        "trigger": lambda: trigger(c, n),
        "preserve": lambda: preserve(c, rounds, n, Q),
        "full": lambda: flat(c, rounds),
    }[policy]()
    return Auction(n=n, q=Q, c=c, rounds=rounds, delta=0.5, reimbursement_policy=rule)


def trigger_gain(n: int, c: float) -> float:
    """Trigger less None in seller net revenue, fin:revenue-gain, with a = 1 so kappa is c."""
    return math.comb(n, 2) * Q**2 * (1 - Q) ** (n - 2) * 2 * c / 9 * (9 - 42 * c + 92 * c**2 - 64 * c**3)


def preserve_saving(n: int, c: float) -> float:
    """Preserve less Full in seller net revenue, pres:binary-saving."""
    return n * Q * min(c, opening_margin(n, Q))


def runs(n: int) -> List[Tuple[str, float]]:
    """The (run, cost) pairs learned at ``n`` bidders, with "trigger", the paper's (0,0,0,c), only at two."""
    learned = ("zero", "trigger", "preserve") if n == 2 else ("zero", "preserve")
    return [(run, c) for run in learned for c in COSTS] + [("frictionless", 0.0)]


def written(run: str, c: float) -> List[Tuple[str, float]]:
    """The (policy, cost) rows one run writes."""
    return [("frictionless", 0.0)] + [("full", k) for k in COSTS] if run == "frictionless" else [(run, c)]


def run_job(job: Tuple[int, str, float, int, int]) -> List[Row]:
    """One seed of one run, scored under each policy and cost it stands for."""
    n, run, c, seed, iterations = job
    scored_under = {(policy, k): auction_at(n, policy, k) for policy, k in written(run, c)}
    theories = {key: sorting_profile(auction) for key, auction in scored_under.items()}
    for key, auction in scored_under.items():
        assert utility_loss(auction, theories[key]) <= certified(n), (n, key)
    started = time.perf_counter()
    profile = solve(auction_at(n, run, c), seed, iterations)
    rows = []
    for (policy, k), auction in scored_under.items():
        theory = theories[(policy, k)]
        learned, target = symmetric.anonymise(auction, profile), symmetric.anonymise(auction, theory)
        distance = float(np.abs(learned - target)[symmetric.bucket_arrival(auction, theory) > RESOLVED].max())
        loss = float(utility_loss(auction, profile))
        if loss > certified(n):
            verdict = Verdict.UNSOLVED
        else:
            verdict = Verdict.DERIVED if distance <= STAY_BAR else Verdict.DIFFERENT
        rows.append({
            "n": n, "policy": policy, "c": k, "seed": seed, "verdict": verdict.value,
            "utility_loss": loss, "distance": distance,
            "Pi": evaluate(auction, profile).revenue, "Pi_theory": evaluate(auction, theory).revenue,
        })
        print(f"  n={n} {policy:13s} c={k:.3f} seed={seed} {rows[-1]['verdict']:9s} "
              f"loss {rows[-1]['utility_loss']:.2e}  Pi {rows[-1]['Pi']:.4f}  "
              f"{time.perf_counter() - started:.0f} s", flush=True)
    return rows


def key(row: Row) -> Tuple[int, str, float, int]:
    """What identifies a saved row: bidder count, policy, cost and seed."""
    return int(row["n"]), str(row["policy"]), float(row["c"]), int(row["seed"])


def pending(done: Iterable[Row], counts: Sequence[int], seeds: Sequence[int],
            iterations: int = ITERATIONS) -> List[Tuple[int, str, float, int, int]]:
    """Jobs for every run and seed with a row not saved yet, largest bidder count first."""
    saved = {key(row) for row in done}
    return [
        (n, run, c, seed, iterations)
        for n in sorted(counts, reverse=True)
        for run, c in runs(n)
        for seed in seeds
        if any((n, policy, k, seed) not in saved for policy, k in written(run, c))
    ]


def read_rows() -> List[Row]:
    """The saved runs on the current grid, or none before the first."""
    path = os.path.join(RESULTS, "scale.csv")
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    grid = {(n, *written_key) for n in COUNTS for run, c in runs(n) for written_key in written(run, c)}
    return [row for row in rows if (int(row["n"]), row["policy"], float(row["c"])) in grid and int(row["seed"]) < SEEDS]


def claims(rows: Sequence[Row]) -> List[Row]:
    """Per bidder count and cost, the learned revenue differences beside the formulas, over runs that reached an equilibrium."""
    reached = [row for row in rows if row["verdict"] != Verdict.UNSOLVED.value]
    revenue = lambda n, c, policy: [float(row["Pi"]) for row in reached
                                    if (int(row["n"]), row["policy"], float(row["c"])) == (n, policy, c)]
    out = []
    for n in sorted({int(row["n"]) for row in rows}):
        for c in COSTS:
            if revenue(n, c, "trigger") and revenue(n, c, "zero"):
                gain = np.mean(revenue(n, c, "trigger")) - np.mean(revenue(n, c, "zero"))
                out.append({"n": n, "c": c, "claim": "trigger_minus_none", "learned": gain,
                            "predicted": trigger_gain(n, c)})
            if revenue(n, c, "preserve") and revenue(n, c, "full"):
                saving = np.mean(revenue(n, c, "preserve")) - np.mean(revenue(n, c, "full"))
                out.append({"n": n, "c": c, "claim": "preserve_minus_full", "learned": saving,
                            "predicted": preserve_saving(n, c)})
    return out


def experiment(counts: Sequence[int] = COUNTS, seeds: Sequence[int] = range(SEEDS)) -> List[Row]:
    """Run every pending seed, saving after each job, then print the revenue claims."""
    rows = {key(row): row for row in read_rows()}
    jobs = pending(rows.values(), counts, seeds)
    with ProcessPoolExecutor() as pool:
        for finished in as_completed([pool.submit(run_job, job) for job in jobs]):
            rows |= {key(row): row for row in finished.result()}
            write("scale.csv", [rows[k] for k in sorted(rows)])
    for claim in claims(list(rows.values())):
        print("  " + "  ".join(f"{name} {value:.4f}" if isinstance(value, float) else f"{name} {value}"
                               for name, value in claim.items()))
    return list(rows.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", default=",".join(map(str, COUNTS)))
    parser.add_argument("--seeds", default=f"0-{SEEDS - 1}", help="3 or 0-4")
    arguments = parser.parse_args()
    first, _, last = arguments.seeds.partition("-")
    print("== scale ==")
    experiment([int(n) for n in arguments.counts.split(",")], range(int(first), int(last or first) + 1))
