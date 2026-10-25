"""The experiment: whether learning finds the paper's sorting equilibria and revenue gaps at two, four and six bidders."""

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
from auction.mccfr import BUDGET, STAY_BAR, Verdict, solve
COUNTS = (2, 4, 6)
Q = 0.3
COSTS = (0.025, 0.05, 0.1)  # kappa, inside fin:trigger's cost bound at every count
SEEDS = 5
WORKERS = 3  # a six-bidder run peaks near 2.1 GB, and the machine has 16 GB

# Each run learns under its own policy at one cost. Full gives every bidder the frictionless payoff on
# every path, so one frictionless run per seed is scored under Full at every cost instead.
LEARNED = ("zero", "trigger", "preserve")

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


def two_survivor_war(auction: Auction) -> int:
    """The orbit where exactly two bidders stayed at the opening."""
    return auction.orbits.index((1, (0,) * (auction.n - 2)))


def two_survivor_revenue(auction: Auction, sigma: np.ndarray) -> float:
    """Seller net revenue from auctions that reach a two-survivor war, the opening stays included."""
    stay = symmetric.anonymise(auction, sigma)
    paid = auction.orbit_view
    price, payout = symmetric.seller_tables(auction, stay, paid)
    war = two_survivor_war(auction)
    total = 0.0
    for high, weight in enumerate(auction.signal_counts):
        thin = np.outer(symmetric.thinning(high, stay[0, 1]), symmetric.thinning(auction.n - high, stay[0, 0]))
        for staying_high in range(min(high, 2) + 1):
            staying_low = 2 - staying_high
            if staying_low <= auction.n - high:
                net = price[war, staying_high] - payout[war, staying_high] - 2 * paid[0]
                total += weight * thin[staying_high, staying_low] * net
    return total


def row_for(n: int, policy: str, seed: int, auction: Auction, theory: np.ndarray, profile: np.ndarray) -> Row:
    """One learned profile scored against theory under one policy."""
    learned, target = symmetric.anonymise(auction, profile), symmetric.anonymise(auction, theory)
    distance = float(np.abs(learned - target)[symmetric.bucket_arrival(auction, theory) > RESOLVED].max())
    loss = float(utility_loss(auction, profile))
    if loss > certified(n):
        verdict = Verdict.UNSOLVED
    else:
        verdict = Verdict.DERIVED if distance <= STAY_BAR else Verdict.DIFFERENT
    return {
        "n": n, "policy": policy, "c": auction.c, "seed": seed, "verdict": verdict.value,
        "utility_loss": loss, "distance": distance,
        "Pi": evaluate(auction, profile).revenue, "Pi_theory": evaluate(auction, theory).revenue,
        "Pi_two_survivor": two_survivor_revenue(auction, profile),
        "open_high": learned[0, 1], "open_low": learned[0, 0],
    }


RUNS = [(run, c) for run in LEARNED for c in COSTS] + [("frictionless", 0.0)]


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
        rows.append(row_for(n, policy, seed, auction, theories[(policy, k)], profile))
        print(f"  n={n} {policy:13s} c={k:.3f} seed={seed} {rows[-1]['verdict']:9s} "
              f"loss {rows[-1]['utility_loss']:.2e}  Pi {rows[-1]['Pi']:.4f}  "
              f"{time.perf_counter() - started:.0f} s", flush=True)
    return rows


def key(row: Row) -> Tuple[int, str, float, int]:
    """What identifies a saved row: bidder count, policy, cost and seed."""
    return int(row["n"]), str(row["policy"]), float(row["c"]), int(row["seed"])


def pending(done: Iterable[Row], counts: Sequence[int], seeds: Sequence[int],
            iterations: int = BUDGET) -> List[Tuple[int, str, float, int, int]]:
    """Jobs for every run and seed with a row not saved yet, largest bidder count first."""
    saved = {key(row) for row in done}
    return [
        (n, run, c, seed, iterations)
        for n in sorted(counts, reverse=True)
        for run, c in RUNS
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
    grid = {written_key for run, c in RUNS for written_key in written(run, c)}
    return [row for row in rows if (row["policy"], float(row["c"])) in grid and int(row["seed"]) < SEEDS]


def claims(rows: Sequence[Row]) -> List[Row]:
    """Per bidder count and cost, the learned revenue differences beside the formulas, over runs that reached an equilibrium."""
    reached = [row for row in rows if row["verdict"] != Verdict.UNSOLVED.value]
    revenue = lambda n, c, policy, column="Pi": [float(row[column]) for row in reached
                                                 if (int(row["n"]), row["policy"], float(row["c"])) == (n, policy, c)]
    out = []
    for n in sorted({int(row["n"]) for row in rows}):
        for c in COSTS:
            if not all(revenue(n, c, p) for p in ("zero", "trigger")):
                continue
            gain = np.mean(revenue(n, c, "trigger")) - np.mean(revenue(n, c, "zero"))
            war = (np.mean(revenue(n, c, "trigger", "Pi_two_survivor"))
                   - np.mean(revenue(n, c, "zero", "Pi_two_survivor")))
            out.append({
                "n": n, "c": c, "claim": "trigger_minus_none", "learned": gain, "predicted": trigger_gain(n, c),
                "two_survivor_part": war, "elsewhere_part": gain - war,
                "passes": bool(gain > 0 and abs(gain - trigger_gain(n, c)) <= 0.25 * trigger_gain(n, c)),
            })
            if revenue(n, c, "preserve") and revenue(n, c, "full"):
                saving = np.mean(revenue(n, c, "preserve")) - np.mean(revenue(n, c, "full"))
                out.append({"n": n, "c": c, "claim": "preserve_minus_full", "learned": saving,
                            "predicted": preserve_saving(n, c)})
    return out


def experiment(counts: Sequence[int] = COUNTS, seeds: Sequence[int] = range(SEEDS)) -> List[Row]:
    """Run every pending seed, saving after each job, then print the revenue claims."""
    rows = {key(row): row for row in read_rows()}
    jobs = pending(rows.values(), counts, seeds)
    with ProcessPoolExecutor(WORKERS) as pool:
        for finished in as_completed([pool.submit(run_job, job) for job in jobs]):
            rows |= {key(row): row for row in finished.result()}
            write("scale.csv", [rows[k] for k in sorted(rows)])
    for claim in claims(list(rows.values())):
        print("  " + "  ".join(f"{name} {value:.4f}" if isinstance(value, float) else f"{name} {value}"
                               for name, value in claim.items()))
    return list(rows.values())


def seed_range(text: str) -> range:
    """``3`` or ``0-4``."""
    first, _, last = text.partition("-")
    return range(int(first), int(last or first) + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", default=",".join(map(str, COUNTS)))
    parser.add_argument("--seeds", default=f"0-{SEEDS - 1}")
    arguments = parser.parse_args()
    print("== scale ==")
    experiment([int(n) for n in arguments.counts.split(",")], seed_range(arguments.seeds))
