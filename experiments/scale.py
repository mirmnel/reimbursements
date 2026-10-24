"""Experiment 2: whether learning finds the paper's sorting equilibria at two, four and six bidders."""

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
from experiments.two_bidder import RESULTS, SEEDS, write

COUNTS = (2, 4, 6)
Q, KAPPA = 0.3, 0.05  # inside fin:trigger's cost bound up to eight bidders
ABOVE_MARGIN = 0.05  # how far the high cost sits above the opening margin, so Preserve pays c - w0 > 0
WORKERS = 4  # a six-bidder run holds about 3 GB
CHUNK = 3  # seeds per job, so a job builds its tables once for several runs

# Each run learns under its own policy. Full gives every bidder the frictionless payoff on every path,
# so the frictionless runs are scored under Full as well instead of learning it again.
RUNS = ("zero", "trigger", "preserve", "frictionless", "preserve_high")
SCORED_UNDER = {"frictionless": ("frictionless", "full", "full_high")}
WAR_SCORED = ("zero", "trigger")  # theory states their stays only at the opening and in a two-survivor war

Row = Dict[str, object]


def cost(n: int, policy: str) -> float:
    """The cost per stay a policy is run at."""
    if policy == "frictionless":
        return 0.0
    return opening_margin(n, Q) + ABOVE_MARGIN if policy.endswith("_high") else KAPPA


def auction_at(n: int, policy: str) -> Auction:
    """The auction on the theorem's clock: step a/2 from zero to the cap n a, signals independent."""
    rounds, c = 2 * n, cost(n, policy)
    rule = {
        "zero": lambda: flat(c, rounds, 0.0),
        "frictionless": lambda: flat(c, rounds, 0.0),
        "trigger": lambda: trigger(c, n),
        "preserve": lambda: preserve(c, rounds, n, Q),
        "full": lambda: flat(c, rounds),
    }[policy.removesuffix("_high")]()
    return Auction(n=n, q=Q, c=c, rounds=rounds, delta=0.5, reimbursement_policy=rule)


def trigger_gain(n: int) -> float:
    """Trigger less None in seller net revenue, fin:revenue-gain."""
    return math.comb(n, 2) * Q**2 * (1 - Q) ** (n - 2) * 2 * KAPPA / 9 * (
        9 - 42 * KAPPA + 92 * KAPPA**2 - 64 * KAPPA**3)


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


def scored(auction: Auction, policy: str, theory: np.ndarray) -> np.ndarray:
    """(orbit, signal) sets where theory states a stay probability and its own play arrives."""
    mask = symmetric.bucket_arrival(auction, theory) > RESOLVED
    if policy in WAR_SCORED:
        stated = np.array([round_index == 0 or departed.count(0) == auction.n - 2
                           for round_index, departed in auction.orbits])
        mask &= stated[:, None]
    return mask


def row_for(n: int, policy: str, seed: int, auction: Auction, theory: np.ndarray, profile: np.ndarray) -> Row:
    """One learned profile scored against theory under one policy."""
    learned, target = symmetric.anonymise(auction, profile), symmetric.anonymise(auction, theory)
    distance = float(np.abs(learned - target)[scored(auction, policy, theory)].max())
    loss = float(utility_loss(auction, profile))
    if loss > certified(n):
        verdict = Verdict.UNSOLVED
    else:
        verdict = Verdict.DERIVED if distance <= STAY_BAR else Verdict.DIFFERENT
    row: Row = {
        "n": n, "policy": policy, "c": auction.c, "seed": seed, "verdict": verdict.value,
        "utility_loss": loss, "distance": distance,
        "Pi": evaluate(auction, profile).revenue, "Pi_theory": evaluate(auction, theory).revenue,
        "Pi_two_survivor": two_survivor_revenue(auction, profile),
        "open_high": learned[0, 1], "open_low": learned[0, 0],
    }
    for t in (1, 2, 3):  # both survivors still in, so no exits after the opening
        orbit = auction.orbits.index((t, (0,) * (n - 2)))
        row[f"war_t{t}"], row[f"war_t{t}_theory"] = learned[orbit, 1], target[orbit, 1]
    return row


def run_job(job: Tuple[int, str, Tuple[int, ...], int]) -> List[Row]:
    """Every seed of one run at one bidder count, scored under each policy it stands for."""
    n, run, seeds, iterations = job
    learner = auction_at(n, run)
    scorers = {policy: auction_at(n, policy) for policy in SCORED_UNDER.get(run, (run,))}
    theories = {policy: sorting_profile(auction) for policy, auction in scorers.items()}
    for policy, auction in scorers.items():
        assert utility_loss(auction, theories[policy]) <= certified(n), (n, policy)
    rows = []
    for seed in seeds:
        started = time.perf_counter()
        profile = solve(learner, seed=seed, iterations=iterations, trace_every=iterations,
                        symmetry_restricted=True).profile
        for policy, auction in scorers.items():
            rows.append(row_for(n, policy, seed, auction, theories[policy], profile))
            print(f"  n={n} {policy:13s} seed={seed} {rows[-1]['verdict']:9s} "
                  f"loss {rows[-1]['utility_loss']:.2e}  Pi {rows[-1]['Pi']:.4f}  "
                  f"{time.perf_counter() - started:.0f} s", flush=True)
    return rows


def pending(done: Iterable[Row], counts: Sequence[int], seeds: Sequence[int],
            iterations: int = BUDGET) -> List[Tuple[int, str, Tuple[int, ...], int]]:
    """Jobs for every run and seed not saved yet, largest bidder count first."""
    saved = {(int(row["n"]), str(row["policy"]), int(row["seed"])) for row in done}
    jobs = []
    for n in sorted(counts, reverse=True):
        for run in RUNS:
            missing = [seed for seed in seeds if (n, run, seed) not in saved]
            jobs += [(n, run, tuple(missing[i:i + CHUNK]), iterations) for i in range(0, len(missing), CHUNK)]
    return jobs


def read_rows() -> List[Row]:
    """The saved runs, or none before the first."""
    path = os.path.join(RESULTS, "scale.csv")
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle))


def claims(rows: Sequence[Row]) -> List[Row]:
    """Per bidder count, the learned revenue differences beside the formulas, over runs that reached an equilibrium."""
    reached = [row for row in rows if row["verdict"] != Verdict.UNSOLVED.value]
    revenue = lambda n, policy, column="Pi": [float(row[column]) for row in reached
                                              if int(row["n"]) == n and row["policy"] == policy]
    out = []
    for n in sorted({int(row["n"]) for row in rows}):
        if not all(revenue(n, p) for p in ("zero", "trigger")):
            continue
        gain = np.mean(revenue(n, "trigger")) - np.mean(revenue(n, "zero"))
        war = np.mean(revenue(n, "trigger", "Pi_two_survivor")) - np.mean(revenue(n, "zero", "Pi_two_survivor"))
        out.append({
            "n": n, "claim": "trigger_minus_none", "learned": gain, "predicted": trigger_gain(n),
            "two_survivor_part": war, "elsewhere_part": gain - war,
            "passes": bool(gain > 0 and abs(gain - trigger_gain(n)) <= 0.25 * trigger_gain(n)),
        })
        for suffix in ("", "_high"):
            if revenue(n, "preserve" + suffix) and revenue(n, "full" + suffix):
                saving = np.mean(revenue(n, "preserve" + suffix)) - np.mean(revenue(n, "full" + suffix))
                out.append({"n": n, "claim": "preserve_minus_full" + suffix, "learned": saving,
                            "predicted": preserve_saving(n, cost(n, "preserve" + suffix))})
    return out


def experiment(counts: Sequence[int] = COUNTS, seeds: Sequence[int] = range(SEEDS)) -> List[Row]:
    """Run every pending seed, saving after each job, then print the revenue claims."""
    rows = read_rows()
    jobs = pending(rows, counts, seeds)
    with ProcessPoolExecutor(WORKERS) as pool:
        for finished in as_completed([pool.submit(run_job, job) for job in jobs]):
            rows += finished.result()
            rows.sort(key=lambda row: (int(row["n"]), str(row["policy"]), int(row["seed"])))
            write("scale.csv", rows)
    for claim in claims(rows):
        print("  " + "  ".join(f"{key} {value:.4f}" if isinstance(value, float) else f"{key} {value}"
                               for key, value in claim.items()))
    return rows


def seed_range(text: str) -> range:
    """``3`` or ``0-9``."""
    first, _, last = text.partition("-")
    return range(int(first), int(last or first) + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", default=",".join(map(str, COUNTS)))
    parser.add_argument("--seeds", default=f"0-{SEEDS - 1}")
    arguments = parser.parse_args()
    print("== scale ==")
    experiment([int(n) for n in arguments.counts.split(",")], seed_range(arguments.seeds))
