"""Experiment 2: which round-only reimbursement schedule earns the most at learned play, at four bidders."""

from __future__ import annotations

import csv
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Sequence, Tuple

import numpy as np

from auction.exact import certified, evaluate, utility_loss
from auction.game import Auction, State, opening_margin, preserve, schedule, trigger
from auction.mccfr import BUDGET, solve
from experiments.scale import Q, RESULTS, write

N, ROUNDS = 4, 4  # delta 1, so prices 0 to 3 and survivors split at the cap n
C = opening_margin(N, Q) + 0.05  # above the opening margin, so Preserve pays c - w0 > 0
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)  # each rho_t as a share of c
SEEDS = 10
WORKERS = 8

Row = Dict[str, object]
Policy = Tuple[str, Tuple[float, ...]]  # (label, rho), rho empty for the trigger


def policies() -> List[Policy]:
    """Every grid schedule, then the trigger and Preserve as reference rows."""
    grid = [("schedule", tuple(share * C for share in shares)) for shares in itertools.product(LEVELS, repeat=ROUNDS)]
    rho = preserve(C, ROUNDS, N, Q)
    return grid + [("trigger", ()), ("preserve", tuple(rho(State(t, (), ())) for t in range(ROUNDS)))]


def run_job(policy: Policy) -> List[Row]:
    """Every seed under one policy, on the same seeds as every other policy."""
    label, rho = policy
    rule = trigger(C, N) if label == "trigger" else schedule(rho)
    auction = Auction(n=N, q=Q, c=C, rounds=ROUNDS, delta=1.0, reimbursement_policy=rule)
    rows = []
    for seed in range(SEEDS):
        sigma = solve(auction, seed)
        loss = float(utility_loss(auction, sigma))
        rows.append({"policy": label, "c": C, **{f"rho{t}": rho[t] if rho else None for t in range(ROUNDS)},
                     "seed": seed, "iterations": BUDGET, "reached": loss <= certified(N),
                     "utility_loss": loss, "Pi": evaluate(auction, sigma).revenue})
    return rows


def key(row: Row) -> Policy:
    """The policy a row was run under, rounded so rows read back from the CSV match."""
    return str(row["policy"]), tuple(round(float(row[f"rho{t}"]), 9) for t in range(ROUNDS) if row[f"rho{t}"] not in (None, ""))


def read_rows() -> List[Row]:
    """The saved runs, or none before the first."""
    path = os.path.join(RESULTS, "schedule_search.csv")
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle))


def ranking(rows: Sequence[Row]) -> List[Row]:
    """Per policy, how many seeds reached an equilibrium and the mean revenue over them, best first."""
    grouped: Dict[Policy, List[Row]] = {}
    for row in rows:
        grouped.setdefault(key(row), []).append(row)
    out = []
    for policy, at in grouped.items():
        revenue = [float(row["Pi"]) for row in at if str(row["reached"]) == "True"]
        out.append({"policy": policy[0], "rho": policy[1], "reached": len(revenue), "seeds": len(at),
                    "flagged": 2 * len(revenue) < len(at),
                    "Pi": np.mean(revenue) if revenue else np.nan,
                    "se": np.std(revenue, ddof=1) / np.sqrt(len(revenue)) if len(revenue) > 1 else np.nan})
    return sorted(out, key=lambda entry: (np.isnan(entry["Pi"]), -entry["Pi"]))


def paired_gap(rows: Sequence[Row], first: Policy, second: Policy) -> Tuple[float, float, int]:
    """Mean revenue gap over seeds where both reached an equilibrium, its standard error and the seed count."""
    revenue = lambda policy: {int(row["seed"]): float(row["Pi"]) for row in rows
                              if key(row) == policy and str(row["reached"]) == "True"}
    one, other = revenue(first), revenue(second)
    gaps = np.array([one[seed] - other[seed] for seed in one.keys() & other.keys()])
    return float(gaps.mean()), float(gaps.std(ddof=1) / np.sqrt(len(gaps))), len(gaps)


def report(rows: Sequence[Row]) -> None:
    """The top ten schedules, the reference rows, and the winner's paired gap to the runner-up."""
    ranked = ranking(rows)
    grid = [entry for entry in ranked if entry["policy"] == "schedule" and not entry["flagged"]]
    flagged = sum(entry["flagged"] for entry in ranked if entry["policy"] == "schedule")
    print(f"  {len(grid)} schedules ranked, {flagged} flagged")
    for entry in grid[:10] + [entry for entry in ranked if entry["policy"] != "schedule"]:
        shares = " ".join(f"{rho / C:.2f}" for rho in entry["rho"]) or "        -"
        print(f"  {entry['policy']:9s} rho/c {shares}  reached {entry['reached']}/{entry['seeds']}"
              f"  Pi {entry['Pi']:.4f} +- {entry['se']:.4f}")
    if len(grid) > 1:
        gap, se, seeds = paired_gap(rows, ("schedule", grid[0]["rho"]), ("schedule", grid[1]["rho"]))
        print(f"  winner less runner-up {gap:.4f} +- {se:.4f} over {seeds} paired seeds")


def experiment() -> List[Row]:
    """Run every policy not saved yet, saving after each, then print the ranking."""
    rows = read_rows()
    done = set(map(key, rows))
    jobs = [policy for policy in policies() if (policy[0], tuple(round(x, 9) for x in policy[1])) not in done]
    print(f"== schedule search == {len(jobs)} policies to run, c = {C:.4f}")
    with ProcessPoolExecutor(WORKERS) as pool:
        for finished in as_completed([pool.submit(run_job, job) for job in jobs]):
            rows += finished.result()
            write("schedule_search.csv", rows)
    report(rows)
    return rows


if __name__ == "__main__":
    experiment()
