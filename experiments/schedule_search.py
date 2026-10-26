"""Experiment 2: which round-only reimbursement schedule earns the most at learned play, at four bidders."""

from __future__ import annotations

import csv
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Sequence, Tuple

import numpy as np

from auction.exact import certified, evaluate, utility_loss
from auction.game import Auction, State, opening_margin, preserve, schedule
from auction.mccfr import BUDGET, solve
from experiments.scale import Q, RESULTS, write

N, ROUNDS = 4, 4
C = opening_margin(N, Q) + 0.05
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)  # each rho_t as a share of c
SEEDS = 10
NAMED = {"None": (0, 0, 0, 0), "Full": (1, 1, 1, 1), "back-loaded": (0, 0, 0, 1), "war-all": (0, 1, 1, 1)}  # shares of c

Row = Dict[str, object]
Policy = Tuple[str, Tuple[float, ...]]


def policies() -> List[Policy]:
    """Every grid schedule, then Preserve as a reference row."""
    grid = [("schedule", tuple(share * C for share in shares)) for shares in itertools.product(LEVELS, repeat=ROUNDS)]
    rho = preserve(C, ROUNDS, N, Q)
    return grid + [("preserve", tuple(rho(State(t, (), ())) for t in range(ROUNDS)))]


def run_job(policy: Policy) -> List[Row]:
    """Every seed under one policy, on the same seeds as every other policy."""
    label, rho = policy
    auction = Auction(n=N, q=Q, c=C, rounds=ROUNDS, delta=1.0, reimbursement_policy=schedule(rho))
    rows = []
    for seed in range(SEEDS):
        sigma = solve(auction, seed)
        loss = float(utility_loss(auction, sigma))
        rows.append({"policy": label, "c": C, **{f"rho{t}": rho[t] for t in range(ROUNDS)},
                     "seed": seed, "iterations": BUDGET, "reached": loss <= certified(N),
                     "utility_loss": loss, "Pi": evaluate(auction, sigma).revenue})
    return rows


def key(row: Row) -> Policy:
    """The policy a row was run under, rounded so rows read back from the CSV match."""
    return str(row["policy"]), tuple(round(float(row[f"rho{t}"]), 9) for t in range(ROUNDS))


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


def report(rows: Sequence[Row]) -> None:
    """The top ten schedules, the named and Preserve rows, and the paired gaps the paper cites."""
    ranked = ranking(rows)
    grid = [entry for entry in ranked if entry["policy"] == "schedule" and not entry["flagged"]]
    flagged = sum(entry["flagged"] for entry in ranked if entry["policy"] == "schedule")
    print(f"  {len(grid)} schedules ranked, {flagged} flagged")
    named = {name: ("schedule", tuple(round(share * C, 9) for share in shares)) for name, shares in NAMED.items()}
    top = {(entry["policy"], entry["rho"]) for entry in grid[:10]}
    rest = [entry for entry in ranked if (entry["policy"], entry["rho"]) not in top
            and (entry["policy"] != "schedule" or (entry["policy"], entry["rho"]) in named.values())]
    for entry in grid[:10] + rest:
        shares = " ".join(f"{rho / C:.2f}" for rho in entry["rho"])
        print(f"  {entry['policy']:9s} rho/c {shares}  reached {entry['reached']}/{entry['seeds']}"
              f"  Pi {entry['Pi']:.4f} +- {entry['se']:.4f}")
    if len(grid) > 1:
        revenue = lambda policy: {int(row["seed"]): float(row["Pi"]) for row in rows
                                  if key(row) == policy and str(row["reached"]) == "True"}
        best = ("schedule", grid[0]["rho"])
        for label, first, second in (("winner less runner-up", best, ("schedule", grid[1]["rho"])),
                                     ("winner less None", best, named["None"]),
                                     ("back-loaded less None", named["back-loaded"], named["None"])):
            one, other = revenue(first), revenue(second)
            gaps = np.array([one[seed] - other[seed] for seed in one.keys() & other.keys()])
            print(f"  {label} {gaps.mean():.4f} +- {gaps.std(ddof=1) / np.sqrt(len(gaps)):.4f}"
                  f" over {len(gaps)} paired seeds")


def experiment() -> List[Row]:
    """Run every policy not saved yet, saving after each, then print the ranking."""
    rows = read_rows()
    done = set(map(key, rows))
    jobs = [policy for policy in policies() if (policy[0], tuple(round(x, 9) for x in policy[1])) not in done]
    print(f"== schedule search == {len(jobs)} policies to run, c = {C:.4f}")
    with ProcessPoolExecutor() as pool:
        for finished in as_completed([pool.submit(run_job, job) for job in jobs]):
            rows += finished.result()
            write("schedule_search.csv", rows)
    report(rows)
    return rows


if __name__ == "__main__":
    experiment()
