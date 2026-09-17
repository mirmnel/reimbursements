"""Emergence experiments: what independent learners arrive at, scored against certified equilibria."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

import learning
import symmetric
from analysis import (
    ALWAYS,
    CERTIFIED,
    NEVER,
    SEPARATING,
    as_strategy,
    empirical_game,
    utility_loss,
    war_of_attrition,
    write,
)
from solver import evaluate, solve
from game import REIMBURSEMENT_FAMILIES, Auction

COST, SIGNAL_PROB, ROUNDS = 0.025, 0.5, 4
SOLVER_ITERATIONS = 200
EXACTLY_SCORABLE = 4  # bidder count above which an asymmetric profile is out of exact reach


@dataclass
class Certified:
    """What the solver established at one setting, and how well."""

    revenue: float
    utility_loss: float


def certified(auction: Auction) -> Certified:
    """The solver's equilibrium at one setting, verified before anything is scored against it."""
    sigma = solve(auction, iterations=SOLVER_ITERATIONS)
    loss = utility_loss(auction, sigma)
    assert loss < CERTIFIED, (auction.n, loss)
    return Certified(evaluate(auction, sigma).revenue, loss)


def emerged(
    auction: Auction, seed: int, ground: Certified, as_it_stands: bool = False
) -> Dict[str, object]:
    """One learning run, scored exactly on its symmetrisation and by simulation as it stands."""
    learned = learning.learn(auction, seed=seed, episodes=learning.budget(auction.n))
    scored = learning.score(auction, learned, seed=seed)
    opening = symmetric.anonymise(auction, learned.profile)[0, auction.n]
    assert learned.visited > 0, (auction.n, seed)
    thin, median, thick = learned.visit_quantiles(0.1, 0.5, 0.9)
    row: Dict[str, object] = {
        "n": auction.n,
        "seed": seed,
        "c": auction.c,
        "q": auction.q,
        "Pi_certified": ground.revenue,
        "utility_loss_certified": ground.utility_loss,
        "Pi_emergent": scored.revenue,
        "utility_loss_emergent": scored.utility_loss,
        "Pi_sampled": scored.sampled_revenue,
        "Pi_sampled_error": scored.sampled_error,
        "dispersion": scored.dispersion,
        "symmetrising_misleads": scored.misleading,
        "cycling": scored.cycling,
        "opening_stay_H": float(opening[1]),
        "opening_stay_L": float(opening[0]),
        "info_sets": len(auction.info_sets),
        "visited": learned.visited,
        "visits_p10": thin,
        "visits_median": median,
        "visits_p90": thick,
        # Exact only while the enumeration is in reach.
        "utility_loss_as_it_stands": utility_loss(auction, learned.profile) if as_it_stands else "",
    }
    assert row["utility_loss_emergent"] >= 0.0, row
    assert row["opening_stay_H"] > row["opening_stay_L"], row  # a high signal is worth more
    if auction.c > auction.reimburses(0):  # a free opening gives L no reason to exit
        assert float(row["opening_stay_L"]) < 0.2, row
    assert scored.sampled_error < 0.01, row  # otherwise the sampled revenue says too little
    return row


def report(name: str, rows: Sequence[Dict[str, object]], field: str) -> None:
    """Print the spread of one recorded field, which a mean alone would hide."""
    labels = ("min", "lower", "median", "upper", "max")
    quantiles = np.quantile([float(row[field]) for row in rows], [0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"  {name} {field}: " + " ".join(f"{label} {value:.4f}" for label, value in zip(labels, quantiles)))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def recovery(seeds: int = 10, counts: Sequence[int] = (4, 6)) -> List[Dict[str, object]]:
    """Validate the learner where exact ground truth exists, before trusting it at ten bidders."""
    rows = []
    for n in counts:
        auction = Auction(n=n, q=SIGNAL_PROB, c=COST, rounds=ROUNDS, reimbursement_policy="targeted")
        ground = certified(auction)
        for seed in range(seeds):
            rows.append(emerged(auction, seed, ground, as_it_stands=n <= EXACTLY_SCORABLE))
        at = [row for row in rows if row["n"] == n]
        report(f"n={n}", at, "utility_loss_emergent")
        report(f"n={n}", at, "dispersion")
        print(f"  n={n} runs flagged: {sum(row['symmetrising_misleads'] for row in at)} dispersed, "
              f"{sum(row['cycling'] for row in at)} cycling, out of {len(at)}")
    write("recovery.csv", rows)
    return rows


def at_scale(counts: Sequence[int] = (4, 6)) -> List[Dict[str, object]]:
    """The empirical game at scale, with the learned strategy alongside the named benchmarks."""
    rows = []
    for n in counts:
        auction = Auction(n=n, q=SIGNAL_PROB, c=COST, rounds=ROUNDS, reimbursement_policy="targeted")
        learned = learning.learn(auction, seed=0, episodes=learning.budget(n))
        scored = learning.score(auction, learned)
        strategies = {
            "separating": SEPARATING,
            "war_of_attrition": war_of_attrition(COST),
            "learned": as_strategy(auction, symmetric.symmetrise(auction, learned.profile)),
            "always_stay": ALWAYS,
            "always_exit": NEVER,
        }
        names, payoffs, regrets = empirical_game(auction, strategies)
        regret = dict(zip(names, regrets))
        named = [value for name, value in regret.items() if name != "learned"]
        assert regret["learned"] < min(named), (n, regret)  # the learners beat every benchmark
        assert min(regret["always_stay"], regret["always_exit"]) > 1e-6, (n, regret)
        rows += [
            {
                "n": n,
                "c": COST,
                "q": SIGNAL_PROB,
                "strategy": name,
                "payoff": payoffs[index, index],
                "regret": regrets[index],
                "source": "learner" if name == "learned" else "benchmark",
                # A symmetrisation, so it carries the spread symmetrising hid.
                "dispersion": learned.dispersion if name == "learned" else "",
                "symmetrising_misleads": scored.misleading if name == "learned" else "",
            }
            for index, name in enumerate(names)
        ]
        print(f"  n={n} regret: " + " ".join(f"{name}={value:.4f}" for name, value in regret.items()))
    write("empirical_game_at_scale.csv", rows)
    return rows


def emergent(seeds: int = 30, n: int = 10) -> List[Dict[str, object]]:
    """The headline run: thirty seeds at ten bidders across the reimbursement families."""
    rows = []
    for family, build in REIMBURSEMENT_FAMILIES.items():
        auction = Auction(
            n=n, q=SIGNAL_PROB, c=COST, rounds=ROUNDS, reimbursement_policy=build(COST, ROUNDS)
        )
        ground = certified(auction)
        for seed in range(seeds):
            rows.append({"family": family} | emerged(auction, seed, ground))
        at = [row for row in rows if row["family"] == family]
        report(family, at, "Pi_emergent")
        report(family, at, "utility_loss_emergent")
        print(f"  {family} certified Pi {ground.revenue:.4f} at utility loss {ground.utility_loss:.2e}, "
              f"{sum(row['symmetrising_misleads'] for row in at)} of {len(at)} runs too dispersed to symmetrise")
    write("emergence.csv", rows)
    return rows


if __name__ == "__main__":
    for experiment in (recovery, at_scale, emergent):
        print(f"== {experiment.__name__} ==")
        experiment()
