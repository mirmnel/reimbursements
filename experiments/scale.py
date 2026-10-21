"""Experiment 2: the rule the program writes at three, four and six bidders, and whether learning recovers it."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from auction import implement, symmetric
from auction.game import Auction, by_orbit, equicorrelated, payoff_scale
from auction.implement import Implementation
from auction.mccfr import BUDGET, REVENUE_BAR, STAY_BAR, Solved, Verdict, solve
from auction.exact import certified, evaluate, utility_loss
from experiments.two_bidder import SEEDS, write

COUNTS = (3, 4, 6)
ROUNDS = 6  # a finer clock than the example's four, spanning the same range of common values
Q, C, RHO = 0.5, 0.05, 0.25  # C at two bidders, stretched with the value, and RHO a pairwise correlation

REFERENCE_SEED = 0  # the run whose result becomes the reference profile the rule is written against
FREE_SEEDS = 3  # seeds the unrestricted arm gets

# The unrestricted arm runs only below this, since an asymmetric profile's exact utility loss
# takes 46 s at 1728 information sets and days at 93312.
EXACT_INFO_SETS = 5_000


def auction_at(n: int, policy, c: float = C) -> Auction:
    """The experiment's auction at one bidder count, correlated signals and finer clock included."""
    return Auction(
        n=n, q=Q, c=c * payoff_scale(n), rounds=ROUNDS, delta=n / ROUNDS,
        reimbursement_policy=policy, signal_prior=equicorrelated(Q, RHO),
    )


@dataclass
class Reference:
    """One bidder count's frictionless equilibrium, and the cheapest rule implementing it where bidding costs."""

    n: int
    frictionless: Auction
    profile: np.ndarray
    chosen: Implementation
    under: Auction  # the costly auction carrying the rule the program chose


def solved_at(auction: Auction, seed: int, restricted: bool) -> Solved:
    """One solver run at the budget."""
    return solve(auction, seed=seed, iterations=BUDGET, trace_every=BUDGET, symmetry_restricted=restricted)


def reference_at(n: int) -> Reference:
    """Solve the frictionless auction, take the result as the reference, and write the cheapest rule for it."""
    frictionless = auction_at(n, "zero", c=0.0)
    profile = solved_at(frictionless, REFERENCE_SEED, restricted=True).profile
    # Holding both auctions' history tables at six bidders is a gigabyte, and only the reduction reads
    # the frictionless auction from here on.
    frictionless.release_history_tables()
    costly = auction_at(n, "full")
    chosen = implement.minimum_cost(costly, profile)
    under = auction_at(n, by_orbit(dict(zip(costly.orbits, map(float, chosen.refunds)))))
    return Reference(n, frictionless, profile, chosen, under)


def rule_row(reference: Reference) -> Dict[str, object]:
    """What the program chose at one bidder count, beside what paying in full would have cost."""
    chosen, under, free = reference.chosen, reference.under, reference.frictionless
    rounds = np.array([round_index for round_index, _ in under.orbits])
    row: Dict[str, object] = {
        "n": reference.n,
        "c": under.c,
        "reference_loss": float(utility_loss(free, reference.profile)),
        "status": chosen.status,
        "bill": chosen.bill,
        "full_bill": chosen.full_bill,
        "target_revenue": evaluate(under, reference.profile).revenue,
        "target_loss_under_rule": float(utility_loss(under, reference.profile)),
    } | {  # per stay in each round, averaged over that round's orbits
        f"r_t{t}": float(chosen.refunds[rounds == t].mean()) for t in range(under.rounds)
    }
    assert row["status"] == "optimal", row
    assert chosen.bill <= chosen.full_bill + 1e-12, row
    # The reference has to be an equilibrium where bidding is free and stay one under the rule.
    assert float(row["reference_loss"]) <= certified(reference.n), row
    assert float(row["target_loss_under_rule"]) <= certified(reference.n), row
    return row


def recovery_row(reference: Reference, seed: int, restricted: bool) -> Dict[str, object]:
    """One run under the chosen rule, scored against the reference profile it is meant to recover."""
    under, target = reference.under, reference.profile
    started = time.perf_counter()
    solved = solved_at(under, seed, restricted)
    loss = float(utility_loss(under, solved.profile))
    revenue = evaluate(under, solved.profile).revenue
    seconds = time.perf_counter() - started
    # The worst bidder's, so one bidder off the target is not averaged away. Weighted by the target's
    # arrival, since most sets at this scale are arrived at a handful of times in a million.
    arrived = symmetric.bucket_arrival(under, target).ravel()[under.orbit_index]
    bidder = [b for b, _, _ in under.info_sets]
    off = np.abs(solved.profile - target) * arrived
    weighted_distance = float(max(np.bincount(bidder, off) / np.bincount(bidder, arrived)))
    target_revenue = evaluate(under, target).revenue
    row: Dict[str, object] = {
        "n": reference.n,
        "c": under.c,
        "seed": seed,
        "symmetry_restricted": restricted,
        "verdict": (Verdict.UNSOLVED if loss > certified(reference.n)
                    else Verdict.DERIVED if weighted_distance <= STAY_BAR else Verdict.DIFFERENT).value,
        "utility_loss": loss,
        "arrival_weighted_distance": weighted_distance,
        "Pi": revenue,
        "Pi_target": target_revenue,
    }
    print(f"  n={reference.n} seed={seed} {'restricted' if restricted else 'unrestricted':12s} "
          f"{row['verdict']:9s} loss {loss:.2e}  Pi {revenue:.4f}  {seconds:.0f} s")
    return row


def experiment(counts: Sequence[int] = COUNTS) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Every bidder count's reference profile and rule, and the seeds testing whether learning recovers it."""
    rules, rows = [], []
    for n in counts:
        reference = reference_at(n)
        rules.append(rule_row(reference))
        rows += [recovery_row(reference, seed, restricted=True) for seed in range(SEEDS)]
        if len(reference.under.info_sets) <= EXACT_INFO_SETS:
            rows += [recovery_row(reference, seed, restricted=False) for seed in range(FREE_SEEDS)]
        # Rewritten per count, since six bidders alone cost most of an hour.
        write("scale_rules.csv", rules)
        write("scale.csv", rows)
        reference.under.release_history_tables()
    matched = [r for r in rows if r["verdict"] == Verdict.DERIVED.value]
    assert all(abs(r["Pi"] - r["Pi_target"]) <= REVENUE_BAR * payoff_scale(r["n"]) for r in matched), matched
    return rules, rows


if __name__ == "__main__":
    print("== scale ==")
    experiment()
