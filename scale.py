"""Experiment 3: the rule the program writes at three, four and six bidders, and whether learning recovers it."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import analysis
import implement
import learning
import symmetric
from analysis import CERTIFIED, write
from game import Auction, equicorrelated
from implement import Implementation
from mccfr import BUDGET, REVENUE_BAR, STAY_BAR, Solved, Verdict, solve
from solver import evaluate
from validate import HOUR, SEEDS

COUNTS = (3, 4, 6)
ROUNDS = 6  # a finer clock than the example's four, over the same unit interval of mean-unit prices
Q, C, RHO = 0.5, 0.025, 0.25  # the correlation the experiment is specified to run at, as a pairwise rho

REFERENCE_SEED = 0  # the one run whose result becomes the reference profile the rule is written against
FREE_SEEDS = 3  # seeds the unrestricted companion arm gets, where every quantity is a bound or a sample

# Above this an asymmetric profile is read by sampling rather than by traversal, and the drop-out
# price bound stops being cheap enough to measure beside a walk. An exact utility loss at 1728
# information sets takes 46 s and at 93312 it is days, so the two the experiment reaches sit far
# either side of wherever the line is drawn.
EXACT_INFO_SETS = 5_000
DRAWS = 20_000  # episodes a sampled revenue is read off
SEARCH_DRAWS = 4_000  # episodes each candidate drop-out price strategy is read off


def auction_at(n: int, policy, c: float = C) -> Auction:
    """The experiment's auction at one bidder count, correlated signals and finer clock included."""
    return Auction(
        n=n, q=Q, c=c, rounds=ROUNDS, delta=1.0 / ROUNDS,
        reimbursement_policy=policy, signal_prior=equicorrelated(Q, RHO),
    )


def arrival(auction: Auction, sigma: np.ndarray) -> np.ndarray:
    """How often the profile itself arrives at each information set, which is what weights a distance."""
    return symmetric.bucket_arrival(auction, sigma).ravel()[auction.orbit_index]


# ---------------------------------------------------------------------------
# Reading a run, exactly where the tree allows it and by sampling where it does not
# ---------------------------------------------------------------------------


@dataclass
class Read:
    """What a finished profile was worth, and whether traversal or sampling is what says so."""

    utility_loss: float
    restricted_bound: Optional[float]  # the drop-out price bound, where measuring it is affordable
    revenue: float
    revenue_error: float
    exact: bool  # false when the loss is the bound itself and the revenue a sample


def sampled_payoffs(auction: Auction, seed: int) -> analysis.Payoffs:
    """Payoffs by playing the auction out, on one seed so that every candidate meets the same draws."""
    tables = learning.lookups(auction)  # built once, since the search plays several hundred profiles out
    return lambda profile: learning.sampled(
        auction, profile, seed=seed, episodes=SEARCH_DRAWS, tables=tables
    ).outcome.payoff


def read_exactly(auction: Auction, sigma: np.ndarray, measure_bound: bool) -> Read:
    """Walk the tree: an exact utility loss by best response, and an exact seller net revenue."""
    loss = float(analysis.utility_loss(auction, sigma))
    # Walked rather than sampled where it is recorded at all, since only then is it truly a lower
    # bound: sampling noise on a maximum over candidates can push the figure past the loss itself.
    payoff_of = lambda profile: evaluate(auction, profile).payoff
    bound = analysis.restricted_loss(auction, sigma, payoff_of) if measure_bound else None
    return Read(loss, bound, evaluate(auction, sigma).revenue, 0.0, True)


def read_by_sampling(auction: Auction, sigma: np.ndarray, seed: int) -> Read:
    """Play the auction out instead: the bound stands in for the loss, and the revenue is a sample."""
    bound = analysis.restricted_loss(auction, sigma, sampled_payoffs(auction, seed))
    drawn = learning.sampled(auction, sigma, seed=seed, episodes=DRAWS)
    return Read(bound, bound, drawn.outcome.revenue, drawn.revenue_error, False)


def walkable(auction: Auction, sigma: np.ndarray) -> bool:
    """Whether this profile's tree can be walked exactly, which the reduction settles where it applies."""
    return symmetric.applies(auction, sigma) or len(auction.info_sets) <= EXACT_INFO_SETS


def read(auction: Auction, sigma: np.ndarray, seed: int) -> Read:
    """Read a profile the affordable way, which at six bidders and an asymmetric profile is sampling."""
    if not walkable(auction, sigma):
        return read_by_sampling(auction, sigma, seed)
    # Measured beside the walk while the auction is small enough, so how far the bound falls short of
    # the loss it bounds is a number rather than a caveat. Past that it costs more than the walk did.
    return read_exactly(auction, sigma, measure_bound=len(auction.info_sets) <= EXACT_INFO_SETS)


def verdict_on(reading: Read, distance: float, target_revenue: float) -> Verdict:
    """The three-way reading of a run, against the reference profile rather than against a closed form."""
    if reading.utility_loss > CERTIFIED:
        return Verdict.UNSOLVED
    if distance <= STAY_BAR and abs(reading.revenue - target_revenue) <= REVENUE_BAR:
        return Verdict.DERIVED
    return Verdict.DIFFERENT


# ---------------------------------------------------------------------------
# The reference profile and the rule written against it
# ---------------------------------------------------------------------------


@dataclass
class Reference:
    """One bidder count's frictionless equilibrium, and the cheapest rule implementing it where bidding costs."""

    n: int
    frictionless: Auction
    profile: np.ndarray
    solved: Solved
    chosen: Implementation
    under: Auction  # the costly auction carrying the rule the program chose
    seconds: Dict[str, float]


def solved_at(auction: Auction, seed: int, iterations: int, restricted: bool) -> Tuple[Solved, float]:
    """One solver run, timed, since the runtime per bidder count is itself a recorded result."""
    started = time.perf_counter()
    solved = solve(
        auction, seed=seed, iterations=iterations, trace_every=iterations // 2, symmetry_restricted=restricted
    )
    return solved, time.perf_counter() - started


def reference_at(n: int, iterations: int = BUDGET) -> Reference:
    """Solve the frictionless auction, take the result as the reference, and write the cheapest rule for it."""
    frictionless = auction_at(n, "zero", c=0.0)
    solved, solving = solved_at(frictionless, REFERENCE_SEED, iterations, restricted=True)
    # The solver walked the histories and so materialised their tables. Nothing after this reads the
    # frictionless auction off anything but the reduction, and holding both auctions' tables at six
    # bidders is a gigabyte that puts the machine into swap.
    frictionless.release_history_tables()
    costly = auction_at(n, "full")
    started = time.perf_counter()
    chosen = implement.minimum_cost(costly, solved.profile)
    program = time.perf_counter() - started
    under = auction_at(n, implement.as_policy(costly, chosen.refunds))
    return Reference(
        n=n,
        frictionless=frictionless,
        profile=solved.profile,
        solved=solved,
        chosen=chosen,
        under=under,
        seconds={"reference_solve": solving, "linear_program": program},
    )


def rule_row(reference: Reference) -> Dict[str, object]:
    """What the program chose at one bidder count, beside what paying in full would have cost."""
    chosen, under, free = reference.chosen, reference.under, reference.frictionless
    row: Dict[str, object] = {
        "n": reference.n,
        "rounds": ROUNDS,
        "c": C,
        "q": Q,
        "rho": RHO,
        "orbits": len(under.orbits),
        "info_sets": len(under.info_sets),
        "reference_seed": REFERENCE_SEED,
        "reference_loss": float(analysis.utility_loss(free, reference.profile)),
        "reference_price": evaluate(free, reference.profile).price,
        "status": chosen.status,
        "constraints": chosen.constraints,
        "slack": chosen.slack,
        "bill": chosen.bill,
        "full_bill": chosen.full_bill,
        "saving": 1.0 - chosen.bill / chosen.full_bill,
        # What the seller is left with if learning does recover the reference, which is the claim under test.
        "target_revenue": evaluate(under, reference.profile).revenue,
        "target_loss_under_rule": float(analysis.utility_loss(under, reference.profile)),
        **reference.seconds,
    } | implement.schedule_of(under, chosen.refunds)
    assert row["status"] == "optimal", row
    assert chosen.bill <= chosen.full_bill + 1e-12, row
    # The reference has to be an equilibrium where bidding is free, or there is nothing to implement.
    assert float(row["reference_loss"]) <= CERTIFIED, row
    return row


# ---------------------------------------------------------------------------
# Whether learning recovers the reference under the rule written for it
# ---------------------------------------------------------------------------


def recovery_row(reference: Reference, seed: int, restricted: bool, iterations: int = BUDGET) -> Dict[str, object]:
    """One run under the chosen rule, scored against the reference profile it is meant to recover."""
    under, target = reference.under, reference.profile
    started = time.perf_counter()
    solved, solving = solved_at(under, seed, iterations, restricted)
    # Timed around the reading as well: where the tree is too large, playing several hundred candidate
    # strategies out costs more than the solve did, and the hour budget is owed that figure too.
    reading = read(under, solved.profile, seed)
    seconds = time.perf_counter() - started
    # At two bidders any arrival at all is on the path. Here most sets are arrived at a handful of
    # times in a million, and what is played there is a tie-break no budget informs, so the bar the
    # learners' own coverage uses is what separates the two.
    arrived = arrival(under, target)
    on_path = arrived > learning.REACHED
    distance = np.abs(solved.profile - target)
    on_path_distance = float(distance[on_path].max())
    weighted_distance = float(distance @ arrived / arrived.sum())
    target_revenue = evaluate(under, target).revenue
    visited = solved.visits[solved.visits > 0]
    assert visited.size, (reference.n, seed)
    thin, median, thick = np.quantile(visited, [0.1, 0.5, 0.9])
    row: Dict[str, object] = {
        "n": reference.n,
        "rounds": ROUNDS,
        "c": C,
        "q": Q,
        "rho": RHO,
        "seed": seed,
        "symmetry_restricted": restricted,
        "iterations": solved.iterations,
        "seconds": seconds,
        "solve_seconds": solving,
        # False where the tree is too large to walk, so the loss is a bound and the revenue a sample.
        "exact": reading.exact,
        # Read on the arrival-weighted distance, since a maximum over the tens of thousands of sets a
        # scaled auction has is decided by the thinnest of them. The maximum is carried beside it.
        "verdict": verdict_on(reading, weighted_distance, target_revenue).value,
        "verdict_at_the_maximum": verdict_on(reading, on_path_distance, target_revenue).value,
        "utility_loss": reading.utility_loss,
        "certified": reading.utility_loss <= CERTIFIED,
        # Recorded beside the loss where both are affordable, so the gap between them is measured.
        "restricted_bound": "" if reading.restricted_bound is None else reading.restricted_bound,
        "on_path_distance": on_path_distance,
        "off_path_distance": float(distance[~on_path].max()),
        # The same distance without the threshold, weighted by how often the target arrives at all.
        "arrival_weighted_distance": weighted_distance,
        "on_path_sets": int(on_path.sum()),
        "Pi": reading.revenue,
        "Pi_error": reading.revenue_error,
        "Pi_target": target_revenue,
        "revenue_distance": abs(reading.revenue - target_revenue),
        "info_sets": len(under.info_sets),
        "visited": int((solved.visits > 0).sum()),
        "visits_p10": thin,
        "visits_median": median,
        "visits_p90": thick,
    }
    assert 0.0 <= on_path_distance <= 1.0, row
    assert row["exact"] == (reading.revenue_error == 0.0), row
    assert 0.0 <= weighted_distance <= 1.0, row
    assert seconds >= solving, row
    if reading.restricted_bound is not None:
        # Fewer deviations can only find less, which is the whole claim the bound rests on.
        assert reading.restricted_bound <= reading.utility_loss + 1e-12, row
    return row


def report(rows: Sequence[Dict[str, object]], n: int, restricted: bool) -> None:
    """Print what the seeds at one bidder count did, as counts rather than as an average over equilibria."""
    at = [row for row in rows if row["n"] == n and row["symmetry_restricted"] == restricted]
    if not at:
        return
    counts = " ".join(f"{verdict.value} {sum(row['verdict'] == verdict.value for row in at)}" for verdict in Verdict)
    reach = lambda key: (min(float(row[key]) for row in at), max(float(row[key]) for row in at))
    print(
        f"  n={n} {'restricted' if restricted else 'unrestricted':12s} {counts}"
        + "  on-path %.4f to %.4f" % reach("on_path_distance")
        + "  loss %.2e to %.2e" % reach("utility_loss")
        + "  Pi %.4f to %.4f" % reach("Pi")
        + "  %.0f s a run" % (sum(float(row["seconds"]) for row in at) / len(at))
    )


def experiment(counts: Sequence[int] = COUNTS, seeds: int = SEEDS, free_seeds: int = FREE_SEEDS) -> Tuple[
    List[Dict[str, object]], List[Dict[str, object]]
]:
    """Every bidder count's reference profile and rule, and the seeds testing whether learning recovers it."""
    rules, rows = [], []
    for n in counts:
        reference = reference_at(n)
        rules.append(rule_row(reference))
        rows += [recovery_row(reference, seed, restricted=True) for seed in range(seeds)]
        # The companion arm, where nothing coordinates the bidders and no exact reading is affordable.
        rows += [recovery_row(reference, seed, restricted=False) for seed in range(free_seeds)]
        for restricted in (True, False):
            report(rows, n, restricted)
        # Rewritten per bidder count rather than at the end, since six bidders alone cost most of an
        # hour and a count already measured is worth keeping whatever the next one does.
        write("scale_rules.csv", rules)
        write("scale.csv", rows)
        reference.under.release_history_tables()  # before the next count builds its own
    return rules, rows


def check(rules: Sequence[Dict[str, object]], rows: Sequence[Dict[str, object]]) -> None:
    """What experiment 3 claims, checked once the measurements behind it are on disk."""
    seen = [(row["n"], row["seed"], row["symmetry_restricted"]) for row in rows]
    assert len(set(seen)) == len(rows), len(rows)
    assert {row["n"] for row in rules} == {row["n"] for row in rows}, rules

    # The rule is written against a certified reference and leaves it certified, which is what implementing means.
    for rule in rules:
        assert float(rule["target_loss_under_rule"]) <= CERTIFIED, rule
        assert float(rule["saving"]) > 0.0, rule
    print(f"  the program paid {min(float(rule['saving']) for rule in rules):.0%} to "
          f"{max(float(rule['saving']) for rule in rules):.0%} less than reimbursing in full")

    # A run that found another equilibrium is a result about the rule, so it is counted rather than
    # asserted away. What is asserted is that such a run is certified, which is what makes it one,
    # and only where the loss behind it was walked rather than bounded from below.
    exact = [row for row in rows if row["exact"]]
    others = [row for row in exact if row["verdict"] == Verdict.DIFFERENT.value]
    assert all(float(row["utility_loss"]) <= CERTIFIED for row in others), others
    print(f"  runs landing on another equilibrium: {len(others)} of {len(exact)} read exactly")

    # Iteration count certifies nothing past two bidders, so the distance to equilibrium is what does,
    # and the verdict has to agree with the distance rather than with how long the run went on for.
    for row in rows:
        assert bool(row["certified"]) == (row["verdict"] != Verdict.UNSOLVED.value), row
        assert int(row["iterations"]) == BUDGET, row
    bounded = [row for row in rows if not row["exact"]]
    assert all(not row["symmetry_restricted"] for row in bounded), bounded
    assert all(float(row["Pi_error"]) > 0.0 for row in bounded), bounded
    print(f"  runs read by a drop-out price bound and a sampled revenue: {len(bounded)} of {len(rows)}")
    print(f"  of those, {sum(bool(row['certified']) for row in bounded)} are not ruled out as equilibria, "
          "which a lower bound cannot establish either way")

    # How far the bound falls short, at the rows where the loss it bounds was also walked. This is the
    # only measurement of what the bounded rows are worth, so it is printed rather than left implicit.
    measured = [row for row in rows if row["restricted_bound"] != ""]
    short = [float(row["utility_loss"]) - float(row["restricted_bound"]) for row in measured]
    assert all(gap >= -1e-12 for gap in short), measured
    missed = [row for row in measured if float(row["restricted_bound"]) <= CERTIFIED < float(row["utility_loss"])]
    print(f"  the drop-out bound sits {min(short):.2e} to {max(short):.2e} under the loss it bounds, "
          f"on {len(measured)} rows where both were read")
    print(f"  and on {len(missed)} of them it clears the certification bar where the loss does not")

    spent = sum(float(row["seconds"]) for row in rows) + sum(
        float(rule["reference_solve"]) + float(rule["linear_program"]) for rule in rules
    )
    for n in sorted({row["n"] for row in rows}):
        at = [row for row in rows if row["n"] == n]
        solving = sum(float(row["solve_seconds"]) for row in at)
        print(
            f"  n={n}: {sum(float(row['seconds']) for row in at) / 60:.1f} min over {len(at)} runs, "
            f"{solving / 60:.1f} of it solving and the rest reading what was solved"
        )
    # Reported rather than asserted, since the experiment is not trimmed to fit the hour the two-bidder
    # pipelines are held to.
    print(f"  the hour budget: {spent / 60:.1f} min spent, which is {spent / HOUR:.2f} of it")


if __name__ == "__main__":
    print("== scale ==")
    check(*experiment())
