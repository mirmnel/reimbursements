"""External-sampling Monte Carlo counterfactual regret minimisation, and the verdict on what it returns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

import numpy as np

from . import game
from .exact import CERTIFIED, RESOLVED, evaluate, reach, utility_loss
from .game import Auction, State, payoff_scale


STAY = 1

INITIAL_REGRET = 0.1  # width of the interval initial regrets are drawn from at two bidders, one draw per seed
TRACE_EVERY = 20

BUDGET = 100_000  # iterations every run gets

STAY_BAR = 0.05  # how far an on-path stay probability may sit from the closed form.
REVENUE_BAR = 0.02  # how far a match's seller net revenue may sit from theory, checked after the verdict


@dataclass
class Solved:
    """What one run produced, with what is needed to judge it."""

    profile: np.ndarray  # average behavioural strategy, flat over the information sets
    averages: np.ndarray  # (sample, |I|) the average profile along the run
    iterations: int


def matched(regret: np.ndarray) -> np.ndarray:
    """Stay with probability proportional to the positive regret of not having stayed."""
    regret = np.maximum(regret, 0.0)
    total = regret.sum(-1)
    # Nothing regretted yet, so no reason to lean either way.
    return np.divide(regret[..., STAY], total, out=np.full(np.shape(total), 0.5), where=total > 0.0)


def matched_one(regret: list) -> float:
    """``matched`` on one information set's [exit, stay] pair, in plain floats for the traversal."""
    exiting, staying = max(regret[0], 0.0), max(regret[1], 0.0)
    return staying / (exiting + staying) if exiting + staying > 0.0 else 0.5


def averaged(strategy_sum: np.ndarray, regret: np.ndarray) -> np.ndarray:
    """The average behavioural strategy, falling back to the matched one where the average carries no weight."""
    # External sampling averages at opponent nodes only, so a bidder's own off-path sets never
    # accumulate. Their regret did accumulate, by enumeration, and it is what the solver knows there.
    total = strategy_sum.sum(-1)
    weighed = np.divide(strategy_sum[..., STAY], np.where(total > 0.0, total, 1.0))
    return np.where(total > 0.0, weighed, matched(regret))


def solve(
    auction: Auction,
    seed: int = 0,
    iterations: int = BUDGET,
    trace_every: int = TRACE_EVERY,
    symmetry_restricted: bool = False,
) -> Solved:
    """Run external-sampling MCCFR with regret matching and linear averaging, sampling opponents on-policy."""
    # Restricted, every bidder reads and writes one shared table, so the search is symmetric by
    # construction rather than a finished profile being averaged. Unrestricted, each has its own.
    rng = np.random.default_rng(seed)
    initial_regret = INITIAL_REGRET * payoff_scale(auction.n)
    index = auction.info_index
    if symmetry_restricted:  # a set's cell is its orbit under relabelling the bidders
        buckets: Dict[Tuple[int, int], int] = {}
        keys = [(auction.orbit_of_state[state], signal) for _, state, signal in auction.info_sets]
        cells = np.array([buckets.setdefault(key, len(buckets)) for key in keys])
    else:
        cells = np.arange(len(auction.info_sets))
    cell_of = cells.tolist()  # plain ints, which the traversal indexes with far more cheaply
    # Plain lists, as numpy's per-call overhead dominates arithmetic on two floats.
    regret = rng.uniform(0.0, initial_regret, (int(cells.max()) + 1, 2)).tolist()  # a random start per seed
    strategy_sum = [[0.0, 0.0] for _ in regret]
    follow: Dict[State, Dict[int, State]] = {
        state: dict(auction.continuations[state]) for state in auction.states
    }

    def walk(state: State, traverser: int, signals: Tuple[int, ...], drawn: int, weight: float) -> float:
        """One traversal from ``state`` under a drawn signal profile, worth what it pays ``traverser``."""
        active = state.active
        payoffs, onward = auction.branch_payoffs[state], follow[state]
        sampled = 0
        for position, bidder in enumerate(active):
            if bidder == traverser:
                continue
            cell = cell_of[index[(bidder, state, signals[bidder])]]
            stay = matched_one(regret[cell])
            row = strategy_sum[cell]
            row[0] += weight * (1.0 - stay)
            row[1] += weight * stay
            if rng.random() < stay:
                sampled |= 1 << position

        def value(mask: int) -> float:
            """What this round's stayers pay the traverser, plus whatever the auction leads to."""
            got = float(payoffs[mask][drawn, traverser])
            return got + walk(onward[mask], traverser, signals, drawn, weight) if mask in onward else got

        if traverser not in active:  # it has left, and nothing after that pays it
            return value(sampled)
        row = regret[cell_of[index[(traverser, state, signals[traverser])]]]
        stay = matched_one(row)
        staying, exiting = value(sampled | 1 << active.index(traverser)), value(sampled)
        expected = stay * staying + (1.0 - stay) * exiting
        # Not floored as in RM+, which turns sampling noise at an indifferent set into drift.
        row[0] += exiting - expected
        row[1] += staying - expected
        return expected

    averages = []
    for iteration in range(1, iterations + 1):
        for traverser in range(auction.n):
            drawn = int(rng.choice(auction.num_profiles, p=auction.prior))
            # The iteration is its own weight in the average, which is the linear weighting.
            walk(auction.root(), traverser, tuple(auction.profiles[drawn]), drawn, float(iteration))
        if iteration % trace_every == 0:
            averages.append(averaged(np.array(strategy_sum), np.array(regret))[cells])
    return Solved(
        profile=averaged(np.array(strategy_sum), np.array(regret))[cells],
        averages=np.array(averages),
        iterations=iterations,
    )


class Verdict(str, Enum):
    """The three things a scored profile can be, which call for three different responses."""

    DERIVED = "derived"  # certified an equilibrium, and it is the one the example derives
    DIFFERENT = "different"  # certified an equilibrium, and it is another one, which is a result
    UNSOLVED = "unsolved"  # not certified, so the solver or its budget is at fault


@dataclass
class Score:
    """The verdict on a finished profile, with the numbers it was reached from."""

    verdict: Verdict
    utility_loss: float  # exact, by best response over the tree, never reading a regret table
    revenue: float  # seller net revenue


def closed_forms(auction: Auction) -> Tuple[np.ndarray, float]:
    """The two-bidder example's closed-form profile and seller net revenue under this auction's rule."""
    strategy, revenue = game.closed_form(auction)
    return game.profile(auction, strategy), revenue


def stay_schedule(auction: Auction, sigma: np.ndarray) -> Dict[str, float]:
    """The stay probabilities the two-bidder closed forms are stated as, read off one bidder."""
    # At two bidders a round has exactly one history, so the round indexes the states directly.
    stay = lambda round_index, signal: float(sigma[auction.info_index[(0, auction.states[round_index], signal)]])
    return {
        "open_high": stay(0, 1),
        "open_low": stay(0, 0),
        "sigma1": stay(1, 1),
        "sigma2": stay(2, 1),
        "sigma3": stay(3, 1),
    }


def score(auction: Auction, sigma: np.ndarray) -> Score:
    """Judge a behaviour profile against the closed forms, by a path that never reads a regret table."""
    derived, _ = closed_forms(auction)
    loss = float(utility_loss(auction, sigma))
    # Measured where the target plays, as the paper's on-path distance. Off the path the closed form
    # is an arbitrary tie-break the run has no experience of.
    reached = reach(auction, derived)
    visits = np.array([
        float(np.where(auction.profiles[:, bidder] == signal, auction.prior, 0.0) @ reached[state])
        for (bidder, state, signal) in auction.info_sets
    ])
    on_path = visits > RESOLVED
    near = np.abs(sigma - derived)[on_path].max() <= STAY_BAR
    if loss > CERTIFIED:
        verdict = Verdict.UNSOLVED
    elif near and utility_loss(auction, derived) <= CERTIFIED:
        verdict = Verdict.DERIVED
    else:
        verdict = Verdict.DIFFERENT
    return Score(verdict=verdict, utility_loss=loss, revenue=evaluate(auction, sigma).revenue)
