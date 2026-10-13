"""External-sampling Monte Carlo counterfactual regret minimisation, and the verdict on what it returns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from . import game
from .exact import CERTIFIED, RESOLVED, evaluate, reach, utility_loss
from .game import Auction, State, payoff_scale


EXIT, STAY = 0, 1

INITIAL_REGRET = 0.1  # width of the interval initial regrets are drawn from at two bidders, one draw per seed
ITERATIONS = 2000
TRACE_EVERY = 20
TREMBLE = 0.01  # uniform mass mixed into a sampled opponent action, during training only

BUDGET = 40_000  # iterations a validation run gets, where the utility loss bottoms.

STAY_BAR = 0.06  # how far an on-path stay probability may sit from the closed form.
REVENUE_BAR = 0.02  # the same bar on seller net revenue, at two bidders


@dataclass
class Solved:
    """What one run produced, with what is needed to judge it."""

    profile: np.ndarray  # average behavioural strategy, flat over the information sets
    regrets: np.ndarray  # (|I|, 2) regret of exiting and of staying, as the run left them
    visits: np.ndarray  # times sampled play reached each information set, the enumeration aside
    iterates: np.ndarray  # (sample, |I|) the regret-matched profile along the run
    averages: np.ndarray  # the same for the average profile
    seed: int
    iterations: int
    symmetry_restricted: bool = False


def matched(regret: np.ndarray) -> np.ndarray:
    """Stay with probability proportional to the regret of not having stayed, which the table already floors."""
    total = regret.sum(-1)
    # Nothing regretted yet, so no reason to lean either way.
    return np.divide(regret[..., STAY], total, out=np.full(np.shape(total), 0.5), where=total > 0.0)


def averaged(strategy_sum: np.ndarray, regret: np.ndarray) -> np.ndarray:
    """The average behavioural strategy, falling back to the matched one where the average carries no weight."""
    # External sampling averages at opponent nodes only, so a bidder's own off-path sets never
    # accumulate. Their regret did accumulate, by enumeration, and it is what the solver knows there.
    total = strategy_sum.sum(-1)
    weighed = np.divide(strategy_sum[..., STAY], np.where(total > 0.0, total, 1.0))
    return np.where(total > 0.0, weighed, matched(regret))


def shared_cells(auction: Auction) -> np.ndarray:
    """Each information set's cell in one shared table, being its orbit under relabelling the bidders."""
    buckets: Dict[Tuple[int, int], int] = {}
    keys = [(auction.orbit_of_state[state], signal) for _, state, signal in auction.info_sets]
    return np.array([buckets.setdefault(key, len(buckets)) for key in keys])


def solve(
    auction: Auction,
    seed: int = 0,
    iterations: int = ITERATIONS,
    initial_regret: Optional[float] = None,
    trace_every: int = TRACE_EVERY,
    tremble: float = TREMBLE,
    symmetry_restricted: bool = False,
) -> Solved:
    """Run external-sampling MCCFR with regret matching plus, linear weighting and a trembled opponent."""
    # Restricted, every bidder reads and writes one shared table, so the search is symmetric by
    # construction rather than a finished profile being averaged. Unrestricted, each has its own.
    rng = np.random.default_rng(seed)
    if initial_regret is None:
        initial_regret = INITIAL_REGRET * payoff_scale(auction.n)
    index = auction.info_index
    cells = shared_cells(auction) if symmetry_restricted else np.arange(len(auction.info_sets))
    cell_of = cells.tolist()  # plain ints, which the traversal indexes with far more cheaply
    regret = rng.uniform(0.0, initial_regret, (int(cells.max()) + 1, 2))  # a random start per seed
    strategy_sum = np.zeros_like(regret)
    visits = np.zeros(len(auction.info_sets), dtype=np.int64)
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
            # Counted here and not under the traverser, so visits say what play reached rather
            # than what enumeration walked, which is what the tremble is judged on.
            at = index[(bidder, state, signals[bidder])]
            cell = cell_of[at]
            stay = float(matched(regret[cell]))
            strategy_sum[cell] += (weight * (1.0 - stay), weight * stay)
            visits[at] += 1
            # Trembled for the draw only, so the accumulation above stays the untrembled strategy.
            if rng.random() < (1.0 - tremble) * stay + tremble * 0.5:
                sampled |= 1 << position

        def value(mask: int) -> float:
            """What this round's stayers pay the traverser, plus whatever the auction leads to."""
            got = float(payoffs[mask][drawn, traverser])
            return got + walk(onward[mask], traverser, signals, drawn, weight) if mask in onward else got

        if traverser not in active:  # it has left, and nothing after that pays it
            return value(sampled)
        cell = cell_of[index[(traverser, state, signals[traverser])]]
        stay = float(matched(regret[cell]))
        staying, exiting = value(sampled | 1 << active.index(traverser)), value(sampled)
        expected = stay * staying + (1.0 - stay) * exiting
        regret[cell] = np.maximum(regret[cell] + (exiting - expected, staying - expected), 0.0)
        return expected

    iterates, averages = [], []
    for iteration in range(1, iterations + 1):
        for traverser in range(auction.n):
            drawn = int(rng.choice(auction.num_profiles, p=auction.prior))
            # The iteration is its own weight in the average, which is the linear weighting.
            walk(auction.root(), traverser, tuple(auction.profiles[drawn]), drawn, float(iteration))
        if iteration % trace_every == 0:
            iterates.append(matched(regret)[cells])
            averages.append(averaged(strategy_sum, regret)[cells])
    return Solved(
        profile=averaged(strategy_sum, regret)[cells],
        regrets=regret[cells],
        visits=visits,
        iterates=np.array(iterates),
        averages=np.array(averages),
        seed=seed,
        iterations=iterations,
        symmetry_restricted=symmetry_restricted,
    )


# ---------------------------------------------------------------------------
# Scoring, kept apart from solving so that an exact profile can be scored for free
# ---------------------------------------------------------------------------


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
    on_path_distance: float  # furthest a stay probability sits from the closed form, where the target plays
    off_path_distance: float  # the same where it does not, which no budget informs and no bar reads
    revenue: float  # seller net revenue
    revenue_distance: float  # how far that sits from the closed form of this auction's rule


def closed_forms(auction: Auction) -> Tuple[np.ndarray, float]:
    """The two-bidder example's closed-form profile and seller net revenue under this auction's rule."""
    if (auction.n, auction.rounds, auction.delta) != (2, 4, 0.5):
        raise ValueError(
            f"the closed forms describe the two-bidder example, not {auction.n} bidders over "
            f"{auction.rounds} rounds at a tick of {auction.delta}"
        )
    rule = auction.reimbursement_policy
    if rule not in game.PI:
        raise ValueError(f"no closed form for the reimbursement policy {rule!r}")
    if not auction.independent_signals:
        raise ValueError("the published closed forms assume independent signals, so a correlated prior has none")
    derived = game.profile(auction, game.closed_form_strategy(rule, auction.c))
    return derived, game.PI[rule](auction.c, auction.q)


def stay_schedule(auction: Auction, sigma: np.ndarray) -> Dict[str, float]:
    """The stay probabilities the two-bidder closed forms are stated as, read off one bidder."""
    if auction.n != 2:
        raise ValueError(f"the closed forms name these probabilities at two bidders, not {auction.n}")
    # At two bidders a round has exactly one history, so the round indexes the states directly.
    stay = lambda round_index, signal: float(sigma[auction.info_index[(0, auction.states[round_index], signal)]])
    return {
        "open_high": stay(0, 1),
        "open_low": stay(0, 0),
        "sigma1": stay(1, 1),
        "sigma2": stay(2, 1),
        "sigma3": stay(3, 1),
    }


def visit_probability(auction: Auction, sigma: np.ndarray) -> np.ndarray:
    """Probability sampled play under ``sigma`` actually arrives at each information set."""
    reached = reach(auction, sigma)
    return np.array(
        [
            float(np.where(auction.profiles[:, bidder] == signal, auction.prior, 0.0) @ reached[state])
            for (bidder, state, signal) in auction.info_sets
        ]
    )


def score(
    auction: Auction,
    sigma: np.ndarray,
    stay_bar: float = STAY_BAR,
    revenue_bar: float = REVENUE_BAR,
) -> Score:
    """Judge a behaviour profile against the closed forms, by a path that never reads a regret table."""
    derived, earnings = closed_forms(auction)
    loss = float(utility_loss(auction, sigma))
    # Measured where the target plays, as the paper's on-path distance. Off the path the closed form
    # is an arbitrary tie-break the run has no experience of.
    on_path = visit_probability(auction, derived) > RESOLVED
    distance = np.abs(sigma - derived)
    on_path_distance, off_path_distance = float(distance[on_path].max()), float(distance[~on_path].max())
    revenue = evaluate(auction, sigma).revenue
    revenue_distance = abs(revenue - earnings)
    if loss > CERTIFIED:
        verdict = Verdict.UNSOLVED
    elif on_path_distance <= stay_bar and revenue_distance <= revenue_bar:
        verdict = Verdict.DERIVED
    else:
        verdict = Verdict.DIFFERENT
    return Score(
        verdict=verdict,
        utility_loss=loss,
        on_path_distance=on_path_distance,
        off_path_distance=off_path_distance,
        revenue=revenue,
        revenue_distance=revenue_distance,
    )
