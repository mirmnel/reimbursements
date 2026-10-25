"""External-sampling Monte Carlo counterfactual regret minimisation, and the verdicts on what it returns."""

from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple

import numpy as np

from . import game
from .game import Auction, payoff_scale


STAY = 1

INITIAL_REGRET = 0.1  # width of the interval initial regrets are drawn from at two bidders, one draw per seed

BUDGET = 100_000  # iterations every run gets

STAY_BAR = 0.05  # how far an on-path stay probability may sit from the target.


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


def solve(auction: Auction, seed: int, iterations: int = BUDGET) -> np.ndarray:
    """Run external-sampling MCCFR with regret matching and linear averaging, returning the average profile."""
    # Every bidder reads and writes one shared table per orbit and signal, so the search is symmetric
    # by construction rather than a finished profile being averaged.
    rng = np.random.default_rng(seed)
    initial_regret = INITIAL_REGRET * payoff_scale(auction.n)
    buckets: Dict[Tuple[int, int], int] = {}
    keys = [(auction.orbit_of_state[state], signal) for _, state, signal in auction.info_sets]
    cells = np.array([buckets.setdefault(key, len(buckets)) for key in keys])
    cell_of = cells.tolist()  # plain ints, which the traversal indexes with far more cheaply
    # Plain lists, as numpy's per-call overhead dominates arithmetic on two floats.
    regret = rng.uniform(0.0, initial_regret, (int(cells.max()) + 1, 2)).tolist()  # a random start per seed
    strategy_sum = [[0.0, 0.0] for _ in regret]
    # The traversal addresses histories by position, so it never hashes a history, and it works out
    # the traverser's payoff where it needs it, as the per-history tables run to gigabytes at six bidders.
    position_of = {state: k for k, state in enumerate(auction.states)}
    actives = [state.active for state in auction.states]
    onward_at, cells_at, net_at, price_at = [], [], [], []
    slot = 0  # info_sets runs over histories, then their active bidders, then the two signals
    for state in auction.states:
        width = len(state.active)
        onward_at.append({
            mask: position_of[game.after(state, tuple(b for k, b in enumerate(state.active) if mask >> k & 1))]
            for mask in range(1 << width)
            if mask.bit_count() >= 2 and state.round + 1 < auction.rounds
        })
        cells_at.append([(cell_of[slot + 2 * k], cell_of[slot + 2 * k + 1]) for k in range(width)])
        slot += 2 * width
        net_at.append(auction.c - auction.reimbursement_policy(state))
        price_at.append(auction.price(state.round))
    cap = auction.price(auction.rounds)
    values = [float(value) for value in auction.value]
    random = rng.random
    # rng.choice(p=prior) draws one uniform against this cdf, without its per-call validation.
    cdf = auction.prior.cumsum()
    cdf /= cdf[-1]
    signal_rows = [tuple(int(signal) for signal in row) for row in auction.profiles]

    def pays(at: int, mask: int, own: int, drawn: int) -> float:
        """What a round ending in stayers ``mask`` pays the traverser at bit ``own``, before what follows."""
        got = 0.0
        stays = mask >> own & 1
        if stays:
            got = got - net_at[at]
        if mask in onward_at[at]:
            return got
        stayers = mask.bit_count()
        if stayers == 0:  # all exit at once, so every active bidder splits the prize
            return got + (values[drawn] - price_at[at]) / len(actives[at])
        if stays:
            return got + (values[drawn] - (cap if stayers >= 2 else price_at[at])) / stayers
        return got

    def walk(at: int, traverser: int, signals: Tuple[int, ...], drawn: int, weight: float) -> float:
        """One traversal from history ``at`` under a drawn signal profile, worth what it pays ``traverser``."""
        cells = cells_at[at]
        sampled, own = 0, -1
        for position, bidder in enumerate(actives[at]):
            if bidder == traverser:
                own = position
                continue
            cell = cells[position][signals[bidder]]
            exiting, staying = regret[cell]
            exiting, staying = (0.0 if exiting < 0.0 else exiting), (0.0 if staying < 0.0 else staying)
            stay = staying / (exiting + staying) if exiting + staying > 0.0 else 0.5
            row = strategy_sum[cell]
            row[0] += weight * (1.0 - stay)
            row[1] += weight * stay
            if random() < stay:
                sampled |= 1 << position
        onward = onward_at[at]
        if own < 0:  # it has left, and nothing after that pays it
            return 0.0 + walk(onward[sampled], traverser, signals, drawn, weight) if sampled in onward else 0.0
        row = regret[cells[own][signals[traverser]]]
        stay = matched_one(row)
        mask = sampled | 1 << own
        staying = pays(at, mask, own, drawn)
        if mask in onward:
            staying = staying + walk(onward[mask], traverser, signals, drawn, weight)
        exiting = pays(at, sampled, own, drawn)
        if sampled in onward:
            exiting = exiting + walk(onward[sampled], traverser, signals, drawn, weight)
        expected = stay * staying + (1.0 - stay) * exiting
        # Not floored as in RM+, which turns sampling noise at an indifferent set into drift.
        row[0] += exiting - expected
        row[1] += staying - expected
        return expected

    root = position_of[auction.root()]
    for iteration in range(1, iterations + 1):
        for traverser in range(auction.n):
            drawn = int(cdf.searchsorted(random(), side="right"))
            # The iteration is its own weight in the average, which is the linear weighting.
            walk(root, traverser, signal_rows[drawn], drawn, float(iteration))
    return averaged(np.array(strategy_sum), np.array(regret))[cells]


class Verdict(str, Enum):
    """The three things a scored profile can be, which call for three different responses."""

    DERIVED = "derived"  # certified an equilibrium, and it is the one the example derives
    DIFFERENT = "different"  # certified an equilibrium, and it is another one, which is a result
    UNSOLVED = "unsolved"  # not certified, so the solver or its budget is at fault
