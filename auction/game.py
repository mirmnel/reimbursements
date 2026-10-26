"""The state space, the parameterised auction, and a solved outcome."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict, List, NamedTuple, Sequence, Tuple, Union

import numpy as np

ACTIVE = -1  # exit round of a bidder that has not left


class State(NamedTuple):
    """A public bidding history: the round, who is still in, and the round each departed bidder left."""

    round: int
    active: Tuple[int, ...]
    exits: Tuple[int, ...]  # per bidder, the round it exited in, or ``ACTIVE`` while it is still in


Orbit = Tuple[int, Tuple[int, ...]]  # (round, multiset of the departed bidders' exit rounds)
Info = Tuple[int, State, int]  # (bidder, state, own signal)
ReimbursementPolicy = Callable[[State], float]  # public history -> paid per stay

COARSENING_TOLERANCE = 1e-12  # how far a policy may vary inside an orbit or a round and still coarsen onto it
INDIFFERENT = 1e-12  # how far below zero a stay's gain may sit and still count as indifference


def payoff_scale(n: int) -> float:
    """How far payoffs stretch against the two-bidder example, the common value summing ``n`` signals."""
    return n / 2


def opening(n: int) -> State:
    """The history before anyone has acted, with every bidder still in."""
    return State(0, tuple(range(n)), (ACTIVE,) * n)


def after(state: State, stayers: Tuple[int, ...]) -> State:
    """The history one round on, recording everyone who left this round as having left in it."""
    staying = set(stayers)
    exits = tuple(
        state.round if bidder in state.active and bidder not in staying else exited
        for bidder, exited in enumerate(state.exits)
    )
    return State(state.round + 1, stayers, exits)


def bucket_means(index: np.ndarray, size: int, values: np.ndarray) -> np.ndarray:
    """Mean of ``values`` in each of ``size`` buckets, leaving empty buckets at zero."""
    totals = np.bincount(index, weights=values, minlength=size)
    counts = np.bincount(index, minlength=size)
    return np.divide(totals, counts, out=np.zeros(size), where=counts > 0)


def orbit_of(state: State) -> Orbit:
    """What survives relabelling the bidders: the round and the multiset of exit rounds."""
    return (state.round, tuple(sorted(exited for exited in state.exits if exited != ACTIVE)))


def schedule(values: Sequence[float]) -> ReimbursementPolicy:
    """A policy paying a set amount per round, whatever the history within it."""
    paid = tuple(float(value) for value in values)
    return lambda state: paid[state.round]


def flat(c: float, rounds: int, fraction: float = 1.0) -> ReimbursementPolicy:
    """Reimburse the same fraction of the participation cost in every round."""
    return schedule([fraction * c] * rounds)


def opening_margin(n: int, q: float) -> float:
    """What a high type gains at the opening by staying when bidding is free, the paper's w0."""
    return (1 - q) ** (n - 1) * (1 - 1 / n)


def preserve(c: float, rounds: int, n: int, q: float) -> ReimbursementPolicy:
    """The cheapest rule keeping the frictionless sorting strategy an equilibrium."""
    return schedule([max(0.0, c - opening_margin(n, q))] + [c] * (rounds - 1))


def trigger(c: float, n: int) -> ReimbursementPolicy:
    """Pay a stay at 3a/2 only in a war that two bidders entered from the opening."""
    return lambda state: c if state.round == 3 and state.exits.count(0) == n - 2 else 0.0


def states(n: int, rounds: int) -> List[State]:
    """Every public history leaving at least two bidders active, in round order, reachable or not."""
    frontier = [opening(n)]
    every = list(frontier)
    for _ in range(1, rounds):
        onward = [
            after(state, stayers)
            for state in frontier
            for size in range(2, len(state.active) + 1)
            for stayers in itertools.combinations(state.active, size)
        ]
        frontier = onward
        every += onward
    return every


@dataclass
class Auction:
    """A fully specified costly clock auction, signals drawn independently."""

    reimbursement_policy: ReimbursementPolicy
    n: int
    q: float  # Pr(s_i = 1)
    c: float  # participation cost per stay
    rounds: int  # decision rounds
    delta: float  # tick size

    @cached_property
    def profiles(self) -> np.ndarray:
        """One row per signal profile, giving each bidder a high or a low signal."""
        return np.array(list(itertools.product([0, 1], repeat=self.n)))

    @cached_property
    def prior(self) -> np.ndarray:
        """Probability of each signal profile."""
        return np.prod(np.where(self.profiles == 1, self.q, 1 - self.q), axis=1)

    @cached_property
    def signal_counts(self) -> np.ndarray:
        """Probability that exactly ``k`` bidders drew a high signal, for ``k`` up to ``n``."""
        return np.bincount(self.profiles.sum(1), weights=self.prior, minlength=self.n + 1)

    @cached_property
    def own_and_others(self) -> np.ndarray:
        """(2, n) probability that a bidder drew each signal while ``k`` of the others drew high."""
        others, own = self.profiles[:, 1:].sum(1), self.profiles[:, 0]
        return np.array(
            [
                np.bincount(others[own == signal], weights=self.prior[own == signal], minlength=self.n)
                for signal in (0, 1)
            ]
        )

    @cached_property
    def high_probability(self) -> float:
        """A bidder's own probability of a high signal, which exchangeability makes common to all."""
        return float(self.profiles[:, 0] @ self.prior)

    @cached_property
    def value(self) -> np.ndarray:
        """Common value under each signal profile, the sum of the signals as in the paper."""
        return self.profiles.sum(1).astype(float)

    @cached_property
    def states(self) -> List[State]:
        """Every public history of the auction, reachable or not."""
        return states(self.n, self.rounds)

    @cached_property
    def info_sets(self) -> List[Info]:
        """Every information set, one per active bidder, public history and own signal."""
        return [
            (bidder, state, signal) for state in self.states for bidder in state.active for signal in (0, 1)
        ]

    @cached_property
    def orbits(self) -> List[Orbit]:
        """Every orbit of the histories under relabelling the bidders, in round order."""
        return list(dict.fromkeys(orbit_of(state) for state in self.states))

    @cached_property
    def orbit_of_state(self) -> Dict[State, int]:
        """Each history's orbit, as a position in ``orbits``."""
        at = {orbit: position for position, orbit in enumerate(self.orbits)}
        return {state: at[orbit_of(state)] for state in self.states}

    @cached_property
    def orbit_index(self) -> np.ndarray:
        """Each information set's slot in a flat (orbit, signal) table."""
        return np.array([self.orbit_of_state[state] * 2 + signal for _, state, signal in self.info_sets])

    @cached_property
    def orbit_active(self) -> np.ndarray:
        """How many bidders are still in, per orbit."""
        return np.array([self.n - len(departed) for _, departed in self.orbits])

    @cached_property
    def onward_orbit(self) -> np.ndarray:
        """(|orbits|, n + 1) orbit reached when ``k`` of the active bidders leave, or -1 where it ends."""
        at = {orbit: position for position, orbit in enumerate(self.orbits)}
        table = np.full((len(self.orbits), self.n + 1), -1, dtype=np.int64)
        for position, (round_index, departed) in enumerate(self.orbits):
            active = self.n - len(departed)
            for leaving in range(active - 1):  # two or more must stay for the auction to carry on
                if round_index + 1 < self.rounds:
                    onward = (round_index + 1, tuple(sorted(departed + (round_index,) * leaving)))
                    table[position, leaving] = at[onward]
        return table

    def price(self, round_index: int) -> float:
        """Clock price in round ``round_index``."""
        return round_index * self.delta

    @cached_property
    def paid_by_state(self) -> np.ndarray:
        """What the policy pays per stay at each public history, aligned with ``states``."""
        return np.array([float(self.reimbursement_policy(state)) for state in self.states])

    @cached_property
    def orbit_view(self) -> Union[np.ndarray, None]:
        """What the policy pays per orbit, or ``None`` where it separates histories inside one."""
        index = np.array([self.orbit_of_state[state] for state in self.states])
        means = bucket_means(index, len(self.orbits), self.paid_by_state)
        return means if np.max(np.abs(self.paid_by_state - means[index])) <= COARSENING_TOLERANCE else None

    def root(self) -> State:
        """The opening history: round ``0``, with every bidder still in."""
        return opening(self.n)


@dataclass
class Outcome:
    """Exact expectations under one behaviour profile."""

    payoff: np.ndarray  # (n,) bidder payoffs
    price: float
    reimbursement: float

    @property
    def revenue(self) -> float:
        """Seller net revenue: the price less every reimbursement paid."""
        return self.price - self.reimbursement


def sorting_profile(auction: Auction) -> np.ndarray:
    """The symmetric sorting equilibrium at any n: sorting at the opening, then each war solved backward."""
    # Survivors believe the value is the opening stayers' count, and mix where indifferent as in fin:rent-bound.
    paid, onward = auction.orbit_view, auction.onward_orbit
    worth = np.zeros(len(auction.orbits))
    table = np.zeros((len(auction.orbits), 2))
    for orbit in reversed(range(len(auction.orbits))):
        round_index, departed = auction.orbits[orbit]
        if round_index == 0:
            table[orbit, 1] = 1.0
            continue
        active = auction.n - len(departed)
        surplus = auction.n - departed.count(0) - auction.price(round_index)
        rivals = np.arange(active)  # how many of the other active bidders stay
        leaving = active - 1 - rivals
        carries = (onward[orbit][leaving] >= 0) & (rivals >= 1)
        capped = surplus + auction.price(round_index) - auction.price(auction.rounds)
        later = np.where(carries, worth[onward[orbit][leaving]], capped / (rivals + 1))
        later[0] = surplus

        def gain(stay: float) -> float:
            odds = np.array([math.comb(active - 1, k) * stay**k * (1 - stay) ** (active - 1 - k) for k in rivals])
            return odds @ later - (auction.c - paid[orbit]) - odds[0] * surplus / active

        if gain(1.0) >= -INDIFFERENT:
            stay = 1.0
        elif gain(0.0) <= 0.0:
            stay = 0.0
        else:
            low, high = 0.0, 1.0
            for _ in range(100):
                low, high = ((low + high) / 2, high) if gain((low + high) / 2) > 0 else (low, (low + high) / 2)
            stay = low
        table[orbit, 1] = stay
        worth[orbit] = (1 - stay) ** (active - 1) * surplus / active + (gain(stay) if stay == 1.0 else 0.0)
    return table.ravel()[auction.orbit_index]
