"""Game objects: the state space, the parameterised auction, and a solved outcome."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np

State = Tuple[int, Tuple[int, ...]]  # (round, active set)
Info = Tuple[int, State, int]  # (bidder, state, own signal)
Strategy = Callable[[State, int], float]  # Pr(stay) for one bidder
ReimbursementPolicy = Callable[[int], float]  # round -> paid per stay


def schedule(values: Sequence[float]) -> ReimbursementPolicy:
    """A policy paying a set amount per round."""
    paid = tuple(float(value) for value in values)
    return lambda round_index: paid[round_index]


def flat(c: float, rounds: int, fraction: float = 1.0) -> ReimbursementPolicy:
    """Reimburse the same fraction of the participation cost in every round."""
    return schedule([fraction * c] * rounds)


def targeted(c: float, rounds: int, fraction: float = 1.0) -> ReimbursementPolicy:
    """Reimburse in every round but the opening one, where separation has to be paid for."""
    return schedule([0.0] + [fraction * c] * (rounds - 1))


def front_loaded(c: float, rounds: int, fraction: float = 1.0) -> ReimbursementPolicy:
    """Reimburse most in the opening round and least in the last, down a linear ramp."""
    return schedule([fraction * c * (rounds - round_index) / rounds for round_index in range(rounds)])


def back_loaded(c: float, rounds: int, fraction: float = 1.0) -> ReimbursementPolicy:
    """Reimburse least in the opening round and most in the last, up a linear ramp."""
    return schedule([fraction * c * (round_index + 1) / rounds for round_index in range(rounds)])


REIMBURSEMENT_FAMILIES = {
    "flat": flat,
    "targeted": targeted,
    "front_loaded": front_loaded,
    "back_loaded": back_loaded,
}
PUBLISHED_REIMBURSEMENTS = {  # the paper's three rules, as members of the schedule family
    "zero": lambda c, rounds: flat(c, rounds, 0.0),
    "full": flat,
    "targeted": targeted,
}
REIMBURSEMENT_POLICIES = tuple(PUBLISHED_REIMBURSEMENTS)


def states(n: int, rounds: int) -> List[State]:
    """Every state of a round and an active set of at least two bidders, reachable or not."""
    active_sets = [
        stayers for size in range(2, n + 1) for stayers in itertools.combinations(range(n), size)
    ]
    return sorted({(0, tuple(range(n)))} | {(t, active) for t in range(1, rounds) for active in active_sets})


@dataclass
class Auction:
    """A fully specified costly clock auction, defaulting to the paper's two-bidder example."""

    q: float = 0.5  # Pr(s_i = 1)
    c: float = 0.025  # participation cost per stay
    reimbursement_policy: Union[str, ReimbursementPolicy] = "targeted"  # a schedule, or a published name
    n: int = 2
    rounds: int = 4  # decision rounds
    delta: float = 0.25  # tick size

    # Tables below build on first use: active-set tables grow like 6^n, exhausting memory at n = 10.

    @cached_property
    def profiles(self) -> np.ndarray:
        """One row per signal profile, giving each bidder a high or a low signal."""
        return np.array(list(itertools.product([0, 1], repeat=self.n)))

    @cached_property
    def prior(self) -> np.ndarray:
        """Probability of each signal profile."""
        return np.prod(np.where(self.profiles == 1, self.q, 1 - self.q), axis=1)

    @property
    def num_profiles(self) -> int:
        """How many signal profiles there are, without building the table of them."""
        return 1 << self.n

    @cached_property
    def value(self) -> np.ndarray:
        """Common value in mean units under each signal profile, shared by every bidder."""
        return self.profiles.mean(1)

    @cached_property
    def states(self) -> List[State]:
        """Every public state of the auction, reachable or not."""
        return states(self.n, self.rounds)

    @cached_property
    def info_sets(self) -> List[Info]:
        """Every information set, one per active bidder, public state and own signal."""
        return [
            (bidder, state, signal) for state in self.states for bidder in state[1] for signal in (0, 1)
        ]

    @cached_property
    def info_index(self) -> Dict[Info, int]:
        """Position of each information set in a flat behaviour profile."""
        return {info_set: position for position, info_set in enumerate(self.info_sets)}

    @cached_property
    def anonymous_index(self) -> np.ndarray:
        """Each information set's slot in a flat (round, active count, signal) table."""
        width = self.n + 1
        return np.array(
            [(round_index * width + len(active)) * 2 + signal for _, (round_index, active), signal in self.info_sets]
        )

    @cached_property
    def gather(self) -> Dict[State, np.ndarray]:
        """(num_profiles, |A|) index of each active bidder's stay probability per signal profile."""
        return {
            state: np.stack(
                [
                    np.array([self.info_index[(bidder, state, 0)], self.info_index[(bidder, state, 1)]])[
                        self.profiles[:, bidder]
                    ]
                    for bidder in state[1]
                ],
                axis=1,
            )
            for state in self.states
        }

    @cached_property
    def branch_tables(self) -> Tuple[Dict[State, np.ndarray], Dict[State, List[Tuple[int, State]]]]:
        """The active-set branch tables, whose size rules out eager construction."""
        return branches(self)

    @property
    def branch_payoffs(self) -> Dict[State, np.ndarray]:
        """Per state, the fixed part of a round's payoff under each set of stayers."""
        return self.branch_tables[0]

    @property
    def continuations(self) -> Dict[State, List[Tuple[int, State]]]:
        """Per state, the sets of stayers that carry the auction into a further round."""
        return self.branch_tables[1]

    def price(self, round_index: int) -> float:
        """Clock price in round ``round_index``."""
        return round_index * self.delta

    @cached_property
    def reimburses(self) -> ReimbursementPolicy:
        """Reimbursement paid per stay in a round, from a published name if a name was given."""
        if callable(self.reimbursement_policy):
            return self.reimbursement_policy
        return PUBLISHED_REIMBURSEMENTS[self.reimbursement_policy](self.c, self.rounds)

    def root(self) -> State:
        """The opening state: round ``0`` with every bidder active."""
        return (0, tuple(range(self.n)))


@dataclass
class Outcome:
    """Exact expectations under one behaviour profile."""

    payoff: np.ndarray  # (n,) bidder payoffs
    price: float
    reimbursement: float
    stay_rounds: float  # expected number of staying bidder-rounds

    @property
    def revenue(self) -> float:
        """Seller net revenue: the price less every reimbursement paid."""
        return self.price - self.reimbursement


def branches(auction: Auction) -> Tuple[Dict[State, np.ndarray], Dict[State, List[Tuple[int, State]]]]:
    """Per state, the fixed part of a round's payoff and the states it can lead to."""
    branch_payoffs, continuations = {}, {}
    for state in auction.states:
        round_index, active = state
        paid = auction.reimburses(round_index)
        payoff_by_mask = np.zeros((1 << len(active), auction.num_profiles, auction.n + 3))
        continuations[state] = []
        for mask in range(1 << len(active)):
            stayers = tuple(bidder for position, bidder in enumerate(active) if mask >> position & 1)
            payoff = payoff_by_mask[mask]
            for bidder in stayers:
                payoff[:, bidder] -= auction.c - paid
                payoff[:, auction.n + 1] += paid
                payoff[:, auction.n + 2] += 1.0
            if len(stayers) >= 2 and round_index + 1 < auction.rounds:
                continuations[state].append((mask, (round_index + 1, stayers)))
            else:  # sole stayer wins at p_t, all-exit splits at p_t, survivors split at p_T
                winners = stayers or active
                price = auction.price(auction.rounds if len(stayers) >= 2 else round_index)
                for bidder in winners:
                    payoff[:, bidder] += (auction.value - price) / len(winners)
                payoff[:, auction.n] += price
        branch_payoffs[state] = payoff_by_mask
    return branch_payoffs, continuations

