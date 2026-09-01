"""The auction's game objects: the state space, the parameterised auction, and a solved outcome."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

State = Tuple[int, Tuple[int, ...]]  # (round, active set)
Info = Tuple[int, State, int]  # (bidder, state, own signal)
Strategy = Callable[[State, int], float]  # Pr(stay) for one bidder


def states(n: int, rounds: int) -> List[State]:
    """Every public state of a round and an active set of at least two bidders, reachable or not."""
    active_sets = [
        stayers for size in range(2, n + 1) for stayers in itertools.combinations(range(n), size)
    ]
    return sorted({(0, tuple(range(n)))} | {(t, active) for t in range(1, rounds) for active in active_sets})


@dataclass
class Auction:
    """A fully specified costly clock auction, defaulting to the paper's two-bidder example."""

    q: float = 0.5  # Pr(s_i = 1)
    c: float = 0.05  # participation cost per stay
    policy: str = "targeted"  # zero | full | targeted
    rho: float = 0.0  # signal correlation (n = 2 only)
    n: int = 2
    rounds: int = 4  # decision rounds
    delta: float = 0.5  # tick size

    def __post_init__(self) -> None:
        """Build the signal prior, the state and information-set index, and the branch tables."""
        self.profiles = np.array(list(itertools.product([0, 1], repeat=self.n)))
        if self.rho:
            q, covariance = self.q, self.rho * self.q * (1 - self.q)
            off_diagonal = q * (1 - q) - covariance
            joint = np.clip(
                np.array([(1 - q) ** 2 + covariance, off_diagonal, off_diagonal, q**2 + covariance]), 0.0, None
            )
            self.prior = joint / joint.sum()
        else:
            self.prior = np.prod(np.where(self.profiles == 1, self.q, 1 - self.q), axis=1)
        self.num_profiles = len(self.profiles)
        self.value = self.profiles.sum(1).astype(float)  # common value, shared by every bidder
        self.states = states(self.n, self.rounds)
        self.info_sets = [
            (bidder, state, signal) for state in self.states for bidder in state[1] for signal in (0, 1)
        ]
        self.info_index = {info_set: position for position, info_set in enumerate(self.info_sets)}
        # (num_profiles, |A|) index of each active bidder's stay probability under each signal profile
        self.gather = {
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
        self.branch_payoffs, self.continuations = branches(self)

    def price(self, round_index: int) -> float:
        """Clock price in round ``round_index``."""
        return round_index * self.delta

    def reimbursement(self, round_index: int) -> float:
        """Reimbursement paid per stay, which under the targeted policy skips the opening round."""
        return {"zero": 0.0, "full": self.c, "targeted": 0.0 if round_index == 0 else self.c}[self.policy]

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
        paid = auction.reimbursement(round_index)
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

