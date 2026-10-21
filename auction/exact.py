"""Exact evaluation of a profile, and the exact best response that certifies it as an equilibrium."""

from __future__ import annotations

from typing import Dict

import numpy as np

from . import symmetric
from .game import Auction, Outcome, State, payoff_scale

CERTIFIED = 1e-3  # the paper's epsilon: the utility loss a certified two-bidder profile clears
RESOLVED = 1e-12  # the least counterfactual reach a gain can be divided by without amplifying noise


def certified(n: int = 2) -> float:
    """``CERTIFIED`` at ``n`` bidders, stretched with the payoffs so every count is held to one bar."""
    return CERTIFIED * payoff_scale(n)


def evaluate(auction: Auction, sigma: np.ndarray) -> Outcome:
    """Exact payoffs and seller accounting, on active counts when the profile is symmetric."""
    if symmetric.applies(auction, sigma):
        return symmetric.outcome(auction, sigma)
    memo: Dict[State, np.ndarray] = {}

    def state_value(state: State) -> np.ndarray:
        """Expected payoff vector at ``state`` under each signal profile."""
        if state not in memo:
            stay_probs = sigma[auction.gather[state]]
            # (2^|A|, num_profiles): probability of each set of stayers
            mask_probs = np.ones((1, auction.num_profiles))
            for position in range(stay_probs.shape[1]):
                mask_probs = np.concatenate(
                    [mask_probs * (1 - stay_probs[:, position]), mask_probs * stay_probs[:, position]],
                    axis=0,
                )
            payoff_by_mask = auction.branch_payoffs[state]
            if auction.continuations[state]:
                payoff_by_mask = payoff_by_mask.copy()
                for mask, next_state in auction.continuations[state]:
                    payoff_by_mask[mask] += state_value(next_state)
            memo[state] = np.einsum("bm,bmk->mk", mask_probs, payoff_by_mask)
        return memo[state]

    expected = auction.prior @ state_value(auction.root())
    return Outcome(
        expected[: auction.n],
        float(expected[auction.n]),
        float(expected[auction.n + 1]),
    )


def reach(auction: Auction, sigma: np.ndarray) -> Dict[State, np.ndarray]:
    """Probability of reaching each state, under each signal profile."""
    reached = {state: np.zeros(auction.num_profiles) for state in auction.states}
    reached[auction.root()] = np.ones(auction.num_profiles)
    for state in auction.states:  # sorted by round, so predecessors are complete
        stay_probs = sigma[auction.gather[state]]
        for mask, next_state in auction.continuations[state]:
            transition = reached[state].copy()
            for position in range(stay_probs.shape[1]):
                stayed = mask >> position & 1
                transition = transition * (stay_probs[:, position] if stayed else 1.0 - stay_probs[:, position])
            reached[next_state] += transition
    return reached


def utility_loss(auction: Auction, sigma: np.ndarray) -> float:
    """Utility loss: the largest gain any bidder gets from a full best response."""
    if symmetric.applies(auction, sigma):
        return symmetric.utility_loss(auction, sigma)
    baseline = evaluate(auction, sigma).payoff
    gains = []
    for bidder in range(auction.n):
        response = sigma.copy()  # an exact best response, by one backward sweep
        for position in reversed(range(len(auction.info_sets))):  # round-sorted
            owner, state, signal = auction.info_sets[position]
            if owner != bidder:
                continue
            # Gain from staying over exiting, weighted by opponent reach only.
            base = response.copy()
            for other, earlier in auction.info_index.items():
                if other[0] == bidder and other[2] == signal and other[1].round < state.round:
                    base[earlier] = 1.0
            staying, exiting = base.copy(), base.copy()
            staying[position], exiting[position] = 1.0, 0.0
            gain = evaluate(auction, staying).payoff[bidder] - evaluate(auction, exiting).payoff[bidder]
            response[position] = float(gain > 0.0)
        gains.append(evaluate(auction, response).payoff[bidder] - baseline[bidder])
    return max(gains)

