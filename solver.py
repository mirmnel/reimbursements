"""Exact evaluation and the exact-gradient solver for the costly Japanese clock auction."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

import symmetric
from game import REIMBURSEMENT_POLICIES, Auction, Outcome, State


# ---------------------------------------------------------------------------
# Exact evaluation of a behaviour profile
# ---------------------------------------------------------------------------


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
        float(expected[auction.n + 2]),
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


# ---------------------------------------------------------------------------
# Counterfactual deviation values
# ---------------------------------------------------------------------------


def counterfactual_gain(auction: Auction, sigma: np.ndarray, position: int) -> float:
    """Gain from staying rather than exiting at ``position``, weighted by opponent reach only."""
    if symmetric.applies(auction, sigma):
        gain, _ = symmetric.gains(auction, sigma)
        _, state, signal = auction.info_sets[position]
        return float(gain[auction.orbit_of_state[state], signal])
    bidder, state, signal = auction.info_sets[position]
    base = sigma.copy()
    for other, earlier in auction.info_index.items():
        if other[0] == bidder and other[2] == signal and other[1].round < state.round:
            base[earlier] = 1.0
    staying, exiting = base.copy(), base.copy()
    staying[position], exiting[position] = 1.0, 0.0
    return float(evaluate(auction, staying).payoff[bidder] - evaluate(auction, exiting).payoff[bidder])


def counterfactual_reach(auction: Auction, sigma: np.ndarray) -> np.ndarray:
    """Prior times opponents' reach, the weight each ``counterfactual_gain`` carries."""
    if symmetric.applies(auction, sigma):
        return symmetric.gains(auction, sigma)[1].ravel()[auction.orbit_index]
    weights = np.zeros(len(auction.info_sets))
    for bidder in range(auction.n):
        forced = sigma.copy()
        forced[[position for info_set, position in auction.info_index.items() if info_set[0] == bidder]] = 1.0
        reached = reach(auction, forced)
        for position, info_set in enumerate(auction.info_sets):
            if info_set[0] == bidder:
                weights[position] = float(
                    np.where(auction.profiles[:, bidder] == info_set[2], auction.prior, 0.0) @ reached[info_set[1]]
                )
    return weights

# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------


def symmetry_groups(auction: Auction) -> Dict[Tuple[int, int], np.ndarray]:
    """Information sets that exchangeable bidders must play identically, by orbit and signal."""
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for position, (_, state, signal) in enumerate(auction.info_sets):
        buckets.setdefault((auction.orbit_of_state[state], signal), []).append(position)
    return {bucket: np.array(positions) for bucket, positions in buckets.items()}

