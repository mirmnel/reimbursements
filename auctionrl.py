"""Exact solver and self-play learner for the costly Japanese clock auction with reimbursements."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from game import Auction, Outcome, State


POLICIES = ("zero", "full", "targeted")


# ---------------------------------------------------------------------------
# Exact evaluation of a behaviour profile
# ---------------------------------------------------------------------------


def evaluate(auction: Auction, sigma: np.ndarray) -> Outcome:
    """Exact expected payoffs and seller accounting, by full tree enumeration."""
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
    """Gain to the bidder from staying rather than exiting at information set ``position``, weighted by opponent reach only."""
    bidder, (round_index, _), signal = auction.info_sets[position]
    base = sigma.copy()
    for other, earlier in auction.info_index.items():
        if other[0] == bidder and other[2] == signal and other[1][0] < round_index:
            base[earlier] = 1.0
    staying, exiting = base.copy(), base.copy()
    staying[position], exiting[position] = 1.0, 0.0
    return float(evaluate(auction, staying).payoff[bidder] - evaluate(auction, exiting).payoff[bidder])


def counterfactual_reach(auction: Auction, sigma: np.ndarray) -> np.ndarray:
    """Prior times opponents' reach at each information set: the weight ``counterfactual_gain`` carries."""
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
# Learning
# ---------------------------------------------------------------------------


def symmetry_groups(auction: Auction) -> List[np.ndarray]:
    """Information sets that exchangeable bidders must play identically."""
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for position, (_, (round_index, active), signal) in enumerate(auction.info_sets):
        buckets.setdefault((round_index, len(active), signal), []).append(position)
    return [np.array(positions) for positions in buckets.values()]


def learn(auction: Auction, iterations: int = 200, learning_rate: float = 0.5) -> np.ndarray:
    """Self-play gradient ascent on the deviation gain, rescaled by reach and clipped to a probability."""
    sigma = np.full(len(auction.info_sets), 0.5)
    groups = symmetry_groups(auction)
    for _ in range(iterations):
        weights = counterfactual_reach(auction, sigma)
        step = np.zeros_like(sigma)
        for group in groups:
            gain = counterfactual_gain(auction, sigma, group[0])
            step[group] = gain / weights[group[0]] if weights[group[0]] > 1e-12 else 0.0
        sigma = np.clip(sigma + learning_rate * step, 0.0, 1.0)
    return sigma
