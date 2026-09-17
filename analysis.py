"""Benchmark strategies, the empirical game, equilibrium quality, the closed forms, and the results writer."""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import numpy as np

import symmetric
from solver import counterfactual_gain, counterfactual_reach, evaluate
from game import Auction, Strategy


RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CERTIFIED = 5e-4  # the utility loss every certified profile clears, under every family and count


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def profile(auction: Auction, *strategies: Strategy) -> np.ndarray:
    """Flat behaviour profile from one strategy per bidder, or from one shared by all."""
    if len(strategies) == 1:
        strategies *= auction.n
    return np.array([float(strategies[bidder](state, signal)) for (bidder, state, signal) in auction.info_sets])


def as_strategy(auction: Auction, sigma: np.ndarray) -> Strategy:
    """Read a profile back as the one strategy every bidder plays, which needs it symmetric."""
    table = symmetric.anonymise(auction, sigma)
    return lambda state, signal: float(table[state[0], len(state[1]), signal])


SEPARATING: Strategy = lambda state, signal: float(signal == 1)  # L exits at p = 0, H stays
ALWAYS: Strategy = lambda state, signal: 1.0
NEVER: Strategy = lambda state, signal: 0.0


def war_of_attrition(c: float) -> Strategy:
    """The low type exits everywhere, and the high type stays at the opening and then mixes."""
    # The two-bidder closed form, which larger auctions borrow as a benchmark, not an equilibrium.
    schedule = (1.0, sigma1(c), 1.0, sigma3(c))
    return lambda state, signal: schedule[state[0]] if signal == 1 else 0.0


# ---------------------------------------------------------------------------
# Best responses and equilibrium quality
# ---------------------------------------------------------------------------


def best_response(auction: Auction, sigma: np.ndarray, bidder: int) -> np.ndarray:
    """Exact best response of ``bidder``, by one backward sweep on ``counterfactual_gain``."""
    response = sigma.copy()
    for position in reversed(range(len(auction.info_sets))):  # round-sorted, so this sweeps backwards
        if auction.info_sets[position][0] == bidder:
            gain = counterfactual_gain(auction, response, position)
            response[position] = float(gain > 0.0)
    return response


def utility_loss(auction: Auction, sigma: np.ndarray) -> float:
    """Utility loss: the largest gain any bidder gets from a full best response."""
    if symmetric.applies(auction, sigma):
        return symmetric.utility_loss(auction, sigma)
    baseline = evaluate(auction, sigma).payoff
    return max(
        evaluate(auction, best_response(auction, sigma, bidder)).payoff[bidder] - baseline[bidder]
        for bidder in range(auction.n)
    )


def max_regret(auction: Auction, sigma: np.ndarray) -> float:
    """Largest one-shot deviation gain at any information set of positive opponent reach."""
    if symmetric.applies(auction, sigma):
        return symmetric.max_regret(auction, sigma)
    weights, worst = counterfactual_reach(auction, sigma), 0.0
    for position in range(len(auction.info_sets)):
        if weights[position] > 1e-12:
            gain = counterfactual_gain(auction, sigma, position) / weights[position]
            stay_prob = sigma[position]
            worst = max(worst, (1 - stay_prob) * max(gain, 0.0) + stay_prob * max(-gain, 0.0))
    return worst


def deviation_payoff(auction: Auction, own: Strategy, population: Strategy) -> float:
    """Payoff to one bidder playing ``own`` while every other bidder plays ``population``."""
    stay = symmetric.anonymise(auction, profile(auction, population))
    deviation = symmetric.anonymise(auction, profile(auction, own))
    return sum(
        (auction.q if signal else 1 - auction.q) * symmetric.tagged_payoff(auction, stay, deviation, signal)
        for signal in (0, 1)
    )


def empirical_game(auction: Auction, strategies: Dict[str, Strategy]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Symmetric empirical payoff matrix over a strategy set, and each profile's regret."""
    names = list(strategies)
    payoffs = np.array(
        [[deviation_payoff(auction, strategies[row], strategies[column]) for column in names] for row in names]
    )
    return names, payoffs, payoffs.max(axis=0) - np.diag(payoffs)


# ---------------------------------------------------------------------------
# Closed forms for the two-bidder example (ground truth)
# ---------------------------------------------------------------------------

# The paper states these in sum units. Each form below is the published one rewritten in mean units
# (ADR-0002), the same game at half the cost, tick and revenue. PUBLISHED keeps the originals.

sigma1 = lambda c: 1 - 8 * c / 3
sigma3 = lambda c: 1 - 8 * c
R0 = lambda c: 0.25 * (1 - sigma1(c) ** 2) + sigma1(c) ** 2 * (0.75 + 0.25 * sigma3(c) ** 2)
L0 = lambda q: 8 * q**2 + 2 * q * (1 - q)  # expected staying bidder-rounds, frictionless and unitless
PI = {
    "zero": lambda c, q: q**2 * R0(c),
    "full": lambda c, q: q**2 - c * L0(q),
    "targeted": lambda c, q: q**2 * (1 - 6 * c),
}
ic_bound = lambda q: (1 - q) / 4  # r^T preserves separation iff c <= (1-q)/4

published_sigma1 = lambda c: 1 - 4 * c / 3
published_sigma3 = lambda c: 1 - 4 * c
published_R0 = lambda c: (
    0.5 * (1 - published_sigma1(c) ** 2) + published_sigma1(c) ** 2 * (1.5 + 0.5 * published_sigma3(c) ** 2)
)
PUBLISHED = {  # the paper's own forms, in sum units, read only by the rescaling test
    "zero": lambda c, q: q**2 * published_R0(c),
    "full": lambda c, q: 2 * q**2 - c * L0(q),
    "targeted": lambda c, q: 2 * q**2 * (1 - 3 * c),
}
PUBLISHED_TICK = 0.5  # the tick the published forms assume, halved by the move to mean units


# The root of 64c^3 - 128c^2 + 96c - 9 below one quarter, published as 0.108629 and halved here.
# Frozen because root finding drifts by an ulp between builds and this number reaches ranking.csv.
CBAR = 0.05431457670080486


def region(q: float) -> float:
    """The cost bound below which the targeted policy beats both comparison policies."""
    return min(CBAR, ic_bound(q))



# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def write(name: str, rows: List[Dict[str, object]]) -> None:
    """Write ``rows`` as a CSV file of the given name under ``results``."""
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")
