"""Benchmark strategies, equilibrium quality, the closed forms, and the results writer."""

from __future__ import annotations

import csv
import os
from typing import Callable, Dict, List, Tuple, Union

import numpy as np

import symmetric
from solver import counterfactual_gain, counterfactual_reach, evaluate
from game import Auction, Strategy


RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CERTIFIED = 5e-4  # the utility loss every certified profile clears, under every family and count

RESOLVED = 1e-12  # the least counterfactual reach a gain can be divided by without amplifying noise

Number = Union[float, np.ndarray]  # scalar at one information set, or a table of them
Payoffs = Callable[[np.ndarray], np.ndarray]  # a profile -> what it pays each bidder


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def profile(auction: Auction, *strategies: Strategy) -> np.ndarray:
    """Flat behaviour profile from one strategy per bidder, or from one shared by all."""
    if len(strategies) == 1:
        strategies *= auction.n
    return np.array([float(strategies[bidder](state, signal)) for (bidder, state, signal) in auction.info_sets])


SEPARATING: Strategy = lambda state, signal: float(signal == 1)  # L exits at p = 0, H stays
ALWAYS: Strategy = lambda state, signal: 1.0
NEVER: Strategy = lambda state, signal: 0.0


def war_of_attrition(c: float) -> Strategy:
    """The low type exits everywhere, and the high type stays at the opening and then mixes."""
    # The two-bidder closed form, which larger auctions borrow as a benchmark, not an equilibrium.
    schedule = (1.0, sigma1(c), 1.0, sigma3(c))
    return lambda state, signal: schedule[state[0]] if signal == 1 else 0.0


def closed_form_strategy(rule: str, c: float) -> Strategy:
    """The strategy the two-bidder example derives under one published reimbursement rule."""
    return war_of_attrition(c) if rule == "zero" else SEPARATING


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


def one_shot_regret(gain: Number, stay_prob: Number) -> Number:
    """What mixing at ``stay_prob`` costs against taking the better of the two actions outright."""
    return (1 - stay_prob) * np.maximum(gain, 0.0) + stay_prob * np.maximum(-gain, 0.0)


def max_regret(auction: Auction, sigma: np.ndarray) -> float:
    """Largest one-shot deviation gain at any information set of positive opponent reach."""
    if symmetric.applies(auction, sigma):
        return symmetric.max_regret(auction, sigma)
    weights, worst = counterfactual_reach(auction, sigma), 0.0
    for position in range(len(auction.info_sets)):
        if weights[position] > RESOLVED:
            gain = counterfactual_gain(auction, sigma, position) / weights[position]
            worst = max(worst, float(one_shot_regret(gain, sigma[position])))
    return worst


def drop_out(thresholds: Tuple[int, int]) -> Strategy:
    """Stay while the clock sits below this bidder's own drop-out price, which its signal alone sets."""
    return lambda state, signal: float(state.round < thresholds[signal])


def with_drop_out(auction: Auction, sigma: np.ndarray, bidder: int, thresholds: Tuple[int, int]) -> np.ndarray:
    """``sigma`` with one bidder replaced by a drop-out price strategy and the rest left alone."""
    replaced = sigma.copy()
    strategy = drop_out(thresholds)
    for position, (at, state, signal) in enumerate(auction.info_sets):
        if at == bidder:
            replaced[position] = strategy(state, signal)
    return replaced


def restricted_loss(auction: Auction, sigma: np.ndarray, payoff_of: Payoffs) -> float:
    """Lower bound on utility loss, deviating only to drop-out price strategies rather than to any strategy."""
    # A bound rather than the quantity, for the profiles whose tree is too large to best respond over.
    baseline, worst = payoff_of(sigma), 0.0
    for bidder in range(auction.n):
        for low in range(auction.rounds + 1):
            for high in range(auction.rounds + 1):
                deviated = payoff_of(with_drop_out(auction, sigma, bidder, (low, high)))
                worst = max(worst, float(deviated[bidder] - baseline[bidder]))
    return worst


def info_rounds(auction: Auction) -> np.ndarray:
    """The round each information set sits in, aligned with a flat behaviour profile."""
    return np.array([state.round for _, state, _ in auction.info_sets])


def trembled_beliefs(auction: Auction, sigma: np.ndarray, tremble: float, round_index: int) -> np.ndarray:
    """``sigma`` with every move before ``round_index`` mixed toward uniform at rate ``tremble``."""
    return np.where(info_rounds(auction) < round_index, (1 - tremble) * sigma + tremble * 0.5, sigma)


def trembled_regret(auction: Auction, sigma: np.ndarray, tremble: float) -> float:
    """Largest one-shot deviation gain at any information set, off-path ones weighted by a tremble."""
    # On path Bayes already fixes the weights, so only the sets max_regret drops need a tremble.
    worst = max_regret(auction, sigma)
    if tremble == 0.0:
        return worst
    rounds, reached = info_rounds(auction), counterfactual_reach(auction, sigma)
    for round_index in range(auction.rounds):
        unreached = np.flatnonzero((rounds == round_index) & (reached <= RESOLVED))
        if not unreached.size:
            continue
        # Beliefs match sigma from round_index onward, so the gain is the untrembled profile's.
        beliefs = trembled_beliefs(auction, sigma, tremble, round_index)
        weights = counterfactual_reach(auction, beliefs)
        for position in unreached:
            if weights[position] == 0.0:  # no sequence of trembles reaches it, so there is nothing to score
                continue
            if weights[position] <= RESOLVED:
                raise ValueError(
                    f"tremble {tremble} reaches {auction.info_sets[position]} with weight "
                    f"{weights[position]:.2e}, too faint to divide into. Raise the tremble rate."
                )
            gain = counterfactual_gain(auction, beliefs, position) / weights[position]
            worst = max(worst, float(one_shot_regret(gain, sigma[position])))
    return worst


# ---------------------------------------------------------------------------
# Closed forms for the two-bidder example (ground truth)
# ---------------------------------------------------------------------------

# The paper states these in sum units. Each form below is the published one rewritten in mean units,
# the same game at half the cost, tick and revenue. PUBLISHED keeps the originals.

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


def region_claim(c: float, q: float) -> bool:
    """Whether the theory puts the targeted policy ahead of both comparisons at this cost."""
    # The two bounds close the region differently. At the cost root the revenues tie, so the claim
    # needs to sit strictly below it. At the incentive bound separation still holds, so the bound
    # itself is inside the claim.
    return 0.0 < c < CBAR and c <= ic_bound(q)



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
