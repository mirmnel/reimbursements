"""The count-based reduction of ADR-0001, evaluating a symmetric profile on counts of active bidders."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from game import Auction, Outcome


TOLERANCE = 1e-12  # how far a profile may sit from its bucket mean and still be symmetric

Values = Tuple[np.ndarray, np.ndarray]  # a payoff constant and a coefficient on the common value


@lru_cache(maxsize=None)
def _row(count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Binomial coefficients for ``count`` bidders, and the survivor counts they weight."""
    return np.array([math.comb(count, j) for j in range(count + 1)], dtype=float), np.arange(count + 1.0)


def thinning(count: int, probability: float) -> np.ndarray:
    """Probability that ``count`` bidders staying with ``probability`` leave each survivor count."""
    coefficients, survivors = _row(count)
    return coefficients * probability**survivors * (1.0 - probability) ** (count - survivors)


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


def bucket_means(index: np.ndarray, size: int, sigma: np.ndarray) -> np.ndarray:
    """Mean of ``sigma`` in each of ``size`` buckets, leaving empty buckets at zero."""
    totals = np.bincount(index, weights=sigma, minlength=size)
    counts = np.bincount(index, minlength=size)
    return np.divide(totals, counts, out=np.zeros(size), where=counts > 0)


def anonymise(auction: Auction, sigma: np.ndarray) -> np.ndarray:
    """Mean stay probability of ``sigma`` in each (round, active count, signal) bucket."""
    size = auction.rounds * (auction.n + 1) * 2
    return bucket_means(auction.anonymous_index, size, sigma).reshape(auction.rounds, auction.n + 1, 2)


def broadcast(auction: Auction, stay: np.ndarray) -> np.ndarray:
    """The flat profile in which every bidder plays its own bucket's entry of ``stay``."""
    return stay.ravel()[auction.anonymous_index]


def symmetrise(auction: Auction, sigma: np.ndarray) -> np.ndarray:
    """The symmetric profile in which every bidder plays the bucket mean of ``sigma``."""
    return broadcast(auction, anonymise(auction, sigma))


def is_symmetric(auction: Auction, sigma: np.ndarray) -> bool:
    """Whether every bidder plays the same strategy, differing only by signal."""
    return bool(np.max(np.abs(sigma - symmetrise(auction, sigma))) <= TOLERANCE)


def applies(auction: Auction, sigma: np.ndarray) -> bool:
    """Whether the reduction is valid here, which needs symmetry."""
    return is_symmetric(auction, sigma)


# ---------------------------------------------------------------------------
# Seller accounting on counts of active bidders
# ---------------------------------------------------------------------------


def seller_values(auction: Auction, stay: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expected price, reimbursement and staying bidder-rounds from each count state onward."""
    n = auction.n
    price = np.zeros((auction.rounds + 1, n + 1, n + 1))
    payout, stayed = np.zeros_like(price), np.zeros_like(price)
    for round_index in reversed(range(auction.rounds)):
        for high in range(n + 1):
            for low in range(n + 1 - high):
                if high + low < 2:
                    continue
                active = high + low
                refund = auction.reimburses(round_index)
                thin = np.outer(
                    thinning(high, stay[round_index, active, 1]), thinning(low, stay[round_index, active, 0])
                )
                stayers = np.add.outer(np.arange(high + 1), np.arange(low + 1))
                carries = (stayers >= 2) & (round_index + 1 < auction.rounds)
                ended = auction.price(np.where(stayers >= 2, auction.rounds, round_index))
                onward = (round_index + 1, slice(high + 1), slice(low + 1))
                next_price, next_payout, next_stayed = price[onward], payout[onward], stayed[onward]
                price[round_index, high, low] = np.sum(thin * np.where(carries, next_price, ended))
                payout[round_index, high, low] = np.sum(
                    thin * (stayers * refund + np.where(carries, next_payout, 0.0))
                )
                stayed[round_index, high, low] = np.sum(thin * (stayers + np.where(carries, next_stayed, 0.0)))
    return price[0], payout[0], stayed[0]


# ---------------------------------------------------------------------------
# One tagged bidder against an otherwise symmetric population
# ---------------------------------------------------------------------------


def branches_at(
    auction: Auction, stay: np.ndarray, values: Values, round_index: int, high: int, low: int
) -> Tuple[np.ndarray, Values, Values]:
    """Thinning weights at a count state, and the tagged bidder's value under each action."""
    const, coefficient = values
    active = high + low + 1
    thin = np.outer(thinning(high, stay[round_index, active, 1]), thinning(low, stay[round_index, active, 0]))
    stayers = np.add.outer(np.arange(high + 1), np.arange(low + 1))
    carries = (stayers >= 1) & (round_index + 1 < auction.rounds)
    ended = auction.price(np.where(stayers >= 1, auction.rounds, round_index))
    onward = (round_index + 1, slice(high + 1), slice(low + 1))
    next_const, next_coefficient = const[onward], coefficient[onward]
    staying = (
        -(auction.c - auction.reimburses(round_index))
        + np.where(carries, next_const, -ended / (stayers + 1)),
        np.where(carries, next_coefficient, 1.0 / (stayers + 1)),
    )
    alone = stayers == 0  # all exit at once, so every active bidder splits the prize
    exiting = (
        np.where(alone, -auction.price(round_index) / active, 0.0),
        np.where(alone, 1.0 / active, 0.0),
    )
    return thin, staying, exiting


def empty_values(auction: Auction) -> Values:
    """Zeroed tables over rounds and the other bidders' counts, for a backward sweep to fill."""
    shape = (auction.rounds + 1, auction.n, auction.n)
    return np.zeros(shape), np.zeros(shape)


def tagged_values(auction: Auction, stay: np.ndarray, deviation: np.ndarray, signal: int) -> Values:
    """A tagged bidder's payoff from each count state, as a constant and a coefficient on the value."""
    values = empty_values(auction)
    for round_index in reversed(range(auction.rounds)):
        for high in range(auction.n):
            for low in range(auction.n - high):
                thin, staying, exiting = branches_at(auction, stay, values, round_index, high, low)
                own = deviation[round_index, high + low + 1, signal]
                for table, on_stay, on_exit in zip(values, staying, exiting):
                    table[round_index, high, low] = np.sum(thin * (own * on_stay + (1.0 - own) * on_exit))
    return values


def tagged_reach(
    auction: Auction, stay: np.ndarray, signal: int, own: Optional[np.ndarray] = None
) -> Values:
    """Reach of each count state for a tagged bidder playing ``own``, and the same weighted by value."""
    n, q, others = auction.n, auction.q, auction.n - 1
    mass = np.zeros((auction.rounds, n, n))
    value_mass = np.zeros_like(mass)
    opening = (q if signal == 1 else 1 - q) * thinning(others, q)
    for high in range(others + 1):
        mass[0, high, others - high] = opening[high]
        value_mass[0, high, others - high] = opening[high] * (high + signal) / n
    for round_index in range(auction.rounds - 1):
        for high in range(n):
            for low in range(n - high):
                if mass[round_index, high, low] == 0.0:
                    continue
                active = high + low + 1
                thin = np.outer(
                    thinning(high, stay[round_index, active, 1]), thinning(low, stay[round_index, active, 0])
                )
                thin[0, 0] = 0.0  # a lone stayer ends the auction, reaching no later round
                if own is not None:
                    thin = thin * own[round_index, active, signal]
                mass[round_index + 1, : high + 1, : low + 1] += mass[round_index, high, low] * thin
                value_mass[round_index + 1, : high + 1, : low + 1] += value_mass[round_index, high, low] * thin
    return mass, value_mass


def combine(auction: Auction, values: Values, signal: int) -> float:
    """Average a tagged bidder's opening value over how many others drew a high signal."""
    const, coefficient = values
    others = auction.n - 1
    weights = thinning(others, auction.q)
    high = np.arange(others + 1)
    low = others - high
    return float(
        np.sum(weights * (const[0, high, low] + coefficient[0, high, low] * (high + signal) / auction.n))
    )


def tagged_payoff(auction: Auction, stay: np.ndarray, deviation: np.ndarray, signal: int) -> float:
    """Payoff of a tagged bidder holding ``signal`` and playing ``deviation`` against ``stay``."""
    return combine(auction, tagged_values(auction, stay, deviation, signal), signal)


# ---------------------------------------------------------------------------
# The public reduction
# ---------------------------------------------------------------------------


def outcome(auction: Auction, sigma: np.ndarray) -> Outcome:
    """Exact expectations under a symmetric profile, on counts of active bidders rather than sets."""
    stay = anonymise(auction, sigma)
    price, payout, stayed = seller_values(auction, stay)
    n, q = auction.n, auction.q
    weights = thinning(n, q)
    high = np.arange(n + 1)
    seller = [float(np.sum(weights * table[high, n - high])) for table in (price, payout, stayed)]
    payoff = q * tagged_payoff(auction, stay, stay, 1) + (1 - q) * tagged_payoff(auction, stay, stay, 0)
    return Outcome(np.full(n, payoff), seller[0], seller[1], seller[2])


def bucket_reach(auction: Auction, sigma: np.ndarray) -> np.ndarray:
    """Probability a bidder reaches each (round, active count, signal), under ``sigma`` symmetrised."""
    stay = anonymise(auction, sigma)
    reach = np.zeros((auction.rounds, auction.n + 1, 2))
    for signal in (0, 1):
        mass = tagged_reach(auction, stay, signal)[0]
        for high in range(auction.n):
            for low in range(auction.n - high):
                reach[:, high + low + 1, signal] += mass[:, high, low]
    return reach


def gains(
    auction: Auction, sigma: np.ndarray, own: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Deviation gain and the reach weighting it, per active set of that size, at each information set."""
    stay = anonymise(auction, sigma)
    gain = np.zeros((auction.rounds, auction.n + 1, 2))
    reach = np.zeros_like(gain)
    for signal in (0, 1):
        mass, value_mass = tagged_reach(auction, stay, signal, own)
        values = tagged_values(auction, stay, stay, signal)
        for round_index in range(auction.rounds):
            for high in range(auction.n):
                for low in range(auction.n - high):
                    if mass[round_index, high, low] == 0.0:
                        continue
                    thin, staying, exiting = branches_at(auction, stay, values, round_index, high, low)
                    active = high + low + 1
                    gain[round_index, active, signal] += mass[round_index, high, low] * np.sum(
                        thin * (staying[0] - exiting[0])
                    ) + value_mass[round_index, high, low] * np.sum(thin * (staying[1] - exiting[1]))
                    reach[round_index, active, signal] += mass[round_index, high, low]
    sets = np.array([math.comb(auction.n - 1, max(size - 1, 0)) for size in range(auction.n + 1)], dtype=float)
    return gain / sets[None, :, None], reach / sets[None, :, None]


def best_response(auction: Auction, stay: np.ndarray, signal: int) -> np.ndarray:
    """A tagged bidder's best response to a symmetric profile, by a backward sweep on count states."""
    response = stay.copy()
    mass, value_mass = tagged_reach(auction, stay, signal)
    values = empty_values(auction)
    for round_index in reversed(range(auction.rounds)):
        gain, branches = np.zeros(auction.n + 1), {}
        for high in range(auction.n):
            for low in range(auction.n - high):
                branches[(high, low)] = branches_at(auction, stay, values, round_index, high, low)
                thin, staying, exiting = branches[(high, low)]
                gain[high + low + 1] += mass[round_index, high, low] * np.sum(
                    thin * (staying[0] - exiting[0])
                ) + value_mass[round_index, high, low] * np.sum(thin * (staying[1] - exiting[1]))
        response[round_index, :, signal] = (gain > 0.0).astype(float)
        for (high, low), (thin, staying, exiting) in branches.items():
            own = response[round_index, high + low + 1, signal]
            for table, on_stay, on_exit in zip(values, staying, exiting):
                table[round_index, high, low] = np.sum(thin * (own * on_stay + (1.0 - own) * on_exit))
    return response


def best_responses(auction: Auction, stay: np.ndarray) -> np.ndarray:
    """The tagged bidder's best response at every information set, both signals in one table."""
    table = stay.copy()
    for signal in (0, 1):
        table[:, :, signal] = best_response(auction, stay, signal)[:, :, signal]
    return table


def utility_loss(auction: Auction, sigma: np.ndarray) -> float:
    """Utility loss of a symmetric profile, from a tagged bidder's exact best response."""
    stay = anonymise(auction, sigma)
    loss = 0.0
    for signal in (0, 1):
        prior = auction.q if signal == 1 else 1 - auction.q
        response = best_response(auction, stay, signal)
        loss += prior * (
            tagged_payoff(auction, stay, response, signal) - tagged_payoff(auction, stay, stay, signal)
        )
    return loss


def max_regret(auction: Auction, sigma: np.ndarray) -> float:
    """Largest one-shot deviation gain at any information set of positive opponent reach."""
    stay = anonymise(auction, sigma)
    gain, reach = gains(auction, sigma)
    reached = reach > 1e-12
    normalised = np.divide(gain, reach, out=np.zeros_like(gain), where=reached)
    regret = (1 - stay) * np.maximum(normalised, 0.0) + stay * np.maximum(-normalised, 0.0)
    return float(np.max(np.where(reached, regret, 0.0)))
