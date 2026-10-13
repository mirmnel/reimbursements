"""The orbit-based reduction, evaluating a symmetric profile on exit-round orbits rather than histories."""

from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from .game import Auction, Outcome


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
    """Mean stay probability of ``sigma`` in each (orbit, signal) bucket."""
    return bucket_means(auction.orbit_index, len(auction.orbits) * 2, sigma).reshape(-1, 2)


def broadcast(auction: Auction, stay: np.ndarray) -> np.ndarray:
    """The flat profile in which every bidder plays its own orbit's entry of ``stay``."""
    return stay.ravel()[auction.orbit_index]


def symmetrise(auction: Auction, sigma: np.ndarray) -> np.ndarray:
    """The symmetric profile in which every bidder plays the bucket mean of ``sigma``."""
    return broadcast(auction, anonymise(auction, sigma))


def is_symmetric(auction: Auction, sigma: np.ndarray) -> bool:
    """Whether every bidder plays the same strategy, differing only by signal."""
    return bool(np.max(np.abs(sigma - symmetrise(auction, sigma))) <= TOLERANCE)


def applies(auction: Auction, sigma: np.ndarray) -> bool:
    """Whether the reduction is valid here, which needs symmetry, an exchangeable prior and an orbit rule."""
    return auction.exchangeable_signals and auction.orbit_measurable and is_symmetric(auction, sigma)


def requires_exchangeability(auction: Auction) -> None:
    """Refuse the reduction where relabelling the bidders changes the prior or the reimbursement rule."""
    if not auction.exchangeable_signals:
        raise ValueError(
            "the orbit reduction reads the signal profile through its high-signal count alone, which a "
            "prior that is not exchangeable breaks. Evaluate on histories instead, by way of "
            "exact.evaluate or exact.utility_loss"
        )
    if not auction.orbit_measurable:
        raise ValueError(
            "the orbit reduction pays one reimbursement per orbit, which a rule separating histories "
            "inside an orbit breaks. Evaluate on histories instead, by way of exact.evaluate"
        )


def refunds_of(auction: Auction, refunds: Optional[np.ndarray]) -> np.ndarray:
    """The per-orbit reimbursement to evaluate under, being the auction's own unless one is supplied."""
    return auction.orbit_reimburses if refunds is None else refunds


@lru_cache(maxsize=None)
def _arrangements(n: int, departed: Tuple[int, ...]) -> float:
    """How many ways the other bidders fill one orbit's exit rounds, a given bidder staying in."""
    total, remaining = math.comb(n - 1, n - 1 - len(departed)), len(departed)
    for count in Counter(departed).values():
        total *= math.comb(remaining, count)
        remaining -= count
    return float(total)


def arrangements(auction: Auction) -> np.ndarray:
    """Per orbit, how many histories in it leave a given bidder active."""
    return np.array([_arrangements(auction.n, departed) for _, departed in auction.orbits])


# ---------------------------------------------------------------------------
# Seller accounting on orbits
# ---------------------------------------------------------------------------


def _onward(auction: Auction, orbit: int, leaving: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Where each number of departures leads, and whether the auction carries on at all."""
    onward = auction.onward_orbit[orbit][leaving]
    return np.where(onward >= 0, onward, 0), onward >= 0


def seller_values(
    auction: Auction, stay: np.ndarray, refunds: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Expected price and reimbursement from each orbit onward."""
    requires_exchangeability(auction)
    paid = refunds_of(auction, refunds)
    shape = (len(auction.orbits), auction.n + 1)
    price, payout = np.zeros(shape), np.zeros(shape)
    for orbit in reversed(range(len(auction.orbits))):
        round_index = auction.orbits[orbit][0]
        active = int(auction.orbit_active[orbit])
        refund = paid[orbit]
        for high in range(active + 1):
            low = active - high
            thin = np.outer(thinning(high, stay[orbit, 1]), thinning(low, stay[orbit, 0]))
            stayers = np.add.outer(np.arange(high + 1), np.arange(low + 1))
            safe, carries = _onward(auction, orbit, active - stayers)
            ended = auction.price(np.where(stayers >= 2, auction.rounds, round_index))
            survivors = np.broadcast_to(np.arange(high + 1)[:, None], stayers.shape)
            price[orbit, high] = np.sum(thin * np.where(carries, price[safe, survivors], ended))
            payout[orbit, high] = np.sum(
                thin * (stayers * refund + np.where(carries, payout[safe, survivors], 0.0))
            )
    return price, payout


# ---------------------------------------------------------------------------
# One tagged bidder against an otherwise symmetric population
# ---------------------------------------------------------------------------


def branches_at(
    auction: Auction, stay: np.ndarray, values: Values, orbit: int, high: int, refund: float
) -> Tuple[np.ndarray, Values, Values]:
    """Thinning weights at an orbit, and the tagged bidder's value under each action."""
    const, coefficient = values
    round_index = auction.orbits[orbit][0]
    active = int(auction.orbit_active[orbit])
    low = active - 1 - high
    thin = np.outer(thinning(high, stay[orbit, 1]), thinning(low, stay[orbit, 0]))
    stayers = np.add.outer(np.arange(high + 1), np.arange(low + 1))  # the other bidders who stay
    safe, carries = _onward(auction, orbit, (active - 1) - stayers)
    ended = auction.price(np.where(stayers >= 1, auction.rounds, round_index))
    survivors = np.broadcast_to(np.arange(high + 1)[:, None], stayers.shape)
    staying = (
        -(auction.c - refund)
        + np.where(carries, const[safe, survivors], -ended / (stayers + 1)),
        np.where(carries, coefficient[safe, survivors], 1.0 / (stayers + 1)),
    )
    alone = stayers == 0  # all exit at once, so every active bidder splits the prize
    exiting = (
        np.where(alone, -auction.price(round_index) / active, 0.0),
        np.where(alone, 1.0 / active, 0.0),
    )
    return thin, staying, exiting


def empty_values(auction: Auction) -> Values:
    """Zeroed tables over the orbits and the other bidders' high-signal counts, for a backward sweep."""
    shape = (len(auction.orbits), auction.n)
    return np.zeros(shape), np.zeros(shape)


def tagged_values(
    auction: Auction, stay: np.ndarray, deviation: np.ndarray, signal: int,
    refunds: Optional[np.ndarray] = None,
) -> Values:
    """A tagged bidder's payoff from each orbit, as a constant and a coefficient on the value."""
    requires_exchangeability(auction)
    paid = refunds_of(auction, refunds)
    values = empty_values(auction)
    for orbit in reversed(range(len(auction.orbits))):
        own = deviation[orbit, signal]
        for high in range(int(auction.orbit_active[orbit])):
            thin, staying, exiting = branches_at(auction, stay, values, orbit, high, paid[orbit])
            for table, on_stay, on_exit in zip(values, staying, exiting):
                table[orbit, high] = np.sum(thin * (own * on_stay + (1.0 - own) * on_exit))
    return values


def tagged_reach(
    auction: Auction, stay: np.ndarray, signal: int, own: Optional[np.ndarray] = None
) -> Values:
    """Reach of each orbit for a tagged bidder playing ``own``, and the same weighted by value."""
    requires_exchangeability(auction)
    n, others = auction.n, auction.n - 1
    mass = np.zeros((len(auction.orbits), n))
    value_mass = np.zeros_like(mass)
    # Joint in the tagged bidder's own signal and how many of the others drew high, which under a
    # correlated prior is no longer its marginal times a binomial.
    start = auction.own_and_others[signal]
    mass[0, : others + 1] = start
    value_mass[0, : others + 1] = start * (np.arange(others + 1.0) + signal)
    for orbit in range(len(auction.orbits)):
        active = int(auction.orbit_active[orbit])
        for high in range(active):
            if mass[orbit, high] == 0.0 and value_mass[orbit, high] == 0.0:
                continue
            low = active - 1 - high
            thin = np.outer(thinning(high, stay[orbit, 1]), thinning(low, stay[orbit, 0]))
            stayers = np.add.outer(np.arange(high + 1), np.arange(low + 1))
            safe, carries = _onward(auction, orbit, (active - 1) - stayers)
            if own is not None:
                thin = thin * own[orbit, signal]
            survivors = np.broadcast_to(np.arange(high + 1)[:, None], stayers.shape)
            target = (safe[carries], survivors[carries])
            np.add.at(mass, target, mass[orbit, high] * thin[carries])
            np.add.at(value_mass, target, value_mass[orbit, high] * thin[carries])
    return mass, value_mass


def signal_weight(auction: Auction, signal: int) -> float:
    """A tagged bidder's own probability of drawing ``signal``."""
    return auction.high_probability if signal == 1 else 1.0 - auction.high_probability


def combine(auction: Auction, values: Values, signal: int) -> float:
    """Average a tagged bidder's opening value over how many others drew a high signal."""
    requires_exchangeability(auction)
    const, coefficient = values
    high = np.arange(auction.n)
    # Conditional on the tagged bidder's own signal, since the caller weights by that separately.
    weights = auction.own_and_others[signal] / signal_weight(auction, signal)
    return float(np.sum(weights * (const[0, high] + coefficient[0, high] * (high + signal))))


def tagged_payoff(
    auction: Auction, stay: np.ndarray, deviation: np.ndarray, signal: int,
    refunds: Optional[np.ndarray] = None,
) -> float:
    """Payoff of a tagged bidder holding ``signal`` and playing ``deviation`` against ``stay``."""
    return combine(auction, tagged_values(auction, stay, deviation, signal, refunds), signal)


# ---------------------------------------------------------------------------
# The public reduction
# ---------------------------------------------------------------------------


def outcome(auction: Auction, sigma: np.ndarray, refunds: Optional[np.ndarray] = None) -> Outcome:
    """Exact expectations under a symmetric profile, on exit-round orbits rather than histories."""
    stay = anonymise(auction, sigma)
    price, payout = seller_values(auction, stay, refunds)
    n = auction.n
    high = np.arange(n + 1)
    seller = [float(np.sum(auction.signal_counts * table[0, high])) for table in (price, payout)]
    payoff = sum(
        signal_weight(auction, signal) * tagged_payoff(auction, stay, stay, signal, refunds)
        for signal in (0, 1)
    )
    return Outcome(np.full(n, payoff), seller[0], seller[1])


def bucket_arrival(auction: Auction, sigma: np.ndarray) -> np.ndarray:
    """Probability a bidder actually arrives at each (orbit, signal), its own stays folded in."""
    stay = anonymise(auction, sigma)
    return np.stack(
        [tagged_reach(auction, stay, signal, stay)[0].sum(axis=1) for signal in (0, 1)], axis=1
    )


def gains(
    auction: Auction, sigma: np.ndarray, own: Optional[np.ndarray] = None,
    refunds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deviation gain and the reach weighting it, per history of that orbit, at each information set."""
    stay = anonymise(auction, sigma)
    paid = refunds_of(auction, refunds)
    gain = np.zeros((len(auction.orbits), 2))
    reach = np.zeros_like(gain)
    for signal in (0, 1):
        mass, value_mass = tagged_reach(auction, stay, signal, own)
        values = tagged_values(auction, stay, stay, signal, refunds)
        for orbit in range(len(auction.orbits)):
            for high in range(int(auction.orbit_active[orbit])):
                if mass[orbit, high] == 0.0 and value_mass[orbit, high] == 0.0:
                    continue
                thin, staying, exiting = branches_at(auction, stay, values, orbit, high, paid[orbit])
                gain[orbit, signal] += mass[orbit, high] * np.sum(
                    thin * (staying[0] - exiting[0])
                ) + value_mass[orbit, high] * np.sum(thin * (staying[1] - exiting[1]))
                reach[orbit, signal] += mass[orbit, high]
    spread = arrangements(auction)[:, None]
    return gain / spread, reach / spread


def best_response(auction: Auction, stay: np.ndarray, signal: int) -> np.ndarray:
    """A tagged bidder's best response to a symmetric profile, by a backward sweep on orbits."""
    response = stay.copy()
    mass, value_mass = tagged_reach(auction, stay, signal)
    paid = auction.orbit_reimburses
    values = empty_values(auction)
    for orbit in reversed(range(len(auction.orbits))):
        gain, branches = 0.0, {}
        for high in range(int(auction.orbit_active[orbit])):
            branches[high] = branches_at(auction, stay, values, orbit, high, paid[orbit])
            thin, staying, exiting = branches[high]
            gain += mass[orbit, high] * np.sum(
                thin * (staying[0] - exiting[0])
            ) + value_mass[orbit, high] * np.sum(thin * (staying[1] - exiting[1]))
        response[orbit, signal] = float(gain > 0.0)
        for high, (thin, staying, exiting) in branches.items():
            own = response[orbit, signal]
            for table, on_stay, on_exit in zip(values, staying, exiting):
                table[orbit, high] = np.sum(thin * (own * on_stay + (1.0 - own) * on_exit))
    return response


def utility_loss(auction: Auction, sigma: np.ndarray) -> float:
    """Utility loss of a symmetric profile, from a tagged bidder's exact best response."""
    stay = anonymise(auction, sigma)
    loss = 0.0
    for signal in (0, 1):
        prior = signal_weight(auction, signal)
        response = best_response(auction, stay, signal)
        loss += prior * (
            tagged_payoff(auction, stay, response, signal) - tagged_payoff(auction, stay, stay, signal)
        )
    return loss

