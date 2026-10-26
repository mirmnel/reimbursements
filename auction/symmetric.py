"""The orbit-based reduction, evaluating a symmetric profile on exit-round orbits rather than histories."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .game import Auction, Outcome, bucket_means


TOLERANCE = 1e-12  # how far a profile may sit from its bucket mean and still be symmetric

Values = Tuple[np.ndarray, np.ndarray]  # a payoff constant and a coefficient on the common value


def thinning(count: int, probability: float) -> np.ndarray:
    """Probability that ``count`` bidders staying with ``probability`` leave each survivor count."""
    coefficients = np.array([math.comb(count, j) for j in range(count + 1)], dtype=float)
    survivors = np.arange(count + 1.0)
    return coefficients * probability**survivors * (1.0 - probability) ** (count - survivors)


def anonymise(auction: Auction, sigma: np.ndarray) -> np.ndarray:
    """Mean stay probability of ``sigma`` in each (orbit, signal) bucket."""
    return bucket_means(auction.orbit_index, len(auction.orbits) * 2, sigma).reshape(-1, 2)


def applies(auction: Auction, sigma: np.ndarray) -> bool:
    """Whether the reduction is valid here, which needs symmetry and an orbit rule."""
    symmetric = np.max(np.abs(sigma - anonymise(auction, sigma).ravel()[auction.orbit_index])) <= TOLERANCE
    return auction.orbit_view is not None and bool(symmetric)


def _onward(auction: Auction, orbit: int, leaving: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Where each number of departures leads, and whether the auction carries on at all."""
    onward = auction.onward_orbit[orbit][leaving]
    return np.where(onward >= 0, onward, 0), onward >= 0


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
    alone = stayers == 0
    exiting = (
        np.where(alone, -auction.price(round_index) / active, 0.0),
        np.where(alone, 1.0 / active, 0.0),
    )
    return thin, staying, exiting


def empty_values(auction: Auction) -> Values:
    """Zeroed tables over the orbits and the other bidders' high-signal counts, for a backward sweep."""
    shape = (len(auction.orbits), auction.n)
    return np.zeros(shape), np.zeros(shape)


def tagged_reach(
    auction: Auction, stay: np.ndarray, signal: int, own: Optional[np.ndarray] = None
) -> Values:
    """Reach of each orbit for a tagged bidder playing ``own``, and the same weighted by value."""
    n, others = auction.n, auction.n - 1
    mass = np.zeros((len(auction.orbits), n))
    value_mass = np.zeros_like(mass)
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


def tagged_payoff(auction: Auction, stay: np.ndarray, deviation: np.ndarray, signal: int) -> float:
    """Payoff of a tagged bidder holding ``signal`` and playing ``deviation`` against ``stay``."""
    paid = auction.orbit_view
    values = empty_values(auction)
    for orbit in reversed(range(len(auction.orbits))):
        own = deviation[orbit, signal]
        for high in range(int(auction.orbit_active[orbit])):
            thin, staying, exiting = branches_at(auction, stay, values, orbit, high, paid[orbit])
            for table, on_stay, on_exit in zip(values, staying, exiting):
                table[orbit, high] = np.sum(thin * (own * on_stay + (1.0 - own) * on_exit))
    return root_payoff(auction, values, signal)


def root_payoff(auction: Auction, values: Values, signal: int) -> float:
    """A tagged bidder's payoff at the root, from its filled orbit tables."""
    const, coefficient = values
    high = np.arange(auction.n)
    # Conditional on the tagged bidder's own signal, since the caller weights by that separately.
    weights = auction.own_and_others[signal] / signal_weight(auction, signal)
    return float(np.sum(weights * (const[0, high] + coefficient[0, high] * (high + signal))))


def outcome(auction: Auction, sigma: np.ndarray) -> Outcome:
    """Exact expectations under a symmetric profile, on exit-round orbits rather than histories."""
    stay = anonymise(auction, sigma)
    n = auction.n
    # Expected price and reimbursement from each orbit onward, per count of high bidders still in.
    price, payout = np.zeros((len(auction.orbits), n + 1)), np.zeros((len(auction.orbits), n + 1))
    for orbit in reversed(range(len(auction.orbits))):
        round_index = auction.orbits[orbit][0]
        active = int(auction.orbit_active[orbit])
        refund = auction.orbit_view[orbit]
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
    high = np.arange(n + 1)
    seller = [float(np.sum(auction.signal_counts * table[0, high])) for table in (price, payout)]
    payoff = sum(
        signal_weight(auction, signal) * tagged_payoff(auction, stay, stay, signal)
        for signal in (0, 1)
    )
    return Outcome(np.full(n, payoff), seller[0], seller[1])


def bucket_arrival(auction: Auction, sigma: np.ndarray) -> np.ndarray:
    """Probability a bidder actually arrives at each (orbit, signal), its own stays folded in."""
    stay = anonymise(auction, sigma)
    return np.stack(
        [tagged_reach(auction, stay, signal, stay)[0].sum(axis=1) for signal in (0, 1)], axis=1
    )


def utility_loss(auction: Auction, sigma: np.ndarray) -> float:
    """Utility loss of a symmetric profile, from a tagged bidder's exact best response."""
    stay = anonymise(auction, sigma)
    paid = auction.orbit_view
    loss = 0.0
    for signal in (0, 1):
        prior = signal_weight(auction, signal)
        # The best response, by a backward pass over the orbits.
        response = stay.copy()
        mass, value_mass = tagged_reach(auction, stay, signal)
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
        loss += prior * (root_payoff(auction, values, signal) - tagged_payoff(auction, stay, stay, signal))
    return loss

