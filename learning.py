"""Monte Carlo play of the auction, for profiles too large or too dispersed to evaluate exactly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from game import ACTIVE, Auction, Outcome


REACHED = 1e-6  # reach below which a bucket is too rare to read

Lookups = Tuple[np.ndarray, np.ndarray, np.ndarray]  # info-set positions, reimbursement by history, radix


def history_codes(auction: Auction) -> np.ndarray:
    """Place value of each bidder's exit round, which codes a public history as one integer."""
    return (auction.rounds + 1) ** np.arange(auction.n)


def lookups(auction: Auction) -> Lookups:
    """Information-set positions and reimbursement by coded public history, and the radix that codes one."""
    radix = history_codes(auction)
    # The round is its own axis: while nobody has left yet, the exit rounds alone do not say which it is.
    shape = (auction.rounds, (auction.rounds + 1) ** auction.n)
    positions = np.full((auction.n, *shape, 2), -1, dtype=np.int64)
    refunds = np.zeros(shape)
    for position, (bidder, state, signal) in enumerate(auction.info_sets):
        left = [auction.rounds if exited == ACTIVE else exited for exited in state.exits]
        positions[bidder, state.round, int(radix @ left), signal] = position
        refunds[state.round, int(radix @ left)] = auction.reimburses(state)
    return positions, refunds, radix


@dataclass
class Tally:
    """What a simulated batch of auctions realised, one entry per episode."""

    price: np.ndarray
    reimbursement: np.ndarray
    stay_rounds: np.ndarray
    payoff: np.ndarray  # (episode, bidder)


class Played(NamedTuple):
    """One simulated batch: experience for the learner, tally for the sampler."""

    experience: List[tuple]  # per round: info set, action, chance of that action, reward-to-go
    tally: Tally


def play(
    auction: Auction, played: np.ndarray, rng: np.random.Generator, episodes: int, tables: Lookups
) -> Played:
    """Simulate a batch of auctions, returning what each bidder did and what each episode realised."""
    positions, refunds, radix = tables
    n, rounds = auction.n, auction.rounds
    signals = auction.profiles[rng.choice(auction.num_profiles, size=episodes, p=auction.prior)]
    value = signals.mean(1)
    active = np.ones((episodes, n), dtype=bool)
    live = np.ones(episodes, dtype=bool)
    left = np.full((episodes, n), rounds, dtype=np.int64)  # exit round per bidder, ``rounds`` while in
    bidders = np.arange(n)[None, :]
    terminal = np.zeros((episodes, n))
    paid, refunded, rounds_stayed = np.zeros(episodes), np.zeros(episodes), np.zeros(episodes)
    record, costs = [], []
    for round_index in range(rounds):
        acting = active & live[:, None]
        coded = left @ radix
        spot = positions[bidders, round_index, coded[:, None], signals]
        probability = np.where(acting, played[spot], 0.0)
        stayed = acting & (rng.random((episodes, n)) < probability)
        chance = np.where(stayed, probability, 1.0 - probability)
        count = acting.sum(1)
        refund = refunds[round_index, coded]
        cost = np.where(stayed, (auction.c - refund)[:, None], 0.0)
        stayers = stayed.sum(1)
        carries = live & (stayers >= 2) & (round_index + 1 < rounds)
        ending = live & ~carries
        winners = np.where((stayers >= 1)[:, None], stayed, acting)
        share = np.where(stayers >= 1, stayers, np.maximum(count, 1))
        price = auction.delta * np.where(stayers >= 2, rounds, round_index)
        terminal += np.where(winners & ending[:, None], ((value - price) / share)[:, None], 0.0)
        paid += np.where(ending, price, 0.0)
        refunded += stayers * refund
        rounds_stayed += stayers
        record.append((spot, stayed, chance, acting))
        costs.append(cost)
        left = np.where(acting & ~stayed, round_index, left)
        active, live = stayed, carries
        if not live.any():
            break
    togo, rewards = terminal, [None] * len(costs)
    for round_index in reversed(range(len(costs))):
        togo = togo - costs[round_index]
        rewards[round_index] = togo
    tally = Tally(paid, refunded, rounds_stayed, rewards[0])
    return Played([entry + (reward,) for entry, reward in zip(record, rewards)], tally)


@dataclass
class Sampled:
    """A Monte Carlo evaluation and its standard errors, for profiles too dispersed to symmetrise."""

    outcome: Outcome
    revenue_error: float
    payoff_error: np.ndarray
    episodes: int


def sampled(
    auction: Auction, sigma: np.ndarray, seed: int = 0, episodes: int = 20000, batch: int = 2000,
    tables: Optional[Lookups] = None,
) -> Sampled:
    """Evaluate any profile by simulation, so one too dispersed to symmetrise is still reportable."""
    rng = np.random.default_rng(seed)
    tables = lookups(auction) if tables is None else tables
    parts = []
    for start in range(0, episodes, batch):
        parts.append(play(auction, sigma, rng, min(batch, episodes - start), tables).tally)
    price = np.concatenate([part.price for part in parts])
    reimbursement = np.concatenate([part.reimbursement for part in parts])
    stay_rounds = np.concatenate([part.stay_rounds for part in parts])
    payoff = np.concatenate([part.payoff for part in parts])
    root = np.sqrt(len(price))
    return Sampled(
        outcome=Outcome(payoff.mean(0), price.mean(), reimbursement.mean(), stay_rounds.mean()),
        revenue_error=float((price - reimbursement).std() / root),
        payoff_error=payoff.std(0) / root,
        episodes=len(price),
    )

