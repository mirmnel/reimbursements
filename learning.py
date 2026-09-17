"""Independent tabular learners, per ADR-0003: each bidder updates from its own experience alone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple, Tuple

import numpy as np

import symmetric
from solver import evaluate
from game import Auction, Outcome


DISPERSED = 0.05  # across-bidder spread above which symmetrising would mislead
CYCLING = 0.05  # late-run spread of the iterates above which a run is cycling
REACHED = 1e-6  # reach below which a bucket is too rare to read
CONFIDENT = 4.0  # standard errors a sampled revenue must move to be believed
EPISODES = 256  # auctions played per iteration at the routine budget
AT_SCALE = 10  # bidder count at which coverage is thin enough to be worth buying more of

Lookups = Tuple[np.ndarray, np.ndarray]  # info-set positions, and reimbursement by round


@dataclass
class Learned:
    """What one run produced, with the traces needed to judge it."""

    profile: np.ndarray  # time-average of the iterates, flat over the information sets
    iterates: np.ndarray  # (sample, bidder, bucket) stay probability along the run
    averages: np.ndarray  # the same for the time-average
    dispersion: float  # across-bidder spread of the time-average, weighted by reach
    visits: np.ndarray  # times the run reached each information set
    seed: int
    iterations: int
    episodes: int  # auctions played per iteration

    @property
    def visited(self) -> int:
        """How many information sets the run reached at least once."""
        return int(np.count_nonzero(self.visits))

    def visit_quantiles(self, *quantiles: float) -> np.ndarray:
        """Visits per information set reached, at the given quantiles, so the tail is readable."""
        return np.quantile(self.visits[self.visits > 0], quantiles)


# ---------------------------------------------------------------------------
# Reading a profile as one table per bidder
# ---------------------------------------------------------------------------


def bidder_buckets(auction: Auction) -> Tuple[np.ndarray, int]:
    """Slot of each information set in a (bidder, round, active count, signal) table, and its size."""
    width = auction.rounds * (auction.n + 1) * 2
    bidders = np.array([bidder for bidder, _, _ in auction.info_sets])
    return bidders * width + auction.anonymous_index, auction.n * width


def per_bidder(auction: Auction, sigma: np.ndarray) -> np.ndarray:
    """Each bidder's mean stay probability per anonymised bucket, which is its strategy table."""
    index, size = bidder_buckets(auction)
    return symmetric.bucket_means(index, size, sigma).reshape(auction.n, -1)


def weights(reach: np.ndarray) -> np.ndarray:
    """How much each anonymised bucket counts, which is how often a bidder reaches it."""
    reachable = np.where(reach.ravel() > REACHED, reach.ravel(), 0.0)
    total = reachable.sum()
    return reachable / total if total > 0.0 else reachable


def spread(auction: Auction, sigma: np.ndarray, reach: np.ndarray) -> float:
    """Across-bidder spread of stay probability, averaged over buckets by their reach."""
    return float(per_bidder(auction, sigma).std(axis=0) @ weights(reach))


# ---------------------------------------------------------------------------
# Playing the auction out
# ---------------------------------------------------------------------------


def lookups(auction: Auction) -> Lookups:
    """Information-set positions by bidder, round, active set and signal, and reimbursement by round."""
    positions = np.full((auction.n, auction.rounds, 1 << auction.n, 2), -1, dtype=np.int64)
    for position, (bidder, (round_index, active), signal) in enumerate(auction.info_sets):
        positions[bidder, round_index, sum(1 << member for member in active), signal] = position
    refunds = np.array([auction.reimburses(t) for t in range(auction.rounds)])
    return positions, refunds


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
    positions, refunds = tables
    n, rounds = auction.n, auction.rounds
    signals = auction.profiles[rng.choice(auction.num_profiles, size=episodes, p=auction.prior)]
    value = signals.mean(1)
    active = np.ones((episodes, n), dtype=bool)
    live = np.ones(episodes, dtype=bool)
    powers, bidders = 1 << np.arange(n), np.arange(n)[None, :]
    terminal = np.zeros((episodes, n))
    paid, refunded, rounds_stayed = np.zeros(episodes), np.zeros(episodes), np.zeros(episodes)
    record, costs = [], []
    for round_index in range(rounds):
        acting = active & live[:, None]
        masks = active.astype(np.int64) @ powers
        spot = positions[bidders, round_index, masks[:, None], signals]
        probability = np.where(acting, played[spot], 0.0)
        stayed = acting & (rng.random((episodes, n)) < probability)
        chance = np.where(stayed, probability, 1.0 - probability)
        count = acting.sum(1)
        cost = np.where(stayed, auction.c - refunds[round_index], 0.0)
        stayers = stayed.sum(1)
        carries = live & (stayers >= 2) & (round_index + 1 < rounds)
        ending = live & ~carries
        winners = np.where((stayers >= 1)[:, None], stayed, acting)
        share = np.where(stayers >= 1, stayers, np.maximum(count, 1))
        price = auction.delta * np.where(stayers >= 2, rounds, round_index)
        terminal += np.where(winners & ending[:, None], ((value - price) / share)[:, None], 0.0)
        paid += np.where(ending, price, 0.0)
        refunded += stayers * refunds[round_index]
        rounds_stayed += stayers
        record.append((spot, stayed, chance, acting))
        costs.append(cost)
        active, live = stayed, carries
        if not live.any():
            break
    togo, rewards = terminal, [None] * len(costs)
    for round_index in reversed(range(len(costs))):
        togo = togo - costs[round_index]
        rewards[round_index] = togo
    tally = Tally(paid, refunded, rounds_stayed, rewards[0])
    return Played([entry + (reward,) for entry, reward in zip(record, rewards)], tally)


# ---------------------------------------------------------------------------
# Independent regret-matching bidders
# ---------------------------------------------------------------------------


def matching(regret: np.ndarray) -> np.ndarray:
    """Regret matching: stay with probability proportional to the regret of not having stayed."""
    positive = np.maximum(regret, 0.0)
    total = positive.sum(1)
    # Nothing regretted yet, so no reason to lean either way.
    return np.divide(positive[:, 1], total, out=np.full(len(regret), 0.5), where=total > 0.0)


def budget(n: int) -> int:
    """Episodes per iteration, raised at the scale where ADR-0004 says coverage is too thin to leave."""
    return 10 * EPISODES if n >= AT_SCALE else EPISODES


def learn(
    auction: Auction,
    seed: int = 0,
    iterations: int = 400,
    episodes: int = EPISODES,
    exploration: float = 0.1,
    initial_regret: float = 0.05,
    trace_every: int = 4,
) -> Learned:
    """Run independent tabular regret-matching bidders, each updating from its own experience alone."""
    rng = np.random.default_rng(seed)
    regret = initial_regret * rng.uniform(0.0, 1.0, (len(auction.info_sets), 2))  # a random start per bidder
    tables = lookups(auction)
    played = matching(regret)
    average = played.copy()
    visits = np.zeros(len(auction.info_sets), dtype=np.int64)
    iterates, averages = [], []
    for iteration in range(1, iterations + 1):
        # Bidders explore off their own strategy, but hold the matched one.
        behaviour = (1.0 - exploration) * played + exploration * 0.5
        for spot, stayed, chance, acting, reward in play(auction, behaviour, rng, episodes, tables).experience:
            position = spot[acting]
            visits += np.bincount(position, minlength=len(visits))
            action = stayed[acting].astype(np.int64)
            got = reward[acting] / chance[acting]
            np.add.at(regret, (position, action), got - reward[acting])
            np.add.at(regret, (position, 1 - action), -reward[acting])
        played = matching(regret)
        average += (played - average) / (iteration + 1)
        if iteration % trace_every == 0:
            iterates.append(per_bidder(auction, played))
            averages.append(per_bidder(auction, average))
    reach = symmetric.bucket_reach(auction, average)
    return Learned(
        profile=average,
        iterates=np.array(iterates),
        averages=np.array(averages),
        dispersion=spread(auction, average, reach),
        visits=visits,
        seed=seed,
        iterations=iterations,
        episodes=episodes,
    )


# ---------------------------------------------------------------------------
# Scoring what emerged
# ---------------------------------------------------------------------------


@dataclass
class Sampled:
    """A Monte Carlo evaluation and its standard errors, for profiles too dispersed to symmetrise."""

    outcome: Outcome
    revenue_error: float
    payoff_error: np.ndarray
    episodes: int


def sampled(auction: Auction, sigma: np.ndarray, seed: int = 0, episodes: int = 20000, batch: int = 2000) -> Sampled:
    """Evaluate any profile by simulation, so one too dispersed to symmetrise is still reportable."""
    rng = np.random.default_rng(seed)
    tables = lookups(auction)
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


@dataclass
class Score:
    """What the learners arrived at, with the caveats the number carries."""

    revenue: float  # exact, from the symmetrised profile
    utility_loss: float  # exact, from the symmetrised profile
    dispersion: float
    misleading: bool  # symmetrising misrepresents what emerged, by spread or by revenue
    cycling: bool  # iterates keep moving, so the time-average is not a resting point
    sampled_revenue: float  # simulated on the profile as it stands, symmetric or not
    sampled_error: float


def cycles(auction: Auction, learned: Learned, reach: np.ndarray) -> bool:
    """Whether the iterates keep moving late in the run, so averaging them conceals a cycle."""
    late = learned.iterates[len(learned.iterates) // 2 :]
    if len(late) < 2:
        return False
    return bool(np.mean(late.std(axis=0) @ weights(reach)) > CYCLING)


def score(auction: Auction, learned: Learned, seed: int = 0) -> Score:
    """Evaluate the symmetrised learned profile exactly, and say when symmetrising misrepresents it."""
    symmetrised = symmetric.symmetrise(auction, learned.profile)
    reach = symmetric.bucket_reach(auction, symmetrised)
    drawn = sampled(auction, learned.profile, seed=seed)
    revenue = evaluate(auction, symmetrised).revenue
    # Symmetrising misleads when bidders disagree, or when its revenue leaves the sampled interval.
    moved = abs(revenue - drawn.outcome.revenue) > CONFIDENT * drawn.revenue_error
    return Score(
        revenue=revenue,
        utility_loss=symmetric.utility_loss(auction, symmetrised),
        dispersion=learned.dispersion,
        misleading=learned.dispersion > DISPERSED or moved,
        cycling=cycles(auction, learned, reach),
        sampled_revenue=drawn.outcome.revenue,
        sampled_error=drawn.revenue_error,
    )
