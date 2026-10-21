"""The state space, the parameterised auction, a solved outcome, and the two-bidder closed forms."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict, List, Mapping, NamedTuple, Sequence, Tuple, Union

import numpy as np

ACTIVE = -1  # exit round of a bidder that has not left


class State(NamedTuple):
    """A public bidding history: the round, who is still in, and the round each departed bidder left."""

    round: int
    active: Tuple[int, ...]
    exits: Tuple[int, ...]  # per bidder, the round it exited in, or ``ACTIVE`` while it is still in


Orbit = Tuple[int, Tuple[int, ...]]  # (round, multiset of the departed bidders' exit rounds)
Info = Tuple[int, State, int]  # (bidder, state, own signal)
Strategy = Callable[[State, int], float]  # Pr(stay) for one bidder
ReimbursementPolicy = Callable[[State], float]  # public history -> paid per stay
SignalPrior = Callable[[np.ndarray], np.ndarray]  # signal profiles -> probability of each

PRIOR_TOLERANCE = 1e-12  # how far a prior may sit from a product of its marginals and still factorise
COARSENING_TOLERANCE = 1e-12  # how far a policy may vary inside an orbit or a round and still coarsen onto it


def independent(q: float) -> SignalPrior:
    """Every bidder draws a high signal on its own, with probability ``q``."""
    return lambda profiles: np.prod(np.where(profiles == 1, q, 1 - q), axis=1)


def equicorrelated(q: float, rho: float) -> SignalPrior:
    """Draw one signal for everyone with probability ``rho``, otherwise draw independently."""
    # Every pair then correlates at exactly rho, and every bidder still draws high with probability q.
    def prior(profiles: np.ndarray) -> np.ndarray:
        high = profiles.sum(axis=1)
        shared = np.where(high == profiles.shape[1], q, np.where(high == 0, 1 - q, 0.0))
        return rho * shared + (1.0 - rho) * independent(q)(profiles)

    return prior


def payoff_scale(n: int) -> float:
    """How far payoffs stretch against the two-bidder example, the common value summing ``n`` signals."""
    return n / 2


def opening(n: int) -> State:
    """The history before anyone has acted, with every bidder still in."""
    return State(0, tuple(range(n)), (ACTIVE,) * n)


def after(state: State, stayers: Tuple[int, ...]) -> State:
    """The history one round on, recording everyone who left this round as having left in it."""
    staying = set(stayers)
    exits = tuple(
        state.round if bidder in state.active and bidder not in staying else exited
        for bidder, exited in enumerate(state.exits)
    )
    return State(state.round + 1, stayers, exits)


def bucket_means(index: np.ndarray, size: int, values: np.ndarray) -> np.ndarray:
    """Mean of ``values`` in each of ``size`` buckets, leaving empty buckets at zero."""
    totals = np.bincount(index, weights=values, minlength=size)
    counts = np.bincount(index, minlength=size)
    return np.divide(totals, counts, out=np.zeros(size), where=counts > 0)


def orbit_of(state: State) -> Orbit:
    """What survives relabelling the bidders: the round and the multiset of exit rounds."""
    return (state.round, tuple(sorted(exited for exited in state.exits if exited != ACTIVE)))


def schedule(values: Sequence[float]) -> ReimbursementPolicy:
    """A policy paying a set amount per round, whatever the history within it."""
    paid = tuple(float(value) for value in values)
    return lambda state: paid[state.round]


def by_orbit(values: Mapping[Orbit, float]) -> ReimbursementPolicy:
    """A policy paying a set amount per orbit, which is the finest public rule the reduction survives."""
    paid = {orbit: float(value) for orbit, value in values.items()}
    return lambda state: paid[orbit_of(state)]


def flat(c: float, rounds: int, fraction: float = 1.0) -> ReimbursementPolicy:
    """Reimburse the same fraction of the participation cost in every round."""
    return schedule([fraction * c] * rounds)


def least_cost(c: float, rounds: int, fraction: float = 1.0) -> ReimbursementPolicy:
    """Reimburse in every round but the opening one, where separation has to be paid for."""
    return schedule([0.0] + [fraction * c] * (rounds - 1))


def backloaded(c: float, rounds: int) -> ReimbursementPolicy:
    """Reimburse the last decision round alone."""
    return schedule([0.0] * (rounds - 1) + [c])


def war(pattern: str) -> Callable[[float, int], ReimbursementPolicy]:
    """Nothing at the opening, then the full cost at each two-bidder war price marked ``c`` in ``pattern``."""
    def policy(c: float, rounds: int) -> ReimbursementPolicy:
        assert len(pattern) == rounds - 1, (pattern, rounds)
        return schedule([0.0] + [c * (mark == "c") for mark in pattern])

    return policy


PUBLISHED_REIMBURSEMENTS = {
    "zero": lambda c, rounds: flat(c, rounds, 0.0),
    "full": flat,
    "least_cost": least_cost,
    "backloaded": backloaded,
    # The other corners of the price-only class, which the back-loaded policy has to beat.
    **{f"war_{pattern}": war(pattern) for pattern in ("c00", "0c0", "cc0", "c0c", "0cc")},
}
REIMBURSEMENT_POLICIES = tuple(PUBLISHED_REIMBURSEMENTS)
VALIDATED_POLICIES = ("zero", "full", "least_cost", "backloaded")  # those the paper prints closed forms for


def states(n: int, rounds: int) -> List[State]:
    """Every public history leaving at least two bidders active, in round order, reachable or not."""
    frontier = [opening(n)]
    every = list(frontier)
    for _ in range(1, rounds):
        onward = [
            after(state, stayers)
            for state in frontier
            for size in range(2, len(state.active) + 1)
            for stayers in itertools.combinations(state.active, size)
        ]
        frontier = onward
        every += onward
    return every


@dataclass
class Auction:
    """A fully specified costly clock auction, defaulting to the paper's two-bidder example."""

    q: float = 0.5  # Pr(s_i = 1)
    c: float = 0.05  # participation cost per stay
    reimbursement_policy: Union[str, ReimbursementPolicy] = "least_cost"  # a schedule, or a published name
    n: int = 2
    rounds: int = 4  # decision rounds
    delta: float = 0.5  # tick size
    signal_prior: Union[SignalPrior, None] = None  # the joint law, defaulting to independent draws at q

    # Tables below build on first use: history tables outgrow memory past seven bidders.

    @cached_property
    def profiles(self) -> np.ndarray:
        """One row per signal profile, giving each bidder a high or a low signal."""
        return np.array(list(itertools.product([0, 1], repeat=self.n)))

    @cached_property
    def prior(self) -> np.ndarray:
        """Probability of each signal profile, drawn jointly so that bidders may correlate."""
        law = self.signal_prior if self.signal_prior is not None else independent(self.q)
        return np.asarray(law(self.profiles), dtype=float)

    @cached_property
    def independent_signals(self) -> bool:
        """Whether the prior is independent draws at ``q``, which the published closed forms assume."""
        product = independent(self.q)(self.profiles)
        return bool(np.max(np.abs(self.prior - product)) <= PRIOR_TOLERANCE)

    @cached_property
    def exchangeable_signals(self) -> bool:
        """Whether the prior survives relabelling the bidders, which the orbit reduction needs."""
        # With binary signals that is exactly the prior depending on the high-signal count alone.
        high = self.profiles.sum(1)
        ways = np.array([math.comb(self.n, count) for count in high], dtype=float)
        return bool(np.max(np.abs(self.prior - self.signal_counts[high] / ways)) <= PRIOR_TOLERANCE)

    @cached_property
    def signal_counts(self) -> np.ndarray:
        """Probability that exactly ``k`` bidders drew a high signal, for ``k`` up to ``n``."""
        return np.bincount(self.profiles.sum(1), weights=self.prior, minlength=self.n + 1)

    @cached_property
    def own_and_others(self) -> np.ndarray:
        """(2, n) probability that a bidder drew each signal while ``k`` of the others drew high."""
        others, own = self.profiles[:, 1:].sum(1), self.profiles[:, 0]
        return np.array(
            [
                np.bincount(others[own == signal], weights=self.prior[own == signal], minlength=self.n)
                for signal in (0, 1)
            ]
        )

    @cached_property
    def high_probability(self) -> float:
        """A bidder's own probability of a high signal, which exchangeability makes common to all."""
        return float(self.profiles[:, 0] @ self.prior)

    @property
    def num_profiles(self) -> int:
        """How many signal profiles there are, without building the table of them."""
        return 1 << self.n

    @cached_property
    def value(self) -> np.ndarray:
        """Common value under each signal profile, the sum of the signals as in the paper."""
        return self.profiles.sum(1).astype(float)

    @cached_property
    def states(self) -> List[State]:
        """Every public history of the auction, reachable or not."""
        return states(self.n, self.rounds)

    @cached_property
    def info_sets(self) -> List[Info]:
        """Every information set, one per active bidder, public history and own signal."""
        return [
            (bidder, state, signal) for state in self.states for bidder in state.active for signal in (0, 1)
        ]

    @cached_property
    def info_index(self) -> Dict[Info, int]:
        """Position of each information set in a flat behaviour profile."""
        return {info_set: position for position, info_set in enumerate(self.info_sets)}

    @cached_property
    def orbits(self) -> List[Orbit]:
        """Every orbit of the histories under relabelling the bidders, in round order."""
        return list(dict.fromkeys(orbit_of(state) for state in self.states))

    @cached_property
    def orbit_of_state(self) -> Dict[State, int]:
        """Each history's orbit, as a position in ``orbits``."""
        at = {orbit: position for position, orbit in enumerate(self.orbits)}
        return {state: at[orbit_of(state)] for state in self.states}

    @cached_property
    def orbit_index(self) -> np.ndarray:
        """Each information set's slot in a flat (orbit, signal) table."""
        return np.array([self.orbit_of_state[state] * 2 + signal for _, state, signal in self.info_sets])

    @cached_property
    def orbit_active(self) -> np.ndarray:
        """How many bidders are still in, per orbit."""
        return np.array([self.n - len(departed) for _, departed in self.orbits])

    @cached_property
    def onward_orbit(self) -> np.ndarray:
        """(|orbits|, n + 1) orbit reached when ``k`` of the active bidders leave, or -1 where it ends."""
        at = {orbit: position for position, orbit in enumerate(self.orbits)}
        table = np.full((len(self.orbits), self.n + 1), -1, dtype=np.int64)
        for position, (round_index, departed) in enumerate(self.orbits):
            active = self.n - len(departed)
            for leaving in range(active - 1):  # two or more must stay for the auction to carry on
                if round_index + 1 < self.rounds:
                    onward = (round_index + 1, tuple(sorted(departed + (round_index,) * leaving)))
                    table[position, leaving] = at[onward]
        return table

    @cached_property
    def gather(self) -> Dict[State, np.ndarray]:
        """(num_profiles, |A|) index of each active bidder's stay probability per signal profile."""
        return {
            state: np.stack(
                [
                    np.array([self.info_index[(bidder, state, 0)], self.info_index[(bidder, state, 1)]])[
                        self.profiles[:, bidder]
                    ]
                    for bidder in state.active
                ],
                axis=1,
            )
            for state in self.states
        }

    @cached_property
    def branch_tables(self) -> Tuple[Dict[State, np.ndarray], Dict[State, List[Tuple[int, State]]]]:
        """The active-set branch tables, whose size rules out eager construction."""
        return branches(self)

    @property
    def branch_payoffs(self) -> Dict[State, np.ndarray]:
        """Per state, the fixed part of a round's payoff under each set of stayers."""
        return self.branch_tables[0]

    @property
    def continuations(self) -> Dict[State, List[Tuple[int, State]]]:
        """Per state, the sets of stayers that carry the auction into a further round."""
        return self.branch_tables[1]

    def price(self, round_index: int) -> float:
        """Clock price in round ``round_index``."""
        return round_index * self.delta

    @cached_property
    def reimburses(self) -> ReimbursementPolicy:
        """Reimbursement paid per stay at a history, from a published name if a name was given."""
        if callable(self.reimbursement_policy):
            return self.reimbursement_policy
        return PUBLISHED_REIMBURSEMENTS[self.reimbursement_policy](self.c, self.rounds)

    @cached_property
    def paid_by_state(self) -> np.ndarray:
        """What the policy pays per stay at each public history, aligned with ``states``."""
        return np.array([float(self.reimburses(state)) for state in self.states])

    @cached_property
    def orbit_view(self) -> Union[np.ndarray, None]:
        """What the policy pays per orbit, or ``None`` where it separates histories inside one."""
        index = np.array([self.orbit_of_state[state] for state in self.states])
        means = bucket_means(index, len(self.orbits), self.paid_by_state)
        return means if np.max(np.abs(self.paid_by_state - means[index])) <= COARSENING_TOLERANCE else None

    def release_history_tables(self) -> None:
        """Drop the per-history tables, which rebuild on demand and cost half a gigabyte at six bidders."""
        for table in ("branch_tables", "gather"):
            self.__dict__.pop(table, None)

    def root(self) -> State:
        """The opening history: round ``0``, with every bidder still in."""
        return opening(self.n)


@dataclass
class Outcome:
    """Exact expectations under one behaviour profile."""

    payoff: np.ndarray  # (n,) bidder payoffs
    price: float
    reimbursement: float

    @property
    def revenue(self) -> float:
        """Seller net revenue: the price less every reimbursement paid."""
        return self.price - self.reimbursement


def branches(auction: Auction) -> Tuple[Dict[State, np.ndarray], Dict[State, List[Tuple[int, State]]]]:
    """Per state, the fixed part of a round's payoff and the states it can lead to."""
    branch_payoffs, continuations = {}, {}
    for state in auction.states:
        round_index, active = state.round, state.active
        paid = auction.reimburses(state)
        payoff_by_mask = np.zeros((1 << len(active), auction.num_profiles, auction.n + 2))
        continuations[state] = []
        for mask in range(1 << len(active)):
            stayers = tuple(bidder for position, bidder in enumerate(active) if mask >> position & 1)
            payoff = payoff_by_mask[mask]
            for bidder in stayers:
                payoff[:, bidder] -= auction.c - paid
                payoff[:, auction.n + 1] += paid
            if len(stayers) >= 2 and round_index + 1 < auction.rounds:
                continuations[state].append((mask, after(state, stayers)))
            else:  # sole stayer wins at p_t, all-exit splits at p_t, survivors split at p_T
                winners = stayers or active
                price = auction.price(auction.rounds if len(stayers) >= 2 else round_index)
                for bidder in winners:
                    payoff[:, bidder] += (auction.value - price) / len(winners)
                payoff[:, auction.n] += price
        branch_payoffs[state] = payoff_by_mask
    return branch_payoffs, continuations


ic_bound = lambda q: (1 - q) / 2  # paying nothing at the opening keeps separation iff c <= (1-q)/2


def region(q: float) -> float:
    """The cost bound below which the back-loaded policy is the revenue-maximising price-only rule."""
    return min(0.25, ic_bound(q))  # past 1/4 the unreimbursed war has no interior mix


def profile(auction: Auction, *strategies: Strategy) -> np.ndarray:
    """Flat behaviour profile from one strategy per bidder, or from one shared by all."""
    if len(strategies) == 1:
        strategies *= auction.n
    return np.array([float(strategies[bidder](state, signal)) for (bidder, state, signal) in auction.info_sets])


def closed_form(auction: Auction) -> Tuple[Strategy, float]:
    """The two-bidder symmetric sorting equilibrium under the auction's policy, and its seller net revenue."""
    # The back-loading theorem's backward recursion over the high-high war, in units of a.
    assert (auction.n, auction.rounds, auction.delta) == (2, 4, 0.5) and auction.independent_signals, auction
    paid = auction.paid_by_state  # one history per round at two bidders
    value, revenue, stays = 0.0, 2.0, [1.0, 0.0, 0.0, 0.0]  # W_4 and R_4: both reach 2a and split it
    for t in (3, 2, 1):
        price, net = auction.price(t), auction.c - paid[t]
        conceded = (2.0 - price) / 2  # what staying earns when the rival exits, less what exiting does
        if net <= value:
            stay, value = 1.0, value - net
        else:
            stay = (conceded - net) / (conceded - value)
            value = (1 - stay) * conceded
        revenue = (1 - stay**2) * price - 2 * stay * paid[t] + stay**2 * revenue
        stays[t] = stay
    strategy: Strategy = lambda state, signal: stays[state.round] if signal == 1 else 0.0
    return strategy, auction.q**2 * revenue - 2 * auction.q * paid[0]


def closed_form_revenue(rule: str, c: float, q: float) -> float:
    """Closed-form seller net revenue under one published policy at one cell."""
    return closed_form(Auction(q=q, c=c, reimbursement_policy=rule))[1]
