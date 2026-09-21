"""The cheapest reimbursement rule that leaves a target profile sequentially rational, as a linear program."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

import symmetric
from analysis import CERTIFIED, RESOLVED, one_shot_regret
from game import Auction, ReimbursementPolicy, by_orbit


BELIEF_TREMBLE = 0.01  # rate the off-path beliefs are trembled at, so every orbit carries a constraint

# A prescribed probability this near pure leaves the constraint on deviating the other way vacuous,
# since the allowance is divided by it. Emitting those rows anyway spread the bounds over forty
# orders of magnitude at six bidders and the program came back infeasible.
MIXED_FLOOR = 1e-6
# Paying in full satisfies every constraint with equality on the ones that bind, so the feasible set
# touches it at a vertex and floating point can put it just outside. Far below analysis.CERTIFIED.
FEASIBILITY_MARGIN = 1e-9


@dataclass
class Implementation:
    """The rule the linear program chose, and what it cost against what full reimbursement costs."""

    refunds: np.ndarray  # what the rule pays per stay at each orbit
    bill: float  # expected reimbursement under the target, which is the objective
    full_bill: float  # the same under full reimbursement, which is always feasible here
    slack: float  # how far from optimal a prescribed action was allowed to sit
    constraints: int  # sequential-rationality constraints the program carried
    status: str


def stay_mass(auction: Auction, sigma: np.ndarray) -> np.ndarray:
    """Expected number of bidder-stays at each orbit, which is what a reimbursement there is paid on."""
    # Arrival rather than counterfactual reach, since a bidder is only paid where it stayed.
    stay = symmetric.anonymise(auction, sigma)
    return auction.n * (symmetric.bucket_arrival(auction, sigma) * stay).sum(axis=1)


def gain_coefficients(
    auction: Auction, sigma: np.ndarray, beliefs: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The deviation gain at each (orbit, signal) as an affine function of the per-orbit reimbursement."""
    # The gain is affine in what the rule pays, so a difference against the zero rule is the exact
    # coefficient rather than an approximation to it. Reach does not move with the rule at all.
    played = sigma if beliefs is None else beliefs
    basis = np.eye(len(auction.orbits))
    base, reach = symmetric.gains(auction, played, refunds=np.zeros(len(auction.orbits)))
    slope = np.array([symmetric.gains(auction, played, refunds=row)[0] - base for row in basis])
    return base, slope, reach


def trembled(sigma: np.ndarray, tremble: float) -> np.ndarray:
    """``sigma`` mixed toward uniform, which is the belief every unreached orbit is judged under."""
    return (1.0 - tremble) * sigma + tremble * 0.5


def sequential_rationality(
    auction: Auction, sigma: np.ndarray, floor: float, tremble: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row per action the target prescribes, bounding what deviating from it may be worth."""
    stay = np.clip(symmetric.anonymise(auction, sigma), 0.0, 1.0)
    base, slope, reach = gain_coefficients(auction, sigma)
    off_base, off_slope, off_reach = gain_coefficients(auction, sigma, trembled(sigma, tremble))
    full = np.full(len(auction.orbits), auction.c)
    rows, bounds, allowed = [], [], np.zeros_like(stay)
    for orbit in range(len(auction.orbits)):
        for signal in (0, 1):
            on_path = reach[orbit, signal] > RESOLVED
            weight = reach[orbit, signal] if on_path else off_reach[orbit, signal]
            if weight <= RESOLVED:  # no sequence of trembles arrives, so there is nothing to prescribe
                continue
            constant = (base if on_path else off_base)[orbit, signal]
            coefficient = (slope if on_path else off_slope)[:, orbit, signal]
            # A set at a time, since one rarely reached set the target already plays badly would
            # otherwise set the allowance everywhere and let the rule underpay where it is reached.
            # Normalised by the reach, which a gain is always read at, so that a set reached once in a
            # trillion does not enter the program with coefficients a trillion times smaller.
            normalised, slopes = constant / weight, coefficient / weight
            at_full = one_shot_regret(normalised + slopes @ full, stay[orbit, signal])
            allowed[orbit, signal] = max(floor, at_full) + FEASIBILITY_MARGIN
            # analysis.one_shot_regret at this set stays inside the allowance exactly on this pair.
            for sign, mixed in ((1.0, 1.0 - stay[orbit, signal]), (-1.0, stay[orbit, signal])):
                if mixed > MIXED_FLOOR:
                    rows.append(sign * slopes)
                    bounds.append(allowed[orbit, signal] / mixed - sign * normalised)
    return np.array(rows), np.array(bounds), allowed


def minimum_cost(
    auction: Auction, sigma: np.ndarray, floor: float = CERTIFIED, tremble: float = BELIEF_TREMBLE
) -> Implementation:
    """Solve for the cheapest per-orbit reimbursement leaving every action the target prescribes optimal."""
    symmetric.requires_exchangeability(auction)
    objective = stay_mass(auction, sigma)
    rows, bounds, allowed = sequential_rationality(auction, sigma, floor, tremble)
    solution = linprog(objective, A_ub=rows, b_ub=bounds, bounds=(0.0, auction.c), method="highs")
    chosen = solution.x if solution.success else np.full(len(auction.orbits), auction.c)
    refunds = np.clip(chosen, 0.0, auction.c)  # the bounds the program was given, past its own rounding
    return Implementation(
        refunds=refunds,
        bill=float(objective @ refunds),
        full_bill=float(objective.sum() * auction.c),
        slack=float(allowed.max()),
        constraints=len(bounds),
        status="optimal" if solution.success else solution.message,
    )


def as_policy(auction: Auction, refunds: np.ndarray) -> ReimbursementPolicy:
    """Read a per-orbit reimbursement vector back as a policy an ``Auction`` can be built on."""
    return by_orbit(dict(zip(auction.orbits, (float(value) for value in refunds))))


def schedule_of(auction: Auction, refunds: np.ndarray) -> Dict[str, float]:
    """What the rule pays per stay in each round, averaged over the orbits of that round."""
    rounds = np.array([round_index for round_index, _ in auction.orbits])
    return {
        f"r_t{round_index}": float(refunds[rounds == round_index].mean())
        for round_index in range(auction.rounds)
    }
