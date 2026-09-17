"""Game-level claims: what the seller's books must close to, and the reimbursement schedule families."""

from __future__ import annotations

import numpy as np

import analysis as AN
import solver as A
import game
from test_paper import main

C, ROUNDS, TIGHT = 0.025, 4, 1e-12


def random_symmetric(auction, seed):
    """A random symmetric profile, drawn once per (round, active count, signal) bucket."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.15, 0.9, auction.rounds * (auction.n + 1) * 2)[auction.anonymous_index]


# --- the payoffs close the seller's books -----------------------------------


def test_bidder_payoffs_close_the_seller_accounting():
    """Total bidder payoff is the prize less net revenue less what participation cost, at every scale."""
    for n in (2, 3, 4, 6, 10):
        for q in (0.3, 0.5):
            for name in A.REIMBURSEMENT_POLICIES:
                auction = A.Auction(n=n, q=q, c=C, reimbursement_policy=name)
                outcome = A.evaluate(auction, random_symmetric(auction, 5))
                books = auction.q - outcome.revenue - auction.c * outcome.stay_rounds
                assert abs(float(np.sum(outcome.payoff)) - books) < 1e-13, (auction, books)


# --- reimbursement as a schedule --------------------------------------------


def test_the_published_policies_are_members_of_the_schedule_family():
    """Each published rule, rebuilt out of the schedule families, reproduces its published revenue."""
    built = {"zero": game.flat(C, ROUNDS, 0.0), "full": game.flat(C, ROUNDS), "targeted": game.targeted(C, ROUNDS)}
    for name, policy in built.items():
        auction = A.Auction(q=0.5, c=C, reimbursement_policy=policy)
        sigma = AN.profile(auction, AN.war_of_attrition(C) if name == "zero" else AN.SEPARATING)
        assert abs(A.evaluate(auction, sigma).revenue - AN.PI[name](C, 0.5)) < TIGHT, name


def test_the_four_families_are_four_policies_to_sweep():
    """Flat, targeted, front-loaded and back-loaded pay differently, so a sweep over them is a sweep."""
    revenues = {}
    for name, build in game.REIMBURSEMENT_FAMILIES.items():
        auction = A.Auction(n=4, c=C, reimbursement_policy=build(C, ROUNDS))
        revenues[name] = round(A.evaluate(auction, random_symmetric(auction, 7)).revenue, 12)
    assert len(set(revenues.values())) == len(game.REIMBURSEMENT_FAMILIES), revenues


if __name__ == "__main__":
    raise SystemExit(main(globals()))
