"""Claims beyond two bidders: the speed of ten of them, and the count-based reduction of ADR-0001."""

from __future__ import annotations

import time

import numpy as np

import analysis as AN
import solver as A
import game
import symmetric as S
from test_game import random_symmetric
from test_paper import main

TIGHT = 1e-12


def settings(counts):
    """Every auction in the equivalence sweep, over bidder counts, policies and signal probabilities."""
    return [
        A.Auction(n=n, q=q, c=0.025, reimbursement_policy=name)
        for n in counts
        for q in (0.3, 0.5)
        for name in A.REIMBURSEMENT_POLICIES
    ]


# --- construction at scale --------------------------------------------------


def test_ten_bidders_construct_without_the_active_set_tables():
    """An auction at ten bidders builds at once, and never materialises the active-set branch tables."""
    started = time.perf_counter()
    auction = A.Auction(n=10)
    sigma = random_symmetric(auction, 0)
    A.evaluate(auction, sigma)
    assert "branch_tables" not in auction.__dict__, sorted(auction.__dict__)
    assert "gather" not in auction.__dict__, sorted(auction.__dict__)
    assert time.perf_counter() - started < 1.0


def test_ten_bidders_evaluate_and_verify_well_within_a_second():
    """Exact evaluation and exact utility loss at ten bidders each finish well inside a second."""
    auction = A.Auction(n=10)
    sigma = random_symmetric(auction, 1)
    for measure in (A.evaluate, AN.utility_loss, AN.max_regret):
        started = time.perf_counter()
        measure(auction, sigma)
        assert time.perf_counter() - started < 1.0, measure.__name__


# --- the solver certifies every family --------------------------------------


def test_the_solver_clears_the_certification_bar_under_every_family():
    """Every family at every bidder count the study uses, so no row is the least certified one."""
    for n in (2, 4, 6, 10):
        for family, build in game.REIMBURSEMENT_FAMILIES.items():
            auction = A.Auction(n=n, q=0.5, c=0.025, reimbursement_policy=build(0.025, 4))
            loss = AN.utility_loss(auction, A.solve(auction))
            assert loss < AN.CERTIFIED, (n, family, loss)


# --- the two paths agree ----------------------------------------------------


def test_the_two_paths_agree_on_every_field_of_the_outcome():
    """Count-based and active-set evaluation agree to machine precision on random symmetric profiles."""
    for auction in settings((2, 3, 4, 5, 6)):
        for seed in (0, 1):
            sigma = random_symmetric(auction, seed)
            fast, slow = S.outcome(auction, sigma), enumerated(A.evaluate, auction, sigma)
            assert abs(fast.price - slow.price) < TIGHT, auction
            assert abs(fast.reimbursement - slow.reimbursement) < TIGHT, auction
            assert abs(fast.stay_rounds - slow.stay_rounds) < TIGHT, auction
            assert np.max(np.abs(fast.payoff - slow.payoff)) < TIGHT, auction


def test_utility_loss_and_regret_agree_with_the_active_set_path():
    """The tagged-deviator path reproduces the enumerated utility loss and one-shot regret."""
    for auction in settings((2, 3, 4)):
        sigma = random_symmetric(auction, 2)
        assert abs(S.utility_loss(auction, sigma) - enumerated(AN.utility_loss, auction, sigma)) < TIGHT, auction
        assert abs(S.max_regret(auction, sigma) - enumerated(AN.max_regret, auction, sigma)) < 1e-10, auction


def test_the_deviation_gain_agrees_at_every_information_set():
    """Every information set's gain and counterfactual reach match the enumerated ones."""
    for auction in settings((2, 3, 4)):
        sigma = random_symmetric(auction, 3)
        gain, reach = S.gains(auction, sigma)
        for position, (_, (round_index, active), signal) in enumerate(auction.info_sets):
            want = enumerated(A.counterfactual_gain, auction, sigma, position)
            assert abs(gain[round_index, len(active), signal] - want) < TIGHT, (auction, position)


# --- the fast path knows when it does not apply -----------------------------


def test_an_asymmetric_profile_still_routes_to_the_active_set_path():
    """One bidder always staying while the other always exits gives the payoffs it plainly should."""
    auction = A.Auction(n=2, q=0.5, c=0.025, reimbursement_policy="full")
    sigma = np.array(
        [1.0 if bidder == 0 else 0.0 for (bidder, _, _) in auction.info_sets]
    )
    assert not S.applies(auction, sigma)
    outcome = A.evaluate(auction, sigma)
    assert abs(outcome.price - 0.0) < TIGHT  # the sole stayer wins at the opening price
    assert abs(outcome.stay_rounds - 1.0) < TIGHT
    assert abs(outcome.reimbursement - auction.c) < TIGHT
    assert abs(outcome.payoff[0] - (auction.q - (auction.c - auction.c))) < TIGHT
    assert abs(outcome.payoff[1]) < TIGHT


# --- reaching the enumerated path on purpose --------------------------------


def enumerated(call, *arguments):
    """Run ``call`` with the reduction switched off, so that it takes the active-set path."""
    saved, S.TOLERANCE = S.TOLERANCE, -1.0
    try:
        return call(*arguments)
    finally:
        S.TOLERANCE = saved


if __name__ == "__main__":
    raise SystemExit(main(globals()))
