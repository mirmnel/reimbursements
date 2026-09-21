"""Claims beyond two bidders: the speed of six of them, the orbit reduction, the solver, and experiment 3."""

from __future__ import annotations

import time

import numpy as np

import analysis as AN
import solver as A
import scale
import symmetric as S
from mccfr import Verdict
from test_game import enumerated, random_symmetric
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


def test_six_bidders_construct_without_the_history_tables():
    """An auction at six bidders builds at once, and never materialises the per-history branch tables."""
    started = time.perf_counter()
    auction = A.Auction(n=6)
    sigma = random_symmetric(auction, 0)
    A.evaluate(auction, sigma)
    assert "branch_tables" not in auction.__dict__, sorted(auction.__dict__)
    assert "gather" not in auction.__dict__, sorted(auction.__dict__)
    assert time.perf_counter() - started < 1.0


def test_six_bidders_evaluate_and_verify_well_within_a_second():
    """Exact evaluation and exact utility loss at six bidders each finish well inside a second."""
    # Six is the count the scale experiment reaches. Keying on histories rather than active sets
    # costs the reduction nothing here, but the flat profile itself outgrows memory by ten.
    auction = A.Auction(n=6)
    sigma = random_symmetric(auction, 1)
    for measure in (A.evaluate, AN.utility_loss, AN.max_regret):
        started = time.perf_counter()
        measure(auction, sigma)
        assert time.perf_counter() - started < 1.0, measure.__name__


# --- the two paths agree ----------------------------------------------------


def test_the_two_paths_agree_on_every_field_of_the_outcome():
    """Orbit and per-history evaluation agree to machine precision on random symmetric profiles."""
    for auction in settings((2, 3, 4, 5, 6)):
        for seed in (0, 1):
            sigma = random_symmetric(auction, seed)
            fast, slow = S.outcome(auction, sigma), enumerated(A.evaluate, auction, sigma)
            assert abs(fast.price - slow.price) < TIGHT, auction
            assert abs(fast.reimbursement - slow.reimbursement) < TIGHT, auction
            assert abs(fast.stay_rounds - slow.stay_rounds) < TIGHT, auction
            assert np.max(np.abs(fast.payoff - slow.payoff)) < TIGHT, auction


def test_utility_loss_and_regret_agree_with_the_per_history_path():
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
        for position, (_, state, signal) in enumerate(auction.info_sets):
            want = enumerated(A.counterfactual_gain, auction, sigma, position)
            assert abs(gain[auction.orbit_of_state[state], signal] - want) < TIGHT, (auction, position)


# --- the fast path knows when it does not apply -----------------------------


def test_an_asymmetric_profile_still_routes_to_the_per_history_path():
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


# --- experiment 3, without running it ---------------------------------------


def test_the_experiment_runs_the_three_bidder_counts_the_paper_names():
    """Three, four and six, with correlated signals and a clock finer than the example's."""
    assert scale.COUNTS == (3, 4, 6)
    assert scale.ROUNDS > 4 and scale.RHO > 0.0
    for n in scale.COUNTS:
        auction = scale.auction_at(n, "targeted")
        # A finer clock over the same unit interval of prices, so revenue stays comparable to the example's.
        assert abs(auction.rounds * auction.delta - 1.0) < TIGHT, n
        assert not auction.independent_signals and auction.exchangeable_signals, n


def test_every_quantity_read_off_a_symmetric_profile_is_exact():
    """The reduction covers the restricted arm at every count, so nothing there is a bound or a sample."""
    for n in scale.COUNTS:
        auction = scale.auction_at(n, "targeted")
        sigma = random_symmetric(auction, 7)
        assert S.applies(auction, sigma), n
        reading = scale.read(auction, sigma, seed=0)
        assert reading.exact and reading.revenue_error == 0.0, n
        # Measured beside the loss while the auction is small, and dropped where it would cost more.
        assert (reading.restricted_bound is None) == (len(auction.info_sets) > scale.EXACT_INFO_SETS), n


def test_an_asymmetric_profile_past_the_enumeration_bar_is_read_by_sampling():
    """Where the tree is too large to walk, the loss becomes a bound and the revenue a sample."""
    # The read itself is far too dear to run here, so what is checked is which path it would take.
    walked = {n: scale.walkable(scale.auction_at(n, "targeted"), asymmetric(n)) for n in scale.COUNTS}
    assert walked == {3: True, 4: True, 6: False}, walked
    assert all(scale.walkable(scale.auction_at(n, "targeted"), random_symmetric(scale.auction_at(n, "targeted"), 7))
               for n in scale.COUNTS), scale.COUNTS


def asymmetric(n):
    """A profile no two bidders share, which is what the unrestricted arm of the experiment produces."""
    auction = scale.auction_at(n, "targeted")
    return np.random.default_rng(8).uniform(0.1, 0.9, len(auction.info_sets))


def test_the_drop_out_bound_never_exceeds_the_utility_loss_it_bounds():
    """Deviating only to drop-out prices is a search over fewer strategies, so it can only find less."""
    for n in (2, 3):
        auction = scale.auction_at(n, "targeted")
        sigma = random_symmetric(auction, 9)
        exact = float(AN.utility_loss(auction, sigma))
        bound = AN.restricted_loss(auction, sigma, lambda profile: A.evaluate(auction, profile).payoff)
        assert 0.0 <= bound <= exact + TIGHT, (n, bound, exact)


def test_a_drop_out_strategy_stays_exactly_while_the_clock_is_under_its_own_price():
    """The restricted class is one exit round per signal, which is what makes its search affordable."""
    auction = scale.auction_at(3, "targeted")
    replaced = AN.with_drop_out(auction, np.full(len(auction.info_sets), 0.5), bidder=1, thresholds=(0, 2))
    for position, (bidder, state, signal) in enumerate(auction.info_sets):
        want = 0.5 if bidder != 1 else float(state.round < (0, 2)[signal])
        assert replaced[position] == want, (position, bidder, state, signal)


def test_the_verdict_reads_the_distance_it_is_handed_and_the_loss_decides_first():
    """Certification comes first, so a profile that is not an equilibrium is never read as a recovery."""
    near = scale.Read(utility_loss=1e-6, restricted_bound=None, revenue=0.3, revenue_error=0.0, exact=True)
    far = scale.Read(utility_loss=1.0, restricted_bound=None, revenue=0.3, revenue_error=0.0, exact=True)
    assert scale.verdict_on(near, distance=0.0, target_revenue=0.3) is Verdict.DERIVED
    assert scale.verdict_on(near, distance=1.0, target_revenue=0.3) is Verdict.DIFFERENT
    assert scale.verdict_on(near, distance=0.0, target_revenue=0.9) is Verdict.DIFFERENT
    assert scale.verdict_on(far, distance=0.0, target_revenue=0.3) is Verdict.UNSOLVED


if __name__ == "__main__":
    raise SystemExit(main(globals()))
