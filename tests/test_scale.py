"""Claims beyond two bidders: the speed of six of them, the orbit reduction, the solver, and the scale experiment."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from auction import game as G
from auction import exact as A
from experiments import scale
from auction import symmetric as S
from test_game import enumerated, random_symmetric
from test_paper import main

TIGHT = 1e-12


def settings(counts):
    """Every auction in the equivalence sweep, over bidder counts, policies and signal probabilities."""
    return [
        G.Auction(n=n, q=q, c=0.05, reimbursement_policy=name)
        for n in counts
        for q in (0.3, 0.5)
        for name in G.VALIDATED_POLICIES
    ]


# --- construction at scale --------------------------------------------------


def test_six_bidders_evaluate_and_verify_well_within_a_second():
    """Exact evaluation and exact utility loss at six bidders each finish well inside a second."""
    # Six is the count the scale experiment reaches. Keying on histories rather than active sets
    # costs the reduction nothing here, but the flat profile itself outgrows memory by ten.
    auction = G.Auction(n=6)
    sigma = random_symmetric(auction, 1)
    for measure in (A.evaluate, A.utility_loss):
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
            assert np.max(np.abs(fast.payoff - slow.payoff)) < TIGHT, auction


def test_utility_loss_agrees_with_the_per_history_path():
    """The tagged-deviator path reproduces the enumerated utility loss."""
    for auction in settings((2, 3, 4)):
        sigma = random_symmetric(auction, 2)
        assert abs(S.utility_loss(auction, sigma) - enumerated(A.utility_loss, auction, sigma)) < TIGHT, auction


def test_the_deviation_gain_agrees_at_every_information_set():
    """Every information set's gain and counterfactual reach match the enumerated ones."""
    for auction in settings((2, 3, 4)):
        sigma = random_symmetric(auction, 3)
        gain, reach = S.gains(auction, sigma)
        for position, (bidder, state, signal) in enumerate(auction.info_sets):
            # Per history: own earlier sets stay, so only opponent reach weights the gain.
            base = sigma.copy()
            for (owner, earlier, own), at in auction.info_index.items():
                if owner == bidder and own == signal and earlier.round < state.round:
                    base[at] = 1.0
            staying, exiting = base.copy(), base.copy()
            staying[position], exiting[position] = 1.0, 0.0
            want = (enumerated(A.evaluate, auction, staying).payoff[bidder]
                    - enumerated(A.evaluate, auction, exiting).payoff[bidder])
            assert abs(gain[auction.orbit_of_state[state], signal] - want) < TIGHT, (auction, position)


# --- the fast path knows when it does not apply -----------------------------


def test_an_asymmetric_profile_still_routes_to_the_per_history_path():
    """One bidder always staying while the other always exits gives the payoffs it plainly should."""
    auction = G.Auction(n=2, q=0.5, c=0.05, reimbursement_policy="full")
    sigma = np.array(
        [1.0 if bidder == 0 else 0.0 for (bidder, _, _) in auction.info_sets]
    )
    assert not S.applies(auction, sigma)
    outcome = A.evaluate(auction, sigma)
    assert abs(outcome.price - 0.0) < TIGHT  # the sole stayer wins at the opening price
    assert abs(outcome.reimbursement - auction.c) < TIGHT
    assert abs(outcome.payoff[0] - (2 * auction.q - (auction.c - auction.c))) < TIGHT
    assert abs(outcome.payoff[1]) < TIGHT


# --- the scale experiment, without running it -------------------------------


def test_the_experiment_runs_the_three_bidder_counts_the_paper_names():
    """Three, four and six, with correlated signals and a clock finer than the example's."""
    for n in scale.COUNTS:
        auction = scale.auction_at(n, "least_cost")
        # A finer clock over the same range of common values, and a cost stretched with them.
        assert abs(auction.rounds * auction.delta - n) < TIGHT, n
        assert abs(auction.c - scale.C * n / 2) < TIGHT, n
        assert not auction.independent_signals and auction.exchangeable_signals, n


def test_the_reduction_covers_the_restricted_arm_at_every_count():
    """Every symmetric profile is read on orbits, so the restricted arm is exact even at six bidders."""
    for n in scale.COUNTS:
        auction = scale.auction_at(n, "least_cost")
        assert S.applies(auction, random_symmetric(auction, 7)), n


def test_the_unrestricted_arm_runs_only_where_its_tree_can_be_walked():
    """Six bidders are past the enumeration bar, so the unrestricted arm stops at four."""
    walked = {n: len(scale.auction_at(n, "least_cost").info_sets) <= scale.EXACT_INFO_SETS for n in scale.COUNTS}
    assert walked == {3: True, 4: True, 6: False}, walked


def test_one_bidder_off_the_target_counts_as_fully_as_all_of_them():
    """The distance is the worst bidder's, so the others matching cannot average one deviant away."""
    auction = scale.auction_at(3, "full")
    target = random_symmetric(auction, 4)
    reference = scale.Reference(3, None, target, None, auction)
    one = np.where([b == 0 for b, _, _ in auction.info_sets], 1.0 - target, target)
    distance, solved_at = {}, scale.solved_at
    try:
        for name, profile in (("one", one), ("all", 1.0 - target)):
            scale.solved_at = lambda *_, profile=profile: SimpleNamespace(profile=profile)
            distance[name] = scale.recovery_row(reference, 0, True)["arrival_weighted_distance"]
    finally:
        scale.solved_at = solved_at
    assert abs(distance["one"] - distance["all"]) < TIGHT, distance


if __name__ == "__main__":
    raise SystemExit(main(globals()))
