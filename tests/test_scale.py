"""Claims beyond two bidders: the speed of six of them, the orbit reduction, the solver, and the scale experiment."""

from __future__ import annotations

import time

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


# --- the theory the scale experiment scores against --------------------------

AUCTIONS = {}


def auction_at(n, policy):
    """``scale.auction_at``, built once per test session since six bidders take seconds."""
    if (n, policy) not in AUCTIONS:
        AUCTIONS[(n, policy)] = scale.auction_at(n, policy)
    return AUCTIONS[(n, policy)]


POLICIES = ("zero", "trigger", "preserve", "full", "frictionless", "preserve_high", "full_high")


def test_the_sorting_profile_reproduces_the_two_bidder_closed_forms():
    """At two bidders the general backward solution is the example's closed form under every validated policy."""
    for q in (0.6, 0.75):
        for c in (0.01, 0.1):
            for name in G.VALIDATED_POLICIES:
                auction = G.Auction(q=q, c=c, reimbursement_policy=name)
                closed = G.profile(auction, G.closed_form(auction)[0])
                assert np.max(np.abs(G.sorting_profile(auction) - closed)) < TIGHT, (name, q, c)


def test_theory_is_an_equilibrium_under_every_policy_at_every_count():
    """Each policy's sorting profile has no profitable deviation, sorting at the opening included."""
    for n in scale.COUNTS:
        for policy in POLICIES:
            auction = auction_at(n, policy)
            assert A.utility_loss(auction, G.sorting_profile(auction)) < 1e-9, (n, policy)


def test_the_costs_sit_where_the_theorems_apply():
    """The main cost is under fin:trigger's bound, and the high cost leaves Preserve a positive opening payment."""
    for n in scale.COUNTS:
        assert scale.KAPPA < min(0.25, G.opening_margin(n, scale.Q)), n
        high = auction_at(n, "preserve_high")
        assert abs(high.reimburses(high.root()) - scale.ABOVE_MARGIN) < TIGHT, n
        assert auction_at(n, "preserve").reimburses(auction_at(n, "preserve").root()) == 0.0, n


def revenue(n, policy, column=None):
    """Exact seller net revenue of the policy's sorting profile, or its two-survivor part."""
    auction = auction_at(n, policy)
    sigma = G.sorting_profile(auction)
    return scale.two_survivor_revenue(auction, sigma) if column else A.evaluate(auction, sigma).revenue


def test_trigger_gains_exactly_fin_revenue_gain_and_only_in_the_two_survivor_war():
    """Trigger less None is the theorem's polynomial, and all of it comes from two-survivor wars."""
    for n in scale.COUNTS:
        gain = revenue(n, "trigger") - revenue(n, "zero")
        war = revenue(n, "trigger", "war") - revenue(n, "zero", "war")
        assert abs(gain - scale.trigger_gain(n)) < TIGHT, n
        assert abs(war - gain) < TIGHT, n


def test_preserve_saves_exactly_the_opening_payment_full_makes():
    """Preserve less Full is n q min(c, w0) at both costs."""
    for n in scale.COUNTS:
        for suffix in ("", "_high"):
            saving = revenue(n, "preserve" + suffix) - revenue(n, "full" + suffix)
            assert abs(saving - scale.preserve_saving(n, scale.cost(n, "preserve" + suffix))) < TIGHT, (n, suffix)


# --- running the experiment ------------------------------------------------


def test_saved_seeds_are_not_run_again():
    """A later call runs only seeds missing from the saved rows, in chunks."""
    done = [{"n": 2, "policy": run, "seed": seed} for run in scale.RUNS for seed in (0, 1)]
    jobs = scale.pending(done, (2,), range(5))
    assert {(n, run) for n, run, _, _ in jobs} == {(2, run) for run in scale.RUNS}
    assert all(seeds == (2, 3, 4) for _, _, seeds, _ in jobs), jobs


def test_a_frictionless_run_is_scored_under_full_at_both_costs():
    """One short run gives a row each for the frictionless game and for Full, with the same play."""
    rows = scale.run_job((2, "frictionless", (0,), 2_000))
    assert [row["policy"] for row in rows] == ["frictionless", "full", "full_high"]
    assert len({row["open_high"] for row in rows}) == 1
    assert all(row["verdict"] in {v.value for v in scale.Verdict} for row in rows)


if __name__ == "__main__":
    raise SystemExit(main(globals()))
