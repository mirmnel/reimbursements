"""Claims beyond two bidders: the speed of six of them, and the scale experiment."""

from __future__ import annotations

import time

import numpy as np

from auction import game as G
from auction import exact as A
from experiments import scale
from test_paper import main

TIGHT = 1e-12


# --- construction at scale --------------------------------------------------


def test_six_bidders_evaluate_and_verify_well_within_a_second():
    """Exact evaluation and exact utility loss at six bidders each finish well inside a second."""
    # Six is the count the scale experiment reaches. Keying on histories rather than active sets
    # costs the reduction nothing here, but the flat profile itself outgrows memory by ten.
    auction = G.Auction(reimbursement_policy=G.flat(0.05, 4, 0.0), n=6, q=0.5, c=0.05, rounds=4, delta=0.5)
    sigma = np.random.default_rng(1).uniform(0.15, 0.9, len(auction.orbits) * 2)[auction.orbit_index]
    for measure in (A.evaluate, A.utility_loss):
        started = time.perf_counter()
        measure(auction, sigma)
        assert time.perf_counter() - started < 1.0, measure.__name__


# --- the theory the scale experiment scores against --------------------------

AUCTIONS = {}


def auction_at(n, policy, c):
    """``scale.auction_at``, built once per test session since six bidders take seconds."""
    if (n, policy, c) not in AUCTIONS:
        AUCTIONS[(n, policy, c)] = scale.auction_at(n, policy, c)
    return AUCTIONS[(n, policy, c)]


SETTINGS = [(policy, c) for run, cost in scale.RUNS for policy, c in scale.written(run, cost)]


def test_theory_is_an_equilibrium_under_every_policy_at_every_count_and_cost():
    """Each policy's sorting profile has no profitable deviation, sorting at the opening included."""
    for n in scale.COUNTS:
        for policy, c in SETTINGS:
            auction = auction_at(n, policy, c)
            assert A.utility_loss(auction, G.sorting_profile(auction)) < 1e-9, (n, policy, c)


def test_the_costs_sit_where_the_theorems_apply():
    """Every cost is under fin:trigger's bound at every count, so Preserve pays nothing at the opening."""
    for n in scale.COUNTS:
        for c in scale.COSTS:
            assert c < min(0.25, G.opening_margin(n, scale.Q)), (n, c)
            assert auction_at(n, "preserve", c).paid_by_state[0] == 0.0, (n, c)


def revenue(n, policy, c, column=None):
    """Exact seller net revenue of the policy's sorting profile, or its two-survivor part."""
    auction = auction_at(n, policy, c)
    sigma = G.sorting_profile(auction)
    return scale.two_survivor_revenue(auction, sigma) if column else A.evaluate(auction, sigma).revenue


def test_trigger_gains_exactly_fin_revenue_gain_and_only_in_the_two_survivor_war():
    """Trigger less None is the theorem's polynomial, and all of it comes from two-survivor wars."""
    for n in scale.COUNTS:
        for c in scale.COSTS:
            gain = revenue(n, "trigger", c) - revenue(n, "zero", c)
            war = revenue(n, "trigger", c, "war") - revenue(n, "zero", c, "war")
            assert abs(gain - scale.trigger_gain(n, c)) < TIGHT, (n, c)
            assert abs(war - gain) < TIGHT, (n, c)


def test_preserve_saves_exactly_the_opening_payment_full_makes():
    """Preserve less Full is n q min(c, w0) at every cost."""
    for n in scale.COUNTS:
        for c in scale.COSTS:
            saving = revenue(n, "preserve", c) - revenue(n, "full", c)
            assert abs(saving - scale.preserve_saving(n, c)) < TIGHT, (n, c)


# --- running the experiment ------------------------------------------------


def test_saved_seeds_are_not_run_again():
    """A later call runs only the runs and seeds with a row missing from the saved ones."""
    done = [{"n": 2, "policy": policy, "c": c, "seed": seed}
            for run, cost in scale.RUNS for policy, c in scale.written(run, cost) for seed in (0, 1)]
    jobs = scale.pending(done, (2,), range(3))
    assert {(n, run, c) for n, run, c, _, _ in jobs} == {(2, run, c) for run, c in scale.RUNS}
    assert all(seed == 2 for _, _, _, seed, _ in jobs), jobs


def test_a_frictionless_run_is_scored_under_full_at_every_cost():
    """One short run gives a row for the frictionless game and one for Full at each cost, with the same play."""
    rows = scale.run_job((2, "frictionless", 0.0, 0, 2_000))
    assert [(row["policy"], row["c"]) for row in rows] == [("frictionless", 0.0)] + [("full", c) for c in scale.COSTS]
    assert len({row["open_high"] for row in rows}) == 1
    assert all(row["verdict"] in {v.value for v in scale.Verdict} for row in rows)


if __name__ == "__main__":
    raise SystemExit(main(globals()))
