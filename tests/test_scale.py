"""The scale experiment at two, three and four bidders."""

from __future__ import annotations

from auction import game as G
from auction import exact as A
from experiments import scale
from test_paper import main

TIGHT = 1e-12


# --- the theory the scale experiment scores against --------------------------

AUCTIONS = {}


def auction_at(n, policy, c):
    """``scale.auction_at``, built once per test session."""
    if (n, policy, c) not in AUCTIONS:
        AUCTIONS[(n, policy, c)] = scale.auction_at(n, policy, c)
    return AUCTIONS[(n, policy, c)]


def test_theory_is_an_equilibrium_under_every_policy_at_every_count_and_cost():
    """Each policy's sorting profile has no profitable deviation, sorting at the opening included."""
    for n in scale.COUNTS:
        for policy, c in [setting for run, cost in scale.runs(n) for setting in scale.written(run, cost)]:
            auction = auction_at(n, policy, c)
            assert A.utility_loss(auction, G.sorting_profile(auction)) < 1e-9, (n, policy, c)


def test_the_costs_sit_where_the_theorems_apply():
    """Every cost is under fin:trigger's bound at every count, so Preserve pays nothing at the opening."""
    for n in scale.COUNTS:
        for c in scale.COSTS:
            assert c < min(0.25, G.opening_margin(n, scale.Q)), (n, c)
            assert auction_at(n, "preserve", c).paid_by_state[0] == 0.0, (n, c)


def revenue(n, policy, c):
    """Exact seller net revenue of the policy's sorting profile."""
    auction = auction_at(n, policy, c)
    return A.evaluate(auction, G.sorting_profile(auction)).revenue


def test_trigger_gains_exactly_fin_revenue_gain():
    """At two bidders, trigger less None is the theorem's polynomial."""
    for c in scale.COSTS:
        gain = revenue(2, "trigger", c) - revenue(2, "zero", c)
        assert abs(gain - scale.trigger_gain(2, c)) < TIGHT, c


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
            for run, cost in scale.runs(2) for policy, c in scale.written(run, cost) for seed in (0, 1)]
    jobs = scale.pending(done, (2,), range(3))
    assert {(n, run, c) for n, run, c, _, _ in jobs} == {(2, run, c) for run, c in scale.runs(2)}
    assert all(seed == 2 for _, _, _, seed, _ in jobs), jobs


def test_a_frictionless_run_is_scored_under_full_at_every_cost():
    """One short run gives a row for the frictionless game and one for Full at each cost."""
    rows = scale.run_job((2, "frictionless", 0.0, 0, 2_000))
    assert [(row["policy"], row["c"]) for row in rows] == [("frictionless", 0.0)] + [("full", c) for c in scale.COSTS]
    assert all(row["verdict"] in {v.value for v in scale.Verdict} for row in rows)


if __name__ == "__main__":
    raise SystemExit(main(globals()))
