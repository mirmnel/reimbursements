"""Every claim of "A Two-Bidder Example", checked against an exact solver never told the closed forms."""

from __future__ import annotations

import numpy as np

import analysis as AN
import solver as A

Q, C, TIGHT = 0.5, 0.025, 1e-9


def target(reimbursement_policy, c=C, q=Q, **overrides):
    """The game under a reimbursement policy, and the equilibrium the paper targets in it."""
    auction = A.Auction(q=q, c=c, reimbursement_policy=reimbursement_policy, **overrides)
    war = AN.war_of_attrition(c) if reimbursement_policy == "zero" else AN.SEPARATING
    return auction, AN.profile(auction, war)


def schedule(auction, sigma, signal):
    """Stay probability of one signal type at each round, both bidders still active."""
    return [
        sigma[auction.info_index[(0, (round_index, (0, 1)), signal)]] for round_index in range(auction.rounds)
    ]


# --- the closed forms are internally consistent ----------------------------


def test_cbar_is_the_root_the_paper_reports():
    """cbar roots the revenue-difference cubic below one quarter, where the difference changes sign."""
    cbar = AN.CBAR
    assert abs(cbar - 0.108629 / 2) < 5e-7, cbar  # mean units halve every cost
    gap = lambda c: AN.PI["targeted"](c, Q) - AN.PI["zero"](c, Q)
    assert gap(cbar - 1e-4) > 0 > gap(cbar + 1e-4)  # the revenue difference changes sign exactly there


def test_targeted_beats_full_by_exactly_two_q_c():
    """The closed-form gap between the targeted and full policies holds at every cost and probability."""
    for q in (0.25, 0.5, 0.75):
        for c in (0.005, 0.025, 0.075):
            assert abs(AN.PI["targeted"](c, q) - AN.PI["full"](c, q) - 2 * q * c) < 1e-12


# --- the exact solver reproduces the paper, without being told the answer ---


def test_frictionless_benchmark_price_and_stay_rounds():
    """The frictionless benchmark reproduces the closed-form expected price and staying bidder-rounds."""
    for q in (0.25, 0.5, 0.75):
        outcome = A.evaluate(*target("full", q=q))
        assert abs(outcome.price - q**2) < TIGHT
        assert abs(outcome.stay_rounds - AN.L0(q)) < TIGHT


def test_seller_revenue_matches_every_closed_form():
    """Exact evaluation of each target reproduces ``Pi_0``, ``Pi_full`` and ``Pi_T``."""
    for q in (0.25, 0.5, 0.75):
        for c in (0.005, 0.025, 0.05, 0.075, 0.1):
            for name in A.REIMBURSEMENT_POLICIES:
                got = A.evaluate(*target(name, c=c, q=q)).revenue
                assert abs(got - AN.PI[name](c, q)) < TIGHT, (name, c, q, got)


def test_rescaling_to_mean_units_reproduces_the_published_revenues():
    """Halving the cost and the tick under mean units reproduces every published two-bidder revenue."""
    for q in (0.25, 0.5, 0.75):
        for c in (0.01, 0.05, 0.1, 0.15, 0.2):  # the published cost grid, stated in sum units
            for name in A.REIMBURSEMENT_POLICIES:
                auction, sigma = target(name, c=c / 2, q=q, delta=AN.PUBLISHED_TICK / 2)
                got = 2 * A.evaluate(auction, sigma).revenue
                assert abs(got - AN.PUBLISHED[name](c, q)) < TIGHT, (name, c, q, got)


# --- the targets are equilibria --------------------------------------------


def test_paper_targets_are_exact_equilibria():
    """Zero utility loss and zero deviation gain everywhere: sequential rationality."""
    for name in A.REIMBURSEMENT_POLICIES:
        auction, sigma = target(name)
        assert AN.utility_loss(auction, sigma) < 1e-12, name
        assert AN.max_regret(auction, sigma) < 1e-12, name


def test_endgame_is_knife_edge_and_the_opening_is_not():
    """The mixed rounds are exact indifference points, but the opening is strict."""
    auction, sigma = target("zero")
    at = auction.info_index
    for round_index in (1, 3):  # sigma_1 and sigma_3 are interior
        assert abs(A.counterfactual_gain(auction, sigma, at[(0, (round_index, (0, 1)), 1)])) < 1e-12, round_index
    assert A.counterfactual_gain(auction, sigma, at[(0, (0, (0, 1)), 1)]) > 1e-6  # H strictly stays at p = 0
    assert A.counterfactual_gain(auction, sigma, at[(0, (0, (0, 1)), 0)]) < -1e-6  # L strictly exits at p = 0


def test_opening_incentive_constraint_binds_at_one_minus_q_over_four():
    """Under the targeted policy the opening gain is the incentive bound less the cost."""
    for q in (0.25, 0.5, 0.75):
        for c in (AN.ic_bound(q) - 0.05, AN.ic_bound(q) + 0.05):
            auction, sigma = target("targeted", c=c, q=q)
            position = auction.info_index[(0, (0, (0, 1)), 1)]
            gain = A.counterfactual_gain(auction, sigma, position) / A.counterfactual_reach(auction, sigma)[position]
            assert abs(gain - (AN.ic_bound(q) - c)) < TIGHT
            assert (gain > 0) == (c < AN.ic_bound(q))


# --- the revenue ranking ----------------------------------------------------


def revenues(c, q):
    """Seller net revenue under each reimbursement policy at ``(c, q)``, with the target imposed."""
    return {name: A.evaluate(*target(name, c=c, q=q)).revenue for name in A.REIMBURSEMENT_POLICIES}


def test_ranking_against_no_reimbursement_fails_above_cbar():
    """Above ``cbar`` the targeted policy loses to doing nothing, but never to full."""
    assert AN.CBAR < AN.ic_bound(0.5)  # at q = 1/2 the binding bound is cbar
    for c in (0.065, 0.075, 0.1):
        revenue = revenues(c, 0.5)
        assert revenue["zero"] > revenue["targeted"] > revenue["full"], (c, revenue)


# --- solving ----------------------------------------------------------------


def test_the_solver_recovers_the_war_of_attrition_and_its_revenue():
    """The exact-gradient solver finds the war-of-attrition stay schedule, and the low type exiting."""
    auction = A.Auction(q=Q, c=C, reimbursement_policy="zero")
    sigma = A.solve(auction)
    assert np.allclose(schedule(auction, sigma, 1), [1, AN.sigma1(C), 1, AN.sigma3(C)], atol=1e-6)
    assert np.allclose(schedule(auction, sigma, 0), 0.0, atol=1e-6)
    assert AN.utility_loss(auction, sigma) < 1e-9 and AN.max_regret(auction, sigma) < 1e-9
    assert abs(A.evaluate(auction, sigma).revenue - AN.PI["zero"](C, Q)) < 1e-9


def main(namespace=None) -> int:
    """Run every ``test_`` function in ``namespace`` and report how many passed."""
    tests = [(name, fn) for name, fn in sorted((namespace or globals()).items()) if name.startswith("test_")]
    failed = [name for name, test_fn in tests if not run(name, test_fn)]
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


def run(name, test_fn) -> bool:
    """Run one test, print PASS or FAIL, and report whether it passed."""
    try:
        test_fn()
    except AssertionError as exc:
        print(f"  FAIL  {name}: {exc}")
        return False
    print(f"  PASS  {name}")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
