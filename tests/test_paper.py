"""Every claim of "A Two-Bidder Example", checked against an exact solver never told the closed forms."""

from __future__ import annotations


from auction import game as G
from auction import exact as A

Q, C, TIGHT = 0.5, 0.05, 1e-9


def target(reimbursement_policy, c=C, q=Q, **overrides):
    """The game under a reimbursement policy, and the equilibrium the paper targets in it."""
    auction = G.Auction(q=q, c=c, reimbursement_policy=reimbursement_policy, **overrides)
    return auction, G.profile(auction, G.closed_form_strategy(reimbursement_policy, c))


# --- the closed forms are internally consistent ----------------------------


def test_cbar_is_the_root_the_paper_reports():
    """cbar roots the revenue-difference cubic below one quarter, where the difference changes sign."""
    cbar = G.CBAR
    assert abs(cbar - 0.108629) < 5e-7, cbar
    gap = lambda c: G.PI["targeted"](c, Q) - G.PI["zero"](c, Q)
    assert gap(cbar - 1e-4) > 0 > gap(cbar + 1e-4)  # the revenue difference changes sign exactly there


def test_the_region_closes_differently_at_each_of_its_two_bounds():
    """At the cost root the revenues tie, and at the incentive bound they do not, so one end is open."""
    q = 0.9  # past 0.783, where the incentive bound is the lesser of the two
    bound = G.ic_bound(q)
    margin = lambda c: G.PI["targeted"](c, q) - max(G.PI["zero"](c, q), G.PI["full"](c, q))
    assert margin(bound) > 0, bound  # separation holds at the bound itself
    assert margin(G.CBAR) < 1e-12, G.CBAR
    # Past the incentive bound the formulas still favour the targeted rule, at a profile nobody plays.
    past = (bound + G.CBAR) / 2
    assert margin(past) > 0, past


def test_targeted_beats_full_by_exactly_two_q_c():
    """The closed-form gap between the targeted and full policies holds at every cost and probability."""
    for q in (0.25, 0.5, 0.75):
        for c in (0.01, 0.05, 0.15):
            assert abs(G.PI["targeted"](c, q) - G.PI["full"](c, q) - 2 * q * c) < 1e-12


# --- the exact solver reproduces the paper, without being told the answer ---


def test_frictionless_benchmark_price():
    """The frictionless benchmark reproduces the closed-form expected price."""
    for q in (0.25, 0.5, 0.75):
        outcome = A.evaluate(*target("full", q=q))
        assert abs(outcome.price - 2 * q**2) < TIGHT


def test_seller_revenue_matches_every_closed_form():
    """Exact evaluation of each target reproduces ``Pi_0``, ``Pi_full`` and ``Pi_T``."""
    for q in (0.25, 0.5, 0.75):
        for c in (0.01, 0.05, 0.1, 0.15, 0.2):  # the published cost grid
            for name in G.REIMBURSEMENT_POLICIES:
                got = A.evaluate(*target(name, c=c, q=q)).revenue
                assert abs(got - G.PI[name](c, q)) < TIGHT, (name, c, q, got)


# --- the targets are equilibria --------------------------------------------


def test_paper_targets_are_exact_equilibria():
    """Zero utility loss for every published target."""
    for name in G.REIMBURSEMENT_POLICIES:
        auction, sigma = target(name)
        assert A.utility_loss(auction, sigma) < 1e-12, name


def test_endgame_is_knife_edge_and_the_opening_is_not():
    """The mixed rounds are exact indifference points, but the opening is strict."""
    auction, sigma = target("zero")
    at = auction.info_index
    for round_index in (1, 3):  # sigma_1 and sigma_3 are interior
        assert abs(A.counterfactual_gain(auction, sigma, at[(0, auction.states[round_index], 1)])) < 1e-12, round_index
    assert A.counterfactual_gain(auction, sigma, at[(0, auction.root(), 1)]) > 1e-6  # H strictly stays at p = 0
    assert A.counterfactual_gain(auction, sigma, at[(0, auction.root(), 0)]) < -1e-6  # L strictly exits at p = 0


def test_opening_incentive_constraint_binds_at_one_minus_q_over_two():
    """Under the targeted policy the opening gain is the incentive bound less the cost."""
    for q in (0.25, 0.5, 0.75):
        for c in (G.ic_bound(q) - 0.1, G.ic_bound(q) + 0.1):
            auction, sigma = target("targeted", c=c, q=q)
            position = auction.info_index[(0, auction.root(), 1)]
            gain = A.counterfactual_gain(auction, sigma, position) / q  # the opening's reach is the prior on H
            assert abs(gain - (G.ic_bound(q) - c)) < TIGHT
            assert (gain > 0) == (c < G.ic_bound(q))


# --- the revenue ranking ----------------------------------------------------


def revenues(c, q):
    """Seller net revenue under each reimbursement policy at ``(c, q)``, with the target imposed."""
    return {name: A.evaluate(*target(name, c=c, q=q)).revenue for name in G.REIMBURSEMENT_POLICIES}


def test_ranking_against_no_reimbursement_fails_above_cbar():
    """Above ``cbar`` the targeted policy loses to doing nothing, but never to full."""
    assert G.CBAR < G.ic_bound(0.5)  # at q = 1/2 the binding bound is cbar
    for c in (0.13, 0.15, 0.2):
        revenue = revenues(c, 0.5)
        assert revenue["zero"] > revenue["targeted"] > revenue["full"], (c, revenue)


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
