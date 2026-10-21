"""Every claim of "A Two-Bidder Example", checked against an exact solver never told the closed forms."""

from __future__ import annotations


from auction import game as G
from auction import exact as A
from auction import symmetric as S

Q, C, TIGHT = 0.5, 0.05, 1e-9


def gain_at(auction, sigma, state, signal):
    """Gain from staying rather than exiting at ``state`` on ``signal``, weighted by opponent reach only."""
    return S.gains(auction, sigma)[0][auction.orbit_of_state[state], signal]


def target(reimbursement_policy, c=C, q=Q, **overrides):
    """The game under a reimbursement policy, and the equilibrium the paper targets in it."""
    auction = G.Auction(q=q, c=c, reimbursement_policy=reimbursement_policy, **overrides)
    return auction, G.profile(auction, G.closed_form(auction)[0])


# --- the closed forms are the published ones -------------------------------


def published(c, q):
    """Per published policy, the high type's stay probability per round and seller net revenue, as printed."""
    s1, s3 = 1 - 4 * c / 3, 1 - 4 * c
    return {
        "zero": ((1, s1, 1, s3), q**2 * (0.5 * (1 - s1**2) + s1**2 * (1.5 + 0.5 * s3**2))),
        "full": ((1, 1, 1, 1), 2 * q**2 - c * (8 * q**2 + 2 * q * (1 - q))),
        "least_cost": ((1, 1, 1, 1), 2 * q**2 * (1 - 3 * c)),
        "backloaded": ((1, 1, 1 - 2 * c, 1), q**2 * (1 + (1 - 2 * c) ** 3)),
    }


def test_the_recursion_reproduces_every_published_closed_form():
    """One backward recursion over what a policy pays gives every closed form the paper prints."""
    for q in (0.6, 0.75, 0.9):
        for c in (0.01, 0.05, 0.1):
            for name, (stays, earned) in published(c, q).items():
                strategy, got = G.closed_form(G.Auction(q=q, c=c, reimbursement_policy=name))
                rounds = G.states(2, 4)
                assert all(abs(strategy(rounds[t], 1) - stays[t]) < TIGHT for t in range(4)), (name, c, q)
                assert all(strategy(state, 0) == 0.0 for state in rounds), name
                assert abs(got - earned) < TIGHT, (name, c, q, got, earned)


def test_least_cost_beats_full_by_exactly_two_q_c():
    """The closed-form gap between the least-cost and full policies holds at every cost and probability."""
    for q in (0.25, 0.5, 0.75):
        for c in (0.01, 0.05, 0.15):
            assert abs(G.closed_form_revenue("least_cost", c, q) - G.closed_form_revenue("full", c, q) - 2 * q * c) < 1e-12


# --- the exact solver reproduces the paper, without being told the answer ---


def test_frictionless_benchmark_price():
    """The frictionless benchmark reproduces the closed-form expected price."""
    for q in (0.25, 0.5, 0.75):
        outcome = A.evaluate(*target("full", q=q))
        assert abs(outcome.price - 2 * q**2) < TIGHT


def test_seller_revenue_matches_every_closed_form():
    """Exact evaluation of each target reproduces the closed-form seller net revenue."""
    for q in (0.25, 0.5, 0.75):
        for c in (0.01, 0.05, 0.1, 0.15, 0.2):  # the published cost grid
            for name in G.REIMBURSEMENT_POLICIES:
                got = A.evaluate(*target(name, c=c, q=q)).revenue
                assert abs(got - G.closed_form_revenue(name, c, q)) < TIGHT, (name, c, q, got)


# --- the targets are equilibria --------------------------------------------


def test_paper_targets_are_exact_equilibria():
    """Zero utility loss for every published target."""
    for name in G.REIMBURSEMENT_POLICIES:
        auction, sigma = target(name)
        assert A.utility_loss(auction, sigma) < 1e-12, name


def test_endgame_is_knife_edge_and_the_opening_is_not():
    """The mixed rounds are exact indifference points, but the opening is strict."""
    auction, sigma = target("zero")
    for round_index in (1, 3):  # sigma_1 and sigma_3 are interior
        assert abs(gain_at(auction, sigma, auction.states[round_index], 1)) < 1e-12, round_index
    assert gain_at(auction, sigma, auction.root(), 1) > 1e-6  # H strictly stays at p = 0
    assert gain_at(auction, sigma, auction.root(), 0) < -1e-6  # L strictly exits at p = 0


def test_opening_incentive_constraint_binds_at_one_minus_q_over_two():
    """Under the least-cost policy the opening gain is the incentive bound less the cost."""
    for q in (0.25, 0.5, 0.75):
        for c in (G.ic_bound(q) - 0.1, G.ic_bound(q) + 0.1):
            auction, sigma = target("least_cost", c=c, q=q)
            gain = gain_at(auction, sigma, auction.root(), 1) / q  # the opening's reach is the prior on H
            assert abs(gain - (G.ic_bound(q) - c)) < TIGHT
            assert (gain > 0) == (c < G.ic_bound(q))


def test_the_backloaded_target_is_an_equilibrium_earning_the_published_revenue():
    """Paying only the last stay, H stays at a/2, mixes at 1 - 2 kappa at a, and the seller earns q^2[1 + (1 - 2 kappa)^3]."""
    auction, sigma = target("backloaded", c=0.1, q=0.6)
    assert A.utility_loss(auction, sigma) < 1e-12
    assert abs(A.evaluate(auction, sigma).revenue - 0.54432) < TIGHT  # 0.36 * (1 + 0.8^3)


def test_the_backloaded_gaps_match_the_published_polynomials():
    """The gaps to least-cost and to no reimbursement are the theorem's two polynomials in kappa."""
    for q in (0.6, 0.75, 0.9):
        for c in (0.01, 0.03, G.ic_bound(q)):
            backloaded = G.closed_form_revenue("backloaded", c, q)
            assert abs(backloaded - G.closed_form_revenue("least_cost", c, q) - 4 * q**2 * c**2 * (3 - 2 * c)) < TIGHT, (c, q)
            gap = 2 * q**2 * c / 9 * (9 - 42 * c + 92 * c**2 - 64 * c**3)
            assert abs(backloaded - G.closed_form_revenue("zero", c, q) - gap) < TIGHT, (c, q)


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
