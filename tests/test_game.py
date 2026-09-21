"""Game-level claims: what the seller's books must close to, and the reimbursement schedule families."""

from __future__ import annotations

import numpy as np

import analysis as AN
import solver as A
import game
import symmetric as S
from test_paper import main

C, ROUNDS, TIGHT = 0.025, 4, 1e-12


def random_symmetric(auction, seed):
    """A random symmetric profile, drawn once per (orbit, signal) bucket."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.15, 0.9, len(auction.orbits) * 2)[auction.orbit_index]


def enumerated(call, *arguments):
    """Run ``call`` with the reduction switched off, so that it enumerates the histories."""
    saved, S.TOLERANCE = S.TOLERANCE, -1.0
    try:
        return call(*arguments)
    finally:
        S.TOLERANCE = saved


# --- the payoffs close the seller's books -----------------------------------


def test_bidder_payoffs_close_the_seller_accounting():
    """Total bidder payoff is the prize less net revenue less what participation cost, at every scale."""
    for n in (2, 3, 4, 6):  # ten no longer builds, now that a profile is flat over histories
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


# --- the signal profile is drawn jointly ------------------------------------


def shared(q):
    """The perfectly correlated prior, in which every bidder draws the one same signal."""
    return lambda profiles: np.where(
        profiles.sum(1) == profiles.shape[1], q, np.where(profiles.sum(1) == 0, 1 - q, 0.0)
    )


def test_the_independent_prior_is_what_the_default_still_draws():
    """Correlating at zero is the independent prior bit for bit, so no existing result can move."""
    for n in (2, 3, 4):
        for q in (0.3, 0.5):
            plain = A.Auction(n=n, q=q)
            zero = A.Auction(n=n, q=q, signal_prior=game.equicorrelated(q, 0.0))
            assert np.array_equal(plain.prior, zero.prior), (n, q)
            assert plain.independent_signals and zero.independent_signals


def test_a_correlated_prior_keeps_the_marginals_and_sets_the_pairwise_correlation():
    """Each bidder still draws high with probability q, and every pair correlates at exactly rho."""
    for q in (0.3, 0.5):
        for rho in (0.0, 0.25, 0.6, 1.0):
            auction = A.Auction(n=4, q=q, signal_prior=game.equicorrelated(q, rho))
            signals, prior = auction.profiles, auction.prior
            assert abs(prior.sum() - 1.0) < TIGHT and prior.min() >= 0.0
            assert np.max(np.abs(auction.marginals - q)) < TIGHT, (q, rho)
            covariance = float((signals[:, 0] * signals[:, 1]) @ prior - q * q)
            assert abs(covariance / (q * (1 - q)) - rho) < TIGHT, (q, rho)
            assert auction.independent_signals == (rho == 0.0)


def test_exact_evaluation_and_gains_are_linear_in_the_joint_prior():
    """A mixture prior evaluates to the mixture of its parts, which is what respecting it means."""
    # equicorrelated is exactly this mixture, so the parts pin the whole rather than restating it.
    q, rho = 0.5, 0.4
    mixed = A.Auction(n=3, q=q, c=C, signal_prior=game.equicorrelated(q, rho))
    parts = [A.Auction(n=3, q=q, c=C, signal_prior=prior) for prior in (shared(q), game.independent(q))]
    sigma = random_symmetric(mixed, 11)
    whole, pieces = A.evaluate(mixed, sigma), [A.evaluate(part, sigma) for part in parts]
    for field in ("price", "reimbursement", "stay_rounds"):
        want = rho * getattr(pieces[0], field) + (1 - rho) * getattr(pieces[1], field)
        assert abs(getattr(whole, field) - want) < TIGHT, field
    want = rho * pieces[0].payoff + (1 - rho) * pieces[1].payoff
    assert np.max(np.abs(whole.payoff - want)) < TIGHT
    reach = rho * A.counterfactual_reach(parts[0], sigma) + (1 - rho) * A.counterfactual_reach(parts[1], sigma)
    assert np.max(np.abs(A.counterfactual_reach(mixed, sigma) - reach)) < TIGHT
    for position in range(0, len(mixed.info_sets), 7):
        gain = sum(
            weight * A.counterfactual_gain(part, sigma, position)
            for weight, part in zip((rho, 1 - rho), parts)
        )
        assert abs(A.counterfactual_gain(mixed, sigma, position) - gain) < TIGHT, position


def test_the_orbit_reduction_answers_under_an_exchangeable_correlated_prior():
    """Correlation moves the signal counts off the binomial, which the reduction now reads rather than refuses."""
    auction = A.Auction(n=3, q=0.5, c=C, signal_prior=game.equicorrelated(0.5, 0.4))
    sigma = random_symmetric(auction, 12)
    assert not auction.independent_signals and auction.exchangeable_signals
    assert S.applies(auction, sigma)
    fast, slow = S.outcome(auction, sigma), enumerated(A.evaluate, auction, sigma)
    assert abs(fast.price - slow.price) < 1e-12 and abs(fast.reimbursement - slow.reimbursement) < 1e-12
    assert np.max(np.abs(fast.payoff - slow.payoff)) < 1e-12
    assert abs(S.utility_loss(auction, sigma) - enumerated(AN.utility_loss, auction, sigma)) < 1e-12


def test_the_orbit_reduction_refuses_a_prior_that_relabelling_the_bidders_changes():
    """Counting signals loses which bidder drew what, so a prior that is not exchangeable has no orbit view."""
    lopsided = lambda profiles: np.prod(np.where(profiles == 1, [0.3, 0.7, 0.7], [0.7, 0.3, 0.3]), axis=1)
    auction = A.Auction(n=3, q=0.5, c=C, signal_prior=lopsided)
    sigma = random_symmetric(auction, 12)
    assert not auction.exchangeable_signals and not S.applies(auction, sigma)
    for call in (lambda: S.outcome(auction, sigma), lambda: S.utility_loss(auction, sigma)):
        try:
            call()
        except ValueError as exc:
            assert "exchangeable" in str(exc), exc
        else:
            raise AssertionError("the reduction answered under a prior that is not exchangeable")


def test_the_orbit_reduction_refuses_a_rule_that_separates_histories_inside_an_orbit():
    """One reimbursement per orbit is what the reduction pays, so a finer rule falls through to the histories."""
    auction = A.Auction(n=3, q=0.5, c=C, reimbursement_policy=lambda state: C * (state.active[0] == 0))
    sigma = random_symmetric(auction, 12)
    assert not auction.orbit_measurable and not S.applies(auction, sigma)
    try:
        S.outcome(auction, sigma)
    except ValueError as exc:
        assert "orbit" in str(exc), exc
    else:
        raise AssertionError("the reduction answered under a rule with no orbit view")


def test_a_prior_that_is_not_a_distribution_is_refused():
    """A prior that does not sum to one, or that goes negative, is a mistake rather than a weighting."""
    for broken in (lambda p: np.full(len(p), 0.1), lambda p: np.full(len(p), -0.25), lambda p: np.ones(3)):
        try:
            A.Auction(n=2, signal_prior=broken).prior
        except ValueError:
            continue
        raise AssertionError("a prior that is not a distribution was accepted")


if __name__ == "__main__":
    raise SystemExit(main(globals()))
