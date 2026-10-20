"""Game-level claims: the signal profile is drawn jointly."""

from __future__ import annotations

import numpy as np

from auction import exact as A
from auction import game
from auction import symmetric as S
from test_paper import main

C, TIGHT = 0.05, 1e-12


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
            plain = game.Auction(n=n, q=q)
            zero = game.Auction(n=n, q=q, signal_prior=game.equicorrelated(q, 0.0))
            assert np.array_equal(plain.prior, zero.prior), (n, q)
            assert plain.independent_signals and zero.independent_signals


def test_a_correlated_prior_keeps_the_marginals_and_sets_the_pairwise_correlation():
    """Each bidder still draws high with probability q, and every pair correlates at exactly rho."""
    for q in (0.3, 0.5):
        for rho in (0.0, 0.25, 0.6, 1.0):
            auction = game.Auction(n=4, q=q, signal_prior=game.equicorrelated(q, rho))
            signals, prior = auction.profiles, auction.prior
            assert abs(prior.sum() - 1.0) < TIGHT and prior.min() >= 0.0
            assert np.max(np.abs(signals.T @ prior - q)) < TIGHT, (q, rho)
            covariance = float((signals[:, 0] * signals[:, 1]) @ prior - q * q)
            assert abs(covariance / (q * (1 - q)) - rho) < TIGHT, (q, rho)
            assert auction.independent_signals == (rho == 0.0)


def test_the_orbit_reduction_answers_under_an_exchangeable_correlated_prior():
    """Correlation moves the signal counts off the binomial, which the reduction now reads rather than refuses."""
    auction = game.Auction(n=3, q=0.5, c=C, signal_prior=game.equicorrelated(0.5, 0.4))
    sigma = random_symmetric(auction, 12)
    assert not auction.independent_signals and auction.exchangeable_signals
    assert S.applies(auction, sigma)
    fast, slow = S.outcome(auction, sigma), enumerated(A.evaluate, auction, sigma)
    assert abs(fast.price - slow.price) < 1e-12 and abs(fast.reimbursement - slow.reimbursement) < 1e-12
    assert np.max(np.abs(fast.payoff - slow.payoff)) < 1e-12
    assert abs(S.utility_loss(auction, sigma) - enumerated(A.utility_loss, auction, sigma)) < 1e-12


if __name__ == "__main__":
    raise SystemExit(main(globals()))
