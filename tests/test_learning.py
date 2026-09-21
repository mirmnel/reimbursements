"""The sampled suite: Monte Carlo play checked against exact evaluation, with its own tolerances."""

from __future__ import annotations

import numpy as np

import analysis as AN
import solver as A
import learning as L
import symmetric as S
from test_paper import main

C, Q = 0.025, 0.5


# --- simulation and exact evaluation are the same auction -------------------


def test_simulation_agrees_with_exact_evaluation_on_a_symmetric_profile():
    """Playing the auction out reproduces the exactly evaluated revenue, within its own standard error."""
    for n in (2, 4, 6):
        auction = A.Auction(n=n, q=Q, c=C)
        sigma = AN.profile(auction, AN.SEPARATING)
        exact = A.evaluate(auction, sigma)
        drawn = L.sampled(auction, sigma, seed=0, episodes=40000)
        assert abs(drawn.outcome.revenue - exact.revenue) < 4 * drawn.revenue_error + 1e-12, (n, drawn.outcome)
        assert abs(drawn.outcome.payoff[0] - exact.payoff[0]) < 4 * drawn.payoff_error[0] + 1e-12, n


def test_simulation_agrees_with_exact_evaluation_on_an_asymmetric_profile():
    """Headline rows report a sampled revenue off a dispersed profile, so that path is checked too."""
    rng = np.random.default_rng(0)
    for n in (2, 3, 4):
        auction = A.Auction(n=n, q=Q, c=C)
        sigma = rng.uniform(0.1, 0.9, len(auction.info_sets))
        assert not S.applies(auction, sigma)  # so the enumeration, not the reduction, is the ground truth
        exact = A.evaluate(auction, sigma)
        drawn = L.sampled(auction, sigma, seed=0, episodes=60000)
        assert abs(drawn.outcome.revenue - exact.revenue) < 4 * drawn.revenue_error + 1e-12, (n, drawn.outcome)
        assert np.all(np.abs(drawn.outcome.payoff - exact.payoff) < 4 * drawn.payoff_error + 1e-12), n


if __name__ == "__main__":
    raise SystemExit(main(globals()))
