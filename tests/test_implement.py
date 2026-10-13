"""The minimum-cost linear program: that its algebra is exact, and that what it returns implements the target."""

from __future__ import annotations

import numpy as np

from auction import implement as I
from auction import symmetric as S
from auction.game import Auction
from test_game import random_symmetric
from test_paper import main

C, TIGHT = 0.05, 1e-12


def costly(n=3, c=C, rounds=4, **overrides):
    """The costly auction the program writes a rule for, whose own policy it never reads."""
    settings = {"reimbursement_policy": "full", "delta": n / rounds} | overrides
    return Auction(n=n, c=c, rounds=rounds, **settings)


# --- the algebra the program rests on ---------------------------------------


def test_the_objective_is_the_reimbursement_the_reduction_already_reports():
    """Expected stays per orbit times what is paid there is the bill, read off an independent path."""
    for n in (2, 3, 4):
        auction = costly(n=n)
        sigma = random_symmetric(auction, 4)
        mass = I.stay_mass(auction, sigma)
        rng = np.random.default_rng(n)
        for refunds in (np.full(len(auction.orbits), C), rng.uniform(0.0, C, len(auction.orbits))):
            want = S.outcome(auction, sigma, refunds).reimbursement
            assert abs(float(mass @ refunds) - want) < TIGHT, (n, want)


def test_the_gain_coefficients_reproduce_the_gain_at_a_rule_they_never_saw():
    """The gain is affine in what the rule pays, so one difference per orbit pins it everywhere."""
    for n in (2, 3):
        auction = costly(n=n)
        sigma = random_symmetric(auction, 6)
        base, slope, _ = I.gain_coefficients(auction, sigma)
        refunds = np.random.default_rng(n).uniform(0.0, C, len(auction.orbits))
        want = S.gains(auction, sigma, refunds=refunds)[0]
        assert np.max(np.abs(base + np.einsum("jos,j->os", slope, refunds) - want)) < TIGHT, n


if __name__ == "__main__":
    raise SystemExit(main(globals()))
