"""The solver suite: what one run returns."""

from __future__ import annotations

import itertools

import numpy as np

from auction import game
from auction import exact as A
from auction import mccfr as M
from test_paper import main

C, Q, TIGHT = 0.05, 0.5, 1e-12


def auction(rule, n=2, c=C):
    """The two-bidder example's clock under ``rule``, at ``n`` bidders."""
    return game.Auction(reimbursement_policy=rule, n=n, q=Q, c=c, rounds=4, delta=0.5)
ITERATIONS = 2000


# --- the easy targets: the plumbing they were chosen to clear ----------------


def test_full_reimbursement_leaves_the_bidders_the_frictionless_game():
    """Paying every stay back is the zero-cost auction to a bidder."""
    frictionless = auction(game.flat(0.0, 4, 0.0), c=0.0)
    reimbursed = auction(game.flat(C, 4))
    for seed in range(3):
        free, paid = M.solve(frictionless, seed, 400), M.solve(reimbursed, seed, 400)
        assert np.array_equal(free, paid), seed
        outcome = A.evaluate(reimbursed, paid)
        assert np.allclose(outcome.payoff, A.evaluate(frictionless, free).payoff, atol=TIGHT)


# --- the symmetry restriction ------------------------------------------------


def orbits(auction, table):
    """Rows of ``table`` grouped by the orbit of their information set under renaming the bidders."""
    # Built from the permutations rather than the solver's bucket key, so a wrong key cannot agree with itself.
    grouped = {}
    for position, (bidder, state, signal) in enumerate(auction.info_sets):
        images = set()
        for permutation in itertools.permutations(range(auction.n)):
            exits = [0] * auction.n
            for member, exited in enumerate(state.exits):
                exits[permutation[member]] = exited
            active = tuple(sorted(permutation[member] for member in state.active))
            images.add((permutation[bidder], game.State(state.round, active, tuple(exits)), signal))
        grouped.setdefault(frozenset(images), []).append(table[position])
    return grouped


def test_under_the_restriction_every_bidder_plays_the_same_strategy_exactly():
    """One shared table means the bidders agree by construction, so the agreement is exact and not close."""
    # Three bidders as well as two, where the active sets differ and the orbits are wider than a pair.
    for setting in (auction(game.flat(C, 4, 0.0)), auction(game.flat(C, 4), n=3)):
        for key, rows in orbits(setting, M.solve(setting, 0, 200)).items():
            assert len(set(rows)) == 1, (setting.n, key, rows)


# --- the average profile ----------------------------------------------------


def test_a_set_the_average_never_weighed_is_read_off_its_regret():
    """External sampling averages at opponent nodes only, so a set no opponent play reached falls back to its regret."""
    regret = np.array([[0.3, 0.1], [0.2, 0.6]])
    strategy_sum = np.array([[0.0, 0.0], [1.0, 3.0]])
    assert np.allclose(M.averaged(strategy_sum, regret), [0.25, 0.75])


def test_sampling_noise_does_not_hold_sigma2_short_of_one():
    """At sigma2 = 1 both actions tie, and a floor on regret once turned noise there into drift below 1."""
    setting = auction(game.flat(C, 4, 0.0))
    profile = M.solve(setting, 0, ITERATIONS)
    sigma2 = profile[setting.info_sets.index((0, setting.states[2], 1))]
    assert sigma2 > 0.99


if __name__ == "__main__":
    raise SystemExit(main(globals()))
