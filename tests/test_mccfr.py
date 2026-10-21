"""The solver suite: what one run returns, the verdict scoring reaches on it, and the grid it is spent on."""

from __future__ import annotations

import itertools

import numpy as np

from auction import game
from auction import exact as A
from auction import symmetric as S
from auction import mccfr as M
from experiments import two_bidder as T
from test_paper import main, target

C, Q, TIGHT = 0.05, 0.5, 1e-12
ITERATIONS = 2000


# --- the easy targets: the plumbing they were chosen to clear ----------------


def test_full_reimbursement_leaves_the_bidders_the_frictionless_game():
    """Paying every stay back is the zero-cost auction to a bidder."""
    frictionless = game.Auction(q=Q, c=0.0, reimbursement_policy="zero")
    reimbursed = game.Auction(q=Q, c=C, reimbursement_policy="full")
    for seed in range(3):
        free, paid = M.solve(frictionless, seed=seed, iterations=400), M.solve(reimbursed, seed=seed, iterations=400)
        assert np.array_equal(free.profile, paid.profile), seed
        outcome = A.evaluate(reimbursed, paid.profile)
        assert np.allclose(outcome.payoff, A.evaluate(frictionless, free.profile).payoff, atol=TIGHT)


# --- the symmetry restriction ------------------------------------------------


def symmetrise(auction, sigma):
    """The symmetric profile in which every bidder plays the bucket mean of ``sigma``."""
    return S.anonymise(auction, sigma).ravel()[auction.orbit_index]


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
    for auction in (game.Auction(q=Q, c=C, reimbursement_policy="zero"), game.Auction(q=Q, c=C, n=3)):
        solved = M.solve(auction, seed=0, iterations=200, trace_every=20, symmetry_restricted=True)
        for profile in (solved.profile, *solved.averages):  # symmetric along the run, not only at the end
            for key, rows in orbits(auction, profile).items():
                assert len(set(rows)) == 1, (auction.n, key, rows)


def test_the_restriction_is_on_the_search_and_not_an_average_taken_afterwards():
    """Sharing a table changes what the run explores, so the answer is not the default run's bucket mean."""
    auction = game.Auction(q=Q, c=C, reimbursement_policy="zero")
    apart, restricted = M.solve(auction, seed=0, iterations=ITERATIONS), M.solve(
        auction, seed=0, iterations=ITERATIONS, symmetry_restricted=True
    )
    assert not S.applies(auction, apart.profile)
    assert S.applies(auction, restricted.profile)
    assert not np.allclose(restricted.profile, symmetrise(auction, apart.profile), atol=0.01)


# --- the average profile ----------------------------------------------------


def test_a_set_the_average_never_weighed_is_read_off_its_regret():
    """External sampling averages at opponent nodes only, so a set no opponent play reached falls back to its regret."""
    regret = np.array([[0.3, 0.1], [0.2, 0.6]])
    strategy_sum = np.array([[0.0, 0.0], [1.0, 3.0]])
    assert np.allclose(M.averaged(strategy_sum, regret), [0.25, 0.75])


def test_sampling_noise_does_not_hold_sigma2_short_of_one():
    """At sigma2 = 1 both actions tie, and a floor on regret once turned noise there into drift below 1."""
    auction = game.Auction(q=Q, c=C, reimbursement_policy="zero")
    profile = M.solve(auction, iterations=ITERATIONS, symmetry_restricted=True).profile
    assert M.stay_schedule(auction, profile)["sigma2"] > 0.99


# --- scoring a finished profile ---------------------------------------------


def bully(auction):
    """One bidder staying on either signal while the other concedes, an equilibrium the paper does not derive."""
    return game.profile(auction, lambda state, signal: 1.0, lambda state, signal: 0.0)


def test_scoring_an_exact_target_finds_the_derived_equilibrium():
    """A profile that is already the closed form scores as the paper's equilibrium, under every rule."""
    for name in game.REIMBURSEMENT_POLICIES:
        got = M.score(*target(name))
        assert got.utility_loss < TIGHT, (name, got)
        assert got.verdict is M.Verdict.DERIVED, (name, got)


def test_the_three_cases_are_told_apart_by_the_returned_verdict():
    """Right, a different equilibrium, and not an equilibrium call for different responses, so each is named."""
    auction, sigma = target("zero")
    uniform = np.full(len(auction.info_sets), 0.5)
    verdicts = [M.score(auction, profile).verdict for profile in (sigma, bully(auction), uniform)]
    assert verdicts == [M.Verdict.DERIVED, M.Verdict.DIFFERENT, M.Verdict.UNSOLVED], verdicts


def test_nothing_in_the_scoring_path_symmetrises_the_profile():
    """The bidders disagree over who concedes, and the bucket mean of that is a profile neither of them plays."""
    auction, _ = target("zero")
    asymmetric = bully(auction)
    mean = symmetrise(auction, asymmetric)
    assert not np.allclose(asymmetric, mean)
    got, bucket = M.score(auction, asymmetric), M.score(auction, mean)
    assert abs(got.utility_loss - A.utility_loss(auction, asymmetric)) < TIGHT, got
    assert abs(got.revenue - A.evaluate(auction, asymmetric).revenue) < TIGHT, got
    # Symmetrising first would have reported the bucket mean's numbers, and lost the equilibrium with them.
    assert got.utility_loss < TIGHT and bucket.utility_loss > 1e-6, (got, bucket)
    assert got.verdict is M.Verdict.DIFFERENT and bucket.verdict is M.Verdict.UNSOLVED, (got, bucket)


def test_the_reported_schedule_names_the_probabilities_the_closed_forms_are_stated_as():
    """The pipelines report a run this way, so the names have to sit on the rounds the paper gives them."""
    auction, _ = target("zero")
    derived, _ = M.closed_forms(auction)
    expected = {"open_high": 1.0, "open_low": 0.0, "sigma1": 1 - 4 * C / 3, "sigma2": 1.0, "sigma3": 1 - 4 * C}
    got = M.stay_schedule(auction, derived)
    assert got.keys() == expected.keys() and all(abs(got[k] - expected[k]) < TIGHT for k in expected), got


def test_the_distance_bar_is_read_on_the_path_the_closed_forms_take():
    """Off the path the run has no experience, so a set it never visits must not decide the verdict."""
    auction, sigma = target("zero")
    stray = sigma.copy()
    # The low type's last round, which the target leaves at zero reach and no budget ever informs.
    stray[auction.info_index[(0, auction.states[3], 0)]] = 1.0
    got = M.score(auction, stray)
    assert got.verdict is M.Verdict.DERIVED, got


# --- the grid ---------------------------------------------------------------


def test_every_closed_form_is_an_equilibrium_inside_the_region():
    """Below the bound the theorem gives every price-only corner a sorting equilibrium, so matching it is reachable."""
    for q, _, c in T.CELLS:
        if c > game.region(q):
            continue
        for rule in game.REIMBURSEMENT_POLICIES:
            auction = game.Auction(q=q, c=c, reimbursement_policy=rule)
            sigma = game.profile(auction, game.closed_form(auction)[0])
            assert float(A.utility_loss(auction, sigma)) <= A.CERTIFIED, (q, c, rule)


if __name__ == "__main__":
    raise SystemExit(main(globals()))
