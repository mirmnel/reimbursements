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


def relabelled(permutation, info_set):
    """Where an information set lands when the bidders are renamed by ``permutation``."""
    bidder, state, signal = info_set
    exits = [0] * len(state.exits)
    for member, exited in enumerate(state.exits):
        exits[permutation[member]] = exited
    active = tuple(sorted(permutation[member] for member in state.active))
    return (permutation[bidder], game.State(state.round, active, tuple(exits)), signal)


def same_orbit(auction, left, right):
    """Whether some renaming of the bidders carries one information set onto the other."""
    return any(
        relabelled(permutation, auction.info_sets[left]) == auction.info_sets[right]
        for permutation in itertools.permutations(range(auction.n))
    )


def orbits(auction, table):
    """Rows of ``table`` grouped by the orbit of the information set they sit at."""
    grouped = {}
    for position, (_, state, signal) in enumerate(auction.info_sets):
        grouped.setdefault((auction.orbit_of_state[state], signal), []).append(table[position])
    return grouped


def test_the_shared_cells_are_the_orbits_of_relabelling_the_bidders():
    """Two information sets share a cell exactly when an exchangeable bidder could be at either."""
    # Orbits are built from the permutations themselves rather than from a bucket key, so a map
    # keyed on the wrong thing fails here instead of agreeing with the solver's own reading.
    for auction in (game.Auction(q=Q, c=C), game.Auction(q=Q, c=C, n=3)):
        cells = M.shared_cells(auction)
        for left in range(len(auction.info_sets)):
            for right in range(len(auction.info_sets)):
                assert (cells[left] == cells[right]) == same_orbit(auction, left, right), (left, right)


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
    assert not S.is_symmetric(auction, apart.profile)
    assert S.is_symmetric(auction, restricted.profile)
    assert not np.allclose(restricted.profile, S.symmetrise(auction, apart.profile), atol=0.01)


# --- the average profile ----------------------------------------------------


def test_a_set_the_average_never_weighed_is_read_off_its_regret():
    """External sampling averages at opponent nodes only, so a set no opponent play reached falls back to its regret."""
    regret = np.array([[0.3, 0.1], [0.2, 0.6]])
    strategy_sum = np.array([[0.0, 0.0], [1.0, 3.0]])
    assert np.allclose(M.averaged(strategy_sum, regret), [0.25, 0.75])


# --- scoring a finished profile ---------------------------------------------


def bully(auction):
    """One bidder staying on either signal while the other concedes, an equilibrium the paper does not derive."""
    return game.profile(auction, lambda state, signal: 1.0, lambda state, signal: 0.0)


def test_scoring_an_exact_target_finds_the_derived_equilibrium():
    """Both distances fall to machine zero on a profile that is already the closed form."""
    for name in game.REIMBURSEMENT_POLICIES:
        got = M.score(*target(name))
        assert got.utility_loss < TIGHT, (name, got)
        assert got.on_path_distance == 0.0, (name, got)
        assert got.revenue_distance < TIGHT, (name, got)
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
    mean = S.symmetrise(auction, asymmetric)
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
    assert M.stay_schedule(auction, derived) == {
        "open_high": 1.0,
        "open_low": 0.0,
        "sigma1": game.sigma1(C),
        "sigma2": 1.0,
        "sigma3": game.sigma3(C),
    }


def test_the_distance_bar_is_read_on_the_path_the_closed_forms_take():
    """Off the path the run has no experience, so a set it never visits must not decide the verdict."""
    auction, sigma = target("zero")
    stray = sigma.copy()
    # The low type's last round, which the target leaves at zero reach and no budget ever informs.
    stray[auction.info_index[(0, auction.states[3], 0)]] = 1.0
    got = M.score(auction, stray)
    assert got.on_path_distance == 0.0, got
    assert got.verdict is M.Verdict.DERIVED, got


# --- the grid ---------------------------------------------------------------


def test_the_targeted_closed_form_stops_being_an_equilibrium_past_the_incentive_bound():
    """Where separation is not preserved the derived verdict is out of reach, and different is the answer."""
    for q, _, c in T.grid(T.VALIDATION) + T.grid(T.ROBUSTNESS):
        for rule in game.REIMBURSEMENT_POLICIES:
            auction = game.Auction(q=q, c=c, reimbursement_policy=rule)
            sigma = game.profile(auction, game.closed_form_strategy(rule, c))
            loss = float(A.utility_loss(auction, sigma))
            assert (loss > A.CERTIFIED) == (rule == "targeted" and c > game.ic_bound(q)), (q, c, rule, loss)


if __name__ == "__main__":
    raise SystemExit(main(globals()))
