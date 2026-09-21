"""The solver suite: what one run returns, the verdict scoring reaches on it, and the grid it is spent on."""

from __future__ import annotations

import ast
import itertools
import pathlib

import numpy as np

import analysis as AN
import game
import solver as A
import symmetric as S
import mccfr as M
import validate as V
from test_paper import main, schedule, target

C, Q, TIGHT = 0.025, 0.5, 1e-12
ITERATIONS = 2000  # enough for the sign of a strict gain to be settled, not for convergence
STRICT = 0.01  # a normalised gain below which the information set is indifferent, not decided


# --- what a run returns -----------------------------------------------------


def test_a_run_returns_a_stay_probability_for_every_information_set():
    """The average behavioural profile is flat over the information sets, as the rest of the repo reads it."""
    auction = A.Auction(q=Q, c=C)
    solved = M.solve(auction, seed=0, iterations=20)
    assert solved.profile.shape == (len(auction.info_sets),)


def test_the_result_carries_what_a_missed_target_has_to_be_read_from():
    """Regrets say whether the run was wandering or confidently wrong, and the traces say which way it moved."""
    auction = A.Auction(q=Q, c=C)
    width = len(auction.info_sets)
    solved = M.solve(auction, seed=3, iterations=40, trace_every=4)
    assert solved.regrets.shape == (width, 2)
    assert solved.visits.shape == (width,)
    assert solved.iterates.shape == (10, width)
    assert solved.averages.shape == solved.iterates.shape
    assert solved.seed == 3


def test_every_returned_probability_is_one_a_bidder_could_play():
    """The profile and both traces are behavioural, so anything downstream can evaluate them as they stand."""
    auction = A.Auction(q=Q, c=C)
    solved = M.solve(auction, seed=1, iterations=60, trace_every=4)
    for name, table in (("profile", solved.profile), ("iterates", solved.iterates), ("averages", solved.averages)):
        assert np.all(np.isfinite(table)), name
        assert np.all((table >= 0.0) & (table <= 1.0)), (name, table.min(), table.max())


def test_the_same_seed_repeats_and_a_different_seed_does_not():
    """The seed is the whole of the randomness, initial regrets included, so any run can be rerun exactly."""
    auction = A.Auction(q=Q, c=C)
    once, again = M.solve(auction, seed=7, iterations=50), M.solve(auction, seed=7, iterations=50)
    other = M.solve(auction, seed=8, iterations=50)
    assert np.array_equal(once.profile, again.profile)
    assert np.array_equal(once.regrets, again.regrets)
    assert not np.array_equal(once.profile, other.profile)
    fresh = [M.solve(auction, seed=seed, iterations=0).regrets for seed in (7, 8)]
    assert not np.array_equal(*fresh)  # the initial draw differs by seed, so ties break differently


# --- the easy targets: the plumbing they were chosen to clear ----------------


def test_at_zero_participation_cost_the_regret_table_leans_the_way_the_exact_gain_does():
    """Separation is decided at the opening, where the sign convention is checked against the exact path."""
    # The opening is the one state whose counterfactual reach is the prior itself, so no set is
    # stale there and every seed is comparable.
    auction = A.Auction(q=Q, c=0.0, reimbursement_policy="zero")
    for seed in range(3):
        solved = M.solve(auction, seed=seed, iterations=ITERATIONS)
        weights = A.counterfactual_reach(auction, solved.profile)
        for signal in (0, 1):
            for bidder in range(auction.n):
                position = auction.info_index[(bidder, auction.root(), signal)]
                gain = A.counterfactual_gain(auction, solved.profile, position) / weights[position]
                if abs(gain) <= STRICT:  # indifferent, so the table has no sign to agree with
                    continue
                lean = solved.regrets[position][M.STAY] - solved.regrets[position][M.EXIT]
                assert np.sign(lean) == np.sign(gain), (seed, bidder, signal, gain, lean)


def test_full_reimbursement_leaves_the_bidders_the_frictionless_game():
    """Paying every stay back is the zero-cost auction to a bidder, and costs the seller each stay exactly once."""
    frictionless = A.Auction(q=Q, c=0.0, reimbursement_policy="zero")
    reimbursed = A.Auction(q=Q, c=C, reimbursement_policy="full")
    for seed in range(3):
        free, paid = M.solve(frictionless, seed=seed, iterations=400), M.solve(reimbursed, seed=seed, iterations=400)
        assert np.array_equal(free.profile, paid.profile), seed
        assert np.array_equal(free.regrets, paid.regrets), seed
        assert np.array_equal(free.visits, paid.visits), seed
        outcome = A.evaluate(reimbursed, paid.profile)
        assert np.allclose(outcome.payoff, A.evaluate(frictionless, free.profile).payoff, atol=TIGHT)
        assert abs(outcome.reimbursement - C * outcome.stay_rounds) < TIGHT, outcome
        assert abs(outcome.revenue - (outcome.price - C * outcome.stay_rounds)) < TIGHT, outcome


def test_chance_is_sampled_from_the_prior_rather_than_enumerated():
    """Signals are drawn, so the opening splits between the types in the prior's proportion and not evenly."""
    for q in (0.25, 0.75):
        auction = A.Auction(q=q, c=C)
        solved = M.solve(auction, seed=0, iterations=ITERATIONS)
        opening = [solved.visits[auction.info_index[(0, auction.root(), signal)]] for signal in (0, 1)]
        assert abs(opening[1] / sum(opening) - q) < 0.05, (q, opening)


def test_visits_count_what_play_reached_and_not_what_enumeration_walked():
    """A set its own play abandons falls quiet while its regret keeps moving, which is what the tremble lifts."""
    auction = A.Auction(q=Q, c=0.0, reimbursement_policy="zero")
    solved = M.solve(auction, seed=0, iterations=ITERATIONS, tremble=0.0)
    quiet = np.flatnonzero(solved.visits == 0)
    assert quiet.size, solved.visits
    moved = np.max(np.abs(solved.regrets[quiet]), axis=1)
    assert np.all(moved > M.INITIAL_REGRET), solved.regrets[quiet]


def test_the_bidders_are_never_made_to_agree():
    """Each bidder keeps its own table, so a run can disagree across bidders rather than hiding it."""
    auction = A.Auction(q=Q, c=0.0, reimbursement_policy="zero")
    apart = [
        abs(solved.profile[auction.info_index[(0, auction.root(), 0)]]
            - solved.profile[auction.info_index[(1, auction.root(), 0)]])
        for solved in (M.solve(auction, seed=seed, iterations=ITERATIONS) for seed in range(3))
    ]
    assert max(apart) > 0.1, apart


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
    for auction in (A.Auction(q=Q, c=C), A.Auction(q=Q, c=C, n=3)):
        cells = M.shared_cells(auction)
        for left in range(len(auction.info_sets)):
            for right in range(len(auction.info_sets)):
                assert (cells[left] == cells[right]) == same_orbit(auction, left, right), (left, right)


def test_under_the_restriction_every_bidder_plays_the_same_strategy_exactly():
    """One shared table means the bidders agree by construction, so the agreement is exact and not close."""
    # Three bidders as well as two, where the active sets differ and the orbits are wider than a pair.
    for auction in (A.Auction(q=Q, c=C, reimbursement_policy="zero"), A.Auction(q=Q, c=C, n=3)):
        solved = M.solve(auction, seed=0, iterations=200, trace_every=20, symmetry_restricted=True)
        for name, table in (("profile", solved.profile), ("regrets", solved.regrets)):
            for key, rows in orbits(auction, table).items():
                assert all(np.array_equal(row, rows[0]) for row in rows), (auction.n, name, key, rows)
        for step, iterate in enumerate(solved.iterates):  # symmetric along the run, not only at the end
            for key, rows in orbits(auction, iterate).items():
                assert len(set(rows)) == 1, (auction.n, step, key, rows)


def test_the_restriction_is_on_the_search_and_not_an_average_taken_afterwards():
    """Sharing a table changes what the run explores, so the answer is not the default run's bucket mean."""
    auction = A.Auction(q=Q, c=C, reimbursement_policy="zero")
    apart, restricted = M.solve(auction, seed=0, iterations=ITERATIONS), M.solve(
        auction, seed=0, iterations=ITERATIONS, symmetry_restricted=True
    )
    assert not S.is_symmetric(auction, apart.profile)
    assert S.is_symmetric(auction, restricted.profile)
    assert not np.allclose(restricted.profile, S.symmetrise(auction, apart.profile), atol=0.01)


def test_the_restriction_is_off_unless_it_is_asked_for_and_changes_nothing_when_off():
    """The default keeps one table per bidder, drawing the same initial regrets from the same stream."""
    auction = A.Auction(q=Q, c=C, reimbursement_policy="zero")
    width = len(auction.info_sets)
    cells = M.shared_cells(auction)
    assert cells.max() + 1 < width  # the restricted table is the smaller one
    for restricted, rows in ((False, width), (True, int(cells.max()) + 1)):
        # A table of the wrong width would consume a different stretch of the stream, so pinning the
        # draw is what keeps the default run the one it was before the restriction existed.
        drawn = np.random.default_rng(5).uniform(0.0, M.INITIAL_REGRET, (rows, 2))
        started = M.solve(auction, seed=5, iterations=0, symmetry_restricted=restricted)
        assert np.array_equal(started.regrets, drawn[cells] if restricted else drawn), restricted
    assert not S.is_symmetric(auction, M.solve(auction, seed=5, iterations=100).profile)


def test_a_restricted_run_reports_that_it_was_restricted():
    """The two searches answer different questions, so a row has to say which of them produced it."""
    auction = A.Auction(q=Q, c=C, reimbursement_policy="zero")
    assert M.solve(auction, seed=0, iterations=20, symmetry_restricted=True).symmetry_restricted
    assert not M.solve(auction, seed=0, iterations=20).symmetry_restricted


def test_a_restricted_run_returns_the_same_shapes_as_an_unrestricted_one():
    """Everything downstream reads a run flat over the information sets, whichever search produced it."""
    auction = A.Auction(q=Q, c=C, reimbursement_policy="zero")
    width = len(auction.info_sets)
    solved = M.solve(auction, seed=1, iterations=40, trace_every=4, symmetry_restricted=True)
    assert solved.profile.shape == (width,)
    assert solved.regrets.shape == (width, 2)
    assert solved.visits.shape == (width,)
    assert solved.iterates.shape == solved.averages.shape == (10, width)
    assert np.all((solved.profile >= 0.0) & (solved.profile <= 1.0)), solved.profile


# --- the method: regret matching plus, linear weighting, the tremble --------


def test_stored_regret_is_never_negative_anywhere_along_a_run():
    """Regret matching plus floors the stored table, so a set that fell out of favour climbs back from zero."""
    free = A.Auction(q=Q, c=0.0, reimbursement_policy="zero")
    for auction in (free, A.Auction(q=Q, c=C, reimbursement_policy="zero")):
        for seed in range(3):
            for iterations in (1, 2, 5, 20, 100, 500):
                solved = M.solve(auction, seed=seed, iterations=iterations, trace_every=iterations)
                assert solved.regrets.min() >= 0.0, (seed, iterations, solved.regrets.min())
                # The last policy of the run was read off the table that came back, so the floor was
                # applied as the run went rather than once on the way out.
                assert np.array_equal(solved.iterates[-1], M.matched(solved.regrets)), (seed, iterations)


def test_the_tremble_lifts_the_off_path_sets_and_without_it_they_stay_quiet():
    """Sampled play reaches an abandoned set only by trembling into it, which is what the tremble exists to do."""
    # Pooled over seeds because reaching a set costs one tremble per round of depth, so a single run
    # lifts the shallow sets and can leave the deepest ones at zero.
    auction = A.Auction(q=Q, c=C, reimbursement_policy="zero")
    quiet, lifted = 0, 0
    for seed in range(4):
        off_path = np.flatnonzero(M.solve(auction, seed=seed, iterations=ITERATIONS, tremble=0.0).visits == 0)
        trembled = M.solve(auction, seed=seed, iterations=ITERATIONS, tremble=M.TREMBLE)
        quiet, lifted = quiet + off_path.size, lifted + int(trembled.visits[off_path].sum())
    assert quiet, "steady play abandoned no set, so the tremble has nothing to lift"
    assert lifted > 0, (quiet, lifted)


def test_the_trembled_probability_is_never_the_one_reported():
    """The report accumulates the untrembled strategy, though trembled play decides how often it accumulates."""
    auction = A.Auction(q=Q, c=0.0, reimbursement_policy="zero")
    tremble = 0.5  # exaggerated, so accumulating the trembled figure would pin the report to the middle half
    solved = M.solve(auction, seed=0, iterations=ITERATIONS, tremble=tremble)
    assert np.any((solved.profile < tremble / 2) | (solved.profile > 1.0 - tremble / 2)), solved.profile


def test_the_average_weights_later_iterations_more_heavily_than_earlier_ones():
    """Linear weighting is what lets the average follow a mixture that is still settling."""
    # A short run, where the iterates are still swinging and the two weightings are far enough apart
    # to tell which one the reported average was built with.
    auction = A.Auction(q=Q, c=C, reimbursement_policy="zero")
    for seed in range(4):
        solved = M.solve(auction, seed=seed, iterations=200, trace_every=2)
        steps = np.arange(1, len(solved.iterates) + 1)[:, None]
        flat, leaning = solved.iterates.mean(0), (steps * solved.iterates).sum(0) / steps.sum()
        moved = np.flatnonzero(np.abs(leaning - flat) > 0.02)  # sets that drifted enough to tell the two apart
        assert moved.size, (seed, np.abs(leaning - flat).max())
        reported = solved.averages[-1][moved]
        nearer_leaning = np.mean(np.abs(reported - flat[moved])) - np.mean(np.abs(reported - leaning[moved]))
        assert nearer_leaning > 0.0, (seed, nearer_leaning)


def solver_source():
    """Every node of the solver module's syntax tree, for the two tests that read it as text."""
    return ast.walk(ast.parse(pathlib.Path(M.__file__).read_text()))


def test_no_learned_probability_is_rounded_to_a_pure_action():
    """The targets are genuinely mixed, so rounding the report would select a different equilibrium."""
    rounding = {"round", "rint", "floor", "ceil", "trunc", "heaviside", "sign"}
    called = set()
    for node in solver_source():
        if isinstance(node, ast.Call):
            called.add(node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", ""))
    assert not called & rounding, sorted(called & rounding)


def test_the_solver_module_does_not_reach_for_the_symmetric_reduction():
    """The reduction fires only on symmetric profiles, and the one path here that can be handed one is scoring."""
    modules = set()
    for node in solver_source():
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert "symmetric" not in modules, sorted(modules)


# --- scoring a finished profile ---------------------------------------------


def bully(auction):
    """One bidder staying on either signal while the other concedes, an equilibrium the paper does not derive."""
    return AN.profile(auction, AN.ALWAYS, AN.NEVER)


def test_scoring_an_exact_target_finds_the_derived_equilibrium():
    """Both distances fall to machine zero on a profile that is already the closed form."""
    for name in A.REIMBURSEMENT_POLICIES:
        got = M.score(*target(name))
        assert got.utility_loss < TIGHT, (name, got)
        assert got.on_path_distance == 0.0, (name, got)
        assert got.revenue_distance < TIGHT, (name, got)
        assert got.verdict is M.Verdict.DERIVED, (name, got)


def test_scoring_reports_the_trembled_gain_alongside_the_ordinary_one():
    """The sets the ordinary gain drops for want of reach are where a conceding bidder's mistake sits."""
    auction, sigma = target("zero")
    assert M.score(auction, sigma).trembled_gain < TIGHT
    got = M.score(auction, bully(auction))
    assert got.one_shot_gain < TIGHT, got
    assert got.trembled_gain > 1e-6, got


def test_the_three_cases_are_told_apart_by_the_returned_verdict():
    """Right, a different equilibrium, and not an equilibrium call for different responses, so each is named."""
    auction, sigma = target("zero")
    uniform = np.full(len(auction.info_sets), 0.5)
    verdicts = [M.score(auction, profile).verdict for profile in (sigma, bully(auction), uniform)]
    assert verdicts == [M.Verdict.DERIVED, M.Verdict.DIFFERENT, M.Verdict.UNSOLVED], verdicts
    assert len(set(verdicts)) == 3


def test_nothing_in_the_scoring_path_symmetrises_the_profile():
    """The bidders disagree over who concedes, and the bucket mean of that is a profile neither of them plays."""
    auction, _ = target("zero")
    asymmetric = bully(auction)
    mean = S.symmetrise(auction, asymmetric)
    assert not np.allclose(asymmetric, mean)
    got, bucket = M.score(auction, asymmetric), M.score(auction, mean)
    assert abs(got.utility_loss - AN.utility_loss(auction, asymmetric)) < TIGHT, got
    assert abs(got.revenue - A.evaluate(auction, asymmetric).revenue) < TIGHT, got
    # Symmetrising first would have reported the bucket mean's numbers, and lost the equilibrium with them.
    assert got.utility_loss < TIGHT and bucket.utility_loss > 1e-6, (got, bucket)
    assert got.verdict is M.Verdict.DIFFERENT and bucket.verdict is M.Verdict.UNSOLVED, (got, bucket)


def test_scoring_a_supplied_profile_never_runs_the_solver():
    """A profile already known to be exact is scored without paying for a run to produce one."""
    auction, sigma = target("zero")
    saved = M.solve

    def refuse(*arguments, **keywords):
        raise AssertionError("scoring ran the solver")

    M.solve = refuse
    try:
        got = M.score(auction, sigma)
    finally:
        M.solve = saved
    assert got.verdict is M.Verdict.DERIVED


def test_the_revenue_scored_against_is_the_mean_unit_closed_form():
    """The published revenues are twice these, so scoring against them would miss every target by its own size."""
    for name in A.REIMBURSEMENT_POLICIES:
        got = M.score(*target(name))
        assert abs(got.revenue - AN.PI[name](C, Q)) < TIGHT, (name, got.revenue)
        assert abs(got.revenue - AN.PUBLISHED[name](C, Q)) > 0.1, (name, got.revenue)


def test_the_stay_probabilities_scored_against_are_the_mean_unit_closed_forms():
    """The published schedule halves each cost differently, so it sits a visible distance from the one used."""
    auction, _ = target("zero")
    derived, _ = M.closed_forms(auction)
    assert np.allclose(schedule(auction, derived, 1), [1.0, AN.sigma1(C), 1.0, AN.sigma3(C)])
    assert not np.allclose(
        schedule(auction, derived, 1), [1.0, AN.published_sigma1(C), 1.0, AN.published_sigma3(C)]
    )


def test_the_reported_schedule_names_the_probabilities_the_closed_forms_are_stated_as():
    """The pipelines report a run this way, so the names have to sit on the rounds the paper gives them."""
    auction, _ = target("zero")
    derived, _ = M.closed_forms(auction)
    assert M.stay_schedule(auction, derived) == {
        "open_high": 1.0,
        "open_low": 0.0,
        "sigma1": AN.sigma1(C),
        "sigma2": 1.0,
        "sigma3": AN.sigma3(C),
    }


def test_the_verdict_reads_the_two_distance_bars_it_was_given():
    """The bars are read off a long run rather than picked in advance, so scoring takes them as it is handed them."""
    auction, _ = target("zero")
    far = bully(auction)  # a certified equilibrium, a full unit of stay probability from the closed form
    assert M.score(auction, far, stay_bar=1.0, revenue_bar=0.3).verdict is M.Verdict.DERIVED
    assert M.score(auction, far, stay_bar=1.0).verdict is M.Verdict.DIFFERENT  # the revenue is still far
    assert M.score(auction, far, revenue_bar=0.3).verdict is M.Verdict.DIFFERENT  # the profile is still far


def test_the_distance_bar_is_read_on_the_path_the_closed_forms_take():
    """Off the path the run has no experience, so a set it never visits must not decide the verdict."""
    auction, sigma = target("zero")
    stray = sigma.copy()
    # The low type's last round, which the target leaves at zero reach and no budget ever informs.
    stray[auction.info_index[(0, auction.states[3], 0)]] = 1.0
    got = M.score(auction, stray)
    assert got.on_path_distance == 0.0, got
    assert got.off_path_distance == 1.0, got
    assert got.verdict is M.Verdict.DERIVED, got


def test_the_off_path_distance_is_reported_rather_than_folded_into_the_bar():
    """It ranges over the whole unit interval at any budget, so it is recorded and left to the trembled gain."""
    auction, sigma = target("zero")
    stray = sigma.copy()
    stray[auction.info_index[(0, auction.states[3], 0)]] = 0.6
    assert M.score(auction, stray).off_path_distance == 0.6
    assert M.score(auction, sigma).off_path_distance == 0.0


def test_the_closed_forms_refuse_what_they_do_not_describe():
    """Scoring against a form that does not describe the auction would be nonsense, so it fails rather than reports."""
    try:
        M.closed_forms(A.Auction(n=3, q=Q, c=C, reimbursement_policy="zero"))
    except ValueError as exc:
        assert "two-bidder" in str(exc), exc
    else:
        assert False, "an auction the closed forms do not describe was scored rather than refused"
    try:
        M.closed_forms(A.Auction(q=Q, c=C, reimbursement_policy=lambda round_index: 0.5 * C))
    except ValueError as exc:
        assert "no closed form" in str(exc), exc
    else:
        assert False, "a reimbursement policy with no closed form was scored rather than refused"


def test_a_belief_tremble_too_faint_to_reach_an_off_path_set_stops_the_score():
    """Scoring passes on the refusal rather than reporting a gain of zero at a set nothing reached."""
    auction = A.Auction(q=Q, c=0.4, reimbursement_policy="zero")  # dear enough that every bidder exits at once
    sigma = AN.profile(auction, AN.NEVER)
    try:
        M.score(auction, sigma, belief_tremble=1e-7)
    except ValueError as exc:
        assert "too faint" in str(exc), exc
    else:
        assert False, "a tremble that reaches no off-path set was scored rather than refused"


def test_a_run_is_scored_from_the_profile_it_returned():
    """The solver's own output goes into scoring as it stands, which is how the pipeline reads a run."""
    auction = A.Auction(q=Q, c=C, reimbursement_policy="zero")
    got = M.score(auction, M.solve(auction, seed=0, iterations=ITERATIONS).profile)
    assert got.verdict in tuple(M.Verdict)
    measured = (got.utility_loss, got.one_shot_gain, got.trembled_gain, got.on_path_distance, got.revenue_distance)
    assert all(np.isfinite(value) and value >= 0.0 for value in measured), got


# --- the grid the validation pipeline spends the budget on ------------------


def test_every_grid_cost_sits_where_its_placement_names():
    """Costs are placed against the region bound, which moves with q, so the names have to move with it."""
    for q, placement, c in V.grid():
        bound = AN.region(q)
        sits = {"inside": c < bound, "bound": c == bound, "outside": c > bound}
        assert sits[placement], (q, placement, c, bound)


def test_the_grid_reaches_the_half_of_the_region_the_incentive_bound_closes():
    """ADR-0004 measured at q = 0.5 alone, where cbar binds, so the other bound has to be exercised too."""
    probabilities = sorted({q for q, _, _ in V.grid()})
    assert [q for q in probabilities if AN.ic_bound(q) < AN.CBAR], probabilities
    assert [q for q in probabilities if AN.ic_bound(q) > AN.CBAR], probabilities


def test_the_targeted_closed_form_stops_being_an_equilibrium_past_the_incentive_bound():
    """Where separation is not preserved the derived verdict is out of reach, and different is the answer."""
    for q, _, c in V.grid():
        for rule in A.REIMBURSEMENT_POLICIES:
            auction = A.Auction(q=q, c=c, reimbursement_policy=rule)
            sigma = AN.profile(auction, AN.closed_form_strategy(rule, c))
            loss = float(AN.utility_loss(auction, sigma))
            assert (loss > AN.CERTIFIED) == (rule == "targeted" and c > AN.ic_bound(q)), (q, c, rule, loss)


if __name__ == "__main__":
    raise SystemExit(main(globals()))
