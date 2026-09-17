"""The sampled suite: what independent learners do, with its own tolerances, apart from the exact one."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import analysis as AN
import solver as A
import learning as L
import symmetric as S
from test_paper import main

C, Q = 0.025, 0.5


# --- each bidder learns from its own experience only ------------------------


def test_the_learner_never_reads_a_payoff_it_was_not_handed():
    """With every payoff function disabled, a run still completes, so no counterfactual is ever read."""
    auction = A.Auction(n=4, q=Q, c=C)
    saved = (L.evaluate, S.tagged_values, S.seller_values, S.gains, S.utility_loss)

    def refuse(*arguments, **keywords):
        raise AssertionError("the learner asked the game for a payoff")

    L.evaluate = S.tagged_values = S.seller_values = S.gains = S.utility_loss = refuse
    try:
        learned = L.learn(auction, seed=0, iterations=20)
    finally:
        L.evaluate, S.tagged_values, S.seller_values, S.gains, S.utility_loss = saved
    assert learned.profile.shape == (len(auction.info_sets),)


def test_each_bidder_holds_its_own_table():
    """Bidders are not forced to agree, so whether they agree is something the run can report."""
    auction = A.Auction(n=4, q=Q, c=C)
    learned = L.learn(auction, seed=0, iterations=60)
    tables = L.per_bidder(auction, learned.profile)
    assert np.max(np.abs(tables - tables[0])) > 1e-6, tables


def test_strategies_start_at_random_rather_than_uniformly():
    """Before any experience the bidders already differ, so the basins are visible rather than shared."""
    auction = A.Auction(n=4, q=Q, c=C)
    start = L.learn(auction, seed=3, iterations=0).profile
    assert np.max(np.abs(start - 0.5)) > 0.05, start
    tables = L.per_bidder(auction, start)
    assert np.max(np.abs(tables - tables[0])) > 0.05, tables


def test_the_same_seed_repeats_and_a_different_seed_does_not():
    """The seed is recorded and it is the whole of the randomness, so a run can be rerun."""
    auction = A.Auction(n=4, q=Q, c=C)
    once, again = L.learn(auction, seed=7, iterations=40), L.learn(auction, seed=7, iterations=40)
    other = L.learn(auction, seed=8, iterations=40)
    assert once.seed == 7
    assert np.array_equal(once.profile, again.profile)
    assert not np.array_equal(once.profile, other.profile)


def test_the_result_carries_the_traces_convergence_is_judged_on():
    """The time-average, the raw iterates and the dispersion all come back from the one run."""
    auction = A.Auction(n=4, q=Q, c=C)
    learned = L.learn(auction, seed=0, iterations=40, trace_every=4)
    buckets = auction.rounds * (auction.n + 1) * 2
    assert learned.iterates.shape == (10, auction.n, buckets)
    assert learned.averages.shape == learned.iterates.shape
    assert 0.0 <= learned.dispersion <= 1.0
    assert learned.visits.shape == (len(auction.info_sets),)  # coverage rides the same object


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


# --- what a run discloses about its coverage --------------------------------


def test_a_run_reports_how_many_information_sets_it_visited():
    """At two bidders every information set carries reach and the run reaches all of them."""
    auction = A.Auction(n=2, q=Q, c=C)
    learned = L.learn(auction, seed=0, iterations=40)
    reachable = int(np.count_nonzero(A.counterfactual_reach(auction, learned.profile)))
    assert reachable == len(auction.info_sets)
    assert learned.visited == reachable, (learned.visited, reachable)


def test_a_run_visits_only_the_sets_its_own_play_actually_reaches():
    """The sets it misses are the ones the profile barely reaches, which is the shape of thin coverage."""
    auction = A.Auction(n=6, q=Q, c=C)
    learned = L.learn(auction, seed=0, iterations=40)
    weights = A.counterfactual_reach(auction, learned.profile)
    seen = learned.visits > 0
    assert learned.visited <= int(np.count_nonzero(weights))
    assert np.median(weights[seen]) > 5 * np.median(weights[~seen])


def test_a_run_at_scale_leaves_most_of_the_space_near_empty():
    """At ten bidders a full run reaches a fraction of the sets and puts a handful of visits in each."""
    auction = A.Auction(n=10, q=Q, c=C)
    learned = L.learn(auction, seed=0)
    assert 0 < learned.visited < 0.75 * len(auction.info_sets), (learned.visited, len(auction.info_sets))
    assert learned.visit_quantiles(0.5) < 20  # a median that thin is why the budget rises at this scale


def test_a_run_reports_the_distribution_of_visits_and_not_only_a_total():
    """Visits are carried per information set, so a median and a thin tail are both readable."""
    auction = A.Auction(n=6, q=Q, c=C)
    learned = L.learn(auction, seed=0, iterations=40)
    assert learned.visits.shape == (len(auction.info_sets),)
    assert learned.visits.sum() > learned.visited  # a total alone would hide the spread
    low, median, high = learned.visit_quantiles(0.1, 0.5, 0.9)
    assert low <= median <= high
    assert high > median, (low, median, high)


def test_coverage_rises_with_the_episode_budget():
    """Buying more experience at the same seed reaches more sets and puts more visits into each."""
    auction = A.Auction(n=6, q=Q, c=C)
    thin = L.learn(auction, seed=0, iterations=40, episodes=64)
    thick = L.learn(auction, seed=0, iterations=40, episodes=640)
    assert thick.visited > thin.visited, (thin.visited, thick.visited)
    assert thick.visit_quantiles(0.5) > thin.visit_quantiles(0.5)


def test_ten_bidder_runs_that_reach_a_table_buy_the_raised_budget():
    """The headline scale gets ten times the experience, and the routine counts keep their budget."""
    assert L.budget(10) == 10 * L.EPISODES
    assert all(L.budget(n) == L.EPISODES for n in (2, 4, 6))


# --- what the learners arrive at --------------------------------------------


def test_the_learners_recover_the_two_bidder_equilibrium():
    """Where the paper already knows the answer, independent learners from random starts find it."""
    auction = A.Auction(n=2, q=Q, c=C, reimbursement_policy="targeted")
    losses = [L.score(auction, L.learn(auction, seed=seed)).utility_loss for seed in range(5)]
    assert max(losses) < 5e-3, losses
    assert np.median(losses) < 2e-3, losses


def test_the_learners_reach_a_near_equilibrium_at_four_bidders_but_not_a_symmetric_one():
    """At four bidders the learners settle close to equilibrium, and symmetrising them moves them away."""
    auction = A.Auction(n=4, q=Q, c=C, reimbursement_policy="targeted")
    for seed in range(5):
        learned = L.learn(auction, seed=seed)
        as_it_stands = AN.utility_loss(auction, learned.profile)
        symmetrised = AN.utility_loss(auction, S.symmetrise(auction, learned.profile))
        assert as_it_stands < 2e-3, (seed, as_it_stands)
        assert symmetrised > 4 * as_it_stands, (seed, as_it_stands, symmetrised)


def test_a_dispersed_profile_is_flagged_rather_than_folded_into_the_number():
    """One bidder behaving unlike the rest is reported as dispersion, not averaged into the answer."""
    auction = A.Auction(n=4, q=Q, c=C)
    apart = np.array([1.0 if bidder == 0 else 0.0 for bidder, _, _ in auction.info_sets])
    reach = S.bucket_reach(auction, S.symmetrise(auction, apart))
    assert L.spread(auction, apart, reach) > L.DISPERSED
    assert L.spread(auction, S.symmetrise(auction, apart), reach) < 1e-12


def test_a_cycling_run_is_labelled_rather_than_averaged_away():
    """Iterates that keep swinging late in the run are reported as cycling, not folded into an average."""
    auction = A.Auction(n=2, q=Q, c=C)
    settled = L.learn(auction, seed=0, iterations=200)
    reach = S.bucket_reach(auction, S.symmetrise(auction, settled.profile))
    swinging = np.zeros_like(settled.iterates)
    swinging[1::2] = 1.0
    assert L.cycles(auction, replace(settled, iterates=swinging), reach)
    assert not L.cycles(auction, settled, reach)


if __name__ == "__main__":
    raise SystemExit(main(globals()))
