"""Experiment 2's suite: the revenue ranking over the price-only corners, read off learned play."""

from __future__ import annotations

from auction import game as G
from auction import mccfr as M
from experiments import two_bidder as T
from auction.game import REIMBURSEMENT_POLICIES
from auction.mccfr import Verdict
from test_paper import main


def run_row(rule, q, placement, c, revenue, seed, verdict=Verdict.DERIVED):
    """One run row of the shape the pipeline writes, carrying what the ranking reads off it."""
    return {
        "reimbursement_policy": rule,
        "placement": placement,
        "c": c,
        "q": q,
        "seed": seed,
        "iterations": M.BUDGET,
        "verdict": verdict.value,
        "Pi": revenue,
        "Pi_closed_form": 0.20 + 0.01 * (rule == "backloaded"),
    }


def at_cell(q=0.7, placement="inside", seed=0, unsolved=(), **revenues):
    """One cell and seed's runs, one per policy, back-loaded earning 0.21 and the rest 0.20 unless given."""
    cost = G.region(q) * T.PLACEMENTS[placement]
    earned = {rule: 0.20 for rule in REIMBURSEMENT_POLICIES} | {"backloaded": 0.21} | revenues
    return [
        run_row(rule, q, placement, cost, earned[rule], seed, Verdict.UNSOLVED if rule in unsolved else Verdict.DERIVED)
        for rule in REIMBURSEMENT_POLICIES
    ]


def raises(claim, *args):
    """Whether ``claim`` refuses its arguments."""
    try:
        claim(*args)
    except AssertionError:
        return True
    return False


# --- the ranking at learned play --------------------------------------------


def test_the_ranking_pairs_every_policy_at_one_cell_and_seed():
    """A ranking is a comparison, so the policies it compares have to be the same cell and the same seed."""
    ranking = T.ranked(at_cell() + at_cell(seed=1, backloaded=0.19))
    first, second = sorted(ranking, key=lambda row: row["seed"])
    assert (first["seed"], second["seed"]) == (0, 1)
    assert first["backloaded_wins"] and not second["backloaded_wins"], ranking


def test_back_loading_has_to_beat_every_corner_and_not_only_the_named_policies():
    """The theorem claims the whole price-only class, so one unnamed corner earning more is a loss."""
    (ranking,) = T.ranked(at_cell(war_0c0=0.22))
    assert not ranking["backloaded_wins"], ranking


def test_the_ranking_reads_learned_revenue_and_not_the_closed_form():
    """The point of experiment 2 is the ranking at learned play, which can disagree with the derived one."""
    (ranking,) = T.ranked(at_cell(backloaded=0.19))
    assert ranking["margin_closed_form"] > 0 > ranking["margin"], ranking
    assert abs(ranking["margin"] + 0.01) < 1e-12, ranking


def test_the_ranking_refuses_policies_that_were_given_different_budgets():
    """Runs at different iteration limits are not one comparison, whatever they earned."""
    rows = at_cell()
    rows[0]["iterations"] = 20_000
    assert raises(T.ranked, rows)


def test_a_seed_where_any_policy_missed_an_equilibrium_is_recorded_but_not_counted():
    """Revenue off a profile that is no equilibrium predicts nothing, so that seed cannot count as a win."""
    ranking = T.ranked(at_cell() + at_cell(seed=1, unsolved=("war_cc0",)))
    assert len(ranking) == 2, ranking
    counted = {row["seed"]: row["counted"] for row in ranking}
    assert counted == {0: True, 1: False}, ranking
    assert not [row for row in ranking if row["seed"] == 1][0]["backloaded_wins"], ranking


if __name__ == "__main__":
    raise SystemExit(main(globals()))
