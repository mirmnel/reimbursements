"""Experiment 2's suite: the revenue ranking read off learned play."""

from __future__ import annotations

from auction import game as G
from auction import mccfr as M
from experiments import two_bidder as T
from auction.game import REIMBURSEMENT_POLICIES
from auction.mccfr import Verdict
from test_paper import main


def run_row(rule, q, placement, c, revenue, seed=0, verdict=Verdict.DERIVED, closed_form_loss=0.0):
    """One run row of the shape the pipeline writes, carrying what the ranking reads off it."""
    return {
        "reimbursement_policy": rule,
        "placement": placement,
        "c": c,
        "q": q,
        "seed": seed,
        "iterations": M.BUDGET,
        "verdict": verdict.value,
        "closed_form_loss": closed_form_loss,
        "Pi": revenue,
    }


def at_cell(revenues, q=0.5, placement="inside", c=None, seed=0, **overrides):
    """One cell and seed's three runs, one per rule, at the revenues given."""
    cost = G.region(q) * T.PLACEMENTS[placement] if c is None else c
    return [run_row(rule, q, placement, cost, revenues[rule], seed, **overrides) for rule in REIMBURSEMENT_POLICIES]


# --- the ranking at learned play --------------------------------------------


def test_the_ranking_pairs_the_three_rules_at_one_cell_and_seed():
    """A ranking is a comparison, so the three rules it compares have to be the same cell and the same seed."""
    rows = at_cell({"zero": 0.20, "full": 0.18, "targeted": 0.21}) + at_cell(
        {"zero": 0.20, "full": 0.18, "targeted": 0.19}, seed=1
    )
    ranking = T.ranked(rows)
    assert len(ranking) == 2, ranking
    first, second = sorted(ranking, key=lambda row: row["seed"])
    assert (first["seed"], second["seed"]) == (0, 1)
    assert first["targeted_wins"] and not second["targeted_wins"], ranking


def test_the_ranking_reads_learned_revenue_and_not_the_closed_form():
    """The point of experiment 2 is the ranking at learned play, which can disagree with the derived one."""
    rows = at_cell({"zero": 0.21, "full": 0.18, "targeted": 0.20})  # inside the region, where theory says targeted
    (ranking,) = T.ranked(rows)
    assert ranking["margin_closed_form"] > 0, ranking
    assert not ranking["targeted_wins"], ranking
    assert abs(float(ranking["margin"]) + 0.01) < 1e-12, ranking



def test_the_ranking_refuses_rules_that_were_given_different_budgets():
    """A ranking is a comparison, so three runs at three budgets are not one, whatever they earned."""
    rows = at_cell({"zero": 0.20, "full": 0.18, "targeted": 0.21})
    rows[0]["iterations"] = 20_000
    try:
        T.ranked(rows)
    except AssertionError:
        return
    raise AssertionError("three rules at two budgets were ranked against each other")


def test_the_ranking_carries_the_verdict_each_revenue_was_read_off():
    """A ranking read off a rule that landed on another equilibrium is a different claim, and has to say so."""
    rows = at_cell({"zero": 0.20, "full": 0.18, "targeted": 0.21}, verdict=Verdict.UNSOLVED)
    (ranking,) = T.ranked(rows)
    assert not ranking["certified"], ranking


def test_the_ranking_says_whether_the_targeted_target_is_still_an_equilibrium():
    """Past the incentive bound the profile the targeted revenue is stated at is not one anybody would play."""
    kept = at_cell({"zero": 0.20, "full": 0.18, "targeted": 0.21})
    lost = at_cell({"zero": 0.20, "full": 0.18, "targeted": 0.21}, closed_form_loss=10 * T.CERTIFIED)
    assert T.ranked(kept)[0]["separation_survives"], kept
    assert not T.ranked(lost)[0]["separation_survives"], lost



if __name__ == "__main__":
    raise SystemExit(main(globals()))
