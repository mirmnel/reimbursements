"""Experiment 2's suite: the refined grid, and the revenue ranking read off learned play."""

from __future__ import annotations

import analysis as AN
import mccfr as M
import robustness as R
import validate as V
from game import REIMBURSEMENT_POLICIES
from mccfr import Verdict
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
    cost = AN.region(q) * R.PLACEMENTS[placement] if c is None else c
    return [run_row(rule, q, placement, cost, revenues[rule], seed, **overrides) for rule in REIMBURSEMENT_POLICIES]


# --- the grid the experiment spends the budget on ---------------------------


def test_every_grid_cost_sits_where_its_placement_names():
    """Costs are placed against the region bound, which moves with q, so the names move with it too."""
    for q, placement, c in R.grid():
        bound = AN.region(q)
        sits = {"inside": c < bound, "under": c < bound, "bound": c == bound, "over": c > bound, "outside": c > bound}
        assert sits[placement], (q, placement, c, bound)


def test_the_grid_refines_on_both_sides_of_the_bound_at_every_signal_probability():
    """Refinement is the ticket, so each signal probability needs a cost just under and just over the bound."""
    for q in R.SIGNAL_PROBABILITIES:
        near = [c / AN.region(q) for at, placement, c in R.grid() if at == q and placement in ("under", "over")]
        assert len(near) == 2, (q, near)
        assert min(near) < 1.0 < max(near), (q, near)
        assert all(abs(multiple - 1.0) <= 0.25 for multiple in near), (q, near)


def test_the_grid_refines_where_the_validation_grid_takes_one_point():
    """Experiment 2 extends experiment 1's grid rather than replacing it, so its placements contain them."""
    assert set(V.PLACEMENTS) <= set(R.PLACEMENTS)
    assert all(V.PLACEMENTS[placement] == R.PLACEMENTS[placement] for placement in V.PLACEMENTS)
    assert len(R.PLACEMENTS) > len(V.PLACEMENTS)


def test_the_grid_exercises_the_cost_root_and_the_incentive_bound():
    """The region has two bounds and the claim under test is refined at both, not at whichever binds at one q."""
    binds = {q: AN.ic_bound(q) < AN.CBAR for q in R.SIGNAL_PROBABILITIES}
    assert set(binds.values()) == {True, False}, binds


def test_the_run_count_is_the_grid_that_runs():
    """The hour is asserted against this number, so a regrown grid has to move it rather than pass quietly."""
    cells = R.grid()
    assert len(set(cells)) == len(cells), cells
    assert R.RUNS == len(REIMBURSEMENT_POLICIES) * len(cells) * R.SEEDS


# --- the ranking at learned play --------------------------------------------


def test_the_ranking_pairs_the_three_rules_at_one_cell_and_seed():
    """A ranking is a comparison, so the three rules it compares have to be the same cell and the same seed."""
    rows = at_cell({"zero": 0.20, "full": 0.18, "targeted": 0.21}) + at_cell(
        {"zero": 0.20, "full": 0.18, "targeted": 0.19}, seed=1
    )
    ranking = R.ranked(rows)
    assert len(ranking) == 2, ranking
    first, second = sorted(ranking, key=lambda row: row["seed"])
    assert (first["seed"], second["seed"]) == (0, 1)
    assert first["targeted_wins"] and not second["targeted_wins"], ranking


def test_the_ranking_reads_learned_revenue_and_not_the_closed_form():
    """The point of experiment 2 is the ranking at learned play, which can disagree with the derived one."""
    rows = at_cell({"zero": 0.21, "full": 0.18, "targeted": 0.20})  # inside the region, where theory says targeted
    (ranking,) = R.ranked(rows)
    assert ranking["theory_wins"] and ranking["margin_closed_form"] > 0, ranking
    assert not ranking["targeted_wins"], ranking
    assert ranking["argmax"] == "zero", ranking
    assert abs(float(ranking["margin"]) + 0.01) < 1e-12, ranking
    assert not ranking["agrees"], ranking


def test_the_ranking_reads_the_region_claim_at_the_bound_it_is_placed_on():
    """At the incentive bound separation still holds, so the theory has the targeted rule winning there."""
    q, c = 0.9, AN.ic_bound(0.9)
    (ranking,) = R.ranked(at_cell({"zero": 0.20, "full": 0.18, "targeted": 0.21}, q=q, c=c))
    assert c == AN.region(q) and ranking["theory_wins"], ranking
    (root,) = R.ranked(at_cell({"zero": 0.20, "full": 0.18, "targeted": 0.21}, q=0.5, c=AN.CBAR))
    assert not root["theory_wins"], root  # at the cost root the revenues tie rather than ranking


def test_the_ranking_refuses_rules_that_were_given_different_budgets():
    """A ranking is a comparison, so three runs at three budgets are not one, whatever they earned."""
    rows = at_cell({"zero": 0.20, "full": 0.18, "targeted": 0.21})
    rows[0]["iterations"] = 20_000
    try:
        R.ranked(rows)
    except AssertionError:
        return
    raise AssertionError("three rules at two budgets were ranked against each other")


def test_the_probe_rereads_the_cells_the_ranking_turns_on_at_the_other_budget():
    """ADR-0007 rests on the probe, so it has to run the bound and outside it at the bottom of the loss."""
    assert R.BOTTOM != M.BUDGET, R.BOTTOM
    assert set(R.PROBED) <= set(R.PLACEMENTS), R.PROBED
    assert 0 < R.PROBE_SEEDS <= R.SEEDS, R.PROBE_SEEDS


def test_the_ranking_carries_the_verdict_each_revenue_was_read_off():
    """A ranking read off a rule that landed on another equilibrium is a different claim, and has to say so."""
    rows = at_cell({"zero": 0.20, "full": 0.18, "targeted": 0.21}, verdict=Verdict.UNSOLVED)
    (ranking,) = R.ranked(rows)
    assert [ranking[f"verdict_{rule}"] for rule in REIMBURSEMENT_POLICIES] == [Verdict.UNSOLVED.value] * 3, ranking
    assert not ranking["certified"], ranking


def test_the_ranking_says_whether_the_targeted_target_is_still_an_equilibrium():
    """Past the incentive bound the profile the targeted revenue is stated at is not one anybody would play."""
    kept = at_cell({"zero": 0.20, "full": 0.18, "targeted": 0.21})
    lost = at_cell({"zero": 0.20, "full": 0.18, "targeted": 0.21}, closed_form_loss=10 * AN.CERTIFIED)
    assert R.ranked(kept)[0]["separation_survives"], kept
    assert not R.ranked(lost)[0]["separation_survives"], lost


def test_a_cell_missing_a_rule_is_refused_rather_than_ranked():
    """Two rules out of three is not a ranking, and reading one as though it were is the failure to avoid."""
    rows = [row for row in at_cell({"zero": 0.20, "full": 0.18, "targeted": 0.21}) if row["reimbursement_policy"] != "full"]
    try:
        R.ranked(rows)
    except ValueError:
        return
    raise AssertionError("a cell missing a rule was ranked")


if __name__ == "__main__":
    raise SystemExit(main(globals()))
