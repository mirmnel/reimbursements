"""Experiment 2: the three reimbursement rules over a refined grid, and the revenue ranking at learned play."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from analysis import CERTIFIED, PI, ic_bound, region, region_claim, write
from game import REIMBURSEMENT_POLICIES, Auction
from mccfr import Verdict
from validate import HOUR, SEEDS, cells_of, check_borrowed_budget, check_unreachable_targets, sweep, validated

BOTTOM = 20_000  # where ADR-0005 found the median loss bottoms at the widened signal probabilities
PROBED = ("bound", "outside")  # the two cells the budget probe rereads, being where the ranking turns
PROBE_SEEDS = 3

# Experiment 1's three placements with one added either side of the bound, where the ranking turns.
PLACEMENTS = {"inside": 0.5, "under": 0.9, "bound": 1.0, "over": 1.1, "outside": 1.5}

# One signal probability per bound of analysis.region: at 0.5 the cost root binds, at 0.9 the
# incentive bound does, so each bound is refined where it closes the region.
SIGNAL_PROBABILITIES = (0.5, 0.9)
CELLS = {q: tuple(PLACEMENTS) for q in SIGNAL_PROBABILITIES}  # crossed, unlike the sampled grid of ticket 14


def grid() -> List[Tuple[float, str, float]]:
    """The cells the experiment runs, each cost placed as a multiple of the region bound at its q."""
    return cells_of(CELLS, PLACEMENTS)


RUNS = len(REIMBURSEMENT_POLICIES) * len(grid()) * SEEDS


def paired(rows: Sequence[Dict[str, object]]) -> Dict[Tuple[object, object, object], Dict[str, Dict[str, object]]]:
    """Each cell and seed's runs keyed by rule, which is the unit a ranking can be read at."""
    runs: Dict[Tuple[object, object, object], Dict[str, Dict[str, object]]] = {}
    for row in rows:
        runs.setdefault((row["q"], row["placement"], row["seed"]), {})[str(row["reimbursement_policy"])] = row
    return runs


def ranked(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """The revenue ranking at learned play, one row per cell and seed, with the derived one beside it."""
    ranking = []
    for (q, placement, seed), runs in paired(rows).items():
        if set(runs) != set(REIMBURSEMENT_POLICIES):
            raise ValueError(f"cell q={q} {placement} seed {seed} ran {sorted(runs)}, not the three rules")
        q, c = float(q), float(runs["targeted"]["c"])
        budgets = {int(runs[rule]["iterations"]) for rule in REIMBURSEMENT_POLICIES}
        # A ranking is a comparison, so the three rules have to have been given the same budget.
        assert len(budgets) == 1, (q, placement, seed, budgets)
        learned = {rule: float(runs[rule]["Pi"]) for rule in REIMBURSEMENT_POLICIES}
        derived = {rule: float(PI[rule](c, q)) for rule in REIMBURSEMENT_POLICIES}
        margin = lambda earned: earned["targeted"] - max(earned["zero"], earned["full"])
        row: Dict[str, object] = {
            "q": q,
            "placement": placement,
            "c": c,
            "seed": int(seed),
            "iterations": budgets.pop(),
            **{f"Pi_{rule}": learned[rule] for rule in REIMBURSEMENT_POLICIES},
            "argmax": max(learned, key=learned.get),
            "margin": margin(learned),
            "targeted_wins": margin(learned) > 0.0,
            "argmax_closed_form": max(derived, key=derived.get),
            "margin_closed_form": margin(derived),
            "theory_wins": region_claim(c, q),
            "theory_bound": region(q),
            **{f"verdict_{rule}": runs[rule]["verdict"] for rule in REIMBURSEMENT_POLICIES},
            "certified": all(runs[rule]["verdict"] != Verdict.UNSOLVED.value for rule in REIMBURSEMENT_POLICIES),
            # Whether the targeted revenue is stated at an equilibrium of this auction at all.
            "separation_survives": float(runs["targeted"]["closed_form_loss"]) <= CERTIFIED,
        }
        row["agrees"] = row["targeted_wins"] == row["theory_wins"]
        assert not row["targeted_wins"] or row["argmax"] == "targeted", row
        ranking.append(row)
    return ranking


def report_ranking(ranking: Sequence[Dict[str, object]], q: float, placement: str) -> None:
    """Print how the seeds at one cell ranked, as counts rather than as an average over seeds."""
    at = [row for row in ranking if row["q"] == q and row["placement"] == placement]
    margins = [float(row["margin"]) for row in at]
    print(
        f"  q={q:.2f} {placement:8s} c={float(at[0]['c']):.4f}"
        + f"  targeted wins {sum(bool(row['targeted_wins']) for row in at)}/{len(at)}"
        + "  learned margin %+.4f to %+.4f" % (min(margins), max(margins))
        + "  closed form %+.4f" % float(at[0]["margin_closed_form"])
        + f"  theory says {'targeted' if at[0]['theory_wins'] else 'not targeted'}"
        + f"  certified {sum(bool(row['certified']) for row in at)}/{len(at)}"
    )


def probe(q: float = 0.9, iterations: int = BOTTOM) -> List[Dict[str, object]]:
    """The same ranking at the budget that signal probability wants, which ADR-0007 rests its decision on."""
    cells = [(at, placement, c) for (at, placement, c) in grid() if at == q and placement in PROBED]
    rows = [
        validated(Auction(q=at, c=c, reimbursement_policy=rule), placement, seed, iterations=iterations)
        for rule in REIMBURSEMENT_POLICIES
        for (at, placement, c) in cells
        for seed in range(PROBE_SEEDS)
    ]
    ranking = ranked(rows)
    for placement in PROBED:
        report_ranking(ranking, q, placement)
    write("budget_probe.csv", ranking)
    return ranking


def check_probe(ranking: Sequence[Dict[str, object]], probed: Sequence[Dict[str, object]]) -> None:
    """Whether the budget being past the bottom of the loss is what decided the ranking."""
    at = {(row["q"], row["placement"], row["seed"]): row for row in ranking}
    for row in probed:
        budgeted = at[(row["q"], row["placement"], row["seed"])]
        assert budgeted["iterations"] != row["iterations"], (row, budgeted)
        assert budgeted["targeted_wins"] == row["targeted_wins"], (row, budgeted)
    print(f"  the ranking is the same at {probed[0]['iterations']} iterations on {len(probed)} probed rows")


def experiment() -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Ten seeds at each cell of the refined grid under each rule, and the ranking those runs imply."""
    cells = grid()
    rows = sweep(cells)
    # Written before the ranking is read off them, as calibrate.main is, since an hour of runs is
    # worth keeping even when the claim read off them no longer holds.
    write("robustness.csv", rows)
    ranking = ranked(rows)
    for q, placement, _ in cells:
        report_ranking(ranking, q, placement)
    write("learned_ranking.csv", ranking)
    return rows, ranking


def check(rows: Sequence[Dict[str, object]], ranking: Sequence[Dict[str, object]]) -> None:
    """What experiment 2 claims, checked once the measurements behind it are on disk."""
    seen = [(row["reimbursement_policy"], row["q"], row["placement"], row["seed"]) for row in rows]
    assert len(set(seen)) == len(rows) == RUNS, len(rows)
    assert sum(float(row["seconds"]) for row in rows) < HOUR, sum(float(row["seconds"]) for row in rows)
    assert len(ranking) == len(grid()) * SEEDS, len(ranking)

    # A rule that found another equilibrium is a result about the rule, so it is counted rather than
    # asserted away. What is asserted is that such a run is certified, which is what makes it one.
    others = [row for row in rows if row["verdict"] == Verdict.DIFFERENT.value]
    assert all(float(row["utility_loss"]) <= CERTIFIED for row in others), others
    print(f"  runs landing on another equilibrium: {len(others)} of {len(rows)}")

    check_unreachable_targets(rows)
    check_borrowed_budget(rows, SIGNAL_PROBABILITIES)
    past = [row for row in ranking if float(row["c"]) > ic_bound(float(row["q"]))]
    assert not any(row["separation_survives"] for row in past), past

    # The two claims the grid is placed to test. Well inside the region the ranking survives being
    # learned rather than derived, and outside it the targeted rule does not win at learned play.
    inside = [row for row in ranking if row["placement"] == "inside" and row["q"] == 0.5]
    assert len(inside) == SEEDS and all(row["targeted_wins"] for row in inside), inside
    outside = [row for row in ranking if row["placement"] == "outside"]
    assert len(outside) == len(SIGNAL_PROBABILITIES) * SEEDS, outside
    assert not any(row["targeted_wins"] for row in outside), outside

    disagreed = [row for row in ranking if not row["agrees"]]
    print(f"  cells where learned play and the region claim disagree: {len(disagreed)} of {len(ranking)}")


if __name__ == "__main__":
    print("== robustness ==")
    rows, ranking = experiment()
    check(rows, ranking)
    print("== budget probe ==")
    check_probe(ranking, probe())
