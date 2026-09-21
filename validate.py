"""The validation experiment: whether MCCFR recovers the two-bidder closed forms from sampled play."""

from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from analysis import CERTIFIED, PI, ic_bound, region, utility_loss, write
from game import REIMBURSEMENT_POLICIES, Auction
from mccfr import BUDGET, CALIBRATED_Q, STAY_BAR, Verdict, closed_forms, score, solve, stay_schedule

SEEDS = 10
HOUR = 3600.0

# Costs are placed against analysis.region(q), since which of its two bounds binds moves with q.
PLACEMENTS = {"inside": 0.5, "bound": 1.0, "outside": 1.5}

# Sampled rather than crossed. Three costs by three signal probabilities is 270 runs, which fills the
# hour on its own and leaves nothing for the calibration re-read the new signal probabilities need.
# q = 0.5 keeps ticket 08's three placements, and the widened rows take the bound and outside it.
CELLS = {
    0.5: ("inside", "bound", "outside"),  # cbar binds, the incentive bound sitting far above at 0.125
    0.75: ("bound", "outside"),  # cbar still binds, at the same two costs as the row above
    0.9: ("bound", "outside"),  # past q = 0.783, so the incentive bound binds instead, at 0.025
}


def costs(q: float, placements: Dict[str, float] = PLACEMENTS) -> Dict[str, float]:
    """Every placement's cost at one signal probability, whether or not the grid runs all of them."""
    return {placement: multiple * region(q) for placement, multiple in placements.items()}


def cells_of(
    cells: Dict[float, Sequence[str]], placements: Dict[str, float] = PLACEMENTS
) -> List[Tuple[float, str, float]]:
    """A grid's cells, each cost placed as a multiple of the region bound at its signal probability."""
    return [(q, placement, costs(q, placements)[placement]) for q, named in cells.items() for placement in named]


def grid() -> List[Tuple[float, str, float]]:
    """The cells the experiment runs, each cost placed as a multiple of the region bound at its q."""
    return cells_of(CELLS)


RUNS = len(REIMBURSEMENT_POLICIES) * len(grid()) * SEEDS


def validated(auction: Auction, placement: str, seed: int, iterations: int = BUDGET) -> Dict[str, object]:
    """One run at the calibrated budget, scored against the closed forms and reported on its own."""
    started = time.perf_counter()
    solved = solve(auction, seed=seed, iterations=iterations, trace_every=iterations // 2, symmetry_restricted=True)
    seconds = time.perf_counter() - started
    scored = score(auction, solved.profile)
    derived, _ = closed_forms(auction)
    schedule = stay_schedule(auction, solved.profile)
    target = stay_schedule(auction, derived)
    reached = solved.visits[solved.visits > 0]
    assert reached.size, auction  # nothing below is worth reading off a run that reached nothing
    thin, median, thick = np.quantile(reached, [0.1, 0.5, 0.9])
    row: Dict[str, object] = {
        "reimbursement_policy": auction.reimbursement_policy,
        "placement": placement,
        "c": auction.c,
        "q": auction.q,
        "seed": seed,
        "iterations": solved.iterations,
        # Whether ADR-0005 measured the budget at this signal probability, rather than it being borrowed.
        "budget_calibrated": auction.q in CALIBRATED_Q,
        "seconds": seconds,
        "verdict": scored.verdict.value,
        "utility_loss": scored.utility_loss,
        # Halfway, so a run that was still descending when the budget ran out is not read as a plateau.
        "utility_loss_halfway": score(auction, solved.averages[0]).utility_loss,
        # The target's own loss, which past the incentive bound puts the derived verdict out of reach.
        "closed_form_loss": float(utility_loss(auction, derived)),
        "on_path_distance": scored.on_path_distance,
        "off_path_distance": scored.off_path_distance,
        "one_shot_gain": scored.one_shot_gain,
        "trembled_gain": scored.trembled_gain,
        "Pi": scored.revenue,
        "Pi_closed_form": PI[auction.reimbursement_policy](auction.c, auction.q),
        "revenue_distance": scored.revenue_distance,
        "info_sets": len(auction.info_sets),
        "visited": int((solved.visits > 0).sum()),
        "visits_p10": thin,
        "visits_median": median,
        "visits_p90": thick,
    } | schedule

    assert all(0.0 <= value <= 1.0 for value in schedule.values()), row
    assert float(row["open_high"]) > float(row["open_low"]), row  # a high signal is worth more
    assert scored.trembled_gain >= scored.one_shot_gain, row  # the same maximum over more sets
    assert abs(scored.revenue - float(row["Pi_closed_form"])) == scored.revenue_distance, row
    if scored.verdict is Verdict.DERIVED:
        # A second reading of the same profile, so the reported schedule cannot disagree with the verdict.
        assert all(abs(schedule[name] - target[name]) <= STAY_BAR for name in target), (row, target)
    return row


def cell(rows: Sequence[Dict[str, object]], rule: str, q: float, placement: str) -> List[Dict[str, object]]:
    """The seeds run at one grid cell, which is the unit every count here is read at."""
    return [
        row
        for row in rows
        if row["reimbursement_policy"] == rule and row["q"] == q and row["placement"] == placement
    ]


def report(rows: Sequence[Dict[str, object]], rule: str, q: float, placement: str) -> None:
    """Print what the seeds at one cell did, as counts rather than as an average over equilibria."""
    at = cell(rows, rule, q, placement)
    counts = " ".join(
        f"{verdict.value} {sum(row['verdict'] == verdict.value for row in at)}" for verdict in Verdict
    )
    reach = lambda key: (min(float(row[key]) for row in at), max(float(row[key]) for row in at))
    print(
        f"  {rule:9s} q={q:.2f} {placement:8s} c={float(at[0]['c']):.4f}  {counts}"
        + "  on-path %.4f to %.4f" % reach("on_path_distance")
        + "  loss %.2e to %.2e" % reach("utility_loss")
    )


def unreachable_targets(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """The rows whose target is not itself an equilibrium, where derived cannot be the answer."""
    return [row for row in rows if float(row["closed_form_loss"]) > CERTIFIED]


def sweep(cells: Sequence[Tuple[float, str, float]]) -> List[Dict[str, object]]:
    """Every published rule at every cell at every seed, each run scored and reported on its own."""
    rows = [
        validated(Auction(q=q, c=c, reimbursement_policy=rule), placement, seed)
        for rule in REIMBURSEMENT_POLICIES
        for (q, placement, c) in cells
        for seed in range(SEEDS)
    ]
    for rule in REIMBURSEMENT_POLICIES:
        for q, placement, _ in cells:
            report(rows, rule, q, placement)
    return rows


def check_unreachable_targets(rows: Sequence[Dict[str, object]]) -> None:
    """Past the incentive bound the scored-against profile is not an equilibrium, so derived is out of reach."""
    beyond = unreachable_targets(rows)
    assert {row["reimbursement_policy"] for row in beyond} == {"targeted"}, beyond
    assert all(float(row["c"]) > ic_bound(float(row["q"])) for row in beyond), beyond
    assert all(row["verdict"] != Verdict.DERIVED.value for row in beyond), beyond
    print(f"  targets that are not themselves equilibria: {len(beyond)} rows, none of them derived")


def check_borrowed_budget(rows: Sequence[Dict[str, object]], probabilities: Sequence[float]) -> None:
    """The budget was measured at some of these signal probabilities and borrowed at the rest."""
    borrowed = sorted({float(row["q"]) for row in rows if not row["budget_calibrated"]})
    assert borrowed == sorted(set(probabilities) - set(CALIBRATED_Q)), borrowed
    print(f"  signal probabilities running on a borrowed budget: {borrowed}")


def validation() -> List[Dict[str, object]]:
    """Ten seeds at each grid cell under each published rule, every one of them reported separately."""
    rows = sweep(grid())
    # Written before the claims are checked, as calibrate.main is, since an hour of runs is worth
    # keeping even when a claim no longer holds on them.
    write("validation.csv", rows)

    seen = [(row["reimbursement_policy"], row["q"], row["placement"], row["seed"]) for row in rows]
    assert len(set(seen)) == len(rows) == RUNS, len(rows)
    assert sum(float(row["seconds"]) for row in rows) < HOUR, sum(float(row["seconds"]) for row in rows)
    # The claim the experiment exists to back, which ADR-0004 measured certifying on ten of ten seeds.
    # Certification rather than the verdict, since a different equilibrium is a result and not a failure.
    at_target = cell(rows, "zero", 0.5, "inside")
    assert all(float(row["utility_loss"]) <= CERTIFIED for row in at_target), at_target
    check_unreachable_targets(rows)
    check_borrowed_budget(rows, tuple(CELLS))
    return rows


if __name__ == "__main__":
    print("== validation ==")
    validation()
