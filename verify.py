"""Diagnostics, benchmark strategies, closed forms, and the experiments behind the paper's tables."""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from auctionrl import POLICIES, counterfactual_gain, counterfactual_reach, evaluate, learn
from game import Auction, Outcome, Strategy


RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def profile(auction: Auction, *strategies: Strategy) -> np.ndarray:
    """Flat behaviour profile from one strategy per bidder, or from a lone strategy shared by all."""
    if len(strategies) == 1:
        strategies *= auction.n
    return np.array([float(strategies[bidder](state, signal)) for (bidder, state, signal) in auction.info_sets])


def as_strategy(auction: Auction, sigma: np.ndarray) -> Strategy:
    """Read bidder 0's rows of a profile back as a strategy (valid when symmetric)."""
    return lambda state, signal: float(sigma[auction.info_index[(0, state, signal)]])


SEPARATING: Strategy = lambda state, signal: float(signal == 1)  # L exits at p = 0, H stays
ALWAYS: Strategy = lambda state, signal: 1.0
NEVER: Strategy = lambda state, signal: 0.0


def war_of_attrition(c: float) -> Strategy:
    """The low type exits everywhere, and the high type stays at the opening and then mixes."""
    schedule = (1.0, 1 - 4 * c / 3, 1.0, 1 - 4 * c)
    return lambda state, signal: schedule[state[0]] if signal == 1 else 0.0

# ---------------------------------------------------------------------------
# Best responses and equilibrium quality
# ---------------------------------------------------------------------------


def best_response(auction: Auction, sigma: np.ndarray, bidder: int) -> np.ndarray:
    """Exact best response of ``bidder``, by one backward sweep on ``counterfactual_gain``."""
    response = sigma.copy()
    for position in reversed(range(len(auction.info_sets))):  # info sets are round-sorted, so this sweeps backwards
        if auction.info_sets[position][0] == bidder:
            gain = counterfactual_gain(auction, response, position)
            response[position] = float(gain > 0.0)
    return response


def epsilon(auction: Auction, sigma: np.ndarray) -> float:
    """Utility loss: the largest predicted gain any bidder gets from a full best response."""
    baseline = evaluate(auction, sigma).payoff
    return max(
        evaluate(auction, best_response(auction, sigma, bidder)).payoff[bidder] - baseline[bidder]
        for bidder in range(auction.n)
    )


def max_regret(auction: Auction, sigma: np.ndarray) -> float:
    """Largest one-shot deviation gain at any information set of positive opponent reach."""
    weights, worst = counterfactual_reach(auction, sigma), 0.0
    for position in range(len(auction.info_sets)):
        if weights[position] > 1e-12:
            gain = counterfactual_gain(auction, sigma, position) / weights[position]
            stay_prob = sigma[position]
            worst = max(worst, (1 - stay_prob) * max(gain, 0.0) + stay_prob * max(-gain, 0.0))
    return worst


def empirical_game(auction: Auction, strategies: Dict[str, Strategy]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Symmetric empirical payoff matrix over a strategy set, and each profile's regret."""
    names = list(strategies)
    payoffs = np.array(
        [
            [evaluate(auction, profile(auction, strategies[row], strategies[column])).payoff[0] for column in names]
            for row in names
        ]
    )
    return names, payoffs, payoffs.max(axis=0) - np.diag(payoffs)

# ---------------------------------------------------------------------------
# Closed forms for the two-bidder example (ground truth)
# ---------------------------------------------------------------------------

sigma1 = lambda c: 1 - 4 * c / 3
sigma3 = lambda c: 1 - 4 * c
R0 = lambda c: 0.5 * (1 - sigma1(c) ** 2) + sigma1(c) ** 2 * (1.5 + 0.5 * sigma3(c) ** 2)
L0 = lambda q: 8 * q**2 + 2 * q * (1 - q)  # expected staying bidder-rounds, frictionless
PI = {
    "zero": lambda c, q: q**2 * R0(c),
    "full": lambda c, q: 2 * q**2 - c * L0(q),
    "targeted": lambda c, q: 2 * q**2 * (1 - 3 * c),
}
ic_bound = lambda q: (1 - q) / 2  # r^T preserves separation iff c <= (1-q)/2


def cbar() -> float:
    """The unique root of the revenue-difference cubic below one quarter, which is 0.108629."""
    return float(
        min(root.real for root in np.roots([64, -128, 96, -9]) if abs(root.imag) < 1e-9 and 0 < root.real < 0.25)
    )


def region(q: float) -> float:
    """The cost bound below which the targeted policy beats both comparison policies."""
    return min(cbar(), ic_bound(q))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def write(name: str, rows: List[Dict[str, object]]) -> None:
    """Write ``rows`` as a CSV file of the given name under ``results``."""
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def solved(c: float, q: float, rho: float = 0.0) -> Dict[str, Tuple[Auction, np.ndarray, Outcome]]:
    """Learn an equilibrium under each reimbursement policy, and evaluate it."""
    learned = {}
    for policy in POLICIES:
        auction = Auction(q=q, c=c, policy=policy, rho=rho)
        sigma = learn(auction)
        learned[policy] = (auction, sigma, evaluate(auction, sigma))
    return learned


def equilibria(c: float = 0.05, q: float = 0.5) -> List[Dict[str, object]]:
    """The learned equilibrium under each policy, scored against the closed forms."""
    rows, learned = [], solved(c, q)
    for policy, (auction, sigma, outcome) in learned.items():
        row: Dict[str, object] = {"policy": policy, "c": c, "q": q}
        for signal, label in ((1, "H"), (0, "L")):
            row |= {
                f"sigma_{label}_t{round_index}": sigma[auction.info_index[(0, (round_index, (0, 1)), signal)]]
                for round_index in range(auction.rounds)
            }
        gap, error = epsilon(auction, sigma), abs(outcome.revenue - PI[policy](c, q))
        row |= {
            "epsilon": gap,
            "max_regret": max_regret(auction, sigma),
            "price": outcome.price,
            "reimbursement": outcome.reimbursement,
            "stay_rounds": outcome.stay_rounds,
            "Pi": outcome.revenue,
            "Pi_closed_form": PI[policy](c, q),
            "abs_error": error,
        }
        assert error < 1e-5, row
        assert gap < 1e-9, row
        rows.append(row)
    write("equilibria.csv", rows)
    return rows


def compare(c: float, q: float, rho: float = 0.0) -> Dict[str, object]:
    """Net revenue per policy at one setting, and the worst utility loss behind it."""
    learned = solved(c, q, rho)
    revenue = {policy: outcome.revenue for policy, (_, _, outcome) in learned.items()}
    return {
        **{f"Pi_{policy}": revenue[policy] for policy in POLICIES},
        "argmax": max(revenue, key=revenue.get),
        "targeted_wins": revenue["targeted"] > max(revenue["zero"], revenue["full"]),
        "max_epsilon": max(epsilon(auction, sigma) for auction, sigma, _ in learned.values()),
    }


def ranking(c: Optional[float] = None, q: Optional[float] = None) -> List[Dict[str, object]]:
    """Revenue ranking over a ``(c, q)`` grid, with behaviour learned rather than assumed."""
    costs = (c,) if c is not None else np.linspace(0.01, 0.24, 24)
    signal_probs = (q,) if q is not None else (0.25, 0.5, 0.75)
    rows = [
        {
            "q": signal_prob,
            "c": float(cost),
            **compare(float(cost), signal_prob),
            "theory_wins": 0 < cost < region(signal_prob),
            "theory_bound": region(signal_prob),
        }
        for signal_prob in signal_probs
        for cost in costs
    ]
    for row in rows:
        assert row["targeted_wins"] == row["theory_wins"], row
        assert float(row["max_epsilon"]) < 1e-6, row  # type: ignore[arg-type]
    write("ranking.csv", rows)
    return rows


def correlation(
    c: float = 0.05, q: float = 0.5, rhos: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)
) -> List[Dict[str, object]]:
    """Revenue ranking under symmetric correlated signals, where positive correlation is affiliation."""
    rows = [{"c": c, "q": q, "rho": rho, **compare(c, q, rho)} for rho in rhos]
    wins = [bool(row["targeted_wins"]) for row in sorted(rows, key=lambda row: row["rho"])]
    assert wins == sorted(wins, reverse=True), rows
    for row in rows:
        assert float(row["max_epsilon"]) < 1e-6, row  # type: ignore[arg-type]
        if row["rho"] == 0.0:
            assert row["targeted_wins"] == (0 < c < region(q)), row
    write("correlation.csv", rows)
    return rows


def empirical(c: float = 0.05, q: float = 0.5) -> List[Dict[str, object]]:
    """Payoff and regret of each symmetric profile in the empirical game, under each policy."""
    rows = []
    for policy in POLICIES:
        auction = Auction(q=q, c=c, policy=policy)
        strategies = {
            "separating": SEPARATING,
            "war_of_attrition": war_of_attrition(c),
            "always_stay": ALWAYS,
            "always_exit": NEVER,
            "learned": as_strategy(auction, learn(auction)),
        }
        names, payoffs, regrets = empirical_game(auction, strategies)
        regret = dict(zip(names, regrets))
        want = "war_of_attrition" if policy == "zero" else "separating"
        assert regret[want] < 1e-12, (policy, regret)
        assert min(regret["always_stay"], regret["always_exit"]) > 1e-6, (policy, regret)
        rows += [
            {
                "policy": policy,
                "c": c,
                "q": q,
                "strategy": name,
                "payoff": payoffs[index, index],
                "regret": regrets[index],
            }
            for index, name in enumerate(names)
        ]
    write("empirical_game.csv", rows)
    return rows


if __name__ == "__main__":
    for experiment in (equilibria, ranking, correlation, empirical):
        print(f"== {experiment.__name__} ==")
        experiment()
