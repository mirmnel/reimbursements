"""The deterministic experiments behind the paper's tables."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from analysis import (
    ALWAYS,
    NEVER,
    PI,
    SEPARATING,
    as_strategy,
    empirical_game,
    max_regret,
    region,
    utility_loss,
    war_of_attrition,
    write,
)
from solver import REIMBURSEMENT_POLICIES, evaluate, solve
from game import Auction, Outcome


def solved(c: float, q: float) -> Dict[str, Tuple[Auction, np.ndarray, Outcome]]:
    """Solve for a profile under each reimbursement policy, and evaluate it."""
    computed = {}
    for name in REIMBURSEMENT_POLICIES:
        auction = Auction(q=q, c=c, reimbursement_policy=name)
        sigma = solve(auction)
        computed[name] = (auction, sigma, evaluate(auction, sigma))
    return computed


def equilibria(c: float = 0.025, q: float = 0.5) -> List[Dict[str, object]]:
    """The solved profile under each policy, verified and scored against the closed forms."""
    rows, computed = [], solved(c, q)
    for name, (auction, sigma, outcome) in computed.items():
        row: Dict[str, object] = {"reimbursement_policy": name, "c": c, "q": q}
        for signal, label in ((1, "H"), (0, "L")):
            row |= {
                f"sigma_{label}_t{round_index}": sigma[auction.info_index[(0, (round_index, (0, 1)), signal)]]
                for round_index in range(auction.rounds)
            }
        loss, error = utility_loss(auction, sigma), abs(outcome.revenue - PI[name](c, q))
        row |= {
            "utility_loss": loss,
            "max_regret": max_regret(auction, sigma),
            "price": outcome.price,
            "reimbursement": outcome.reimbursement,
            "stay_rounds": outcome.stay_rounds,
            "Pi": outcome.revenue,
            "Pi_closed_form": PI[name](c, q),
            "abs_error": error,
        }
        assert error < 1e-5, row
        assert loss < 1e-9, row
        rows.append(row)
    write("equilibria.csv", rows)
    return rows


def compare(c: float, q: float) -> Dict[str, object]:
    """Net revenue per policy at one setting, and the worst utility loss behind it."""
    computed = solved(c, q)
    revenue = {name: outcome.revenue for name, (_, _, outcome) in computed.items()}
    return {
        **{f"Pi_{name}": revenue[name] for name in REIMBURSEMENT_POLICIES},
        "argmax": max(revenue, key=revenue.get),
        "targeted_wins": revenue["targeted"] > max(revenue["zero"], revenue["full"]),
        "max_utility_loss": max(utility_loss(auction, sigma) for auction, sigma, _ in computed.values()),
    }


def ranking(c: Optional[float] = None, q: Optional[float] = None) -> List[Dict[str, object]]:
    """Revenue ranking over a ``(c, q)`` grid, with behaviour solved for rather than assumed."""
    costs = (c,) if c is not None else np.linspace(0.005, 0.12, 24)
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
        assert float(row["max_utility_loss"]) < 1e-6, row  # type: ignore[arg-type]
    write("ranking.csv", rows)
    return rows


def empirical(c: float = 0.025, q: float = 0.5) -> List[Dict[str, object]]:
    """Payoff and regret of each strategy in the empirical game, under each policy."""
    rows = []
    for policy_name in REIMBURSEMENT_POLICIES:
        auction = Auction(q=q, c=c, reimbursement_policy=policy_name)
        strategies = {
            "separating": SEPARATING,
            "war_of_attrition": war_of_attrition(c),
            "always_stay": ALWAYS,
            "always_exit": NEVER,
            "solved": as_strategy(auction, solve(auction)),
        }
        names, payoffs, regrets = empirical_game(auction, strategies)
        regret = dict(zip(names, regrets))
        want = "war_of_attrition" if policy_name == "zero" else "separating"
        assert regret[want] < 1e-12, (policy_name, regret)
        assert min(regret["always_stay"], regret["always_exit"]) > 1e-6, (policy_name, regret)
        rows += [
            {
                "reimbursement_policy": policy_name,
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
    for experiment in (equilibria, ranking, empirical):
        print(f"== {experiment.__name__} ==")
        experiment()
