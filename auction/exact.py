"""Exact evaluation of a symmetric profile, and the exact best response that certifies it as an equilibrium."""

from __future__ import annotations

import numpy as np

from . import symmetric
from .game import Auction, Outcome, payoff_scale

CERTIFIED = 1e-3  # utility-loss threshold at two bidders
RESOLVED = 1e-12  # the least counterfactual reach a gain can be divided by without amplifying noise


def certified(n: int = 2) -> float:
    """``CERTIFIED`` at ``n`` bidders, stretched with the payoffs so every count is held to one bar."""
    return CERTIFIED * payoff_scale(n)


def evaluate(auction: Auction, sigma: np.ndarray) -> Outcome:
    """Exact payoffs and seller accounting, on active counts."""
    assert symmetric.applies(auction, sigma)
    return symmetric.outcome(auction, sigma)


def utility_loss(auction: Auction, sigma: np.ndarray) -> float:
    """Utility loss: the largest gain any bidder gets from a full best response."""
    assert symmetric.applies(auction, sigma)
    return symmetric.utility_loss(auction, sigma)
