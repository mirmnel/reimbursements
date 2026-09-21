"""The trembled-belief one-shot check, against targets the untrembled check already certifies."""

from __future__ import annotations

import numpy as np

import analysis as AN
import solver as A
from test_paper import main, target

Q, C, TIGHT = 0.5, 0.025, 1e-12

TREMBLES = (0.001, 0.01, 0.1, 0.5)
DESERTED_C = 0.4  # dear enough that exiting at the opening is the equilibrium, leaving the rest off path


def deserted():
    """An auction whose equilibrium is every bidder exiting at the opening, leaving later rounds off path."""
    auction = A.Auction(q=Q, c=DESERTED_C, reimbursement_policy="zero")
    return auction, AN.profile(auction, AN.NEVER)


def positions_in_round(auction, round_index, bidder=None):
    """Every position in a flat profile whose information set sits in ``round_index``."""
    return [
        position
        for position, (owner, state, _) in enumerate(auction.info_sets)
        if state.round == round_index and bidder in (None, owner)
    ]


def mistaken(auction, sigma, bidder=None):
    """``sigma`` with a bidder staying on at round two, a mistake no untrembled belief ever reaches."""
    planted = sigma.copy()
    planted[positions_in_round(auction, 2, bidder)] = 1.0
    return planted


# --- the equilibria stay equilibria under any tremble -----------------------


def test_an_exact_equilibrium_has_no_trembled_gain():
    """The paper's targets are sequentially rational at every tremble rate."""
    for name in A.REIMBURSEMENT_POLICIES:
        auction, sigma = target(name)
        for tremble in (0.0,) + TREMBLES:
            assert AN.trembled_regret(auction, sigma, tremble) < TIGHT, (name, tremble)


def test_an_equilibrium_with_off_path_sets_has_no_trembled_gain():
    """Exiting at the opening stays optimal at the rounds only a tremble ever reaches."""
    auction, sigma = deserted()
    for tremble in TREMBLES:
        assert AN.trembled_regret(auction, sigma, tremble) < TIGHT, tremble


def test_a_zero_tremble_agrees_with_the_untrembled_check():
    """With no tremble no set gains weight, so the two checks report the same number."""
    for name in A.REIMBURSEMENT_POLICIES:
        auction, aimed = target(name)
        for sigma in (aimed, np.clip(aimed + 0.2, 0.0, 1.0), np.full(len(aimed), 0.5)):
            got, want = AN.trembled_regret(auction, sigma, 0.0), AN.max_regret(auction, sigma)
            assert abs(got - want) < TIGHT, (name, got, want)


# --- the off-path sets the untrembled check cannot see ----------------------


def test_the_deserted_auction_leaves_every_later_round_off_path():
    """The premise of the tests below: an equilibrium whose later rounds carry no reach at all."""
    auction, sigma = deserted()
    assert AN.utility_loss(auction, sigma) < TIGHT
    weights = A.counterfactual_reach(auction, sigma)
    assert np.all(weights[positions_in_round(auction, 0)] > AN.RESOLVED)
    for round_index in range(1, auction.rounds):
        assert np.all(weights[positions_in_round(auction, round_index)] <= AN.RESOLVED), round_index


def test_an_off_path_mistake_is_invisible_without_a_tremble():
    """Staying on at a price no bidder can afford is a mistake the untrembled check scores as zero."""
    auction, sigma = deserted()
    assert AN.max_regret(auction, mistaken(auction, sigma)) < TIGHT
    assert AN.trembled_regret(auction, mistaken(auction, sigma), 0.01) > 1e-6


def test_the_off_path_gain_is_the_cost_of_the_stay_plus_the_loss_on_the_sale():
    """The planted mistake pays ``c`` to buy at ``p_T`` a prize worth less, and the check reports exactly that."""
    auction, sigma = deserted()
    price = auction.price(auction.rounds)
    lost = DESERTED_C + (price - auction.q) / 2  # the low type wins half a prize it values below the price
    assert abs(AN.trembled_regret(auction, mistaken(auction, sigma), 0.01) - lost) < TIGHT


def test_a_larger_tremble_does_not_hide_the_mistake():
    """The mistake is reported at every tremble rate, not only at one lucky one."""
    auction, sigma = deserted()
    for tremble in TREMBLES:
        assert AN.trembled_regret(auction, mistaken(auction, sigma), tremble) > 1e-6, tremble


def test_a_tremble_too_faint_to_reach_a_set_is_refused_rather_than_scored_as_zero():
    """A tremble that cannot carry weight to an off-path set fails loudly, which is the bug being fixed."""
    auction, sigma = deserted()
    try:
        AN.trembled_regret(auction, mistaken(auction, sigma), 1e-7)
    except ValueError as exc:
        assert "too faint" in str(exc), exc
    else:
        assert False, "a tremble that reaches no off-path set was scored rather than refused"


# --- asymmetry --------------------------------------------------------------


def test_an_asymmetric_profile_is_scored_on_the_set_based_path():
    """One bidder separating while the other never exits is scored without being symmetrised."""
    auction = A.Auction(q=Q, c=C, reimbursement_policy="targeted")
    sigma = AN.profile(auction, AN.SEPARATING, AN.ALWAYS)
    assert not np.allclose(sigma, AN.profile(auction, AN.SEPARATING))
    assert abs(AN.trembled_regret(auction, sigma, 0.0) - AN.max_regret(auction, sigma)) < TIGHT
    assert AN.trembled_regret(auction, sigma, 0.01) > 1e-6  # the low type staying on is an on-path mistake


def test_an_asymmetric_off_path_mistake_is_caught():
    """A mistake planted in one bidder alone, off the path, is still reported."""
    auction, sigma = deserted()
    planted = mistaken(auction, sigma, bidder=0)
    assert not np.allclose(planted, AN.profile(auction, AN.NEVER))
    assert AN.max_regret(auction, planted) < TIGHT
    assert AN.trembled_regret(auction, planted, 0.01) > 1e-6


if __name__ == "__main__":
    raise SystemExit(main(globals()))
