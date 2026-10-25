"""The two-bidder claims of the paper, checked against an exact solver never told the closed forms."""

from __future__ import annotations

from auction import exact as A
from auction import game as G
from experiments import scale

TIGHT = 1e-9


def published(c, q):
    """Per policy, the high type's stay probability per round and seller net revenue, as printed."""
    s1, s3 = 1 - 4 * c / 3, 1 - 4 * c
    return {
        "zero": ((1, s1, 1, s3), q**2 * (0.5 * (1 - s1**2) + s1**2 * (1.5 + 0.5 * s3**2))),
        "trigger": ((1, 1, 1 - 2 * c, 1), q**2 * (1 + (1 - 2 * c) ** 3)),
        "preserve": ((1, 1, 1, 1), 2 * q**2 - 6 * c * q**2),
        "full": ((1, 1, 1, 1), 2 * q**2 - 6 * c * q**2 - 2 * q * c),
    }


def test_the_sorting_profile_at_two_bidders_is_every_published_closed_form():
    """At two bidders the general backward solution gives the printed war stays and revenue under each policy."""
    for c in scale.COSTS:
        for name, (stays, earned) in published(c, scale.Q).items():
            auction = scale.auction_at(2, name, c)
            sigma = G.sorting_profile(auction)
            at = {(bidder, state.round, signal): sigma[k] for k, (bidder, state, signal) in enumerate(auction.info_sets)}
            assert all(abs(at[(0, t, 1)] - stays[t]) < TIGHT for t in range(4)), (name, c)
            assert all(at[(0, t, 0)] == 0.0 for t in range(4)), (name, c)
            assert abs(A.evaluate(auction, sigma).revenue - earned) < TIGHT, (name, c)


def main(namespace=None) -> int:
    """Run every ``test_`` function in ``namespace`` and report how many passed."""
    tests = [(name, fn) for name, fn in sorted((namespace or globals()).items()) if name.startswith("test_")]
    failed = [name for name, test_fn in tests if not run(name, test_fn)]
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


def run(name, test_fn) -> bool:
    """Run one test, print PASS or FAIL, and report whether it passed."""
    try:
        test_fn()
    except AssertionError as exc:
        print(f"  FAIL  {name}: {exc}")
        return False
    print(f"  PASS  {name}")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
