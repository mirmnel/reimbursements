# Japanese Auctions with Costly Bidding

Companion code for the paper's three learning experiments.

```bash
pip install -e .                       # numpy, scipy, matplotlib
python3 -m experiments.two_bidder validation  # experiment 1, ~11 min on 8 cores
python3 -m experiments.two_bidder robustness  # experiment 2, ~16 min on 8 cores
python3 -m experiments.scale                  # experiment 3, ~55 min
python3 -m experiments.plots                  # every figure, from the CSVs in results/
python3 tests/test_paper.py                   # the two-bidder closed forms, ~2 s (each test file runs on its own)
```

## The experiments

| experiment | script | writes |
|---|---|---|
| 1. Validation: MCCFR recovers separation, the stay probabilities under `r=0`, and `Pi_0`, `Pi_full`, `Pi_T` | `experiments/two_bidder.py validation` | `validation.csv` |
| 2. Robustness: the ranking `Pi_T > max(Pi_0, Pi_full)` at learned play, refined near both bounds | `experiments/two_bidder.py robustness` | `robustness.csv`, `learned_ranking.csv` |
| 3. Scale: at `n` in 3, 4, 6, solve `Gamma^0`, write the minimum-cost rule `r*`, and test whether learning recovers the reference | `experiments/scale.py` | `scale_rules.csv`, `scale.csv` |

Every run is scored with one of three verdicts. `derived` means it is certified as an equilibrium and sits at the
target. `different` means it is certified but is another equilibrium. `unsolved` means it is not certified.
Certification is an exact best-response utility loss under `exact.CERTIFIED`, the paper's epsilon of `1e-3`. That
check never reads a regret table, so a solver bug cannot certify itself.

## The modules

The library lives in `auction/`. The scripts in `experiments/` run the experiments and draw the figures.

| module | holds |
|---|---|
| `auction/game.py` | the public histories, the auction, the published reimbursement rules, and the two-bidder closed forms |
| `auction/symmetric.py` | exact evaluation of a symmetric profile on exit-round orbits, which is what makes six bidders tractable |
| `auction/exact.py` | exact evaluation on histories, the best response, and utility loss |
| `auction/mccfr.py` | the external-sampling MCCFR solver and the verdict on what it returns |
| `auction/implement.py` | the linear program for the cheapest per-orbit rule that keeps a target profile an equilibrium |
| `experiments/two_bidder.py` | experiments 1 and 2 |
| `experiments/scale.py` | experiment 3 |
| `experiments/plots.py` | every figure |

An information set is a bidder's own signal together with the public history: the round, who is still in, and the
round each departed bidder left in. A symmetric profile under an exchangeable prior and an orbit-measurable rule is
evaluated on orbits. Anything else is evaluated by enumerating the histories, which is exact but out of reach
beyond about seven bidders. `tests/test_scale.py` checks that the two paths agree to machine precision.

`mccfr.solve(symmetry_restricted=True)` gives every bidder one shared regret table. That makes the search symmetric
by construction, which is how the two-bidder closed forms (stated within symmetric profiles) are tested.
Experiment 3 also runs an unrestricted arm at three and four bidders. At six its profiles are too large to walk,
so that arm stops at four.

## Units

The common value is the sum of the signals. The two-bidder example uses a tick of one half and a default cost of
0.05. At `n` bidders, `experiments/scale.py` stretches the cost and every tolerance by `game.payoff_scale(n) = n / 2`, so each
bidder count is held to the same bars relative to the prize.

## Where the method comes from

The utility-loss diagnostic follows Thoma et al 2023. We compute the best response exactly by a backward sweep,
where they learn it.
