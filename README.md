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

Every run gets one of three verdicts. `derived` means it reached an equilibrium that sits at the target, and the
target is itself an equilibrium. `different` means it reached another equilibrium. `unsolved` means it did not reach
an equilibrium. A run reaches an equilibrium when its exact best-response utility loss is under `exact.CERTIFIED`,
the paper's epsilon of `1e-3`. That check never reads a regret table, so a solver bug cannot pass itself.

## The modules

The library lives in `auction/`. The scripts in `experiments/` run the experiments and draw the figures.

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
