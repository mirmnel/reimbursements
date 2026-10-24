# Japanese Auctions with Costly Bidding

Companion code for the paper's two learning experiments.

```bash
pip install -e .                       # numpy, matplotlib
python3 -m experiments.two_bidder     # experiment 1, ~75 min on 4 cores
python3 -m experiments.scale --counts 2,4         # experiment 2 at two and four bidders, ~10 min
python3 -m experiments.scale --counts 6 --seeds 0  # one seed at six bidders, ~65 min (seeds already saved are skipped)
python3 -m experiments.plots          # every figure, from the CSVs in results/
python3 tests/test_paper.py           # the two-bidder closed forms, ~2 s (each test file runs on its own)
```

## The experiments

| experiment | script | writes |
|---|---|---|
| 1. Two bidders: MCCFR recovers the sorting equilibrium and revenue under no, full, war-all and back-loaded reimbursement, and back-loading out-earns every other price-only corner and full reimbursement at learned play | `experiments/two_bidder.py` | `two_bidder.csv`, `two_bidder_trace.csv`, `learned_ranking.csv` |
| 2. Scale: at `n` in 2, 4, 6, whether MCCFR finds the paper's sorting equilibria under None, the two-survivor trigger, Preserve, Full and free bidding, and whether the learned revenue differences match `fin:revenue-gain` and `pres:binary-saving` | `experiments/scale.py` | `scale.csv` |

Every run gets one of three verdicts. `derived` means it reached an equilibrium that sits at the target, and the
target is itself an equilibrium. `different` means it reached another equilibrium. `unsolved` means it did not reach
an equilibrium. A run reaches an equilibrium when its exact best-response utility loss is under `exact.CERTIFIED`,
the paper's epsilon of `1e-3`. That check never reads a regret table, so a solver bug cannot pass itself. Revenue
stays out of the verdict. Experiment 1 asserts separately that every `derived` run's revenue sits within
`REVENUE_BAR` of theory. Experiment 2 compares revenue differences between policies with the paper's formulas,
over the runs that reached an equilibrium (`GLOSSARY.md` names the policies).

## The modules

The library lives in `auction/`. The scripts in `experiments/` run the experiments and draw the figures.

An information set is a bidder's own signal together with the public history: the round, who is still in, and the
round each departed bidder left in. A symmetric profile under an exchangeable prior and an orbit-measurable rule is
evaluated on orbits. Anything else is evaluated by enumerating the histories, which is exact but out of reach
beyond about seven bidders. `tests/test_scale.py` checks that the two paths agree to machine precision.

`mccfr.solve(symmetry_restricted=True)` gives every bidder one shared regret table. That makes the search symmetric
by construction, which is how the closed forms (stated within symmetric profiles) are tested. Both experiments
use it for every run.