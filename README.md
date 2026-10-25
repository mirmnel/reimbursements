# Japanese Auctions with Costly Bidding

Companion code for the paper's two learning experiments.

```bash
pip install -e .                       # numpy, matplotlib
python3 -m experiments.scale --counts 2,4         # experiment 1 at two and four bidders, ~10 min
python3 -m experiments.scale --counts 6 --seeds 0  # one seed at six bidders, ~65 min (seeds already saved are skipped)
python3 -m experiments.schedule_search  # experiment 2, ~4 h on 8 cores (policies already saved are skipped)
python3 -m experiments.plots          # every figure, from the CSVs in results/
python3 tests/test_paper.py           # the two-bidder closed forms, ~1 s (each test file runs on its own)
```

## The experiments

| experiment | script | writes |
|---|---|---|
| 1. Sorting equilibria: at `n` in 2, 4, 6, whether MCCFR finds the paper's sorting equilibria under None, the two-survivor trigger, Preserve, Full and free bidding, and whether the learned revenue differences match `fin:revenue-gain` and `pres:binary-saving` | `experiments/scale.py` | `scale.csv` |
| 2. Schedule search: at four bidders and four rounds, which of the 625 round-only schedules with each `rho_t` in {0, c/4, c/2, 3c/4, c} earns the most seller net revenue at learned play, beside the two-survivor trigger and Preserve | `experiments/schedule_search.py` | `schedule_search.csv` |

Every run gets one of three verdicts. `derived` means it reached an equilibrium that sits at the target, and the
target is itself an equilibrium. `different` means it reached another equilibrium. `unsolved` means it did not reach
an equilibrium. A run reaches an equilibrium when its exact best-response utility loss is under `exact.CERTIFIED`,
the paper's epsilon of `1e-3`. That check never reads a regret table, so a solver bug cannot pass itself. Revenue
stays out of the verdict. Experiment 1 compares revenue differences between policies with the paper's
formulas, over the runs that reached an equilibrium (`GLOSSARY.md` names the policies). Experiment 2 has no
target, so a run there either reached an equilibrium or did not, and a schedule's revenue is its mean over the
seeds that did.

## The modules

The library lives in `auction/`. The scripts in `experiments/` run the experiments and draw the figures.

An information set is a bidder's own signal together with the public history: the round, who is still in, and the
round each departed bidder left in. Profiles are symmetric and every policy pays the same at histories that differ
only by relabelling the bidders, so `exact` evaluates them on those orbits and asserts that both hold.

`mccfr.solve` gives every bidder one shared regret table, which makes the search symmetric by construction.