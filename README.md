# Japanese Auctions with Costly Bidding

```bash
pip install -e .                 # numpy and scipy
python3 tests/test_paper.py      # the two-bidder example, 11 tests, ~2s
python3 tests/test_game.py       # the seller's books, the schedules and the joint prior, 10 tests, ~2s
python3 tests/test_analysis.py   # the trembled-belief one-shot check, 10 tests, ~1s
python3 tests/test_scale.py      # six bidders, the reduction, the solver and experiment 3, 14 tests, ~7 min
python3 tests/test_learning.py   # the sampled suite, 2 tests
python3 tests/test_mccfr.py      # the solver, its scoring and the validation grid, 38 tests, ~13s
python3 tests/test_robustness.py # the refined grid and the learned ranking, 13 tests, ~1s
python3 tests/test_implement.py  # the minimum-cost linear program, 11 tests, ~20s
python3 validate.py              # the MCCFR validation pipeline, writes results/*.csv, ~40 min
python3 robustness.py            # the robustness pipeline, writes results/*.csv, ~52 min
python3 scale.py                 # the scale pipeline, writes results/scale*.csv, over an hour
```

## The modules

| module | holds |
|---|---|
| `game.py` | the state space, the parameterised auction, and the reimbursement schedule families |
| `symmetric.py` | the orbit reduction: exact evaluation of a symmetric profile on exit-round orbits |
| `solver.py` | exact evaluation, deviation gains, and the exact-gradient solver |
| `learning.py` | Monte Carlo play of the auction, for profiles too large or too dispersed to evaluate exactly |
| `mccfr.py` | the external-sampling Monte Carlo counterfactual regret minimisation solver, and the scoring of what it returns |
| `analysis.py` | benchmark strategies, the equilibrium diagnostics, the closed forms, and the results writer |
| `validate.py` | the validation experiment, which asks whether MCCFR recovers the closed forms |
| `robustness.py` | the robustness experiment, which asks whether the revenue ranking survives learned play |
| `implement.py` | the minimum-cost reimbursement rule implementing a target profile, as a linear program over orbits |
| `scale.py` | the scale experiment, which asks whether learning recovers a reference profile at three, four and six bidders |

An information set is a bidder's own signal together with the full public bidding history, which is
the round, who is still active, and the round each departed bidder left in. The exit rounds matter
because the common value is the mean over every bidder's signal, those who have left included.

Evaluation dispatches on symmetry, on the prior and on the rule. A symmetric profile under an
exchangeable prior and an orbit-measurable rule is evaluated on orbits, being the round together with
the multiset of exit rounds, so six bidders take milliseconds rather than the hours enumerating the
histories would need. Anything else routes to the enumeration, which is exact under any prior and out
of reach beyond roughly seven bidders. `tests/test_scale.py` asserts the two paths agree to machine
precision wherever both run.

The signal profile is drawn jointly, so signals may correlate. The default prior is independent draws
at `q` and is bit-identical to what the code drew before. What the orbit reduction needs is that the
prior survive relabelling the bidders, not that it factorise, so `game.equicorrelated` is evaluated
on orbits too and a prior giving one named bidder its own marginal is refused.

A reimbursement policy reads the whole public history rather than the round alone, which is what lets
`implement.py` write a rule finer than the published families. A rule that separates two histories of
one orbit is expressible and refused by the reduction rather than averaged.

`mccfr.score` reads a finished profile and returns one of three verdicts. `derived` means the profile
is certified an equilibrium and sits at the closed forms. `different` means it is certified and is
another equilibrium, which the paper anticipates and which is a result rather than a failure.
`unsolved` means it was not certified, so the solver or its budget is at fault. Certification comes
from `analysis.utility_loss`, an exact best-response path that never reads a regret table, so a solver
bug cannot certify itself. Scoring takes a profile rather than a run, so a profile already known to be
exact is scored without paying for a run.

`mccfr.solve` keeps one regret table per bidder, so what it finds is asymmetric in general. Passing
`symmetry_restricted=True` gives every bidder one shared table instead, which makes the searched
profile symmetric by construction and is how the paper's two-bidder closed forms are validated
against, since they are stated within symmetric profiles. Nothing is averaged either way, and the two
searches answer different questions: the default says what an uncoordinated learner arrives at, and
the restricted one says whether the derived equilibrium is learnable.

## Units

The common value is the mean of the signals, not their sum, so it stays in the unit interval at every
bidder count and one clock calibration serves the whole study. That halves every price and payoff at
two bidders relative to the published closed forms, which are recovered exactly by halving the
participation cost and the tick alongside. `analysis.PUBLISHED` keeps the paper's own forms, and
`test_rescaling_to_mean_units_reproduces_the_published_revenues` asserts the identity on the published
grid.

At the defaults the targeted policy gives `Pi = 0.2125` against the closed form `q^2(1 - 6c)`, with
utility loss and one-shot deviation gain at machine zero.

## Results

Every experiment asserts what it records, so a broken result fails the run.

| file | covers | backs |
|---|---|---|
| `validation.csv` | ten seeds at seven `(c, q)` cells under each published reimbursement rule | Experiment 1, "MCCFR recovers the closed forms to within the distance bar" |
| `robustness.csv` | ten seeds at ten `(c, q)` cells under each published reimbursement rule, refined at both bounds of the region | Experiment 2, "The rules are solved and scored either side of both bounds" |
| `learned_ranking.csv` | the same runs paired by seed, one row per cell and seed | Experiment 2, "The revenue ranking read at learned play rather than at the theoretical profile" |
| `budget_probe.csv` | three seeds at the two `q = 0.9` cells the ranking turns on, at twenty thousand iterations | "The learned ranking does not turn on the budget being past the bottom of the loss" |
| `scale_rules.csv` | one row per bidder count: the reference profile's price, and what the linear program charged against what paying in full would have | Experiment 3, "The minimum-cost rule and the slack it found" |
| `scale.csv` | ten symmetry-restricted seeds and three unrestricted ones at each bidder count, under the rule written for that count | Experiment 3, "Whether learning recovers the reference profile and its net revenue" |

`validation.csv`, `robustness.csv` and `scale.csv` also disclose how
much experience each run had.
`visited` counts the information sets the run reached out of the `info_sets` it could, and
`visits_p10`, `visits_median` and
`visits_p90` give the spread of visits across the sets it reached, which a total would hide.

`scale.csv` says of every run whether its utility loss was walked or bounded from below, in `exact`,
and a run read by sampling carries the standard error of its revenue in `Pi_error` rather than being
reported as though it were exact. Where both are affordable the drop-out price bound is recorded
beside the loss it bounds, in `restricted_bound`, so how far it falls short is a measurement: at four
bidders the unrestricted arm's bound is 0.0 on all three seeds where the exact loss is 6.74e-04 to
9.20e-04, so the bound would have certified all three runs and the walk certifies none of them. Its `verdict` is read at the arrival-weighted distance to the
reference and `verdict_at_the_maximum` at the maximum over the sets the reference reaches, because at
93,312 information sets the two disagree and each says something the other does not.

`validation.csv` is the experiment the first learning claim rests on. Two hundred and ten runs, being
seven `(c, q)` cells by the three published reimbursement rules by ten seeds, each at the budget and
under the symmetry restriction, take forty minutes against the hour. A run takes 7.8 to 49.4 seconds,
the slow end being the high signal probabilities, where both bidders usually hold a high signal and
the auction runs deeper before anyone exits. One hundred and five runs are `derived`, seventy-five are
`unsolved` and thirty are `different`. Every seed is a row of its own, because implementation does not
imply uniqueness and a mean across distinct equilibria would describe none of them. A run reaches 12
to 16 of the 16 information sets at two bidders. `validate.py` reads `mccfr.BUDGET` and
`mccfr.STAY_BAR` rather than choosing numbers of its own.

The grid is sampled rather than crossed. Costs sit at a half, one and one and a half times
`analysis.region(q)`, since which of that region's two bounds binds moves with the signal probability.
`q = 0.5` runs all three placements, where the cost root binds, and `q = 0.75` and `q = 0.9` run the
bound and outside it, where the fragile claim is. The full three by three would be 270 runs and would
fill the hour with nothing left for the calibration the widening needs. Only `q = 0.5` carries a
budget measured at it, and every row says which it is in `budget_calibrated`.

What the signal probability does is the result. At `q = 0.5` seventy-five runs are `derived` and
fifteen `unsolved`. That is close to ticket 08's seventy-seven and thirteen but not the same file,
since placing the costs against `analysis.region(q)` moved the outer two from 0.025 and 0.08 to
0.0272 and 0.0815, and the zero rule outside the region fell from five of ten `derived` to three. At
`q = 0.75` it is thirty and thirty. At `q = 0.9` nothing is `derived` at all, thirty runs being
`unsolved` and thirty `different`. Every `different` verdict in the file is at `q = 0.9`.

The zero rule is where the difficulty is, its closed form being the mixed war of attrition rather
than pure separation. At `q = 0.5` it is derived on ten of ten seeds inside the region, two of ten
on the bound and three of ten outside it. At `q = 0.75` and `q = 0.9` it is `unsolved` on all forty,
and those are the budget falling short rather than a rival equilibrium, since across those forty the
on-path distance never exceeds 0.0388 against a bar of 0.06 while the bully of ADR-0003 sits a full
unit of stay probability away. `utility_loss_halfway` gives the same loss at half the budget, which
is what separates a run still descending from one that has plateaued.

The full rule is derived at `q = 0.5` and `q = 0.75` and `different` on all twenty seeds at `q = 0.9`,
where the profile certifies at a loss near `3e-5` and sits 0.124 to 0.179 from separation, well past
the bar. Its revenue is within 0.007 of the closed form there, so it is different play reaching much
the same money. Within a signal probability it still plays identically at every cost, since
reimbursing the participation cost in full leaves the same game behind, and its revenue still moves
with the cost because the seller pays the reimbursement.

The targeted rule is the one the widening was for. `closed_form_loss` records the target's own utility
loss, and in twenty rows it exceeds `analysis.CERTIFIED`, being the two cells where the cost is past
`analysis.ic_bound(q)` and separation is no longer an equilibrium to recover. `derived` is out of
reach there by construction rather than by budget, and `validate.py` asserts that no such row claims
it. All twenty come back `unsolved`.

One `different` verdict is the bar rather than the play. At `q = 0.9` on the incentive bound the
targeted runs sit 0.0116 to 0.0124 from the closed form in stay probability, well inside the 0.06
bar, and are called `different` only because the revenue distance reaches 0.0162 against
`mccfr.REVENUE_BAR` of 0.01. Seller net revenue is about 0.69 in mean units there against 0.21 at
`q = 0.5`, so an absolute revenue bar is a far tighter test at a high signal probability than at
`q = 1/2`, where it was set. The bar was left where it is, since moving it so that a point passes is
fitting the bar to the data, and the rows are reported so the reading can be made from them.

`trembled_gain` equals `one_shot_gain` in all two hundred and ten rows. `analysis.max_regret` skips a
set only when its counterfactual reach falls under `1e-12`, and that reach is the opponent's doing
rather than the
bidder's own, so at two bidders it stays near 0.2 at every set of every run here. Nothing is skipped
and the trembled belief has nothing left to lift. This is not the off-path distance being small: it
reaches 0.99 under the targeted rule, at sets the closed form never plays, and those sets still carry
reach under the learned profile. The column discriminates where a profile drives its
opponent pure, which is what ADR-0003 found it doing, and it is recorded here so the claim can be read
rather than assumed.

`robustness.csv` and `learned_ranking.csv` are the second learning claim, which is not that the closed
forms are recovered but that the revenue ranking between the three rules survives being learned. Three
hundred runs, being ten `(c, q)` cells by the three rules by ten seeds, take fifty minutes at 8.0 to
14.5 seconds a run. One hundred and thirty-one are `derived`, eighty-nine `unsolved` and eighty
`different`. The costs sit at a half, nine tenths, one, one and a tenth and one and a half times
`analysis.region(q)`, refining experiment 1's three placements on both sides of the bound, at `q = 0.5`
where the cost root binds and `q = 0.9` where the incentive bound does.

`learned_ranking.csv` pairs the three rules at one cell and one seed and asks which of them actually
paid the seller most, with the same comparison between the closed forms beside it. Thirteen of its
hundred rows disagree with the region claim, ten at one cell and three at another, and the two
disagreements point opposite ways.

At `q = 0.5` the ranking flips before the cost root does. The targeted rule wins on ten of ten seeds
well inside the region, by 0.0007 to 0.0027 against a promised 0.0063, and on none from nine tenths of
the bound outward, where the closed forms still put it ahead by 0.0021. Both learned revenues move the
comparison against it there, though from opposite sides of their own closed forms. The zero rule earns
the seller 0.1784 to 0.1811 against a closed form of 0.1746, because learned play does not reach the
war of attrition and a bidder that stays too long pays the seller for it, while the targeted rule
earns 0.0012 under its own. Together that is five to eight thousandths, which is more than the margin
being read.

At `q = 0.9` it flips after the incentive bound. The targeted rule wins on ten of ten seeds at the
bound itself, on three of ten a tenth past it, and on none at one and a half times it, so learned play
carries the rule's advantage past the point where separation stops being an equilibrium without
carrying it as far as the revenue formulas do, those still favouring the targeted rule by 0.017 at the
outermost cell. The reading is not a sharp boundary. The winning margins at the bound are 0.0002 to
0.0024 at the budget, and `budget_probe.csv` reruns that cell and the outermost one at the twenty
thousand iterations `q = 0.9` actually wants, where they are three times larger with the same sign.

`theory_wins` reads `analysis.region_claim`, which is strict at the cost root, where the two revenue
formulas tie, and inclusive at the incentive bound, where separation still holds and the targeted rule
is ahead by 0.020. `ranking.csv` reads the same predicate, and its contents are unchanged by it, since
no cost on that grid lands on a bound.

Only twenty-one of the hundred ranking rows have all three rules certified, which the `certified`
column says per row. The zero rule is `unsolved` at every `q = 0.9` cell but the innermost, where it is
`derived` on three of ten seeds, so most of that row's ranking is a comparison in which one rule never
reached an equilibrium at all. Non-target equilibria are counted rather than failed, and what is
asserted of them is that they certified.

The checked-in `scale.csv` and `scale_rules.csv` hold three and four bidders only. Six bidders is
written and verified, and its linear program solves in 179 s for a saving of 4.5 per cent, but the
thirteen six-bidder runs behind it did not finish on the machine this was developed on: the auction
needs about 650 MB of per-history tables and the run was getting some five per cent of a CPU against
other work on the box. `scale.py` runs all three counts as it stands and needs no change, only a
machine with memory and cores free.

## Where the method comes from

`utility_loss()` and `best_response()` take the diagnostic from Thoma et al 2023, scoring a profile by
its approximate-equilibrium gap. We use an exact backward sweep for simplicity where they use learned
best responses.
