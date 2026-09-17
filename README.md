# Japanese Auctions with Costly Bidding

Companion code for the paper. Staying in the clock auction costs a bidder something every round, and
the seller may reimburse part of it. Which reimbursement policy maximises the seller's net revenue,
and do independent learning bidders converge on the equilibria the theory predicts?

`CONTEXT.md` defines the vocabulary. `docs/adr/` records the decisions that shape the code:
evaluation on counts rather than active sets, the common value in mean units, independent learners
under a symmetric evaluator, identity-conditioned strategies under an anonymised report, and
fictitious play where the solver's ascent stalls.

```bash
pip install -e .                 # numpy only
python3 test_paper.py            # the two-bidder example, 10 tests, ~2s
python3 test_game.py             # the seller's books and the schedules, 3 tests, ~1s
python3 test_scale.py            # ten bidders, the reduction and the solver, 7 tests, ~6 min
python3 test_learning.py         # the sampled suite, 17 tests, ~10s
python3 verify.py                # the deterministic pipeline, writes results/*.csv, ~3 min
python3 emergence.py             # the sampled pipeline, writes results/*.csv, ~17 min
```

## The modules

| module | holds |
|---|---|
| `game.py` | the state space, the parameterised auction, and the reimbursement schedule families |
| `symmetric.py` | the count-based reduction: exact evaluation of a symmetric profile on active counts |
| `solver.py` | exact evaluation, deviation gains, and the exact-gradient solver |
| `learning.py` | independent tabular regret-matching bidders, and the scoring of what they arrive at |
| `analysis.py` | benchmark strategies, the empirical game, the equilibrium diagnostics, the closed forms, and the results writer |
| `verify.py` | the deterministic experiments behind the paper's tables |
| `emergence.py` | the sampled experiments, which compare what emerged against certified equilibria |

Evaluation dispatches on symmetry. A symmetric profile is evaluated on the counts of active
high-signal and low-signal bidders, so ten bidders take milliseconds rather than the day the
active-set enumeration would need. An asymmetric profile still routes to the enumeration, which is
out of reach beyond roughly seven bidders. `test_scale.py` asserts the two paths agree to machine
precision wherever both run.

## Units

The common value is the mean of the signals, not their sum, so it stays in the unit interval at every
bidder count and one clock calibration serves the whole study. That halves every price and payoff at
two bidders relative to the published closed forms, which are recovered exactly by halving the
participation cost and the tick alongside. `analysis.PUBLISHED` keeps the paper's own forms, and
`test_rescaling_to_mean_units_reproduces_the_published_revenues` asserts the identity on the published
grid.

At the defaults the targeted policy gives `Pi = 0.2125` against the closed form `q^2(1 - 6c)`, with
utility loss and one-shot regret at machine zero.

## Results

Every experiment asserts what it records, so a broken result fails the run.

| file | covers | backs |
|---|---|---|
| `equilibria.csv` | one point, `c = 0.025, q = 0.5` | Table 1, "The solver recovers the closed forms" |
| `ranking.csv` | a 72-point `(c, q)` grid | "The ranking survives solving" |
| `empirical_game.csv` | five strategies under each published reimbursement policy at two bidders | Table 2, "Implementation is not uniqueness" |
| `recovery.csv` | ten seeds at four and six bidders | "The learners reach a near-equilibrium, and an asymmetric one" |
| `empirical_game_at_scale.csv` | five strategies at four and six bidders | "The learned strategy beats the benchmarks" |
| `emergence.csv` | thirty seeds at ten bidders, four reimbursement families | the emergence claim |

The first three report seller net revenue, except `empirical_game.csv`, which reports bidder payoff and
is why always-exit leads there while leaving the seller with nothing. The last three carry, per run,
what the solver certified and what the learners found, with the cross-bidder dispersion and whether
symmetrising the learned profile misrepresents it.

`recovery.csv` and `emergence.csv` also disclose how much experience each run had. `visited` counts the
information sets the run reached out of the `info_sets` it could, and `visits_p10`, `visits_median` and
`visits_p90` give the spread of visits across the sets it reached, which a total would hide. At ten
bidders a run reaches 18,850 to 27,171 of its 30,680 sets, with a median of fourteen to ninety-six
visits across the families, on ten times the episodes the smaller counts use. ADR-0004 records why the space is
that large and reported nowhere.

What the learners arrive at is not symmetric. Every run recorded here is flagged as one where
symmetrising misleads, because the bidders agree on the shape of play and not on who plays which part.
On those rows read `Pi_sampled`, the profile as it stands, rather than `Pi_emergent`, its
symmetrisation.

`Pi_certified` clears the same bar under every family, `5e-4`, and every row still carries the loss
the solver actually reached in `utility_loss_certified`. At ten bidders that is `3e-12` under
targeted, a few parts in `1e-6` under flat and front-loaded, and `1e-4` under back-loaded, which is
where the solver stops rather than where it plateaus.

## Where the method comes from

`empirical_game()` takes the empirical game-theoretic analysis method from Wellman, Tuyls and
Greenwald 2024, but enumerates its deviation payoffs exactly, skipping the survey's estimation
problem.

`utility_loss()` and `best_response()` take the diagnostic from Thoma et al 2023, scoring a profile by
its approximate-equilibrium gap. We use an exact backward sweep for simplicity where they use learned
best responses.
