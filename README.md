# Japanese Auctions with Costly Bidding

Companion code for the paper. `auctionrl.py` holds the exact solver and the
self-play learner. `verify.py` reproduces the two-bidder example, scores the
equilibria behind every claim, and re-derives the revenue ranking under learned
behaviour. Tests in `test_paper.py`.

```bash
pip install -e .                 # numpy only
python3 test_paper.py            # 12 tests, ~4s
python3 verify.py                # writes results/*.csv, ~2 min
```

`equilibria()` is the shortest path in: it builds each auction, learns an
equilibrium from a uniform start, and scores it. At the defaults the targeted
policy gives `Pi = 0.42499999999959315` against the closed form `2q^2(1-3c)`,
with `epsilon` and `max_regret` at machine zero.

## Results

Four files, each the evidence behind one claim in the paper's Reinforcement
Learning section, where the methodology and findings are written up.

| file | covers | backs |
|---|---|---|
| `equilibria.csv` | one point, `c = 0.05, q = 0.5` | Table 1, "The learner recovers the closed forms" |
| `ranking.csv` | a 72-point `(c, q)` grid | "The ranking survives learning" |
| `empirical_game.csv` | five strategies under each policy | Table 2, "Implementation is not uniqueness" |
| `correlation.csv` | `rho` from `0` to `0.95` | Table 3, "Affiliation has a limit" |

Three report seller net revenue. `empirical.csv` reports bidder payoff, which
is why always-exit leads there while leaving the seller with nothing.

Every experiment asserts what it records, so a broken result fails the run.

## Where the method comes from

`empirical_game()` takes the empirical game-theoretic analysis method from
Wellman, Tuyls and Greenwald 2024, but enumerates its payoff matrix exactly,
skipping the estimation problem in the survey.

`epsilon()` and `best_response()` take the diagnostic from Thoma et al 2023,
scoring a learned profile by its approximate-equilibrium gap. We use an exact backward sweep for simplicity whereas they use learned best responses.