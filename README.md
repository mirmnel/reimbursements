# Japanese Auctions with Costly Bidding

Companion code for the paper's two learning experiments. The paper (`arxiv.tex`, `www.tex`) describes the
experiments and how runs are scored.

```bash
pip install -e .                        # numpy, matplotlib
python3 -m experiments.scale            # experiment 1, writes results/scale.csv (seeds already saved are skipped)
python3 -m experiments.schedule_search  # experiment 2, writes results/schedule_search.csv (policies already saved are skipped)
python3 -m experiments.plots            # the ranked schedule figure, from results/schedule_search.csv
python3 tests/test_paper.py             # the two-bidder closed forms, ~1 s (each test file runs on its own)
```

As described in the paper's "Use of generative AI" section, generative AI tools helped write this code.