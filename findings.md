# Findings So Far

All experimental findings of the project up to 2026-11-04, with what each one means for the learned allocator as our contribution. The method itself is described in [trajectory-labeling-engine.md](trajectory-labeling-engine.md). The step-by-step history of how it was found is in [experiment-history.md](experiment-history.md).

---

## Summary

- **Budget allocation across trajectories is a large lever** for query accuracy on realistic, variable-length data. Allocating more evenly than size-proportional gains +3 to +5 points on the mean of kNN, range and where-at scores.
- **A small allocator trained on query accuracy** (3 features, no labels) gets the best mean score at every budget. It improves every selector tested (RDP, SED, MLSimp), and it beats the best hand-tuned rule by +0.6 to +1.0 points.
- **Most of the gain comes from allocation itself, not from the learning.** A simple rule (size^0.25), or fixed per-trajectory budgets, captures most of it.
- **The effect depends on the data.** On MLSimp's benchmark, where every trajectory has exactly 1,000 points, the allocator adds nothing over SED.
- **Learned point selection (a ranker) failed** when it scored each point once and kept the top n. A label oracle shows there is headroom at small budgets, but only for step-by-step selection.
- **On our metrics, MLSimp is 7–11 points below plain SED.** On MLSimp's own metrics it wins only when it is given the test queries. Without them, SED is better at almost every budget.
- **Fixed per-trajectory budgets improve MLSimp on both our metrics and its own,** by up to +3.7 points.
- **The learned θ does not transfer** to MLSimp's equal-length benchmark and metrics. There it is worse than plain uniform budgets, so θ must be retrained for the target metrics and data.

---

## Setup

Unless stated otherwise:
- GeoLife, random split (seed 42): 29,064 database trajectories, 16.1M points (mean 554 per trajectory).
- kNN queries come from the held-out pool, split into a training half (2,564) and a test half (2,565), with k = 10. kNN uses discrete Fréchet distance on the kept points.
- Range: 2,000 queries per workload, scored by recall.
- Where-at: 3.72M probes per workload, scored by average hit (25–500 m tolerance).
- "Mean" is the equal-weight average of kNN recall@k, range recall and where-at average hit.
- All scores are on **test** workloads (separate seeds, the kNN test half), never seen in training.
- Budgets are 0.25 / 0.5 / 1 / 2% of the points.
- Tool: `engine/apps/knn_alloc_oracle.cpp`. Commands are listed under each finding.

---

## 1. Allocation matters, especially for kNN

Making allocation flatter than size-proportional improves kNN recall@k by +4.1 to +7.1 points (RDP selector, SE ≈ 0.3).

The cause: Fréchet distances on sparse trajectories inflate, so short trajectories that size-proportional leaves at 2–3 points drop out of top-k lists. Where-at and range also count every trajectory equally, regardless of its length.

`--train 0 --test 0`

## 2. Labels from specific queries overfit

A greedy oracle that allocates points to maximize recall on the training queries gains up to **+15** points on them, but only +1 to +2 on test queries, which is no better than a uniform allocation. Each trajectory appears in only a few queries, so per-trajectory targets are mostly noise.

**Consequence:** the allocator is trained on the evaluator's score, not on labels.

`--train 0 --test 0`

## 3. Three features are enough

| Model (kNN-only training) | Δ test recall@k vs size^0.25 (0.25 / 0.5 / 1 / 2%) |
|---|---|
| Linear, 3 features (size, density, curve) | +0.6 / +0.6 / +0.7 / +2.3 |
| Linear, 9 features | +0.6 / +0.8 / +0.7 / +2.5 |
| 9 features + MLP | +0.6 / +0.8 / +0.9 / +2.5 |

More features and an MLP added at most +0.25 points, within 1–2 standard errors.

`--features 1`, `--model 1 --evaluations 300`

## 4. Query types conflict

- A flat allocation helps kNN and where-at but costs 1–2 points of range.
- An allocator trained on kNN alone gains +1.4 kNN at 2%, but loses 4.3 where-at and 1.2 range, so its mean drops below the hand rule.
- With the same allocation, SED beats RDP on where-at by +4.4 to +5.8 points.

**Consequence:** the allocator must be trained on all three query types together, with SED as the selector.

`--check 1`

## 5. The learned allocator, trained on all three query types

Test scores, SED selector:

| Budget | Size-proportional | size^0.25 | **Learned allocator** | Δ vs size-prop. | Δ vs size^0.25 |
|---|---|---|---|---|---|
| 0.25% | 0.604 | 0.640 | **0.650** | +4.6 | +1.0 |
| 0.5% | 0.685 | 0.727 | **0.736** | +5.1 | +0.8 |
| 1% | 0.778 | 0.818 | **0.826** | +4.8 | +0.8 |
| 2% | 0.860 | 0.887 | **0.893** | +3.3 | +0.6 |

- The training gains match the test gains (+0.8 / +0.9 / +0.7 / +0.6 over size^0.25 on training), so the model generalizes.
- Range comes back to the size-proportional level at 1–2%, while kNN (+5.7 to +7.0) and where-at (+3.4 to +9.2) keep most of their gains.
- The learned weights are stable across budgets:
  - size +0.20 to +0.31 (much flatter than size-proportional's 1.0);
  - density +0.06 to +0.10 (slightly more points in crowded areas);
  - curve −0.31 to −0.45 (fewer points for winding trajectories).

`--combined 1 --evaluations 150` (about 35 minutes)

## 6. The allocator helps every engine selector

Test mean, full database:

| Budget | RDP | SED | Learned + RDP | **Learned + SED** |
|---|---|---|---|---|
| 0.25% | 0.594 | 0.604 | 0.639 | **0.650** |
| 0.5% | 0.664 | 0.685 | 0.715 | **0.736** |
| 1% | 0.747 | 0.778 | 0.789 | **0.826** |
| 2% | 0.827 | 0.860 | 0.855 | **0.893** |

- Learned allocator gains: +2.8 to +5.1 with RDP, +3.3 to +5.1 with SED.
- Simplification time for 16M points:
  - RDP: 0.1–0.2 s;
  - SED: about 2 s;
  - computing the learned budgets adds 0.1–0.3 s, plus about 1 s for the features.

`--compare 1`

## 7. Learned point selection: headroom exists, but a top-n ranker fails

Run on a 2,000-trajectory subset with 300 kNN test queries. Test mean:

| Budget | Size-prop. + SED | Learned alloc + SED | Size-prop. + ranker | Learned alloc + ranker | Learned alloc + label oracle |
|---|---|---|---|---|---|
| 0.25% | 0.726 | 0.777 | 0.690 | 0.722 | 0.820 |
| 0.5% | 0.790 | 0.845 | 0.732 | 0.764 | 0.874 |
| 1% | 0.866 | 0.915 | 0.789 | 0.818 | 0.920 |
| 2% | 0.932 | 0.962 | 0.851 | 0.884 | 0.954 |

- **Label oracle:** within each trajectory, it greedily keeps the point with the largest marginal gain on the training range and where-at queries. On top of the allocator it adds +4.3 / +2.9 / +0.5 / −0.8 points. Most of that comes from range (up to +9.8), and it costs up to 3 points of kNN, because kNN is not in its labels. Unlike the allocation oracle, these labels generalize to new queries. The trajectories were the same ones the labels came from, though, so this is an optimistic bound.
- **Ranker:** a linear pairwise ranker (RankNet / LambdaRank loss, 11 point features, 2-fold cross-fitting over trajectories) that scores each point once and keeps the top n. It is **3.6 to 8.1 points below SED**. It picks clusters of redundant points and leaves long gaps, which hurts where-at most.
- **Consequence:** learned selection only makes sense as a step-by-step selector that re-scores candidates against the points already kept. It needs kNN in its labels, or a shape feature.

`--selection 1`, `--ranker 1` (with `--db-limit 2000 --train 300 --test 300`)

## 8. MLSimp, scored with our metrics

Data: MLSimp's GeoLife test set (2,000 trajectories of exactly 1,000 points each), split into 1,700 database trajectories and a kNN query pool (150 test queries). MLSimp's GNN-TS scores were recomputed with dropout off. Test mean:

| Method | 0.25% | 0.5% | 1% | 2% |
|---|---|---|---|---|
| MLSimp (its own global sampling) | 0.694 | 0.755 | 0.818 | 0.877 |
| MLSimp sampling + fixed size-proportional budgets | 0.708 | 0.769 | 0.840 | 0.896 |
| **MLSimp sampling + learned allocator** | **0.713** | **0.774** | **0.842** | **0.898** |
| SED | 0.765 | 0.857 | **0.930** | **0.969** |
| Learned allocator + SED | **0.772** | **0.858** | 0.928 | 0.967 |

**Allocator with MLSimp:**
- The allocator improves MLSimp by **+1.9 to +2.4** points.
- Fixed per-trajectory budgets give +1.4 to +2.2 of that; the learned weights add +0.2 to +0.5.
- MLSimp's global, score-proportional sampling allocates points unevenly, and that hurts every query type.

**MLSimp vs SED:**
- MLSimp is **7 to 11 points below SED**, mostly on where-at (−13 to −20) and Fréchet kNN (−7 to −9).
- Keeping MLSimp's top-scored points instead of sampling them is even worse (about 0.61–0.64), the same clustering failure as the ranker in section 7.

**Allocator on this dataset:**
- Over SED it adds **+0.6 to −0.2**, i.e. nothing.
- With equal-length trajectories, size-proportional is the same as uniform, so the main effect from sections 1 and 5 cannot appear.

**Corrections to earlier reports:**
- A first run paired cached scores with trajectories from a different copy of MLSimp's `test.pkl`. Those numbers are void.
- The cached scores came from a run with GNN dropout active.

`--mlsimp mlsimp_test_eval.csv --evaluations 80`. The CSV is produced by `mlsimp_eval_scores.py` (session scratchpad; it should be moved into the repo).

## 9. Notes on MLSimp's protocol

- **Allocation:** each point's score is 0.5 × GNN importance plus 0.5 × a bonus if the point lies inside a range query box. MLSimp then samples `cr × N × 1000` points across the database, with probability proportional to score, and keeps the endpoints. There is no explicit per-trajectory budget.
- **The bonus uses the evaluation queries.** `adj_query_data` is identical to `test_query_data`, so the published protocol sees the test queries during simplification. In our comparisons, the bonus is built from training queries.
- **The benchmark is length-normalised:** GeoLife is cut into 1,000-point trajectories. That hides the allocation effect, which explains why MLSimp's results do not show it.
- **Training is query-agnostic** (self-supervised contrastive losses plus mutual learning with a diffusion model). Queries only enter through the bonus at simplification time.

## 10. MLSimp's own metrics

**Setup:**
- MLSimp's own evaluation code (`range_query_operator`, range F1; `knn_edr_query_online`, kNN-EDR with k = 3), its own query workloads (`test_query_*`) and its cached ground truth.
- All 2,000 trajectories of its test set (1,000 points each), at its budgets `cr × 2,000 × 1,000` plus endpoints: 9k / 14k / 24k / 44k points.
- The learned allocator uses the θ trained on our engine's GeoLife split with our metrics. It was **not** retrained for this data or these metrics.
- "Mean" = (range F1 + kNN-EDR) / 2.

**'data' query distribution:**

| Method | cr 0.25% | 0.5% | 1% | 2% |
|---|---|---|---|---|
| MLSimp, published (bonus from the test queries) | 0.848 | 0.867 | 0.882 | 0.891 |
| MLSimp, published + fixed uniform budgets | **0.850** | **0.892** | **0.906** | **0.918** |
| MLSimp, published + learned allocator | 0.829 | 0.878 | 0.895 | 0.900 |
| MLSimp, no query bonus | 0.636 | 0.673 | 0.747 | 0.797 |
| MLSimp, no query bonus + learned allocator | 0.635 | 0.672 | 0.751 | 0.810 |
| SED (= uniform budgets here) | 0.690 | 0.782 | 0.843 | 0.899 |
| Learned allocator + SED | 0.695 | 0.771 | 0.842 | 0.891 |

**'gau' query distribution:**

| Method | cr 0.25% | 0.5% | 1% | 2% |
|---|---|---|---|---|
| MLSimp, published | 0.829 | **0.843** | 0.848 | 0.851 |
| MLSimp, published + fixed uniform budgets | **0.830** | 0.842 | 0.859 | 0.888 |
| MLSimp, published + learned allocator | 0.815 | 0.827 | 0.830 | 0.849 |
| MLSimp, no query bonus | 0.656 | 0.681 | 0.755 | 0.818 |
| MLSimp, no query bonus + learned allocator | 0.710 | 0.739 | 0.764 | 0.812 |
| SED | 0.757 | 0.816 | **0.876** | **0.903** |
| Learned allocator + SED | 0.746 | 0.798 | 0.851 | 0.886 |

**Findings:**
1. **MLSimp's published protocol wins on range F1 because it is told the test queries.**
   - Range F1 is 0.95–0.99 with the bonus.
   - Without the bonus it is 0.59–0.86, and MLSimp falls below SED at every budget on the 'data' distribution (−5 to −11) and at most budgets on 'gau'.
   - Its learned point scores alone are weaker than SED on its own metrics.
2. **Fixed per-trajectory budgets improve MLSimp, even on its own metrics and data:** up to +2.7 points ('data') and +3.7 ('gau' at 2%). This confirms section 8: MLSimp's global score-proportional sampling allocates points badly.
3. **The learned allocator, transferred without retraining, is worse than uniform budgets here.**
   - With published MLSimp: −0.9 to −2.1 vs uniform budgets ('data'), and −1.1 to −3.9 ('gau').
   - With SED: −2.4 to +0.5.
   - The only clear gains are with no-bonus MLSimp at low budgets on 'gau' (+5.3, +5.8).
   - Its density and curve weights were trained for our metrics on variable-length data. On equal-length data, with range F1 and EDR, they move points to the wrong places.

**Consequence:** θ does not transfer across metrics and data. That is consistent with how the method is meant to work (train θ on the workload and data you care about), but it means the "trained once, reuse anywhere" argument does not hold without retraining. The fair test of the allocator on this benchmark is to retrain θ with MLSimp's metrics as the reward. Uniform budgets are inside its search space (all exponents 0), so a retrained θ should at least match them.

Script: `mlsimp_metrics.py` (session scratchpad; it should be moved into the repo), run from the MLSimp directory after the `--mlsimp ... --export DIR` export. About 15 minutes.

---

## What this means for the learned allocator as our contribution

**What strengthened it:**
- A large, robust gain on realistic variable-length data: +3.3 to +5.1 over SED, with the same gain on training and test.
- It improves every selector tested: RDP, SED and MLSimp.
- MLSimp's implicit allocation is a measurable weakness that the allocator fixes.

**What weakened it:**
- On length-normalised data (MLSimp's benchmark) it adds nothing over SED.
- The *learned* part adds less than the *allocation* part:
  - +0.6 to +1.0 over the best hand rule on full GeoLife;
  - +0.2 to +0.5 over fixed budgets with MLSimp.
- **θ does not transfer** to other metrics and data without retraining. Under MLSimp's metrics on MLSimp's data, the transferred θ is worse than uniform budgets (section 10).

**A claim that holds:**

> Budget allocation across trajectories is an overlooked lever in query-driven simplification. On realistic, variable-length data, a small allocator trained on query accuracy improves every selector we tested (RDP, SED, MLSimp) by 2–5 points, and beats hand-tuned rules. On length-normalised benchmarks the effect largely disappears, which explains why prior work has not reported it.

**A claim to avoid:** "a learned model is much better than simple allocation rules".

**How to strengthen it:**
1. Argue that real data is not length-normalised, and that uploads to the service look like the engine's GeoLife split.
2. Repeat the main result on a second real dataset with natural length variation (T-Drive or Porto).
3. Present the learned model as the way to find the best rule automatically, for any dataset and any query weighting. A Pareto front across query weightings shows what a single hand rule cannot do.
4. Retrain θ with MLSimp's metrics as the reward on MLSimp's benchmark. Uniform budgets are inside the search space, so it should at least match them. The external evidence that holds already is that **fixed per-trajectory budgets improve MLSimp under its own metrics** (section 10).

## Caveats that apply throughout

- One seed for the split and the workloads. Standard errors exist only for the kNN-only runs.
- Range and where-at differences below 1 point need a second seed before they are quoted.
- Sections 7 and 8 use small query sets (150–300 kNN queries).
- MLSimp is scored with our metrics in section 8 and with its own in section 10. Its own queries are only 100 grid cells per distribution, so its metrics are noisy as well.
