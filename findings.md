# Findings So Far

All experimental findings of the project up to 2026-11-04, with what each one means for the learned allocator as our contribution. The method itself is described in [trajectory-labeling-engine.md](trajectory-labeling-engine.md). The step-by-step history of how it was found is in [experiment-history.md](experiment-history.md).

---

## Summary

- **Budget allocation across trajectories is a large lever** for query accuracy on realistic, variable-length data. Allocating more evenly than size-proportional gains +3 to +5 points on the mean of kNN, range and where-at scores.
- **A small allocator trained on query accuracy** (3 features, no labels) gets the best mean score at every budget. It improves every selector tested (RDP, SED, MLSimp), and it beats the best hand-tuned rule by +0.6 to +1.0 points.
- **Most of the gain comes from allocation itself, not from the learning.** A simple rule (size^0.25), or fixed per-trajectory budgets, captures most of it.
- **The effect depends on the data.** On MLSimp's benchmark, where every trajectory has exactly 1,000 points, the allocator adds nothing over SED.
- **It holds on a second real dataset.** On Porto taxi trips (less length variation than GeoLife), allocation gains +2 to +4 points at the same points per trajectory, about 70–85% of the GeoLife effect. The learned part again adds little over size^0.25 (−0.0 to +0.8).
- **The gain grows with length variation.** On ship trips (Gulf of Mexico AIS, length CV 1.82), the allocator gains +7 to +12 points at the same points per trajectory, against +3 to +5 on GeoLife (CV 1.12) and +2 to +4 on Porto (CV 0.74). Here the learned model also beats the fixed size^0.25 rule by up to +3.6, because the best size exponent changes with the budget.
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

## 11. Second dataset: Porto taxi trips

**Setup:**
- Porto taxi trajectories (ECML/PKDD 2015, UCI mirror). Random sample of 40,000 trips with `MISSING_DATA = False`, at least 10 points, inside the Porto area, and no step above 50 m/s.
- After the split (seed 42): 34,000 database trajectories, 1.69M points. kNN pool halves of 3,000 train and 3,000 test queries. Range and where-at as in the setup above.
- Porto has no per-point timestamps. Times are the trip start plus 15 s per point.
- The engine's projection is fixed at GeoLife's origin, so coordinates are moved there, with the longitude offset scaled by cos(41.15°)/cos(39.95°) so that metric distances are kept.
- 100 ES evaluations per budget (150 on GeoLife). SED selector.

**Length variation:**

| Dataset | Trajectories | Mean length | Median | p5–p95 | Max | CV |
|---|---|---|---|---|---|---|
| GeoLife | 34,193 | 552 | 352 | 115–1,654 | 12,488 | 1.12 |
| Porto sample | 40,000 | 50 | 43 | 17–102 | 1,099 | 0.74 |

Porto trips are about 10× shorter, so at GeoLife's budget shares almost the whole budget goes to endpoints. The run therefore adds larger shares that match GeoLife's points per trajectory.

**Test scores (mean of kNN, range and where-at):**

| Budget | Kept per trajectory | Size-proportional | size^0.25 | **Learned allocator** | Δ vs size-prop. | Δ vs size^0.25 |
|---|---|---|---|---|---|---|
| 0.25% | 2.1 | 0.367 | 0.373 | 0.373 | +0.6 | −0.0 |
| 1% | 2.5 | 0.411 | 0.423 | 0.427 | +1.6 | +0.4 |
| 3% | 3.4 | 0.520 | 0.549 | 0.553 | +3.3 | +0.4 |
| 6% | 4.9 | 0.643 | 0.678 | 0.686 | +4.3 | +0.8 |
| 12% | 7.7 | 0.788 | 0.821 | 0.823 | +3.5 | +0.2 |
| 24% | 13.5 | 0.899 | 0.919 | 0.921 | +2.3 | +0.2 |

**Compared with GeoLife at the same points per trajectory** (learned allocator vs size-proportional):

| Kept per trajectory | GeoLife | Porto |
|---|---|---|
| about 3.4 | +4.6 (0.25%) | +3.3 (3%) |
| about 4.9 | +5.1 (0.5%) | +4.3 (6%) |
| about 7.6 | +4.8 (1%) | +3.5 (12%) |
| about 13 | +3.3 (2%) | +2.3 (24%) |

**Findings:**
1. **Allocation matters on a second real dataset.** The gain is +2.3 to +4.3 points at 3–24%, about 70–85% of the GeoLife effect at the same points per trajectory. The smaller effect fits Porto's lower length variation, which supports the claim that the effect follows length variation.
2. **At GeoLife's budget shares (0.25–1%), there is little to allocate.** At 0.25%, only about 4k of the 72k points are not endpoints. Budgets should be compared by points per trajectory, not only by share.
3. **The learned part adds −0.0 to +0.8 over size^0.25.** This is consistent with GeoLife (+0.6 to +1.0), and probably within noise at several budgets.
4. **The learned weights agree with GeoLife at 3% and above:**
   - size +0.11 to +0.26;
   - curve −0.16 to −0.41;
   - density about 0, which fits a dataset where all trips are in one dense city.

   At 0.25–1% the weights are unstable (curve +0.85, then −1.29), and at 0.25% the train gain does not carry over to test. With almost nothing to allocate, the ES fits noise.
5. **The GeoLife-trained kNN-only θ** (only available at 0.25% and 1%) is no better than size^0.25 on Porto. This is another sign that θ must be retrained for each dataset (section 10).

**Caveats:** one seed, a 40k-trip sample, 100 evaluations, and no standard errors.

Config: `engine/configs/geolife.json` with `data_path` pointing to the Porto parquet, `export_results` false, and `budget_shares` [0.0025, 0.01, 0.03, 0.06, 0.12, 0.24]. Run with `--train 0 --test 0 --combined 1 --evaluations 100` (about 15 minutes on 4 cores). Converter: `porto_to_parquet.py <train.csv> <out.parquet> 40000` (session scratchpad; it should be moved into the repo).

## 12. Third dataset: ship trips with very varying lengths (AIS)

**Setup:**
- NOAA Marine Cadastre AIS, 15–16 January 2023, Texas–Louisiana Gulf coast (lat 27.5–30.5, lon −95.5 to −89.0). Median sampling 70 s.
- A trip is one vessel's moving segment: split at gaps over 10 minutes and wherever SOG < 0.5 kn. At least 10 points and 500 m extent; steps above 30 m/s dropped.
- 13,167 trips. After the split (seed 42): 11,192 database trajectories, 1.0M points. kNN pool halves of 987 train and 988 test queries.
- Coordinates are moved to the engine's origin as in section 11. `where_at.max_gap_seconds` is 180 (60 elsewhere) to fit the sampling rate.
- 100 ES evaluations per budget. SED selector.

**Length variation:**

| Dataset | Trajectories | Mean length | Median | p5–p95 | Max | CV |
|---|---|---|---|---|---|---|
| Porto sample | 40,000 | 50 | 43 | 17–102 | 1,099 | 0.74 |
| GeoLife | 34,193 | 552 | 352 | 115–1,654 | 12,488 | 1.12 |
| **Gulf AIS** | 13,167 | 89 | 27 | 10–414 | 2,062 | **1.82** |

**Test scores (mean of kNN, range and where-at):**

| Budget | Kept per trajectory | Size-proportional | size^0.25 | **Learned allocator** | Δ vs size-prop. | Δ vs size^0.25 |
|---|---|---|---|---|---|---|
| 0.25% | 2.2 | 0.427 | 0.434 | 0.446 | +1.9 | +1.2 |
| 0.5% | 2.4 | 0.442 | 0.460 | 0.478 | +3.6 | +1.8 |
| 1% | 2.9 | 0.471 | 0.494 | 0.528 | +5.7 | +3.3 |
| 1.6% | 3.4 | 0.503 | 0.571 | 0.581 | +7.7 | +1.0 |
| 2% | 3.7 | 0.523 | 0.600 | 0.610 | +8.7 | +0.9 |
| 3.3% | 4.9 | 0.578 | 0.675 | 0.679 | +10.1 | +0.3 |
| 6.4% | 7.6 | 0.673 | 0.772 | 0.791 | +11.9 | +2.0 |
| 12.6% | 13.0 | 0.815 | 0.848 | 0.884 | +7.0 | +3.6 |

**All three datasets at the same points per trajectory** (learned allocator vs size-proportional):

| Kept per trajectory | Porto (CV 0.74) | GeoLife (CV 1.12) | Gulf AIS (CV 1.82) |
|---|---|---|---|
| about 3.4 | +3.3 | +4.6 | +7.7 |
| about 4.9 | +4.3 | +5.1 | +10.1 |
| about 7.6 | +3.5 | +4.8 | +11.9 |
| about 13 | +2.3 | +3.3 | +7.0 |

**Findings:**
1. **The gain grows with length variation.** At every matched budget the order is Porto < GeoLife < AIS, the same order as their length CV. On AIS the gain over size-proportional is +7 to +12 points, about twice the GeoLife effect. This is the strongest evidence so far that the effect comes from length variation.
2. **The learned model beats the fixed hand rule by more here:** +0.3 to +3.6 points, against +0.6 to +1.0 on GeoLife.
3. **The reason is that the best size exponent depends on the budget.** The learned size weight rises from about 0 at 0.25–1% to +0.51 at 6.4% and +0.70 at 12.6%. A fixed size^0.25 is too flat at large budgets: at 12.6% it drops range recall from 0.76 to 0.60, and the learned model recovers most of that (0.73) while keeping kNN and where-at.
4. **Density weight is positive at every budget** (+0.05 to +0.16). The curve weight is near 0 and unstable.
5. **Range recall is low on this dataset** (0.21–0.76 at all budgets), probably because 2 km boxes in open water hold few trips. Range is the query type most hurt by flat allocation.

**Caveats:**
- One seed, one region, two days, 13k trips, 988 kNN test queries. No standard errors.
- The gain over size^0.25 is partly because α = 0.25 was fixed on GeoLife. A size^α rule tuned per budget on AIS is the fair comparison, and would likely close part of the gap at 6.4–12.6%.

Config: as in section 11, with `data_path` pointing to the AIS parquet, `max_gap_seconds` 180 and `budget_shares` [0.0025, 0.005, 0.01, 0.016, 0.02, 0.033, 0.064, 0.126]. Run with `--train 0 --test 0 --combined 1 --evaluations 100` (about 3 minutes on 4 cores). Converter: `ais_to_parquet.py <out.parquet> 40000 AIS_2023_01_15.csv AIS_2023_01_16.csv` (session scratchpad; it should be moved into the repo).

---

## What this means for the learned allocator as our contribution

**What strengthened it:**
- A large, robust gain on realistic variable-length data: +3.3 to +5.1 over SED, with the same gain on training and test.
- It improves every selector tested: RDP, SED and MLSimp.
- MLSimp's implicit allocation is a measurable weakness that the allocator fixes.
- The allocation effect reproduces on Porto: +2.3 to +4.3 at matched points per trajectory (section 11).
- Across three datasets, the gain follows length variation: Porto < GeoLife < AIS, with +7 to +12 on AIS (section 12). On AIS the learned model also beats the fixed hand rule by up to +3.6, because the best size exponent changes with the budget.

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
2. Repeat the main result on a second real dataset with natural length variation. First evidence on Porto is in section 11. It still needs seeds and standard errors.
3. Present the learned model as the way to find the best rule automatically, for any dataset and any query weighting. A Pareto front across query weightings shows what a single hand rule cannot do.
4. Retrain θ with MLSimp's metrics as the reward on MLSimp's benchmark. Uniform budgets are inside the search space, so it should at least match them. The external evidence that holds already is that **fixed per-trajectory budgets improve MLSimp under its own metrics** (section 10).

## Caveats that apply throughout

- One seed for the split and the workloads. Standard errors exist only for the kNN-only runs.
- Range and where-at differences below 1 point need a second seed before they are quoted.
- Sections 7 and 8 use small query sets (150–300 kNN queries).
- MLSimp is scored with our metrics in section 8 and with its own in section 10. Its own queries are only 100 grid cells per distribution, so its metrics are noisy as well.
