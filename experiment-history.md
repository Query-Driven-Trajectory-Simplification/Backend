# Experiment History: How the Learned Allocator Was Found

The earlier experiments behind [trajectory-labeling-engine.md](trajectory-labeling-engine.md), in the order they were run. Each one led to the design decision noted with it.

**Setup for all runs:**
- GeoLife, random split (seed 42): 29,064 database trajectories, 16.1M points (mean 554 per trajectory).
- The held-out pool is split into kNN train queries (2,564) and kNN test queries (2,565), with k = 10.
- Unless stated otherwise: RDP selector, 500 range queries, 3.72M where-at probes.
- ± values are paired standard errors over queries.
- Tool: `engine/apps/knn_alloc_oracle.cpp`.

The original per-trajectory labeling plan (utility-curve labels as training targets) is in git history, commit `6ff46a7`, Appendix A of `trajectory-labeling-engine.md`.

---

## 1. Allocation matters for kNN

**Question:** does moving points between trajectories change kNN accuracy?

Test recall@k, RDP:

| Allocation | 0.25% | 0.5% | 1% | 2% |
|---|---|---|---|---|
| Size-proportional (current) | 0.479 | 0.544 | 0.617 | 0.701 |
| size^0.5 | 0.514 | 0.598 | 0.669 | 0.737 |
| size^0.25 | 0.550 | 0.615 | 0.681 | 0.743 |
| Uniform | 0.546 | 0.614 | 0.680 | 0.744 |
| size^−0.5 | 0.516 | 0.592 | 0.663 | 0.730 |

A flatter allocation gains +4.1 to +7.1 recall@k (SE ≈ 0.3), with the same gain on train and test. Fréchet distances on sparse trajectories inflate, so short trajectories that size-proportional leaves at 2–3 points drop out of top-k lists.

**Decision:** allocation is worth learning.

## 2. Query-specific oracles overfit

**Question:** can per-query targets (labels) drive the allocation?

A greedy oracle allocated points to maximize recall on the training queries, using precomputed Fréchet distances to each query's 40 nearest candidates.

| Oracle variant | Train Δ vs size-prop. | Test Δ vs size-prop. |
|---|---|---|
| Greedy from 2 points | +1.8 to +9.1 | +1.2 to +2.4 |
| Greedy on half size-proportional | +3.5 to +11.8 | −1.2 to +1.5 |
| Greedy on half uniform | +7.9 to +15.2 | +4.3 to +5.8 (no better than plain uniform) |

Each trajectory appears in only a few training queries, so per-trajectory targets are mostly noise. The oracle's internal recall estimate was also 3–6 points too high, because trajectories outside the 40 candidates do enter the top-k at low budgets.

**Decision:** no labels; train on the evaluator's score with a small model.

## 3. A 3-feature model generalizes

**Question:** do features beyond size help?

Weight = size^a · density^b · curve^c, with exponents fitted by grid and coordinate search on train recall@k. Test recall@k:

| Budget | size^0.25 | Best fit | Fitted (size, density, curve) | Δ vs size^0.25 |
|---|---|---|---|---|
| 0.25% | 0.550 | 0.550 | (+0.25, 0, 0) | +0.0 |
| 0.5% | 0.615 | 0.616 | (0, +0.25, 0) | +0.1 ± 0.3 |
| 1% | 0.681 | 0.688 | (+0.25, +0.25, −0.25) | +0.65 ± 0.19 |
| 2% | 0.743 | 0.763 | (+0.25, +0.5, −0.5) | +1.98 ± 0.26 |

**Decision:** keep size, density and curve.

## 4. More features or an MLP add nothing

**Question:** does more model capacity help?

Fitted with a (1+1) evolution strategy, 300 evaluations per model, on train recall@k.

| Model | Δ test recall@k vs size^0.25 (0.25 / 0.5 / 1 / 2%) |
|---|---|
| Linear, 3 features | +0.6 / +0.6 / +0.7 / +2.3 |
| Linear, 9 features (+ duration, interval, speed, extent, nn1, nn10) | +0.6 / +0.8 / +0.7 / +2.5 |
| 9 features + MLP residual (6 tanh units) | +0.6 / +0.8 / +0.9 / +2.5 |

Compared with linear-3, the extra features and the MLP gained at most +0.25 (within 1–2 SE). The extra features' weights stayed near zero. The evolution strategy is weak for the MLP's 75 parameters, but linear-9 is already a clear negative.

A smoke test on 3,000 trajectories and 200 queries had suggested +1.3 to +1.9 for linear-9. That gain disappeared at full scale, so it was noise from the small sample.

**Decision:** keep the 3-feature linear model.

## 5. A kNN-only model hurts the other query types

**Question:** what does the kNN-tuned allocator do to range and where-at?

Mean of kNN recall@k, range recall and where-at average hit (kNN on test queries; range and where-at with the config seed):

| Budget | RDP size-prop. | SED size-prop. | SED size^0.25 | SED linear-3 (kNN-trained) |
|---|---|---|---|---|
| 0.25% | 0.593 | 0.602 | **0.640** | 0.640 (same as size^0.25) |
| 0.5% | 0.664 | 0.683 | **0.727** | 0.708 |
| 1% | 0.747 | 0.776 | **0.814** | 0.814 |
| 2% | 0.828 | 0.860 | **0.884** | 0.871 |

- At 2%, the kNN-trained model gains +1.4 on kNN over size^0.25 but loses 4.3 on where-at and 1.2 on range.
- A flat allocation helps kNN and where-at but costs 1–2 points of range.
- With the same allocation, SED beats RDP on where-at by +4.4 to +5.8.

The final combined-reward training run confirmed this. Scored on its own test workloads, the kNN-only model reached 0.640 / 0.709 / 0.817 / 0.874, below size^0.25 at 0.5–2%.

**Decisions:** train on the combined reward, and use SED as the selector.

## 6. Ideas not pursued (yet)

- **Learned-priority global simplification.** One heap over all points in the database; remove the point with the lowest learned priority, computed from SED error, perpendicular error, time gap, local density and trajectory features. This would learn which points to keep, not only how many. Kept as a stretch goal because it is riskier and more expensive to train.
- **Feature importance.** Permutation importance or drop-one ablation on the 9-feature model trained on the combined reward, to back the choice of 3 features with evidence.

## Reproducing

From the repo root, after building `knn_alloc_oracle` in `build-release`:

| Section | Command | Time |
|---|---|---|
| 1, 2 | `./build-release/engine/knn_alloc_oracle engine/configs/geolife.json --train 0 --test 0` | ~3 min |
| 3 | `... --train 0 --test 0 --features 1` | ~5 min |
| 4 | `... --train 0 --test 0 --model 1 --evaluations 300` | ~48 min |
| 5 | `... --test 0 --check 1` | ~1 min |
