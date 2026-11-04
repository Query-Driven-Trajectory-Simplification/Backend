# Learned Budget Allocator for Query-Driven Trajectory Simplification

The ML method of the project: a small model that decides how many points each trajectory keeps, trained directly on query accuracy for kNN, range and where-at queries. It needs no labels.

---

## 1. Goal

Minimize query error on the compressed database for all three query types at budgets of 0.25–2% of the points. The objective is the mean of the three engine metrics:

```
reward = mean(knn.recall_at_k, range_recall, where_at.average_hit)
```

## 2. Structure

```
trajectories ─► 3 features ─► linear score ─► weight ─► points per trajectory ─► SED ─► compressed DB
```

**1. Features per trajectory.** Computed from the data alone, without queries:

| Feature | Definition |
|---|---|
| size | number of points |
| density | distinct trajectories per 500 m grid cell, averaged over the cells the trajectory visits |
| curve | path length / bounding-box diagonal (1 = straight line) |

Each feature is log-transformed and z-scored over the dataset.

**2. Model.** Linear, with three learned parameters θ:

```
score_i  = θ₁·size_i + θ₂·density_i + θ₃·curve_i
weight_i = exp(score_i)
```

**3. Points per trajectory.**
- Every trajectory keeps its 2 endpoints (`allocation::kMinPointsPerTrajectory`).
- The rest of the budget is shared in proportion to `weight_i`, capped at the trajectory's length.

**4. Selection.** SED keeps that many points of each trajectory.

## 3. Training

There are no labels. The model is trained on the Evaluator's score:

```
θ = start at size^0.25
repeat ~150 times:
    θ' = θ + σ · noise
    allocate with θ', simplify with SED, run the Evaluator on the training workloads
    if reward improved: keep θ', grow σ;  else: shrink σ
```

- **Optimizer:** a (1+1) evolution strategy. No gradients are needed.
- **Parameters:** one θ per budget share, so 3 × 4 numbers in total.
- **Workloads:**
  - Training: the kNN train half of the held-out pool, plus range queries and where-at probes generated with training seeds.
  - Test: the kNN test half, plus range and where-at generated with other seeds. Test workloads are used only for the final comparison.
- **Ground truth:** the only ground truth is the reference answer of each query on the original data, which the generators already produce.

## 4. Inference

1. Compute the three features (one pass over the data).
2. Apply the stored θ for the chosen budget share.
3. Allocate the budget and run SED.

No queries are needed and the cost is linear in data size, so the method works for new uploads in the service.

## 5. Why this design

Experiments on GeoLife (29,064 trajectories, 16.1M points; kNN scored on 2,565 held-out queries):

- **Allocation matters.** A flatter allocation than size-proportional improves kNN recall@k by +4 to +7 points and where-at by +3 to +8, and costs about 1–2 points of range recall.
- **Labels from specific queries overfit.** An oracle fitted to training queries gained up to +15 points on them, but almost nothing on test queries.
- **Three features are enough.** Nine features or an MLP added nothing over this linear model.
- **Training has to use all three query types.** A model trained on kNN alone improved kNN but lowered where-at and the mean score.

## 6. Results

**Setup:**
- GeoLife, random split (seed 42): 29,064 database trajectories, 16.1M points. SED selector.
- Training workloads: 2,564 kNN queries (k = 10), 2,000 range queries, 3.72M where-at probes.
- Test workloads: 2,565 different kNN queries, plus range and where-at generated with a different seed.
- 150 training evaluations per budget share, about 8 minutes each.

All scores below are on the **test** workloads.

| Budget | Allocation | kNN R@k | Range | Where-at | **Mean** |
|---|---|---|---|---|---|
| 0.25% | Size-proportional | 0.479 | 0.745 | 0.589 | 0.604 |
| | size^0.25 | 0.537 | 0.729 | 0.654 | 0.640 |
| | Trained on kNN only | 0.537 | 0.729 | 0.654 | 0.640 |
| | **Trained on all three** | 0.536 | 0.738 | 0.676 | **0.650** |
| 0.5% | Size-proportional | 0.552 | 0.813 | 0.690 | 0.685 |
| | size^0.25 | 0.619 | 0.793 | 0.770 | 0.727 |
| | Trained on kNN only | 0.613 | 0.771 | 0.744 | 0.709 |
| | **Trained on all three** | 0.620 | 0.806 | 0.782 | **0.736** |
| 1% | Size-proportional | 0.636 | 0.881 | 0.815 | 0.778 |
| | size^0.25 | 0.702 | 0.871 | 0.881 | 0.818 |
| | Trained on kNN only | 0.714 | 0.871 | 0.865 | 0.817 |
| | **Trained on all three** | 0.709 | 0.882 | 0.886 | **0.826** |
| 2% | Size-proportional | 0.721 | 0.937 | 0.921 | 0.860 |
| | size^0.25 | 0.777 | 0.929 | 0.954 | 0.887 |
| | Trained on kNN only | 0.791 | 0.919 | 0.911 | 0.874 |
| | **Trained on all three** | 0.785 | 0.938 | 0.955 | **0.893** |

**Findings:**
- **Best mean score at every budget.**
  - vs size-proportional: +4.6 / +5.1 / +4.8 / +3.3 points (0.25 / 0.5 / 1 / 2%)
  - vs size^0.25: +1.0 / +0.8 / +0.8 / +0.6 points
- **It generalizes.** The gain over size^0.25 is the same on training and test workloads: +0.8 / +0.9 / +0.7 / +0.6 on training.
- **It removes the range penalty of the flat rule** while keeping most of its kNN and where-at gains:
  - range is back to size-proportional level at 1–2%, and within 0.7 points at 0.25–0.5%;
  - kNN is +5.7 to +7.0 over size-proportional;
  - where-at is +3.4 to +9.2 over size-proportional.
- **The kNN-only model is worse on the mean than the hand rule.** Training on all three query types is what makes the model useful.
- **The learned weights are stable across budgets:**

  | Budget | size | density | curve |
  |---|---|---|---|
  | 0.25% | +0.20 | +0.10 | −0.40 |
  | 0.5% | +0.21 | +0.08 | −0.45 |
  | 1% | +0.29 | +0.06 | −0.31 |
  | 2% | +0.31 | +0.10 | −0.43 |

  The pattern: a much flatter allocation than size-proportional, slightly more points in dense areas, and fewer points for winding trajectories. One shared θ for all budgets would likely do almost as well.

**Caveats:**
- One seed and one dataset.
- No standard errors in this run. Earlier kNN runs had a standard error of about 0.3 points on similar differences.
- Range and where-at differences below 1 point should be confirmed with a second seed.

## 7. Comparison

Every method uses the same database split and the same test workloads (fixed seeds), and is scored with the engine's `Evaluator`. Results are reported per metric and as the mean, at each budget share.

| Method | Type |
|---|---|
| RDP, size-proportional | Current pipeline |
| SED, size-proportional | Current pipeline |
| SED, size^0.25 | Hand rule; best non-learned method so far |
| MLSimp | Learned baseline (self-supervised point importance) |
| **Learned allocator + SED** | This method |

## 8. Implementation

| Part | Location |
|---|---|
| Experiments so far | `engine/apps/knn_alloc_oracle.cpp`. The results in section 6 come from `./build-release/engine/knn_alloc_oracle engine/configs/geolife.json --train 0 --test 0 --combined 1 --evaluations 150` (about 35 minutes) |
| Features and allocator | new `engine/src/allocation/learned.cpp`, `AllocatorType::Learned` |
| Training app (writes θ to JSON) | new `engine/apps/train_allocator.cpp` |
| θ per budget share in the config | `pipeline/config.cpp` (`load_config` rejects unknown keys, so extend the key list) |
