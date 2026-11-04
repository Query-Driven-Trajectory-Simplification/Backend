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

**Status:**
- The kNN-only version of this model was trained and tested.
- The combined-reward version above is the next step.

## 6. Comparison

Every method uses the same database split and the same test workloads (fixed seeds), and is scored with the engine's `Evaluator`. Results are reported per metric and as the mean, at each budget share.

| Method | Type |
|---|---|
| RDP, size-proportional | Current pipeline |
| SED, size-proportional | Current pipeline |
| SED, size^0.25 | Hand rule; best non-learned method so far |
| MLSimp | Learned baseline (self-supervised point importance) |
| **Learned allocator + SED** | This method |

## 7. Implementation

| Part | Location |
|---|---|
| Experiments so far | `engine/apps/knn_alloc_oracle.cpp` |
| Features and allocator | new `engine/src/allocation/learned.cpp`, `AllocatorType::Learned` |
| Training app (writes θ to JSON) | new `engine/apps/train_allocator.cpp` |
| θ per budget share in the config | `pipeline/config.cpp` (`load_config` rejects unknown keys, so extend the key list) |
