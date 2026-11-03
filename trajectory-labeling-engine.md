# Learned Multi-Query Trajectory Simplification (engine version)

*Updated 2026-11-03.* Goal, experiment results so far, and the proposed learned model. The original per-trajectory labeling plan is kept as Appendix A; parts of it are superseded by the results below (see A.0).

---

## 1. Goal

Minimize query error on the compressed database for **three query types at once**, at budgets of 0.25–2% of the points:

| Query type | Engine metric | How it is answered on compressed data |
|---|---|---|
| kNN | `knn.recall_at_k` (k = 10) | Discrete Fréchet between the full query and the kept vertices |
| Range | `range_recall` | A trajectory is a hit if a kept point is inside the box and time window |
| Where-at | `where_at.average_hit` | Position interpolated linearly in time between kept points |

Objective: `score = w_knn · knn_recall + w_range · range_recall + w_whereat · where_at_average_hit`. Equal weights by default. The weights are the proposal's "query requirements".

Constraints from the proposal: the method runs as a service on uploaded data, so at simplification time it **cannot use queries or labels**, and its cost must scale roughly linearly with data size.

---

## 2. Findings so far

**Setup for all runs:**
- GeoLife, random split (seed 42): 29,064 database trajectories, 16.1M points (mean 554 per trajectory).
- The held-out pool is split in two: kNN **train** queries (2,564) and kNN **test** queries (2,565), with k = 10.
- 500 range queries and 3.72M where-at probes, generated with the `geolife.json` settings.
- Everything was run with `engine/apps/knn_alloc_oracle.cpp` (section 8).
- The ± values are paired standard errors over queries.

### 2.1 For kNN, how the budget is shared matters more than anything else tried

All numbers are test recall@k, RDP selector.

| Allocation | 0.25% | 0.5% | 1% | 2% |
|---|---|---|---|---|
| Size-proportional (current) | 0.479 | 0.544 | 0.617 | 0.701 |
| Uniform (2 points each, rest equal) | 0.546 | 0.614 | 0.680 | 0.744 |
| size^0.25 | 0.550 | 0.615 | 0.681 | 0.743 |
| Learned linear-3 (2.3) | **0.556** | **0.621** | **0.688** | **0.765** |

A flatter allocation than size-proportional gains **+4.1 to +7.7** recall@k (SE ≈ 0.3). The gain is the same on train and test queries, so it is structural, not overfitting. The cause is that Fréchet distances on sparse trajectories inflate: short trajectories, which size-proportional leaves at 2–3 points, drop out of top-k lists.

### 2.2 Query-specific oracles overfit, so per-query labels are not a usable training target

A greedy oracle that allocates points to maximize recall on the training queries gains up to **+15** points on those queries. On test queries it gains only +1 to +2 over size-proportional, and nothing over uniform. Each trajectory appears in only a few training queries, so per-trajectory targets are mostly noise. This is the main reason the utility-curve labels of Appendix A are not used as training targets.

### 2.3 Three features are enough; more features or an MLP add nothing

Model: each trajectory gets weight `exp(θ · z)`, where z holds z-scored log features. Every trajectory keeps 2 points and the rest of the budget is shared by weight. θ is fitted on train recall@k with a (1+1) evolution strategy against the real evaluator.

| Model | Δ test recall@k vs size^0.25 (0.25 / 0.5 / 1 / 2%) |
|---|---|
| Linear, 3 features (size, density, curve) | +0.6 / +0.6 / +0.7 / **+2.3** |
| Linear, 9 features (+ duration, interval, speed, extent, nn1, nn10) | +0.6 / +0.8 / +0.7 / +2.5 |
| 9 features + MLP residual (6 tanh units) | +0.6 / +0.8 / +0.9 / +2.5 |

Compared with linear-3, the extra features and the MLP gain at most +0.25 (within 1–2 SE). Density gets a positive weight (crowded areas mean fragile rankings) and curve a negative one; the likely cause is that high tortuosity is GPS jitter around stops, which hasn't been verified. The optimizer is weak for the MLP's 75 parameters, but the linear-9 result is already a clear negative.

### 2.4 All three query types: the kNN-tuned allocator conflicts with the others

The kNN numbers here come from test queries; range and where-at were never used for fitting. "mean" is the equal-weight objective.

| Budget | Selector + allocation | kNN R@k | Range | Where-at hit | Mean |
|---|---|---|---|---|---|
| 0.25% | RDP size-proportional (current) | 0.479 | 0.736 | 0.565 | 0.593 |
| | SED size-proportional | 0.479 | 0.738 | 0.589 | 0.602 |
| | RDP size^0.25 | 0.550 | 0.739 | 0.611 | 0.633 |
| | **SED size^0.25** | 0.537 | 0.729 | **0.654** | **0.640** |
| 0.5% | RDP size-proportional | 0.544 | 0.800 | 0.649 | 0.664 |
| | SED size-proportional | 0.552 | 0.808 | 0.690 | 0.683 |
| | RDP size^0.25 | 0.615 | 0.795 | 0.715 | 0.709 |
| | **SED size^0.25** | 0.619 | 0.792 | **0.770** | **0.727** |
| | SED linear-3 | 0.613 | 0.767 | 0.744 | 0.708 |
| 1% | RDP size-proportional | 0.617 | 0.861 | 0.763 | 0.747 |
| | SED size-proportional | 0.636 | 0.877 | 0.815 | 0.776 |
| | RDP size^0.25 | 0.681 | 0.847 | 0.823 | 0.784 |
| | **SED size^0.25** | 0.702 | 0.861 | **0.881** | **0.814** |
| | SED linear-3 | 0.714 | 0.862 | 0.865 | 0.814 |
| 2% | RDP size-proportional | 0.701 | 0.917 | 0.866 | 0.828 |
| | SED size-proportional | 0.721 | 0.937 | 0.921 | 0.860 |
| | RDP size^0.25 | 0.743 | 0.898 | 0.898 | 0.846 |
| | **SED size^0.25** | 0.777 | 0.922 | **0.954** | **0.884** |
| | SED linear-3 | 0.791 | 0.910 | 0.911 | 0.871 |

(Linear-3 here uses the grid-fitted exponents from the `--features` run, (size, density, curve) = (0.25, 0, 0), (0, 0.25, 0), (0.25, 0.25, −0.25), (0.25, 0.5, −0.5), so its kNN numbers differ slightly from the evolution-strategy fit in 2.1. At 0.25% it is identical to size^0.25. RDP linear-3 rows are omitted and show the same pattern.)

**Takeaways:**
1. **A flatter allocation helps kNN and where-at, and costs a little range.** Moving from size-proportional to size^0.25 under SED:
   - kNN: +5.6 to +6.7
   - where-at: +3.2 to +8.0
   - range: −0.8 to −1.7
   - equal-weight mean: **+2.4 to +4.4**

   Where-at averages over trips and range counts trajectories, so every trajectory counts equally no matter how long it is. Size-proportional spends the budget by length instead.
2. **Query types want different allocations.** Compared with size^0.25 at 2%, the kNN-tuned linear-3 gains +1.4 on kNN but loses 4.3 on where-at and 1.2 on range, so the mean goes down. A model fitted to one query type is the wrong tool; it has to be trained on the **combined** objective.
3. **Which points are kept matters for where-at.** With the same allocation, SED beats RDP on where-at by +4.4 to +5.8. Range is mixed (−1.0 to +2.4) and kNN is similar.
4. **The new baseline to beat is SED + size^0.25.** It has the best mean at every budget: 0.640 / 0.727 / 0.814 / 0.884, compared with 0.602 / 0.683 / 0.776 / 0.860 for the current SED + size-proportional.

---

## 3. Proposed model: learned priority, global bottom-up simplification

One small model decides both *how many* points each trajectory keeps (which matters for kNN, 2.1) and *which* points (which matters for where-at and range, 2.4).

**Algorithm:**

```
priority(p) = exp(θ · features(p))          # p: an interior point of any trajectory

put every interior point of every trajectory in one global min-heap
while points kept > total budget B:
    remove the lowest-priority point
    recompute the priorities of its two kept neighbours (their errors changed)
```

- Endpoints are never in the heap, so every trajectory keeps at least 2 points, as `allocation::kMinPointsPerTrajectory` requires.
- There is no separate allocator: a trajectory keeps many points exactly when its points have high priority.
- With θ set to use SED error only, this is "global SED", a new baseline in its own right (SED error compared across trajectories instead of within a fixed per-trajectory budget).

**Features of point p in trajectory i** (all available without queries, log-transformed):

| Feature | Recomputed after a neighbour is removed | Aimed at |
|---|---|---|
| SED error to the segment between its kept neighbours | yes | where-at |
| Perpendicular (RDP-style) error | yes | kNN (Fréchet shape) |
| Time gap between its kept neighbours | yes | where-at |
| Point-local density (other trajectories' points in its 500 m cell) | no | range |
| Trajectory size, density, curve (z-scored, from 2.3) | no | kNN allocation |

The parameter vector θ has about 7–8 numbers. That is small on purpose: 2.2 and 2.3 showed that targets fitted to specific queries, and extra model capacity, do not generalize.

## 4. Training

- **Reward:** the equal-weight mean of kNN recall@k, range recall and where-at average hit on **training workloads**:
  - kNN train half of the pool;
  - range and where-at generated with training seeds.

  Test workloads use other seeds and the kNN test half, and are never seen during fitting.
- **Optimizer:** the same (1+1) evolution strategy used in 2.3. No labels and no gradients; each step simplifies the database and runs the real `Evaluator`.
- **Start point:** SED error only (global SED), so anything accepted is an improvement over that baseline.
- **Cost:** a global heap over 16M points takes seconds per evaluation. Fit on a 3,000–5,000 trajectory subset and evaluate on the full database, which also tests transfer.
- **Budgets:** fit one θ per budget share, or add log(share) as a feature to get a single model.
- **Query requirements:** start with equal weights. Later, fit one θ per weight setting, or add the weights as inputs.

## 5. Evaluation plan

Held-out workloads, every budget share, all three metrics plus the weighted mean:

| Method | Role |
|---|---|
| RDP / SED, size-proportional | Current pipeline |
| RDP / SED, size^0.25 | Simple rule; **SED + size^0.25 is the baseline to beat** |
| Global SED (θ = SED only) | Collective version of SED |
| MLSimp, simplified output run through this engine's `Evaluator` | Learned baseline (self-supervised point importance) |
| Learned priority model | Proposed method |

Also report:
- simplification time against dataset size (the proposal's scalability metric);
- one transfer test (T-Drive or Porto, both named in the proposal).

## 6. Implementation steps

| Step | Files |
|---|---|
| Point and trajectory features | new `engine/src/features/` |
| Global learned-priority simplifier (one heap across trajectories; returns a `core::Simplification`) | new `engine/src/selection/learned_priority.cpp` |
| Training app: ES loop, multi-query reward, writes θ as JSON | new `engine/apps/train_simplifier.cpp` |
| `learned` config section (θ per share, query weights). `load_config` rejects unknown keys, so extend its key list | `pipeline/config.cpp`, `tests/config_test.cpp` |
| Add a size-power allocator, so SED + size^0.25 is a pipeline baseline | `allocation/` |

## 7. Risks and open points

- **Range has only 500 queries,** so differences of about 1 point are noise. Raise `range.query_count` to 5,000 for the next runs.
- All results so far use **one seed and one dataset.** Repeat on a second seed before quoting numbers.
- **Global heap cost** during training (section 4).
- The engine's **range rule** (a kept point must be inside the box, no interpolation) rewards keeping points in busy cells. That is a property of the metric and should be stated in the report.

## 8. Reproducing the experiments

Build in Release (`cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release`, then build the `knn_alloc_oracle` target). Run from the repo root:

| Section | Command | Time |
|---|---|---|
| 2.1, 2.2 | `./build-release/engine/knn_alloc_oracle engine/configs/geolife.json --train 0 --test 0` | ~3 min |
| 2.3 (3-feature grid) | `... --train 0 --test 0 --features 1` | ~5 min |
| 2.3 (linear-9, MLP) | `... --train 0 --test 0 --model 1 --evaluations 300` | ~48 min |
| 2.4 | `... --test 0 --check 1` | ~1 min |

Add `--db-limit 2000 --train 100 --test 100` for a quick smoke run.

---

# Appendix A: Original labeling plan

## A.0 What still applies

- **Still valid:**
  - workload generation and strict train/test seed separation (§3);
  - the scoring rules, which match the `Evaluator` (§5a, §5b, §5c);
  - sanity checks (§9) and pitfalls (§12).
- **Superseded:**
  - per-trajectory utility curves as model training targets, because query-specific targets overfit (2.2);
  - a kNN-only or allocation-only model, because the query types conflict (2.4).

The model in section 3 is trained directly on the evaluator's combined score instead.

---

## 0. What a label is

For every trajectory in the database, the label answers one question:

> **If this trajectory keeps n points, how much does it contribute to each query type's score?**

Example for one trajectory T₇ (illustrative numbers):

| Points kept (n) | Range  | Where-at | kNN (1% budget) |
|-----------------|--------|----------|-----------------|
| 2               | 0.0010 | 0.0002   | 0.0000          |
| 4               | 0.0015 | 0.0003   | 0.0004          |
| 8               | 0.0015 | 0.0004   | 0.0009          |
| 16              | 0.0016 | 0.0005   | 0.0010          |
| all             | 0.0016 | 0.0005   | 0.0008          |

Each column is that trajectory's **utility curve** f_i^o(n) for query type o. Values are the trajectory's *share* of the database-wide score, weighted so that summing over trajectories gives the score the Evaluator reports:

```
Score(n_1, ..., n_N) ≈ Σ_i Σ_o w_o · f_i^o(n_i)
```

A curve that rises steeply means "this trajectory deserves more points"; a curve that flattens early means "extra points are wasted here". The kNN curve can even *fall* (last row): section 5c explains why.

**What each query type's sum equals**

| Query type | Σ_i f_i(n_i) equals | Exact? |
|---|---|---|
| Range | `range_recall` (mean over queries) | Yes |
| Where-at | `where_at.average_hit` (mean over trips) | Yes |
| kNN | an estimate of `knn.recall_at_k` | No: rankings depend on competitors (5c, 9) |

**Assumptions**

- Point choice *within* a trajectory is done by one fixed, nested selector (step 1).
- kNN labels depend on the budget level, because they are measured against how much competitors were simplified. Range and where-at labels do not.

---

## 1. Fix a point order for each trajectory

To ask "what if T₇ keeps 8 points?" you need to know *which* 8. Use one fixed selector that produces a **ranking** of the trajectory's points:

1. Start with the two endpoints (always kept: `allocation::kMinPointsPerTrajectory = 2`, and where-at throws for probes outside the kept time span).
2. Repeatedly add the point the selector prefers next.
3. Record the insertion order: `π_i = [0, len−1, 40, 17, 22, ...]`.

Keeping n points always means **the first n entries of π_i**.

**Why "nested" matters:** the 8-point version must contain the 4-point version. Otherwise going from n to n+1 could swap points, and the curve no longer means "the gain from one more point".

Selectors available in this repo:

| Selector | Where | Nested? | Order available today? |
|---|---|---|---|
| RDP | `selection/rdp.cpp`: top-down, max-error split from a priority queue | Yes | No: `rdp::simplify` returns sorted indices |
| SED | `selection/sed.cpp`: bottom-up min-max removal | Yes, when the removal order is reversed | No |
| Where-at greedy | `apps/conflict_pilot.cpp`: `best_where_at_split` | Yes, per trajectory | Inside the pilot only |

Both engine selectors are already nested, so they only need to **expose the order** (section 13). Add a test that `sorted(order[:n]) == simplify(trajectory, n)` for every n.

**Where-at greedy needs probes.** Its probes come from the original data alone (`query::generate_where_at_queries`), so it can run at inference time, but:

- use a **separate seed** for selector probes and label probes. Otherwise the selector is optimized on the exact probes that score it, and the where-at labels are inflated;
- record the probe config and seed in `meta.json`.

Compute π_i once, store it, and use the **same selector at inference**. The curves are only valid for the selector that produced them.

---

## 2. Choose the grid of n values

Use a geometric grid instead of every n:

```
n ∈ {2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, ...}
```

- Cap at the trajectory's length; always include `n = len(T_i)` as the last point.
- About 10–15 grid points per curve is enough. Interpolate linearly in between.
- At the configured `budget_shares` (0.25–2%) most trajectories get well under 100 points, so the dense low end of the grid is where the labels matter.

---

## 3. Generate training workloads

Generate **several independent workloads** (W = 3–5, different seeds) so labels estimate *expected* usefulness, not usefulness for one specific query set. Use the existing generators with the evaluation config, changing only the seed and the count:

| Query type | Generator | Size per workload | Notes |
|---|---|---|---|
| Range | `query::generate_range_queries(db, config.range, seed)` | `query_count` 5,000+ | Same `box_size_m`, `window_seconds`, `min_results`, `full_accept_at` as evaluation. The pilot's 500 queries ran out of signal after 489 points |
| Where-at | `query::generate_where_at_queries(db, config.where_at, seed)` | 128 probes per trip | Trips with no gap ≤ `max_gap_seconds` get no probes |
| kNN | `query::generate_knn_queries(db, train_pool, config.knn, seed)` | 2,000+ | See below |

**kNN query sources** (cheapest first):

1. **Split the query pool.** The engine's evaluation queries come from the held-out pool (`split.query_percentage`, 15%). Split the pool into a training half and a test half, as the pilot does. No new code.
2. **Leave-one-out from the database**, if the training half is too small. Query with database trajectory j, ask for k+1 neighbours, drop j itself. Same distribution (random split), but needs a small new generator (section 13).

Run every query on the **original, unsimplified** database once and keep the ground truth:

- **Range:** `RangeQuery::reference`, the sorted set R_q of trajectory indices (already filled by the generator).
- **Where-at:** `WhereAtQuery::reference`, the original interpolated position (already filled).
- **kNN:** the true top-k (filled), plus the original Fréchet distance to each candidate. That needs a public distance function (section 13).

**Test workloads are generated separately with their own seeds and never used for labeling.** kNN test queries come from the test half of the pool, which is disjoint from every training query.

---

## 4. For each trajectory, list the queries that involve it

Invert the ground truth so each trajectory has a short list of queries that can affect it:

- **Range:** all queries q with T_i ∈ R_q.
- **Where-at:** all probes with `trajectory_index == i`, sorted by `t`.
- **kNN:** for each query q, the candidate set C_q = true top-k **plus near misses**: `query::knn(db, q, 3k)` filtered to original distance ≤ 1.2 · d_k. T_i's list is every q with T_i ∈ C_q.

Near misses matter: a trajectory just outside the top-k can enter it after simplification and push out a real neighbour.

This inversion makes labeling fast: each trajectory only checks its own list, and trajectories run independently (`core::parallel_for`).

---

## 5. Walk through the prefix and score

For each trajectory, add points in π_i order. At each grid value n, take `S = π_i[:n]` and score S against the trajectory's lists.

### 5a. Range

`query::range` counts a trajectory as a hit when **at least one kept point** lies inside the box and time window. There is no segment interpolation, and kept points are a subset of the original points, so simplification can only lose hits, never create false ones. The score is recall only.

For each query q on the list:

```
hit(q, S)  = 1 if some p in S has t_min ≤ t_p ≤ t_max and p inside the box, else 0
weight(q)  = 1 / (|R_q| · |Q_range|)
f_range(n) = Σ_q weight(q) · hit(q, S)
```

- The weight makes Σ_i f_range,i(n_i) equal `range_recall` exactly. Every generated query has |R_q| ≥ `min_results`, so none is skipped as NaN.
- Pairs whose box contains an endpoint are hits already at n = 2.
- Curves are exactly non-decreasing (S only grows), and usually staircases. Step 7 handles that.

### 5b. Where-at

The Evaluator scores each probe by the distance between the simplified trajectory's position at time t (`core::position_at`, linear in time between kept points) and the reference position:

```
err          = distance(position_at(S, t), reference)
average_hit  = clamp((max_tolerance_m − err) / (max_tolerance_m − min_tolerance_m), 0, 1)
                                                         # 500 m and 25 m in geolife.json
f_whereat(n) = (1 / N_trips) · (1 / P_i) · Σ_t average_hit(t)
```

- P_i = probes on T_i; N_trips = trips with at least one probe (the Evaluator skips the rest).
- Optionally store a second curve for `within_radius` (err ≤ `hit_radius_m`, 100 m) with the same weights.
- Inserting a point only changes probes between its two kept neighbours, so update incrementally.

### 5c. kNN

`query::knn` ranks database trajectories by **discrete Fréchet distance between the full query and the kept points** of each trajectory. With a sparse trajectory, every query point must match a kept vertex, so at 0.25–2% budgets the distances of **all** candidates inflate. Comparing d′ with the *original* d_k would mark nearly every pair as broken at small n, even when the ranking survives because the competitors inflated too.

So measure each trajectory against its competitors **as they would be simplified**:

1. **Reference allocation.** For a budget level s, simplify every trajectory with the same selector under `allocation::create_trajectory_budgets(db, SizeProportional, total_budget(db, s))`.
2. **Competitor distances.** For every q and every j ∈ C_q, compute d″_j = Fréchet(q, T_j under the reference allocation), once.
3. **Score T_i at n.** With d′ = Fréchet(q, π_i[:n]):

```
θ = k-th smallest d″_j over j ∈ C_q, j ≠ i       # where T_i has to land relative to the others

true top-k member:  m = θ − d′                   # should stay inside
near miss:          m = d′ − θ                   # should stay outside

score(q)   = sigmoid(m / τ),   τ = max(0.1 · θ, 10 m)
f_knn(n)   = Σ_q score(q) / (k · |Q_knn|)
```

- One label set per `budget_shares` level (4 in `geolife.json`). Range and where-at are shared across levels.
- **The curve can decrease.** A near miss with many points can beat true neighbours that were simplified, so giving it more points hurts. That is real and should stay in the label (no running max for kNN).
- **Optional refinement:** after step 9, recompute d″ under the true-hull allocation instead of size-proportional, and relabel. One or two rounds is enough.
- **Target metric.** The margin targets `recall_at_k`. It ignores order inside the top-k, so `kendalls_tau_topk` is reported but not labelled.

### 5d. Speed

Pilot timings (2,000 trajectories, 1.09M points, Release build, 8 cores): range 0.1 s, where-at 6.7 s, kNN 63 s for 100 queries.

- The prefix only grows, so each insertion splits one segment. Only probes in that time span and boxes containing the new point can change.
- kNN distances are recomputed at each grid point, but only for the trajectory's candidate queries, and S is small (Fréchet costs |q| · n).
- Trajectories are independent → `core::parallel_for` over trajectories.

---

## 6. Average over workloads

Repeat steps 4–5 for each of the W workloads. For every (trajectory, query type, budget level, n) store:

- **mean** across workloads → the label;
- **standard deviation** → label noise.

High std usually means few queries touched that trajectory. A zero mean with zero std on a trajectory no query touched means "never sampled", not "unimportant": fix it by sampling more queries, or down-weight those labels in the training loss (e.g. weight ∝ 1 / (std² + c)).

---

## 7. Turn curves into training targets

### 7a. Concave hull

Raw curves are often **staircases**: a range pair jumps from 0 to 1 the moment a point lands in the box. Greedy allocation is only optimal when each extra point is worth less than the previous one, so replace each curve by its **upper concave hull**.

Example (range scores ×1000):

| n    | 2   | 4   | 8   | 16  |
|------|-----|-----|-----|-----|
| Raw  | 0.0 | 0.0 | 1.2 | 1.3 |
| Hull | 0.0 | 0.6 | 1.2 | 1.3 |

The hull is the tightest "rubber band" over the curve from above: it never dips below the raw curve and its slope only decreases. Greedy allocation on hulls is the classic Lagrangian solution from rate-distortion bit allocation: optimal at hull vertices, near-optimal in between.

```python
def upper_concave_hull(ns, fs):
    """ns increasing, fs = curve values. Returns hull values at each n."""
    hull = []  # indices of hull vertices
    for i in range(len(ns)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # remove b if it lies on or below the line a -> i
            cross = (ns[b] - ns[a]) * (fs[i] - fs[a]) - (fs[b] - fs[a]) * (ns[i] - ns[a])
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    out = []
    for n in ns:
        for a, b in zip(hull, hull[1:]):
            if ns[a] <= n <= ns[b]:
                t = (n - ns[a]) / (ns[b] - ns[a])
                out.append(fs[a] + t * (fs[b] - fs[a]))
                break
        else:
            out.append(fs[hull[0]])  # single-vertex edge case
    return out
```

Monotonicity per query type:

| Query type | Before the hull |
|---|---|
| Range | Nothing: exactly non-decreasing already |
| Where-at | Running max: rare small dips are interpolation noise |
| kNN | **Nothing**: decreases are real (5c). The hull ends in negative slopes, and greedy simply stops there |

### 7b. Marginal gains

Convert hull values into gain per point between consecutive grid values:

```
gain_k = (hull[k+1] − hull[k]) / (n[k+1] − n[k])
```

For the example: 0.30/pt (2→4), 0.15/pt (4→8), 0.0125/pt (8→16). These decreasing gains are what the model learns to predict.

Keep both versions:
- **hull gains** → training targets;
- **raw curves** → evaluation and the exact oracle (step 9).

---

## 8. Output format

Write with the engine's existing Arrow/Parquet setup (`io/workload_export.cpp` already has a `write_parquet` helper) and nlohmann/json for metadata.

**labels.parquet**: one row per (trajectory, query type, budget level, grid point). `budget_pct` is empty for range and where-at:

```
traj_idx,trajectory_id,query_type,budget_pct,n,raw_mean,raw_std,hull_value,marginal_gain
7,412,range,,2,0.0000,0.0000,0.0000,0.000300
7,412,range,,4,0.0000,0.0000,0.0006,0.000150
7,412,knn,1,4,0.0004,0.0001,0.0004,0.000125
...
```

**features.parquet**: one row per trajectory (model input), all computable from `TrajectoryView` and the database:

```
traj_idx,trajectory_id,len,duration_s,mean_sampling_s,mean_speed,bbox_area,
sel_err_n2,sel_err_n4,...,sel_err_n128,        # selector's own max error at each n
local_point_density,                           # database points in a grid cell around the trajectory
nn_lb_1,nn_lb_5,nn_lb_10,                      # endpoint/bbox lower bounds to nearest trajectories
hotspot_overlap                                # share of points in dense cells
```

The nearest-neighbour features use the cheap endpoint and bounding-box bounds already in `query/knn.cpp`, not full Fréchet distances.

**orders.parquet**: π_i as `(traj_idx, rank, point_idx)`, so the inference-time selector is identical to the labeling-time one.

**meta.json**: selector name and its probe seed, grid, W and workload seeds, workload sizes, k, near-miss factor, τ rule, reference allocator, `budget_shares`, and the full `where_at_scoring` block. Labels are meaningless without these.

---

## 9. Sanity checks (before training any model)

1. **Totals match the Evaluator.** For a few allocations (size-proportional at each budget level, and one random allocation), materialize the simplification with the label selector and run the real `Evaluator` on the training workloads:
   - `range_recall` and `where_at.average_hit` should equal Σ_i f_i(n_i) to rounding.
   - kNN: only approximately. Report the correlation between the label-predicted kNN score and real `recall_at_k` across allocations.
2. **Curves look sane.** Range non-decreasing, all curves end at their full-resolution value, no NaNs. Spot-check plots with `visualization/`.
3. **The labels are worth learning.** On **held-out test workloads**, compare:

| Allocation | Point choice | Tells you |
|---|---|---|
| Size-proportional (current) | Label selector | Baseline |
| Size-proportional | RDP / SED | Does the label selector itself help? |
| Greedy on true training hulls | Label selector | What a perfect budget model could achieve |
| Exact DP on raw training curves (multiple-choice knapsack) | Label selector | Cost of the hull approximation |
| Greedy on hulls from a *different* workload sample | Label selector | Generalization vs memorization |
| Pilot's best rank blend (`conflict_pilot`) | Free per point | What point-level choice adds on top of allocation |

- True-hull greedy doesn't beat size-proportional on test → a budget model is pointless; focus on point selection.
- It does → the gap is your project's motivation and the model's target.
- Hull greedy ≈ exact DP → the concave approximation costs nothing (worth reporting).
- Pilot blend clearly beats true-hull greedy → allocation alone isn't enough, and the model needs a point-level part too. At 0.25% the pilot's range oracle got 0.915 test recall against about 0.81 for everything else, by choosing specific points.

---

## 10. Whole pipeline in pseudocode

```
config  = load_config("engine/configs/geolife.json")
db, pool = load_split(config)
train_pool, test_pool = halves(pool)

# Step 1: nested orderings
selector_probes = generate_where_at_queries(db, config.where_at, seed = SELECTOR_SEED)
parallel for each trajectory i:
    order[i] = insertion_order(db[i], selector, selector_probes)

# Steps 3–5: label under W workloads
for w in 1..W:
    range   = generate_range_queries(db, config.range, seed = w)
    probes  = generate_where_at_queries(db, config.where_at, seed = w)
    knn     = generate_knn_queries(db, train_pool, config.knn, seed = w)
    cands   = knn_candidates(knn, factor = 1.2)                       # Step 4
    lists   = invert(range, probes, cands)
    for s in config.budget_shares:                                   # 5c reference
        ref[s] = competitor_distances(cands, order, size_proportional(db, s))
    parallel for each trajectory i:                                  # Step 5
        for n in grid(len(db[i])):
            S = order[i][:n]
            score[w][i]["range"][n]   = range_score(S, lists[i].range)
            score[w][i]["whereat"][n] = whereat_score(S, lists[i].probes)
            for s in config.budget_shares:
                score[w][i]["knn", s][n] = knn_score(S, lists[i].knn, ref[s])

# Steps 6–7: aggregate and build targets
for each trajectory i, query type o (and budget level for kNN):
    mean, std = mean/std over w
    curve = running_max(mean) if o == "whereat" else mean
    hull  = upper_concave_hull(grid_i, curve)
    gains = diff(hull) / diff(grid_i)
    write_rows(i, o, grid_i, mean, std, hull, gains)                 # Step 8

write_features(); write_orders(); write_meta()
run_sanity_checks(test_workloads)                                    # Step 9
```

---

## 11. Parameters to decide (and record)

| Parameter | Suggested start | Config key / source |
|---|---|---|
| Selector | where-at greedy; SED as alternative | new `labels.selector` |
| Selector probe seed | different from every label seed | new `labels.selector_seed` |
| Grid | geometric, ~1.5× steps | new `labels.grid_factor` |
| W (workloads) | 3–5 | new `labels.workloads` |
| Range queries / workload | 5,000 | `range.query_count` |
| kNN queries / workload | 2,000+ | half the pool, or leave-one-out |
| k | 10 | `knn.k` |
| Near-miss factor | 1.2 · d_k | new `labels.near_miss_factor` |
| τ (kNN sigmoid) | max(0.1 · θ, 10 m) | new `labels.tau_fraction`, `labels.tau_min_m` |
| Reference allocator (5c) | size-proportional | `AllocatorType::SizeProportional` |
| Budget levels | 0.25, 0.5, 1, 2% | `budget_shares` |
| Where-at scoring | 100 m / 25 m / 500 m | `where_at_scoring` (must match evaluation) |

---

## 12. Common pitfalls

- **Labels from too few queries.** A zero then means "never sampled", not "unimportant". This is why the pilot's kNN oracle dropped from 0.956 (train) to 0.835 (test).
- **Comparing kNN distances with the original d_k.** Every simplified distance is larger, so all pairs look broken at small n. Compare with simplified competitors (5c).
- **Wrong distance or range rules.** The engine uses discrete Fréchet on kept vertices for kNN, and kept points only (no interpolation, recall only) for range.
- **Same probes for the selector and the labels.** Inflates where-at labels. Use separate seeds.
- **Different selector at inference than at labeling.** The curves no longer describe what happens.
- **Non-nested selector.** Curves stop being marginal-gain sequences. Test `sorted(order[:n]) == simplify(n)`.
- **Leaking test queries into labels.** Separate seeds for every workload, and disjoint pool halves for kNN.
- **Running max on kNN curves.** Hides the real cost of over-detailing near misses.
- **Mixing units across query types.** The weights in 5a–5c already put each type on its metric's scale (recall, average hit, recall@k); keep them before applying preference weights w.
- **Unoptimized builds.** `build/` has no `CMAKE_BUILD_TYPE`, so it compiles without optimization. Configure the labeling build with `-DCMAKE_BUILD_TYPE=Release`.

---

## 13. What the repo needs

Small, mostly additive changes. Nothing in the evaluation path changes.

| Change | Files | Size |
|---|---|---|
| Expose insertion orders: `selection::insertion_order(view, AlgorithmType)` returning the full order, endpoints first. RDP records its split order; SED reverses its removal order | `selection/selector.hpp/.cpp`, `rdp.cpp`, `sed.cpp`, new test in `tests/selectors/` | Small |
| Move the where-at greedy out of the pilot as a third selector (needs probes as input) | new `selection/where_at_greedy.hpp/.cpp` | Small |
| Public Fréchet distance (the current one is in an anonymous namespace) | `query/knn.hpp/.cpp` | Small |
| Optional leave-one-out kNN generator (k+1 neighbours, drop self) | `query/knn.hpp/.cpp` | Small |
| Labeling library: lists, prefix scoring, workload averaging, hulls | new `engine/include/backend/engine/labels/`, `engine/src/labels/` | Main work |
| Labeling app, plus the step 9 checks | new `engine/apps/label.cpp` (replaces `apps/conflict_pilot.cpp`) | Medium |
| `labels` config section. `load_config` rejects unknown root keys, so the key list needs extending | `pipeline/config.hpp/.cpp`, `tests/config_test.cpp` | Small |
| Expose `write_parquet` for reuse | `io/workload_export.hpp/.cpp` | Small |
| Later, for inference: `AllocatorType::Learned` reading predicted gains, greedy allocation with a priority queue, at least 2 points per trajectory | `allocation/` | Medium |

**Implementation order:** insertion orders and public Fréchet → range and where-at labels (exact, quick to validate with check 9.1) → kNN labels with the reference allocation → step 9 table. Only start the model if the true-hull greedy row clearly beats size-proportional on test workloads.
