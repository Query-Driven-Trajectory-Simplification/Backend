# The teacher

The teacher is a slow, query-aware simplifier. It searches how many points each trajectory gets
and which candidate keeps them: RDP, SED, UNIFORM or one of three greedy variants. Its output is
meant to be a target that a learned model can later imitate.

It sees **only the database** (the 85% split) and workloads built from it with their own seeds. It
never sees the held-out query pool that the pipeline scores on, and it trains for that setting:
its kNN and join queries are original trajectories against the simplified database. Budgets start
from equal shares (α = 0) and are redistributed within the same total.

Self-trained options (compressed kNN queries, a `Self` join term, the held-out term next to it, the
per-trajectory kNN term) and the arms `EQUAL_CHOICE`, `EQUAL_CHOICE_RDP_PARTNERS`, `EQUAL_LOOP`,
`BUDGET_LOOP_REALLOCATE`, `BUDGET_LOOP_SHUFFLED` and `BUDGET_LOOP_NO_UPDATE` were removed on
2026-11-22; they are in git history before that date.

Range scoring changed on 2026-11-20: J_T now uses macro F1, including its endpoint floor,
matching J4. Historical teacher selections and J_T values need a fresh run. `RDP_GLOBAL` is the
query-independent database-wide top-down PED baseline in both experiment runners.

Continuous kNN now uses one numerical ranking in both the evaluator and `KnnState`: the
smallest millimetre-grid radius accepted by the Fréchet decision predicate, with equal keys
ordered by database index. This replaces evaluator-specific overlapping intervals and the
tracker's independently bisected scalar distances. Rankings near ties can change; V6 caches
rebuild the reference neighbours, and earlier kNN/J4 results should be rerun. Generic Fréchet
distances and join decisions are unchanged. Limited kNN watch lists remain an approximation.

## How it fits together

```
pipeline::run                                    (pipeline.cpp)
 ├─ load_or_build_workloads(database, ...)        workloads.cpp: kNN, pairs, boxes, probes; cache
 ├─ add_ranked_neighbours                         workloads.cpp: k + 5 nearest per query, for margin
 ├─ Teacher(database, workloads, ...)             teacher.cpp: builds the Objective once
 │    └─ Objective(db, workloads)                 objective.cpp: inverts workloads, computes floors
 └─ per budget share:
      Teacher::run(budget, arms)
       ├─ make_candidates                         RDP at equal budgets: the curves' partners
       ├─ build_curves → allocate                 value curves, concave-envelope allocation
       │   ├─ BUDGET_RDP                          allocated budgets, RDP's points
       │   └─ BUDGET_CHOICE                       allocated budgets, best candidate each
       │       └─ BUDGET_LOOP                     best response on the tracked J_T
       │           └─ BUDGET_LOOP_TRADE           points traded between trajectories: the teacher
      → each arm evaluated and exported like RDP and SED
```

Files:

| File | Contents |
|---|---|
| `workloads.hpp/.cpp` | The teacher's training workloads, and the cache that saves the slow part |
| `objective.hpp/.cpp` | J_T, the score the teacher maximises, and range, join and kNN state trackers |
| `curves.hpp/.cpp` | Value curves and the budget allocator |
| `greedy.hpp/.cpp` | Query-aware point selection, with optional classic-method seeds |
| `teacher.hpp/.cpp` | The `Teacher` class and the selection steps |

## Words used throughout

- **Kept set:** the sorted point indices one trajectory keeps, endpoints included. `KeptSets` is one
  kept set per trajectory, i.e. a whole simplified database.
- **Budget:** the number of points a trajectory keeps, endpoints included. A budget of 2 means
  endpoints only.
- **Floor:** a metric's value when every trajectory keeps endpoints only.
- **Normalised metric:** `(metric − floor) / (1 − floor)`. 0 means no better than endpoints only;
  1 means as good as the original.
- **J4:** the pipeline's score on the held-out test workloads: the equal mean of four normalised
  metrics (kNN, range, where-at, join). Route clustering is reported as a diagnostic only, because
  it is built from the same Fréchet neighbour test as the join and would count it twice.
  With `join.mode` `HeldOut` (the Geolife config) the join term compares original trajectories
  from the query pool with the simplified database, so all four terms are held out. The database
  self join, whose pairs the teacher trains on, is then reported separately (`norm_self_join`
  and the thresholds in `thresholds.csv`) and is not part of J4. Results quoted
  below predate this and have the self join inside the score.
- **J4-self** (`J4_self`): the in-database setting, where the database's own trajectories are
  the kNN and join queries. Its terms are self kNN (each database trajectory, compressed, searched
  against the compressed database, itself excluded), J4's range and where-at, and the self join
  averaged over `join.score_eps_m` like J4's join. Its kNN and join queries are the teacher's
  training queries by definition, so only range, where-at and untrained thresholds are held out.
  The self-kNN reference is read from the teacher's kNN cache (`self_knn_workload`); it depends
  on the database alone. Without a teacher, `run` reports it as NaN.
- **J5:** the earlier headline, J4 plus clustering. Results quoted below from earlier runs use it;
  `teacher_eval` no longer writes it.
- **J_T:** the teacher's own score on its training workloads: J4's four terms on the database's
  own queries, plus the query-free shape and margin terms, weighted by `term_weights`. `teacher_eval`
  also writes `J_T4`, the equal mean of the four J4-like terms, so training gains can be compared
  with J4 without the priors.
- **J_T's kNN term:** recall@k of the teacher's database queries, as originals, answered on the
  compressed database (`knn_recall`), so competitors are compressed too, like the test. The loop
  and trades track it with `KnnState`. Value curves use a per-trajectory check instead (is the
  simplified neighbour still within the query's original cutoff?), since they score trajectories
  one at a time.
- **J_T's join term:** held out. Every database trajectory acts as an original query against its
  compressed partners, so each pair is tested once in each direction and a trajectory's own choice
  only affects its partners' F1. This is the database-only stand-in for the held-out join; the
  teacher still never sees the query pool. A trajectory's join outcomes do not depend on the
  others' choices, so only range F1 and ranking kNN couple trajectories.
- **Join terms:** by default the teacher trains on `join.eps_m` and `join.score_eps_m`, the
  thresholds J4 averages over, so the join gain is not fitted to one threshold.
  `teacher.extra_join_eps_m` (or `teacher_eval --train-eps`) overrides the extra thresholds. Like
  J4's join grid, each term's macro F1 is normalised by its own endpoints-only floor and the terms
  are averaged equally. A term without true pairs is left out, as J4 skips NaN entries.
- **Join pairs:** the teacher uses every pair at each of its thresholds, because whoever
  compresses the database knows all of it. Whether the join gain generalises to thresholds the
  teacher never saw is tested in `teacher_eval` at 125, 175 and 250 m, between the trained ones.
- **Arm:** one named output of the teacher, e.g. `BUDGET_LOOP_TRADE`.

---

## How selection works, in plain terms

### What is being chosen
- At a given budget, the teacher chooses **one candidate per trajectory**. The six recipes are
  RDP, SED, UNIFORM, unseeded greedy, and greedy seeded with half of the interior budget from
  SED or from RDP. A variant seeded with a quarter from SED was removed: at small budgets it
  kept the same points as unseeded greedy. Seeded candidates combine classic and greedy
  point selection. Different recipes can produce identical kept sets.
- Per-trajectory budgets come from value curves, with the total fixed.
- For example, the three classic candidates for a trajectory with budget 4 might be:

  ```
  RDP     with 4 points → e.g. points {0, 37, 112, 250}
  SED     with 4 points → e.g. points {0, 61, 140, 250}
  UNIFORM with 4 points → e.g. points {0, 83, 166, 250}
  ```

  Candidates at the chosen budgets are built by `make_candidates` and reused during best response.

### Why a trajectory's choice depends on other trajectories
Range macro F1 depends on how many trajectories still answer each box, so the marginal reward
for recovering one trajectory depends on the others. Ranking kNN also depends on competing
trajectories: a neighbour drops out of a query's top k when a competitor moves closer. The
held-out join does not couple trajectories. These dependencies make the best method for i depend
on the current choices elsewhere in the database.

### The steps
1. **Value curves** (`build_curves`): for each trajectory and each budget up to its cap, score
   every candidate with `independent_value` against fixed partners (RDP at equal budgets) and keep
   the best. `BUDGET_RDP` uses RDP-only curves.
2. **Allocation** (`allocate`): fill the total budget with greedy steps along each curve's upper
   concave envelope.
3. **Recipe choice** (`BUDGET_CHOICE`): each trajectory takes its curve's best candidate at its
   allocated budget.
4. **The loop** (`BUDGET_LOOP`, class `Search`): best response on the tracked J_T, against the
   others' current choices.
5. **Trades** (`BUDGET_LOOP_TRADE`, `trade_budgets`): move 1, 2 or 4 points from one trajectory to
   another while the exact J_T rises.

On held-out runs most of the J4 gain comes from allocation, then recipe choice; the loop and
trades add little J4 but win back kNN and join (research notes, 2026-11-22).

`Teacher::single_recipes` gives the matching baselines: each candidate recipe for every trajectory
at equal budgets (`RECIPE_RDP`, …, `RECIPE_GREEDY_RDP2`). `teacher_eval` writes them with the
other baselines. `RECIPE_RDP`, `RECIPE_SED` and `RECIPE_UNIFORM` equal `RDP_a0`, `SED_a0` and
`UNI_a0`; the greedy recipes are the query-aware single recipes that choosing per trajectory must
beat.

### The loop
**In one sentence:** go through the trajectories one at a time, and let each switch to whichever
of its other methods makes the tracked J_T highest, given what everyone else currently uses;
repeat until nobody wants to switch.

1. **Who is visited:** every trajectory with a range membership, a join link, a watching kNN
   query, or any trajectory when shape or margin is weighted.
2. **Setup:** every distinct candidate is prepared once, with its separable value, kNN distances
   to the queries watching it, box coverage and join outcomes; none depend on the others. A
   `JoinState`, a `RangeState` and a `KnnState` hold the current state.
3. **One pass:** visit those trajectories in random order. For trajectory i and each of its other
   distinct candidate kept sets c:

   ```
   gain(c) = [separable(i, c) − separable(i, current)]   where-at, shape, margin
           + join.propose(i, outcomes).gain              exact change in the join F1
           + range.propose(i, coverage).gain             exact change in range macro F1
           + knn.propose(i, distances)                   ranking kNN change
   ```

   **It switches when the best gain is positive** (above 1e-12), then applies the move to all
   trackers, so later visits see i's new method.
4. **Repeat passes** until a pass has no switches (at most 10).

The kNN tracker only watches each query's top k plus its nearest others in the start state and
the candidates, so a trajectory outside those lists can enter a top k unseen. `checked_loop`
therefore compares the result's exact J_T with the start's and keeps the start if it lost.

### Why it always stops, and where
Every switch raises the tracked J_T by a strictly positive amount, and there are finitely many
choices, so it must stop. Requiring a strict gain also means no trajectory flips back and forth on
a tie. In practice there is also a limit of 10 passes.

If a pass has no switches, **no single trajectory can raise J_T by switching on its own** within
the current candidate sets and improvement tolerance. Such a local optimum need not be global. In
optimisation terms this is coordinate ascent; in game-theory terms, best-response dynamics.

### Why it is fast
The trackers keep every pair's outcome, every box's count and every query's ranking, so testing
one switch only rescores that trajectory's own pairs, boxes and watchers, and the result is exact
for range and join: the tests check it against a full recompute.

---

## workloads.hpp / workloads.cpp

### Why

The teacher needs its own queries to optimise against. If it optimised against the test queries,
its score would only show that it can fit its own exam. These workloads come from the database
alone, with different seeds from the test.

The kNN part is slow: every database trajectory is a query against all the others, with Fréchet
distances on full-resolution trajectories, which takes 25–45 minutes on Geolife. So it is cached
on disk.

### Structs

| Struct | Fields | Meaning |
|---|---|---|
| `WorkloadConfig` | `k`, `join_eps_m`, `extra_join_eps_m`, `frechet`, `range`, `where_at` | Settings, taken from the pipeline config (`knn.k`, `join.eps_m`, `teacher.extra_join_eps_m` or else `join.score_eps_m`, `frechet`, `range` with the teacher's own query count, `where_at`) |
| `KnnEntry` | `valid`, `top`, `cutoff` | Database trajectory e as a kNN query: its k nearest other trajectories, and `cutoff`, the Fréchet distance to the (k+1)th. `valid = false` if it has fewer than k+1 neighbours. |
| `JoinPair` | `i < j` | A pair of database trajectories |
| `ExtraJoin` | `eps_m`, `positives`, `negatives` | The self-join pairs at one extra threshold, all of them used |
| `TeacherWorkloads` | `knn`, `positives`, `negatives`, `extra_joins`, `range`, `where_at`, plus the settings | Everything the objective needs |

**Positives** are true self-join pairs: the originals are within `join_eps_m` (Fréchet).
**Negatives** are candidate pairs whose endpoints are within `join_eps_m`, but whose originals are
not. They matter because the join metric is F1: simplification can make negatives falsely match.

### `load_or_build_workloads(db, config, seed, cache_dir)` — public

- **Input:** the database, the settings, the seed, and the directory to look in for a cache.
- **Computation:**
  1. Fingerprints the database (`fingerprint`) and looks for `cache_dir/teacher_cache_seed<seed>.bin`.
  2. Calls `load`. If the file matches the current workload version, fingerprint, seed, `k`,
     `join_eps_m` and `frechet`, it uses the cached kNN entries and pairs. Older cache versions
     are rebuilt once because their metadata-only fingerprints cannot verify the point contents.
  3. Otherwise, it builds everything with `build_knn` and `collect_pairs`, then `save`s it.
  4. Always regenerates the range boxes and where-at probes. They are fast, and use their own seeds
     (`seed + 7001` and `seed + 7002`). Range draws its acceptance stream from its seed + 1, the
     same stream as where-at's; harmless, and kept so existing workloads and results stay valid.
  5. Always regenerates the extra joins, one self join per extra threshold. They are not cached, so
     changing the thresholds does not invalidate the kNN cache.
- **Output:** a `TeacherWorkloads`. It prints whether the cache was hit, and the counts.

### `fingerprint(db)` — internal

- **Why:** a cache is only valid for the exact same database. A different split or data file must
  not reuse it.
- **Input:** the database.
- **Computation:** an FNV-1a hash over the trajectory count, the point count, each trajectory's ID
  and length, and every point's timestamp, latitude, longitude and projected x/y coordinates.
  Including the projected coordinates also invalidates the cache when the projection changes.
- **Output:** a 64-bit number, stored in the cache header.

### `build_knn(db, config)` — internal

- **Why:** this gives the kNN term its training queries. Using database trajectories as queries
  means every trajectory appears in about k queries' answers, so every trajectory gets some kNN
  signal. With only held-out queries, many trajectories would never be in anyone's top k, and the
  teacher would treat them as free to wreck.
- **Input:** the database and the settings.
- **Computation:** in parallel, for each trajectory e:
  1. find its k+2 nearest neighbours (`query::knn`) and remove e itself;
  2. keep the first k as `top`;
  3. compute `cutoff`, the distance to the (k+1)th neighbour, with `query::frechet_distance`. This cutoff remains a proxy used by value curves. Ranking kNN
     uses `query::knn_distance`, shared with the evaluator.
- **Output:** one `KnnEntry` per trajectory.

### `collect_pairs(db, eps_m, frechet, positives, negatives)` — internal

- **Why:** this gives the join term its training pairs: what should match (positives) and what
  shouldn't (negatives).
- **Input:** the database, the threshold, the Fréchet variant, and output vectors.
- **Computation:** runs the engine's `query::generate_self_join`. For every candidate pair of each
  entry, a pair in `reference` is a positive and any other pair is a negative. The pairs at
  `join_eps_m` are cached; the extra thresholds are collected on every load.
- **Output:** the filled vectors.

### `save` / `load` / `write_pairs` / `read_pairs` / `put` / `get` — internal

- **Why:** the cache file format.
- **Version 6** (read and written): a header (magic number, seed, full-content fingerprint, k,
  eps, variant), then the kNN entries, positives and negatives. Version 5 has the same layout
  but an obsolete continuous-kNN ranking policy, so it is rebuilt. Version 4 also stored a seen /
  unseen flag per pair and is rejected. The version covers both file
  layout and cached workload generation and must be bumped when either changes.
- **Versions 2 and 3** are rejected because they used metadata-only fingerprints.
- `save` writes to a `.tmp` file and renames it, so a crash never leaves a half-written cache.
- `load` returns false on any mismatch, which triggers a rebuild.

---

## objective.hpp / objective.cpp

### Why

This is the score the teacher maximises on its training workloads. kNN, range, where-at and join
use J4's metric definitions on database queries; the value curves use a per-trajectory kNN check
instead. Improvements on training workloads must still be checked against held-out J4.

Six terms, each normalised `(metric − floor) / (1 − floor)`, then averaged with
`term_weights` (Geolife: 1, 1, 1, 1, 1, 1):

| Term | Per trajectory i, simplified | Mirrors |
|---|---|---|
| kNN | recall@k of each database query, as original, ranked on the compressed database (`knn_recall`, tracked by `KnnState`) | recall@k |
| Range | average `2 * found / (reference size + found)` over boxes with nonempty references | range macro F1 |
| Where-at | average-hit ramp over i's probes: `clamp((max − err) / (max − min), 0, 1)` | where-at average-hit |
| Join | held-out macro F1 per threshold (original query, compressed partner, both directions), each normalised by its floor, then averaged | J4's join grid mean, computed the same way |
| Shape | ramp 25–500 m on `simplification_bound(original i, simplified i)`, averaged over trajectories | none: a prior for queries the workloads lack |
| Margin | for each query e ranking i among its k + 5 nearest (`workloads.ranked`): 1 if i's deviation bound is within its gap to e's k-th place, else ½(1 + gap / deviation); averaged over memberships | held-out kNN: whether i's rank side survives any query moving it by its deviation |

`simplification_bound` is an upper bound on the continuous Fréchet distance between a trajectory
and its simplification (`certificate_test.cpp`), so for any query Q, |d(Q, simplified i) − d(Q, i)|
is at most it. Shape and margin need no query results, so they also cover held-out queries: without
them the teacher starved trajectories its training workloads never touch. The weights were chosen
on pseudo-held-out splits of the database and confirmed on four seeds (research notes, 2026-11-22).

Where-at, shape and margin are **separable**. Range macro F1 and ranking kNN depend on other
trajectories; the join and kNN are tracked incrementally during the loop and trades.

Why range has no false positives: kept points are a subset of the original points, so a simplified
trajectory can't have a point in a box the original had none in.

### `PreparedTrajectory` and `PreparedTrajectories` — public class

- **Why:** the checks need trajectory i with only its kept points, in the shape the engine's
  Fréchet and interpolation code expect. Building a full simplified database for every candidate
  would be far too slow.
- **Input:** an original `TrajectoryView` and a kept set.
- **Computation:** copies `t`, `x` and `y` at the kept indices into its own vectors and points a
  `TrajectoryView` at them. `lat` and `lon` stay empty, because the Fréchet and interpolation code
  only read `t`, `x` and `y`.
- **Output:** `view()`, valid while the object lives and is unmodified. `assign` replaces the
  retained points and rebuilds bounds while reusing coordinate storage between budgets.
  `PreparedTrajectories` owns a whole set and provides borrowed views. Fixed partners are prepared
  once for curve scoring; the loop prepares every distinct candidate of every trajectory once and
  shares duplicate views.

### Range and join helpers — internal

- `range_f1` and `range_marginal`: per-box F1 and the gain from recovering one more trajectory.
- `f1_of`: per-trajectory join F1, `2·hits / (found + expected)`. This is exactly the formula in
  the engine's join evaluator.

### `Objective(db, workloads, config)` — constructor

- **Why:** everything that doesn't depend on a particular simplification is computed once here,
  so each later score is cheap.
- **Input:** the database, the workloads, and the where-at tolerances (`ObjectiveConfig`, from the
  pipeline's `where_at_scoring`).
- **Computation:**
  1. **Inverts kNN.** The workloads say "query e has top k {i, …}". The constructor turns this into
     `m_knn[i]` = every (e, cutoff) where i is in e's top k, so scoring trajectory i only touches
     its own memberships. `m_knn_weight = 1 / (k · valid queries)`, so the term sums to recall@k.
  2. **Inverts range.** `m_range[i]` contains every box i is in. Its default weight is the
     marginal macro F1 of recovering i when all other reference trajectories are found. Greedy
     candidate construction uses these weights; curve scoring recomputes them against fixed partners.
  3. **Groups where-at probes** per trajectory in `m_probes[i]`, with
     `m_where_at_weight = 1 / trajectories with probes` (averaged per trajectory, like the
     evaluator).
  4. **Builds the join links.** Positives first, then negatives, stored in
     `m_pair_i`/`m_pair_j`. Each pair is linked from both sides in `m_links`. `m_truth[t]` counts
     t's positives.
  5. **Floors:** each join term's macro F1 with endpoints only gives `m_join_floor`, and
     `m_join_weight[k] = 1 / (terms with true pairs · (1 − floor_k))`, so the join metric is already
     normalised and 0 at the floor. Then `metrics(endpoints)` gives each metric's floor. If a floor
     is already 1, it throws, because that metric can't be normalised.
  6. **Scales:** `m_scale[m] = weight_m / (Σ weights · (1 − floor_m))`. This applies the
     normalisation and the weighted average in one factor.
  7. **`m_endpoint_separable[i]`:** trajectory i's scaled separable terms with endpoints only. Subtracting it makes
     `separable` exactly 0 at the floor.
- **Output:** a ready `Objective`. It keeps references to `db` and `workloads`, so both must
  outlive it (`Teacher` guarantees this).

### `raw_separable(i, candidate)` — private

- **Why:** the where-at, shape and margin terms for one trajectory. The kNN slot stays 0: the
  per-trajectory kNN check is done in `independent_value`, the ranking term in `metrics`.
- **Input:** trajectory index and a prepared candidate view.
- **Computation:**
  - **Where-at:** for each probe, interpolate the simplified position at the probe's time
    (`core::position_at`), take the distance to the true position, apply the ramp, and average over
    i's probes times `m_where_at_weight`.
  - **Shape and margin**, when weighted: `simplification_bound` once, then the shape ramp and each
    of i's margins against it.
- **Output:** `Terms`, with i's raw separable contributions (range and join slots 0).

### `separable(i, kept)` — public

- **Input:** trajectory index and kept set.
- **Computation:** `Σ m_scale[m] · raw_separable[m] − m_endpoint_separable[i]` over where-at,
  shape and margin.
- **Output:** i's separable part of J_T, 0 at endpoints only. Summed over all
  trajectories, this equals those terms' share of J_T exactly.

### `independent_value(i, candidate, partners)` — public

- **Why:** the value curves need one number per (trajectory, candidate), computed without knowing
  what the other trajectories will choose.
- **Input:** trajectory index, a prepared candidate view, fixed partner views (empty means
  original partners), and an optional `JoinState` built on those partners. Partners enter the range
  marginals and, via `JoinState`, the join marginals.
- **Computation:** `separable(i, candidate)`, plus the per-trajectory kNN check over i's
  memberships, plus the marginal range F1 of covering i against the partners, plus the exact
  marginal held-out join F1 on i's partners' slots (`JoinState::gain`) when `JoinState` is given.
- **Output:** an approximate per-trajectory score: range is marginal against fixed partners and
  kNN ignores competitors. The loop corrects both.
- `prepare_independent` computes the range, where-at and join terms once, together with
  the upper bound used to order candidates. `independent_value(score, beat)` reuses those terms
  while checking kNN, so a candidate is not rescored just because its bound was promising. The
  prepared score borrows its candidate's coordinates and must be used before they change.

### `match`, `held_out_match`, `original_within` — public

- `match`: `query::frechet_within` with the term's threshold and the workload's Fréchet variant.
- `held_out_match`: whether original `link.other` finds candidate i. Each pair's originals get a
  distance bracket once (discrete Fréchet from above, exact below the largest threshold; Hausdorff
  from below). With dev = `simplification_bound(i)`, upper + dev ≤ ε proves a hit and
  lower − dev > ε a miss, without the test (`certificate_test.cpp`). kNN checks in the curves do
  the same with the ranked neighbours' distances.
- `original_within(j, candidate, r)`: every remaining test of an original against a candidate
  (join, curve kNN checks, `propose_bounded`, `watch`). Each original has a coarse RDP copy with
  deviation μ ≤ 5 m (`kCoarseM`), used only if it at least halves the points. The coarse copy
  within r − μ proves a hit, not within r + μ a miss; only the band between needs the full test.
- All of these are exact: they decide the same outcomes as the full test (continuous Fréchet
  only), and `CoarseScreenAgreesWithTheFullTest` checks it.
- Speed, Geolife seed 42, 24 threads: these screens, one `CandidateMaker` per trajectory shared by
  curves, candidates and trades, the k-th place read from the ranking in `propose_bounded` (and
  the stored distance ± both deviation bounds tried first), the greedy where-at score skipping
  `hypot` outside its ramp, cached edge lengths for the deviation bound, the loop reusing the
  tracker's distances for each start candidate, and the tracker taking its list distances from
  `query::knn_with_distances` took the teacher from 206 s to 104 s at 0.25 % and from 444 s to
  192 s at 1 %, with identical kept sets on seeds 42 and 7 (research notes,
  `Teacher speedups 2026-11-22`).

### `join_metric(state)` — private

- **Input:** a whole simplified database.
- **Computation:** for every pair, in parallel, original i against compressed j and original j
  against compressed i; each outcome counts for its query. Then `f1_of(hits, found, truth)` per
  trajectory and term, averaged over trajectories with any truth or any match, like the evaluator.
- **Output:** the normalised join metric.

### `metrics(kept)` — public

- **Input:** a whole simplified database.
- **Computation:** prepares the trajectories once, sums `raw_separable` over their views (in
  parallel), reuses those views in `RangeState::metric` and `join_metric`, and ranks kNN on the
  materialised database with `knn_recall`.
- **Output:** the six raw metric values. Used for the floors, `teacher_eval`'s `terms.csv`, and
  in tests.

### `total(kept)` — public

- **Input:** a whole simplified database.
- **Computation:** `Σ_m m_scale[m] · (metric_m − floor_m)`.
- **Output:** the exact J_T. It is 0 with endpoints only, and higher is better. The pipeline prints
  it next to each arm.

### `RangeState` — public class

Tracks the found trajectory count in every box and each trajectory's coverage. `propose` computes
an exact scaled macro-F1 change without mutation; `apply` updates the counts. `weights(i)` gives
F1's marginal reward for recovering i with everyone else fixed, reused across candidates and
budgets. Greedy candidates use the original-partner marginals; selection and value curves use
their specified partner state. The loop updates range F1 against the actual current state.

### `JoinState` — public class

#### Why

Best response tries thousands of single-trajectory switches. Recomputing `join_metric` for every
try would rescore about 90,000 pairs each time. `JoinState` keeps every pair's outcome and every
trajectory's counts, so a switch of trajectory i only rescores i's own pairs (usually a handful).
The result is exact, not approximate: the tests check it against a full recompute.

#### `JoinState(objective, state)` — constructor

- **Input:** the objective and a starting simplified database.
- **Computation:** each trajectory's outcomes as the compressed side (`outcomes`, in parallel).
  Each outcome is stored per pair and direction in `m_matched` and counted for the query, then F1
  is summed over counted slots into `m_sum` and `m_counted`, per join term.
- **Output:** a state whose `metric()` equals `join_metric(state)`.

#### `outcomes(i, candidate)` and `propose(i, outcomes)`

- `outcomes`: i's link outcomes with i as candidate. They do not depend on the state, so the loop
  computes them once per candidate.
- `propose`: for every link whose outcome flips, the change in hits and found at the query, then
  each affected slot's F1 swapped in the sum. Returns a `Proposal` with
  `gain = m_scale[Join] · (new join metric − current join metric)`, valid until the next `apply`.

#### `apply(proposal)`

Writes the saved outcomes into `m_matched`, the counts and the sum, without repeating Fréchet
tests.

#### `metric()`, `counted(t, found)`, `f1(t, hits, found)`

Helpers: the current join metric; whether slot t counts (it has truth or matches); and t's F1.

---

## curves.hpp / curves.cpp

### `Curve`

- **Why:** a value curve records what one trajectory is worth at each budget. The allocator uses
  these values to redistribute points; selection reads the method at the chosen budget.
- **Fields:** `value[k]` and `method[k]` for budget `b = 2 + k`: the best candidate's
  `independent_value` and which candidate reached it.
- **Helpers:** `max_budget()` is the largest budget on the curve, and `at(b)` is the value at b.
- **Note:** Fréchet thresholds flip on and off, so a curve is a step function and need not rise at
  every step.

`allocate(curves, total_budget, value)` starts with endpoints and fills the remaining budget with
steps along each curve's upper concave envelope: each time it takes the step, of any length, with
the best gain per point. This is greedy allocation on the concave envelope, exactly optimal for the
relaxed problem and whenever the budget ends on envelope vertices, with no step-length limit to
choose. It replaced greedy steps of up to three points. A donor/receiver
exchange pass used to follow; across the runs of November 2026 it never changed the curve sum by
more than 0.0001, so it was removed. It preserves the exact total budget and respects every curve's maximum budget. It is a
heuristic search of the curve sum, which is distinct from the exact J_T with both partners
simplified.

---

## teacher.hpp / teacher.cpp

### Constants and types

| Name | Meaning |
|---|---|
| `CandidateIndex`, `kCandidates` | Named indices and labels for the six candidate recipes. The index is the "method". |
| `kArms` | the four arm names, listed in `teacher.hpp` |
| `TeacherConfig` | `enabled`, `cache_dir`, `arms`, `range_query_count` (default 128000), `extra_join_eps_m` (default: `join.score_eps_m`), `term_weights` (kNN, range, where-at, join, shape, margin; 4 to 6 values, missing ones 0; default 1, 1, 1, 1, 0, 0) |
| `Arm` | one result: `name`, `simplification` (for the pipeline), `objective` (its J_T) |
| `Candidates` | `candidates[c][i]` = trajectory i simplified with method c |
| `Methods` | one method index per trajectory, i.e. a selection |
| `kMaxResponsePasses` | at most 10 best-response passes |
| `kKnnWatchFactor` | each kNN query watches its 3k nearest trajectories in the start state |
| `kTradeSteps`, `kMaxTradeRounds` | trades move 1, 2 or 4 points; at most 3 rounds |
| `kMinImprovement` | strict gain threshold, 1e-12 |

### `Teacher(db, workloads, config, seed)` — constructor

- **Input:** the database, the workloads (moved in), the objective settings, and the seed.
- **Computation:** stores the workloads, then builds the `Objective` on them. `m_workloads` is
  declared before `m_objective`, so it is constructed first. That ordering matters because the
  objective keeps a reference to it. Prints the pair counts and floors.
- **Output:** a ready teacher. It can't be copied, since that would break the reference.

### `Teacher::run(total_budget, arms)` — public, the main entry

- **Input:** the total point budget for this share, and the arm names to produce.
- **Computation:**
  1. Rejects unknown arm names.
  2. `equal_budgets` = equal budgets per trajectory (the engine's size^α allocator with α = 0).
     Curves are capped at about four times the mean interior budget, at least six interior
     points, never below the equal budget and never above trajectory length.
  3. RDP at those budgets (`make_candidates`): the fixed partners for the value curves.
  4. `build_curves`, RDP only unless a later arm needs the best candidate.
  5. **`BUDGET_RDP`:** `allocate` on the RDP curves.
  6. **`BUDGET_CHOICE`:** `allocate` on the best curves; each trajectory takes its curve's method.
  7. **`BUDGET_LOOP`:** `Search(...).run(seed)` from `BUDGET_CHOICE`, then `checked_loop`.
  8. **`BUDGET_LOOP_TRADE`:** `trade_budgets` from `BUDGET_LOOP`.
  9. `keep` assembles each arm's kept sets, computes its exact J_T, and prints how many
     trajectories each method won. `timed` prints step durations.
- **Output:** one `Arm` per requested name, in the requested order. Each keeps exactly
  `total_budget` points, because every candidate keeps exactly its trajectory's budget.

### `make_candidates(objective, budgets, caps, makers, selected)` — internal

- **Why:** each candidate is simplified once, then reused by every step.
- **Input:** objective, per-trajectory budgets and caps, the shared makers, and optionally
  selected method indices.
- **Computation:** in parallel, each trajectory's `CandidateMaker` builds all six recipes, or only
  the selected one. A maker is created on first use and kept for the whole `run`, so its RDP, SED
  and greedy sequences are built once and serve the curves, the candidates and the trades.
- **Output:** `Candidates`.

### `assemble(candidates, method)` — internal

- **Input:** candidates and a selection.
- **Computation:** for each trajectory, picks the kept set of its chosen method.
- **Output:** a whole simplified database (`KeptSets`).

### `build_curves(db, objective, partners, caps, rdp_only)` — internal

- **Why:** the value curves: for each trajectory and each budget, the best method and its value.
- **Input:** the partner reference state (RDP at equal budgets), curve caps, and whether only RDP
  is needed.
- **Computation:** in parallel per trajectory, build the RDP addition sequence and reversed SED
  removal sequence once, then use sorted prefixes for each budget. Candidate coordinate buffers
  are reused between budgets. For b = 2 … max:
  - simplify with every candidate (at b = 2 all keep the endpoints, so only one is scored);
  - score each with `independent_value` against the fixed partner reference;
  - record the best value and method.
- **Output:** RDP-only and best-candidate curves per trajectory.
- **Why partners as RDP:** a fixed, plausible guess at how partners will look, so range marginals
  reward covering boxes the others will likely miss.

### `curve_methods(curves, budgets)` — internal

- **Input:** curves and per-trajectory budgets.
- **Computation:** `method[b − 2]` for each trajectory. Throws if a budget is off its curve.
- **Output:** `Methods` at the supplied budgets.

### `Search(db, objective, candidates, method)` and `run(seed)` — internal

The loop described above. The constructor prepares every distinct candidate and builds the
trackers; `run` updates them in place, so it runs once.

### `trade_budgets(db, objective, caps, kept, value)` — internal

- **Why:** allocation scores trajectories against fixed partners; trades correct budgets against
  the actual state.
- **Computation:** each round, every trajectory's best recipe at `b ± 1, 2, 4` is scored against
  the current state (kNN by `propose_bounded`, computing a distance only near a query's k-th
  place). The kNN lists then watch every offered version. Buyers are paired with the cheapest
  sellers; each pair is rescored exactly as applied and kept only if the sum gains.
- **Output:** the traded kept sets and their exact J_T, or the input if the exact J_T fell.

### `wins`, `is_arm` — helpers

- `wins`: per-candidate counts for the log.
- `is_arm`: whether a name is in `kArms`. It is also used by the config parser.

---

## In the pipeline

- **`pipeline/config.cpp`, `parse_teacher_config`:** reads the `teacher` section. It rejects
  unknown keys, an empty `cache_dir`, an empty `arms` list when enabled, unknown or duplicate arm
  names, and `range_query_count` = 0.
- **`pipeline.cpp`, in `run`:**
  - If `teacher.enabled`, it builds the workloads from the `database` split, with settings taken
    from the engine config. Then it constructs `teacher_model`.
  - Per budget share, after the regular algorithms, `teacher_model->run` produces the arms. Each
    goes through the same `evaluate` lambda: materialise, score on the held-out workloads, print,
    and export as `<ARM>_<share>pct`.

## Tests

`engine/tests/teacher/teacher_test.cpp` uses a small synthetic database:

- `CacheIsWrittenThenReadBack`: a cache round trip gives identical workloads.
- `CacheForOtherSettingsIsRebuilt`: a cache built with a different `k` isn't reused.
- `ObjectiveIsZeroAtTheFloorAndTracksJoinExactly`: J_T = 0 at endpoints only. After random
  switches, the tracked range and join F1 equal full recomputations, and their gains plus separable predict the
  real change in J_T.
- `SeveralJoinTermsAreTrackedExactly`: the same exactness with two extra thresholds.
- `ContinuousKnnNearTiesMatchFullScoringThroughSwitches`: the former 23/24 versus 1 regression,
  checked through switches and bounded proposals.
- `RankingKnnIsTrackedExactly`: when every query watches every trajectory, `KnnState`'s tracked
  recall equals a full recompute after random switches, and the kNN, range, join and separable changes
  together predict the real change in J_T.
- `RankingKnnArmsKeepTheBudget`: the arms keep exactly the budget with the ranking kNN term.
- `SingleRecipesKeepTheBudget`: the six single-recipe baselines each keep exactly the budget.
- `ArmsKeepTheBudgetAndBestResponseOnlyImproves`: every arm keeps exactly the budget, endpoints
  kept, indices sorted; `BUDGET_LOOP`'s J_T ≥ `BUDGET_CHOICE`'s and the trades' ≥ the loop's.
- `JoinTermsAreNormalisedApartAndAveragedLikeJ4`: the join metric equals the mean over thresholds
  of each threshold's macro F1, normalised by its own endpoints-only floor.
- `ShapeTermRampsTheDeviationBound`, `TwoSidedMarginScoresOutsidersToo`: shape and margin are 1
  with every point, 0 at the floor, separable, and the margin term needs ranked neighbours.
- `CoarseScreenAgreesWithTheFullTest`: on dense trajectories, `original_within` gives the full
  Fréchet test's answer at, near and far from the exact distance.
- `PreparedIndependentScoresMatchFullScoringAndSurvivePruning`: independent scores equal direct
  scoring, with and without ranked-neighbour brackets for the kNN checks.
- `BoundedKnnProposalsMatchFullDistances`: `propose_bounded`, with and without a reach, equals
  the proposal from exact distances.

`engine/tests/teacher/certificate_test.cpp` checks `simplification_bound` never falls below the
continuous Fréchet distance, alone and added to a discrete pair distance.

`engine/tests/pipeline/config_test.cpp` has the three teacher config tests.
