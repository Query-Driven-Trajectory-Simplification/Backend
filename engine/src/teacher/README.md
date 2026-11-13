# The teacher

The teacher is a slow, query-aware simplifier. It searches how many points each trajectory gets
and which candidate keeps them: RDP, SED, UNIFORM or one of four greedy variants. Its output is
meant to be a target that a learned model can later imitate.

It sees **only the database** (the 85% split) and workloads built from it with their own seeds. It
never sees the held-out query pool that the pipeline scores on. The fixed-budget arms use equal
shares (α = 0); the allocation arms redistribute the same total point budget.

## How it fits together

```
pipeline::run                                    (pipeline.cpp)
 ├─ load_or_build_workloads(database, ...)        workloads.cpp: kNN, pairs, boxes, probes; cache
 ├─ Teacher(database, workloads, ...)             teacher.cpp: builds the Objective once
 │    └─ Objective(db, workloads)                 objective.cpp: inverts workloads, computes floors
 └─ per budget share:
      Teacher::run(budget, arms)
       ├─ make_candidates                         seven candidate recipes at equal budgets
       ├─ GATEORIG_a0: independent_choice         best method, partners original
       ├─ GATE_a0 / BR_a0                         fixed budgets, selection / best response
       ├─ build_curves → allocate                 partners fixed at equal-budget RDP
       │   ├─ TEACHER_RDP                         allocation using RDP alone
       │   ├─ TEACHER → BR_TEACHER                 best candidates, then best response
       │   └─ BR_SHUFFLED                         control with relabelled join partners
       └─ ITER_TEACHER                            rebuild curves, reallocate, best response
      → each arm evaluated and exported like RDP and SED
```

Files:

| File | Contents |
|---|---|
| `workloads.hpp/.cpp` | The teacher's training workloads, and the cache that saves the slow part |
| `objective.hpp/.cpp` | J_T, the score the teacher maximises, and `JoinState` for fast exact join updates |
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
  self join, whose pairs the teacher trains on, is then reported separately (`norm_self_join`,
  the seen / unseen recall and the 100 / 300 m thresholds) and is not part of J4. Results quoted
  below predate this and have the self join inside the score.
- **J5:** the earlier headline, J4 plus clustering. Results quoted below from earlier runs use it;
  `teacher_eval` still writes it so old and new runs can be compared.
- **J_T:** the teacher's own score on its training workloads. It has the same shape as J4. kNN and range use training proxies; an optional shape term can also be
  enabled.
- **J_T's join term:** follows `join.mode`, so the teacher trains on the kind of join it is
  scored on. `Self`: both sides of a pair compressed, the self join. `HeldOut`: every database
  trajectory also acts as an original query against its compressed partners, so each pair is
  tested once in each direction and a trajectory's own choice only affects its partners' F1.
  This is the database-only stand-in for the held-out join, as the kNN proxy is for held-out
  kNN; the teacher still never sees the query pool. `teacher.join` (or `teacher_eval
  --teacher-join Self|HeldOut`) overrides it, e.g. to train on the self join for the self-join
  experiment. Held out, `GATE_a0` and `GATEORIG_a0` are the same arm, and best response is
  coupled only through each query's F1. Everything below describing the join term as "both
  sides compressed" is the `Self` mode.
- **Join pairs:** the teacher uses every 150 m pair, because whoever compresses the database knows
  all of it. The pairs are also split in half (seen / unseen); with `all_join_pairs = false` the
  teacher only gets the seen half, which shows how much of its join gain is fitting those
  specific pairs. Whether the join gain generalises is tested at other thresholds (100 m, 300 m).
- **Arm:** one named output of the teacher, e.g. `GATE_a0`.

---

## How selection works, in plain terms

### What is being chosen
- At a given budget, the teacher chooses **one candidate per trajectory**. The seven recipes are
  RDP, SED, UNIFORM, unseeded greedy, greedy seeded with a quarter or half of the interior budget
  from SED, and greedy seeded with half from RDP. Seeded candidates combine classic and greedy
  point selection. Different recipes can produce identical kept sets.
- The fixed-budget arms use equal shares (α = 0). Allocation arms search per-trajectory budgets
  using value curves, while keeping the total fixed.
- For example, the three classic candidates for a trajectory with budget 4 might be:

  ```
  RDP     with 4 points → e.g. points {0, 37, 112, 250}
  SED     with 4 points → e.g. points {0, 61, 140, 250}
  UNIFORM with 4 points → e.g. points {0, 83, 166, 250}
  ```

  Candidates at the chosen budgets are built by `make_candidates` and reused during best response.

### Why a trajectory's choice depends on other trajectories
kNN, range and where-at only involve trajectory i itself, because everyone else is compared in
original form. The join is different: a pair (i, j) is scored with **both** sides simplified. Two
trajectories can gain or lose a join match when either kept set changes. So the best method for i
depends on what its join partners chose.

### Two stages
The fixed-budget arms illustrate two stages, which differ in how the join partners are seen:

```
Stage 1, GATE_a0:  one round, every trajectory on its own, partners ASSUMED to be RDP
Stage 2, BR_a0:    starts from stage 1, partners are their ACTUAL current choices
                   pass 1, pass 2, ... until a pass makes no switches
```

| | Stage 1 (GATE_a0) | Stage 2 (BR) |
|---|---|---|
| Points per trajectory | fixed budget | fixed budget |
| Methods tried for i | all 7 recipes | other distinct kept sets |
| Terms scored | all 4 (kNN, range, where-at, join) | all 4, as the change from i's current method |
| i's join partners seen as | RDP (a guess) | their actual current methods |
| Join change counted for | i only | i and every partner whose pair flips |
| Order | all trajectories at once, independently | one at a time; each switch is visible to the next |
| Runs | once | until no switches, at most 10 passes |

Stage 1 is not needed for correctness: BR could start from all-RDP and still climb. It gives BR a
good starting point, and it is a result of its own (the independent, one-pass teacher), which
shows how much stage 2 adds.

### Stage 1: GATE_a0 (`independent_choice`, or `curve_methods`)
For each trajectory i, independently, score its candidates at its equal budget with
`independent_value`, with join partners fixed at equal-budget RDP. Choose the highest score;
ties go to the earlier candidate. If allocation arms also need curves, read the same choice from
the curve at that budget instead.

The independent scores use a fixed partner reference. Their best choices need not maximise the
actual J_T once both sides of every join pair use their chosen candidates.

### Allocation arms
`build_curves` scores budgets from 2 up to a per-trajectory cap. `allocate` fills the total budget
with greedy additions, then improves the curve sum by moving up to three points between a donor
and a receiver. `TEACHER_RDP` uses RDP-only curves; `TEACHER` uses the best of all candidates.
`BR_TEACHER` applies best response at those allocated budgets. `BR_SHUFFLED` is a control that
searches with relabelled partners, then scores the result against the real workloads.

`ITER_TEACHER` starts from `BR_TEACHER`. Each round rebuilds curves against the current partner
state, reallocates and runs best response. A round is accepted only if the exact J_T improves by
more than 1e-12; at most three rounds are tried.

### Stage 2: BR (`best_response`)
**In one sentence:** go through the trajectories one at a time, and let each switch to whichever
of its other methods makes the *real* J_T highest, given what everyone else currently uses;
repeat until nobody wants to switch.

1. **Who is visited:** only trajectories with at least one join pair. For the others, the
   score doesn't depend on anyone else, so their stage-1 choice is already their best.
2. **Starting state:** stage 1's method for every trajectory, plus a `JoinState` holding every
   pair's outcome (match or not) with both sides on their stage-1 methods.
3. **One pass:** visit those trajectories in random order. For trajectory i and each of its other
   distinct candidate kept sets c:

   ```
   gain(c) = [separable(i, c) − separable(i, current)]   change in i's kNN, range, where-at
           + join.propose(i, c, state).gain              exact change in the join F1
   ```

   `join.propose` rescores **all** of i's pairs against the partners' current kept points, and
   updates the F1 of i and of every partner whose pair flips.

   **It switches when the best gain is positive** (above 1e-12). The two parts can pull in
   opposite directions: a switch that costs i a little where-at can still win if it fixes a join
   pair. After a switch, `join.apply` stores the new pair outcomes, and later visits see i's new
   method.
4. **Repeat passes** until a pass has no switches (at most 10).

**Why it repeats.** A switch changes every pair with that trajectory in it, so its partners may
now prefer a different method too. If a partner comes later in the same pass, it is checked
against the new choice right away; otherwise the next pass picks it up. One switch can ripple
through connected trajectories.

### Example
A reduced example using only the three classic candidates: trajectories 1–4; (1, 4) is a true
join pair, and 1 is also paired with 2.

```
stage 1 (GATE_a0): 1=RDP  2=RDP  3=SED  4=SED
  1 chose RDP assuming its partner 4 would be RDP, but 4 actually chose SED,
  so the pair (1, 4) doesn't match.

stage 2, pass 1 (random order):
  visit 3: try RDP, UNIFORM → no gain                   1=RDP  2=RDP  3=SED  4=SED
  visit 1: try SED, UNIFORM, against 4 as SED:
           SED makes (1, 4) match; it also rescores (1, 2);
           total gain > 0 → switch                       1=SED  2=RDP  3=SED  4=SED
  visit 4: try RDP, UNIFORM → would break (1, 4): stays
  visit 2: scored against 1 as SED (its new method):
           SED makes (1, 2) match again → switch         1=SED  2=SED  3=SED  4=SED
pass 1 had switches → pass 2
pass 2: nobody gains → stop
```

1's switch only became worthwhile once the teacher used 4's actual method, and 2's switch only
became worthwhile after 1 switched. Neither could be found by scoring each trajectory on its own.

### Why it always stops, and where
Every switch raises J_T by a strictly positive amount, and there are finitely many choices, so it
must stop. Requiring a strict gain also means no trajectory flips back and forth on a tie. In
practice there is also a limit of 10 passes.

If a pass has no switches, **no single trajectory can raise J_T by switching on its own** within
the current candidate sets and improvement tolerance. Reaching the pass limit does not guarantee
this. Such a local optimum need not be global: switching two trajectories together might still
help. The random visiting order can change which local optimum it lands in. In
optimisation terms this is coordinate ascent; in game-theory terms, best-response dynamics.

### Why it is fast
`JoinState` keeps every pair's outcome and every trajectory's hit and found counts. Testing one
switch only rescores that trajectory's own pairs (usually a handful) instead of all
pairs, and the result is exact: the tests check it against a full recompute.

### What it shows
In an earlier three-candidate run, seed 42 at 0.25%: GATE_a0 0.506 → BR_a0 0.544 J5, with the same
methods and budgets. The gain is in join (0.522 → 0.621) and clustering. The only difference is using the
partners' actual choices. Part of the join gain is in-sample, though: recall on unseen pairs falls
below seen pairs (0.78 against 0.89).

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
| `WorkloadConfig` | `k`, `join_eps_m`, `frechet`, `range`, `where_at` | Settings, taken from the pipeline config (`knn.k`, `join.eps_m`, `frechet`, `range` with the teacher's own query count, `where_at`) |
| `KnnEntry` | `valid`, `top`, `cutoff` | Database trajectory e as a kNN query: its k nearest other trajectories, and `cutoff`, the Fréchet distance to the (k+1)th. `valid = false` if it has fewer than k+1 neighbours. |
| `JoinPair` | `i < j`, `seen` | A pair of database trajectories |
| `TeacherWorkloads` | `knn`, `positives`, `negatives`, `range`, `where_at`, plus the settings | Everything the objective needs |

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
  3. Otherwise, it builds everything with `build_knn` and `build_pairs`, then `save`s it.
  4. Always regenerates the range boxes and where-at probes. They are fast, and use their own seeds
     (`seed + 7001` and `seed + 7002`).
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
  3. compute `cutoff`, the distance to the (k+1)th neighbour, with `distance()`. For continuous
     Fréchet, the discrete distance is used as the upper bound of the search.
- **Output:** one `KnnEntry` per trajectory.

### `build_pairs(db, config, seed, positives, negatives)` — internal

- **Why:** this gives the join term its training pairs: what should match (positives) and what
  shouldn't (negatives).
- **Input:** the database, the settings, the seed, and output vectors.
- **Computation:** runs the engine's `query::generate_self_join`. For every candidate pair of each
  entry, a pair in `reference` is a positive and any other pair is a negative. Each list then goes
  through `split_seen`.
- **Output:** the filled vectors.

### `split_seen(pairs, seed)` — internal

- **Input:** a pair list and a seed.
- **Computation:** a seeded shuffle; the first half is marked `seen`; then sorted by (i, j).
  Positives use `seed + 7003`, negatives `seed + 7005`.
- **Output:** the same pairs, half of them seen. The split is deterministic for a given seed.

### `save` / `load` / `write_pairs` / `read_pairs` / `put` / `get` — internal

- **Why:** the cache file format.
- **Version 4** (read and written): a header (magic number, seed, full-content fingerprint, k,
  eps, variant), then the kNN entries, positives and negatives. The version covers both file
  layout and cached workload generation and must be bumped when either changes.
- **Versions 2 and 3** are rejected because they used metadata-only fingerprints.
- `save` writes to a `.tmp` file and renames it, so a crash never leaves a half-written cache.
- `load` returns false on any mismatch, which triggers a rebuild.

---

## objective.hpp / objective.cpp

### Why

This is the score the teacher maximises. It has the same shape as J4, so that improving it
improves J4. It is computed only on the teacher's training workloads. Over the arms tested so far,
J_T correlated with the earlier J5 at r ≈ 0.95.

The four terms, each normalised `(metric − floor) / (1 − floor)` and then averaged:

| Term | Per trajectory i, simplified | Mirrors |
|---|---|---|
| kNN | for each query e that has i in its top k: Fréchet(original e, simplified i) ≤ cutoff(e) | recall@k |
| Range | for each box i is in: does a kept point still lie inside the box? | range F1, as recall (there are no false positives) |
| Where-at | average-hit ramp over i's probes: `clamp((max − err) / (max − min), 0, 1)` | where-at average-hit |
| Join | macro F1 over the pairs used, both sides simplified | join F1, computed the same way |

The first three are **separable**: they add up over trajectories, because each check only involves
trajectory i (the others stay original). Join is **not separable**: whether a pair matches depends
on both sides' simplifications.

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
  once for curve scoring; best response prepares each unique candidate once and shares duplicate
  views. Trajectories without join links only need their selected candidate prepared.

### `inside(box, view, p)` and `f1_of(hits, found, expected)` — internal

- `inside`: is point p of the view inside the box's space and time window?
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
  2. **Inverts range.** `m_range[i]` = every box i is in, each with weight
     `1 / (|box reference| · boxes with any reference)`. Summed, this is recall averaged per box.
  3. **Groups where-at probes** per trajectory in `m_probes[i]`, with
     `m_where_at_weight = 1 / trajectories with probes` (averaged per trajectory, like the
     evaluator).
  4. **Builds the join links.** Positives first, then negatives (all, or only the seen half), stored in
     `m_pair_i`/`m_pair_j`. Each pair is linked from both sides in `m_links`. `m_truth[t]` counts
     t's positives.
  5. **Floors:** `metrics(endpoints)` gives each metric's floor. If a floor is already 1, it
     throws, because that metric can't be normalised.
  6. **Scales:** `m_scale[m] = 1 / (4 · (1 − floor_m))`. This applies the normalisation and the
     equal average in one factor.
  7. **`m_v0[i]`:** trajectory i's scaled separable terms with endpoints only. Subtracting it makes
     `separable` exactly 0 at the floor.
- **Output:** a ready `Objective`. It keeps references to `db` and `workloads`, so both must
  outlive it (`Teacher` guarantees this).

### `raw_separable(i, kept)` — private

- **Why:** the actual kNN, range and where-at checks for one trajectory.
- **Input:** trajectory index and kept set.
- **Computation:**
  - **kNN:** for each membership (e, cutoff), test `frechet_within(original e, simplified i,
    cutoff)` and add `m_knn_weight` if it holds. Competitors stay original.
  - **Range:** for each box, add the box's weight if any kept point is `inside`.
  - **Where-at:** for each probe, interpolate the simplified position at the probe's time
    (`core::position_at`), take the distance to the true position, apply the ramp, and average over
    i's probes times `m_where_at_weight`.
- **Output:** `Terms`, with i's raw contribution to each metric (join slot 0).

### `separable(i, kept)` — public

- **Input:** trajectory index and kept set.
- **Computation:** `Σ m_scale[m] · raw_separable[m] − m_v0[i]` over kNN, range and where-at.
- **Output:** i's part of J_T from the three separable terms, 0 at endpoints only. Summed over all
  trajectories, this equals those terms' share of J_T exactly.

### `independent_value(i, candidate, partners)` — public

- **Why:** the value curves need one number per (trajectory, candidate), computed without knowing
  what the other trajectories will choose.
- **Input:** trajectory index, a prepared candidate view, and fixed partner views. An empty
  partner span means original partners (the `GATEORIG_a0` arm); otherwise the views contain RDP
  at equal budgets for `GATE_a0`.
- **Computation:** `separable(i, candidate)` plus i's own join F1 against those fixed partners: count
  `found` and `hits` over i's links using `match`, then add
  `m_scale[Join] · F1_i / (trajectories with truth)`.
- **Output:** an approximate per-trajectory score. It's approximate because the real join depends
  on partners' actual choices and counts each pair on both sides. That is exactly what
  `best_response` corrects.
- `prepare_independent` computes the range, where-at, shape and join terms once, together with
  the upper bound used to order candidates. `independent_value(score, beat)` reuses those terms
  while checking kNN, so a candidate is not rescored just because its bound was promising. The
  prepared score borrows its candidate's coordinates and must be used before they change.

### `match(a, b)` — public

- **Input:** two prepared trajectory views (or an original view).
- **Computation:** `query::frechet_within` on the simplified views, with the workload's threshold
  and Fréchet variant.
- **Output:** whether the pair would be reported by the join.

### `join_metric(state)` — private

- **Input:** a whole simplified database.
- **Computation:**
  1. In parallel, `match` every pair used.
  2. For each matched pair, increment `found` on both sides, and `hits` too if it's a positive.
  3. Average `f1_of(hits, found, truth)` over trajectories with any truth or any match. This is the
     same counting rule as the engine's evaluator.
- **Output:** macro join F1 on the pairs used.

### `metrics(kept)` — public

- **Input:** a whole simplified database.
- **Computation:** prepares the trajectories once, sums `raw_separable` over their views (in
  parallel), and reuses those views in `join_metric`.
- **Output:** the four raw metric values. Used for the floors, and in tests.

### `total(kept)` — public

- **Input:** a whole simplified database.
- **Computation:** `Σ_m m_scale[m] · (metric_m − floor_m)`.
- **Output:** the exact J_T. It is 0 with endpoints only, and higher is better. The pipeline prints
  it next to each arm.

### `JoinState` — public class

#### Why

Best response tries thousands of single-trajectory switches. Recomputing `join_metric` for every
try would rescore about 90,000 pairs each time. `JoinState` keeps every pair's outcome and every
trajectory's counts, so a switch of trajectory i only rescores i's own pairs (usually a handful).
The result is exact, not approximate: the tests check it against a full recompute.

#### `JoinState(objective, state)` — constructor

- **Input:** the objective and a starting simplified database.
- **Computation:** matches every pair used once (from its smaller side, in parallel). It fills
  `m_matched`, `m_hits[t]` and `m_found[t]`, then sums F1 over counted trajectories into `m_sum`
  and `m_counted`.
- **Output:** a state whose `metric()` equals `join_metric(state)`.

#### `propose(i, candidate, state)`

- **Why:** "how much would J_T's join term change if i switched to `kept`?", without changing
  anything.
- **Input:** trajectory index, prepared candidate view, and current partner views.
- **Computation:**
  1. For each of i's links, compare the pair's current outcome with its outcome under the
     candidate.
  2. For every pair that flips, record the changes to hits and found on both trajectories.
  3. For each affected trajectory, remove its old F1 from the sum and add its new one, adjusting
     the counted trajectories when one starts or stops being counted.
- **Output:** a `Proposal` with the trajectory index, pair outcomes, and
  `gain = m_scale[Join] · (new macro F1 − current macro F1)`. Added to the cached `separable`
  difference, this is the exact change in J_T. Proposals remain valid until the next `apply`.

#### `apply(proposal)`

- **Input:** the winning proposal evaluated against the current state.
- **Computation:** writes its saved outcomes into `m_matched`, the counts and the sum, without
  repeating Fréchet tests. The caller also switches the current view to the winning candidate.
- **Output:** nothing; the state now matches the new solution.

#### `metric()`, `counted(t, found)`, `f1(t, hits, found)`

Helpers: the current macro F1; whether trajectory t counts (it has truth or matches); and t's F1.

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

`allocate(curves, total_budget, stats)` starts with endpoints, fills the remaining budget with
greedy steps of up to three points, then makes improving donor/receiver exchanges of up to three
points. It preserves the exact total budget and respects every curve's maximum budget. It is a
heuristic search of the curve sum, which is distinct from the exact J_T with both partners
simplified.

---

## teacher.hpp / teacher.cpp

### Constants and types

| Name | Meaning |
|---|---|
| `CandidateIndex`, `kCandidates` | Named indices and labels for the seven candidate recipes. The index is the "method". |
| `kArms` | the eight valid arm names, listed in `teacher.hpp` |
| `TeacherConfig` | `enabled`, `cache_dir`, `arms`, `range_query_count`, `all_join_pairs`, `shape_weight`, `join` (optional, default: the pipeline's `join.mode`) |
| `Arm` | one result: `name`, `simplification` (for the pipeline), `objective` (its J_T) |
| `Candidates` | `candidates[c][i]` = trajectory i simplified with method c |
| `Methods` | one method index per trajectory, i.e. a selection |
| `kMaxResponsePasses` | at most 10 best-response passes |
| `kMaxAllocationRounds` | at most 3 iterative reallocation rounds |
| `kMinImprovement` | strict gain threshold, 1e-12 |

### `Teacher(db, workloads, config, seed)` — constructor

- **Input:** the database, the workloads (moved in), the where-at tolerances, and the seed.
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
  3. `make_candidates` at those budgets, building only RDP if no fixed-budget arm needs alternatives.
  4. **`GATEORIG_a0`** (if requested): `independent_choice` with `partners = nullptr`.
  5. **`GATE_a0` and `BR_a0`** (if either is requested):
     - score candidates directly against equal-budget RDP partners, or reuse curves if allocation
       arms also need them, to obtain `GATE_a0`;
     - `best_response` starting from `GATE_a0`, which gives `BR_a0`.
  6. Allocation arms build curves against equal-budget RDP, allocate the total budget, and select
     candidates. Response arms refine that selection; iterative rounds rebuild against the current
     partners and keep only improvements in exact J_T.
  7. `keep` assembles each arm's kept sets, computes its exact J_T, and prints how many
     trajectories each method won. `timed` prints step durations.
- **Output:** one `Arm` per requested name, in the requested order. Each keeps exactly
  `total_budget` points, because every candidate keeps exactly its trajectory's budget.

### `make_candidates(objective, budgets, caps, selected)` — internal

- **Why:** each candidate is simplified once, then reused by every step.
- **Input:** objective, per-trajectory budgets and caps, and optionally selected method indices.
- **Computation:** in parallel, `CandidateMaker` builds all seven recipes, or only the selected
  recipe for each trajectory. Classic sequences and greedy sequences with the same seed are reused.
- **Output:** `Candidates`.

### `assemble(candidates, method)` — internal

- **Input:** candidates and a selection.
- **Computation:** for each trajectory, picks the kept set of its chosen method.
- **Output:** a whole simplified database (`KeptSets`).

### `independent_choice(db, objective, candidates, partners)` — internal

- **Why:** the simplest teacher. Each trajectory picks its best method on its own, against a fixed
  partner state.
- **Input:** candidates, and the partner state for `independent_value`.
- **Computation:** in parallel, for each trajectory, the method with the highest
  `independent_value`. Ties go to the earlier method, so RDP wins when nothing differs.
- **Output:** `Methods`. With `partners = nullptr` this gives `GATEORIG_a0`. Original partners
  hide the effect of simplifying both sides of a join pair.

### `build_curves(db, objective, partners, caps, rdp_only)` — internal

- **Why:** the value curves: for each trajectory and each budget, the best method and its value.
- **Input:** the partner reference state (initially RDP at equal budgets, or the current state in
  iterative rounds), curve caps, and whether only RDP is needed.
- **Computation:** in parallel per trajectory, build the RDP addition sequence and reversed SED
  removal sequence once, then use sorted prefixes for each budget. Candidate coordinate buffers
  are reused between budgets. For b = 2 … max:
  - simplify with every candidate (at b = 2 all keep the endpoints, so only one is scored);
  - score each with `independent_value` against the fixed partner reference;
  - record the best value and method.
- **Output:** RDP-only and best-candidate curves per trajectory.
- Runs requesting only `GATE_a0` and/or `BR_a0` score their fixed budgets directly and skip full
  curve construction.
- **Why partners as RDP:** a fixed, plausible guess at how partners will look. It still scores each
  trajectory on its own, but it rewards matching how partners will likely be simplified. This took
  J5 from 0.446 (`GATEORIG_a0`) to 0.506 (`GATE_a0`) in the earlier three-candidate experiment.

### `curve_methods(curves, budgets)` — internal

- **Input:** curves and per-trajectory budgets.
- **Computation:** `method[b − 2]` for each trajectory. Throws if a budget is off its curve.
- **Output:** `Methods` at the supplied budgets.

### `best_response(db, objective, candidates, method, seed)` — internal

- **Why:** this step uses the partners' **actual** choices instead of a guess. A trajectory's best
  method depends on what its join partners chose, so the teacher revisits trajectories until no
  single change helps.
- **Input:** candidates, a starting selection (`GATE_a0` or `TEACHER`), and the seed.
- **Computation:**
  1. `order` = the trajectories with any join link. Only they can gain, since the other terms
     don't depend on anyone else.
  2. Prepares candidate views, caches their separable scores, and builds the starting `JoinState`.
  3. For up to 10 passes, in a seeded random order:
     - for each trajectory i, and each other method c: gain = `separable(c) − separable(current)`
       + `join.propose(c).gain`. That's the exact change in J_T;
     - switches to the best c only if its gain beats 1e-12 (strictly positive, so ties can't
       cycle), and calls `join.apply`.
  4. Stops after a pass with no switches.
- **Output:** the improved `Methods`. Stops after a pass with no switches or the 10-pass limit.

### `to_simplification`, `wins`, `seconds_since`, `is_arm` — helpers

- `to_simplification`: `KeptSets` → `core::Simplification`, the format the pipeline evaluates and
  exports.
- `wins`: per-candidate counts for the log.
- `seconds_since`: wall time for the log.
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

## Earlier three-candidate results (Geolife, 0.25%, seed 42)

These predate J4 and use J5, which includes clustering.

| Arm | J_T | J5 |
|---|---:|---:|
| RDP_a0 (baseline) | — | 0.492 |
| GATEORIG_a0 | 0.458 | 0.446 |
| GATE_a0 | 0.503 | 0.506 |
| BR_a0 | 0.548 | 0.544 |

Historical run time: about 7 s for `GATEORIG_a0`, 16 s for the curves and 66 s for best response.
These timings predate the seven-candidate teacher, allocation search and cache changes.

## Tests

`engine/tests/teacher/teacher_test.cpp` uses a small synthetic database:

- `CacheIsWrittenThenReadBack`: a cache round trip gives identical workloads.
- `CacheForOtherSettingsIsRebuilt`: a cache built with a different `k` isn't reused.
- `ObjectiveIsZeroAtTheFloorAndTracksJoinExactly`: J_T = 0 at endpoints only. After random
  switches, `JoinState`'s tracked F1 equals a full recompute, and `delta + separable` predicts the
  real change in J_T.
- `ArmsKeepTheBudgetAndBestResponseOnlyImproves`: every arm keeps exactly the budget, endpoints
  kept, indices sorted; and `BR_a0`'s J_T ≥ `GATE_a0`'s.

`engine/tests/config_test.cpp` has the three teacher config tests.
