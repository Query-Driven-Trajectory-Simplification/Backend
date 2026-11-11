# The teacher

The teacher is a slow, query-aware simplifier. Each trajectory keeps a fixed number of points, and
the teacher chooses which simplifier picks them: RDP, SED or UNIFORM. Its output is meant to be a
target that a learned model can later imitate.

It sees **only the database** (the 85% split) and workloads built from it with their own seeds. It
never sees the held-out query pool that the pipeline scores on. Allocation (how many points each
trajectory gets) is not part of it yet: every trajectory gets its equal share of the budget
(α = 0).

## How it fits together

```
pipeline::run                                    (pipeline.cpp)
 ├─ load_or_build_workloads(database, ...)        workloads.cpp: kNN, pairs, boxes, probes; cache
 ├─ Teacher(database, workloads, ...)             teacher.cpp: builds the Objective once
 │    └─ Objective(db, workloads)                 objective.cpp: inverts workloads, computes floors
 └─ per budget share:
      Teacher::run(budget, arms)
       ├─ make_candidates                         RDP, SED and UNIFORM for every trajectory
       ├─ GATEORIG_a0: independent_choice         best method, partners original
       ├─ GATE_a0:     build_curves → curve_methods   best method, partners as RDP
       └─ BR_a0:       best_response(from GATE_a0) re-pick against partners' current choice
      → each arm evaluated and exported like RDP and SED
```

Files:

| File | Contents |
|---|---|
| `workloads.hpp/.cpp` | The teacher's training workloads, and the cache that saves the slow part |
| `objective.hpp/.cpp` | J_T, the score the teacher maximises, and `JoinState` for fast exact join updates |
| `curves.hpp` | The `Curve` struct: what a trajectory is worth at each budget |
| `teacher.hpp/.cpp` | The `Teacher` class and the selection steps |

## Words used throughout

- **Kept set:** the sorted point indices one trajectory keeps, endpoints included. `KeptSets` is one
  kept set per trajectory, i.e. a whole simplified database.
- **Budget:** the number of points a trajectory keeps, endpoints included. A budget of 2 means
  endpoints only.
- **Floor:** a metric's value when every trajectory keeps endpoints only.
- **Normalised metric:** `(metric − floor) / (1 − floor)`. 0 means no better than endpoints only;
  1 means as good as the original.
- **J5:** the pipeline's score on the held-out test workloads: the equal mean of five normalised
  metrics (kNN, range, where-at, join, clustering).
- **J_T:** the teacher's own score on its training workloads. It has the same shape as J5 but
  leaves clustering out and only uses the seen join pairs.
- **Seen / unseen pairs:** the self-join pairs are split in half. The teacher only gets the seen
  half, so the unseen half shows how much of its join gain carries over to pairs it never saw.
- **Arm:** one named output of the teacher, e.g. `GATE_a0`.

---

## How selection works, in plain terms

### What is being chosen
- Every trajectory keeps a **fixed number of points**: its equal share of the budget (α = 0).
  At 0.25% that is about 4–5 points. The teacher never changes this number.
- The teacher chooses **one method per trajectory**: RDP, SED or UNIFORM. All of the trajectory's
  kept points come from that one method. Methods are never mixed within a trajectory.
- So each trajectory has exactly **three options**, each a whole kept set with one score. For a
  trajectory with budget 4:

  ```
  RDP     with 4 points → e.g. points {0, 37, 112, 250}
  SED     with 4 points → e.g. points {0, 61, 140, 250}
  UNIFORM with 4 points → e.g. points {0, 83, 166, 250}
  ```

  These are simplified once at the start (`make_candidates`) and then only looked up.

### Why a trajectory's choice depends on other trajectories
kNN, range and where-at only involve trajectory i itself, because everyone else is compared in
original form. The join is different: a pair (i, j) is scored with **both** sides simplified. Two
trajectories simplified with the same method follow the same shape and match far more often than
two simplified differently. So the best method for i depends on what its join partners chose.

### Two stages
Selection happens in two stages, which differ in how the join partners are seen:

```
Stage 1, GATE_a0:  one round, every trajectory on its own, partners ASSUMED to be RDP
Stage 2, BR_a0:    starts from stage 1, partners are their ACTUAL current choices
                   pass 1, pass 2, ... until a pass makes no switches
```

| | Stage 1 (GATE_a0) | Stage 2 (BR) |
|---|---|---|
| Points per trajectory | fixed budget | fixed budget |
| Methods tried for i | all 3 | the other 2 (than i's current one) |
| Terms scored | all 4 (kNN, range, where-at, join) | all 4, as the change from i's current method |
| i's join partners seen as | RDP (a guess) | their actual current methods |
| Join change counted for | i only | i and every partner whose pair flips |
| Order | all trajectories at once, independently | one at a time; each switch is visible to the next |
| Runs | once | repeated passes until nothing changes |

Stage 1 is not needed for correctness: BR could start from all-RDP and still climb. It gives BR a
good starting point, and it is a result of its own (the independent, one-pass teacher), which
shows how much stage 2 adds.

### Stage 1: GATE_a0 (`build_curves`, `curve_methods`)
For each trajectory i, independently:

1. For each budget b from 2 up to i's budget, try all three methods with b points, and score each
   on **all four terms** with `independent_value`. In the join term, i's partners are assumed to
   use RDP. Record the best value and method at each b: that is i's value curve.
2. Read the curve **at i's budget** and take that method.

"RDP" is only the guess about the partners. i itself can come out as RDP, SED or UNIFORM.

The curve values at smaller b are not used yet. They are there for an allocation step (deciding
whether i should get 3 or 5 points instead of 4), which can be added back later. With fixed
budgets and independent scores, picking each trajectory's best method also gives the highest
possible total, so nothing more is needed in stage 1.

### Stage 2: BR (`best_response`)
**In one sentence:** go through the trajectories one at a time, and let each switch to whichever
of its other methods makes the *real* J_T highest, given what everyone else currently uses;
repeat until nobody wants to switch.

1. **Who is visited:** only trajectories with at least one seen join pair. For the others, the
   score doesn't depend on anyone else, so their stage-1 choice is already their best.
2. **Starting state:** stage 1's method for every trajectory, plus a `JoinState` holding every
   seen pair's outcome (match or not) with both sides on their stage-1 methods.
3. **One pass:** visit those trajectories in random order. For trajectory i and each of its other
   two methods c:

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
Trajectories 1–4; (1, 4) is a true join pair, and 1 is also paired with 2.

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
practice (seed 42, 0.25%) it takes 8 passes: 777 switches in the first pass, 91 in the second,
then a handful.

It ends at a state where **no single trajectory can raise J_T by switching on its own**. That is a
local optimum, not necessarily the global one: switching two trajectories at the same time might
still help. The random visiting order changes slightly which local optimum it lands in. In
optimisation terms this is coordinate ascent; in game-theory terms, best-response dynamics.

### Why it is fast
`JoinState` keeps every pair's outcome and every trajectory's hit and found counts. Testing one
switch only rescores that trajectory's own pairs (usually a handful) instead of all ~90,000 seen
pairs, and the result is exact: the tests check it against a full recompute.

### What it shows
On seed 42 at 0.25%: GATE_a0 0.506 → BR_a0 0.544 J5, with the same three methods and the same
budgets. The gain is in join (0.522 → 0.621) and clustering. The only difference is using the
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
  2. Calls `load`. If the file matches the fingerprint, seed, `k`, `join_eps_m` and `frechet`, it
     uses the cached kNN entries and pairs.
     - An older cache file (format 2, from the `init-t1` harness) has no negatives. They are
       recomputed with `build_pairs` (about 10 s). The file isn't rewritten, so the old harness can
       still read it.
  3. Otherwise, it builds everything with `build_knn` and `build_pairs`, then `save`s it.
  4. Always regenerates the range boxes and where-at probes. They are fast, and use their own seeds
     (`seed + 7001` and `seed + 7002`).
- **Output:** a `TeacherWorkloads`. It prints whether the cache was hit, and the counts.

### `fingerprint(db)` — internal

- **Why:** a cache is only valid for the exact same database. A different split or data file must
  not reuse it.
- **Input:** the database.
- **Computation:** an FNV-1a hash over the trajectory count, the point count, and each
  trajectory's ID, length, and first and last timestamps.
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
- **Input:** the database, the settings, the seed, and output vectors. Passing `positives = nullptr`
  builds negatives only, for old cache files.
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
- **Format 3** (written): a header (magic number, seed, fingerprint, k, eps, variant), then the kNN
  entries, positives and negatives.
- **Format 2** (read only): no settings in the header (implicitly k = 10, 150 m, continuous) and
  no negatives.
- `save` writes to a `.tmp` file and renames it, so a crash never leaves a half-written cache.
- `load` returns false on any mismatch, which triggers a rebuild.

---

## objective.hpp / objective.cpp

### Why

This is the score the teacher maximises. It has the same shape as J5, so that improving it
improves J5. It is computed only on the teacher's training workloads. Over the arms tested so far,
J_T correlates with J5 at r ≈ 0.95.

The four terms, each normalised `(metric − floor) / (1 − floor)` and then averaged:

| Term | Per trajectory i, simplified | Mirrors |
|---|---|---|
| kNN | for each query e that has i in its top k: Fréchet(original e, simplified i) ≤ cutoff(e) | recall@k |
| Range | for each box i is in: does a kept point still lie inside the box? | range F1, as recall (there are no false positives) |
| Where-at | average-hit ramp over i's probes: `clamp((max − err) / (max − min), 0, 1)` | where-at average-hit |
| Join | macro F1 over the seen pairs, both sides simplified | join F1, computed the same way |

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
- **Output:** `view()`, valid while the object lives. Bounds are rebuilt from retained points.
  `PreparedTrajectories` owns a whole set and provides borrowed views. Fixed partners are prepared
  once for curve scoring; best response prepares every candidate once and switches view references.

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
  4. **Builds the join links.** Seen positives first, then seen negatives, stored in
     `m_pair_i`/`m_pair_j`. Each pair is linked from both sides in `m_links`. `m_truth[t]` counts
     t's seen positives.
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

### `match(a, b)` — public

- **Input:** two prepared trajectory views (or an original view).
- **Computation:** `query::frechet_within` on the simplified views, with the workload's threshold
  and Fréchet variant.
- **Output:** whether the pair would be reported by the join.

### `join_metric(state)` — private

- **Input:** a whole simplified database.
- **Computation:**
  1. In parallel, `match` every seen pair.
  2. For each matched pair, increment `found` on both sides, and `hits` too if it's a positive.
  3. Average `f1_of(hits, found, truth)` over trajectories with any truth or any match. This is the
     same counting rule as the engine's evaluator.
- **Output:** macro join F1 on the seen pairs.

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
- **Computation:** matches every seen pair once (from its smaller side, in parallel). It fills
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

## curves.hpp

### `Curve`

- **Why:** a value curve records what one trajectory is worth at each budget. The teacher
  currently reads it only at the trajectory's fixed budget, i.e. for selection. The struct is kept
  separate so that an allocation search (moving points to the trajectories whose curves rise
  fastest) can be added back on top later.
- **Fields:** `value[k]` and `method[k]` for budget `b = 2 + k`: the best candidate's
  `independent_value` and which candidate reached it.
- **Helpers:** `max_budget()` is the largest budget on the curve, and `at(b)` is the value at b.
- **Note:** Fréchet thresholds flip on and off, so a curve is a step function and need not rise at
  every step.

---

## teacher.hpp / teacher.cpp

### Constants and types

| Name | Meaning |
|---|---|
| `kCandidates` | RDP, SED, UNIFORM, the methods the teacher chooses from. The index into this array is the "method". |
| `kArms` | the valid arm names: `GATEORIG_a0`, `GATE_a0`, `BR_a0` |
| `TeacherConfig` | the config section: `enabled`, `cache_dir`, `arms`, `range_query_count` |
| `Arm` | one result: `name`, `simplification` (for the pipeline), `objective` (its J_T) |
| `Candidates` | `candidates[c][i]` = trajectory i simplified with method c |
| `Methods` | one method index per trajectory, i.e. a selection |
| `kMaxPasses` | at most 10 best-response passes |

### `Teacher(db, workloads, config, seed)` — constructor

- **Input:** the database, the workloads (moved in), the where-at tolerances, and the seed.
- **Computation:** stores the workloads, then builds the `Objective` on them. `m_workloads` is
  declared before `m_objective`, so it is constructed first. That ordering matters because the
  objective keeps a reference to it. Prints the seen-pair counts and floors.
- **Output:** a ready teacher. It can't be copied, since that would break the reference.

### `Teacher::run(total_budget, arms)` — public, the main entry

- **Input:** the total point budget for this share, and the arm names to produce.
- **Computation:**
  1. Rejects unknown arm names.
  2. `equal` = equal budgets per trajectory (the engine's size^α allocator with α = 0).
  3. `make_candidates` at those budgets.
  4. **`GATEORIG_a0`** (if requested): `independent_choice` with `partners = nullptr`.
  5. **`GATE_a0` and `BR_a0`** (if either is requested):
     - `build_curves` with partners fixed at the RDP candidates, up to each trajectory's budget;
     - `curve_methods` reads each curve at the budget, which gives `GATE_a0`;
     - `best_response` starting from `GATE_a0`, which gives `BR_a0`.
  6. `keep` assembles each arm's kept sets, computes its exact J_T, and prints how many
     trajectories each method won. `timed` prints step durations.
- **Output:** one `Arm` per requested name, in the requested order. Each keeps exactly
  `total_budget` points, because every candidate keeps exactly its trajectory's budget.

### `make_candidates(db, budgets)` — internal

- **Why:** each candidate is simplified once, then reused by every step.
- **Input:** database and per-trajectory budgets.
- **Computation:** in parallel, `selection::simplify_trajectory` with each of the three methods.
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
- **Output:** `Methods`. With `partners = nullptr` this gives `GATEORIG_a0`. It loses on join,
  because judging a match against an *original* partner hides the fact that partners get
  simplified too, and that pairs simplified with different methods match less often.

### `build_curves(db, objective, partners, max_budget)` — internal

- **Why:** the value curves: for each trajectory and each budget, the best method and its value.
- **Input:** the partner reference state (RDP at equal budgets), and how far each curve goes (the
  trajectory's budget, never more than its length).
- **Computation:** in parallel per trajectory, build the RDP addition sequence and reversed SED
  removal sequence once, then use sorted prefixes for each budget. Single-budget requests keep
  using the ordinary selectors. For b = 2 … max:
  - simplify with every candidate (at b = 2 all keep the endpoints, so only one is scored);
  - score each with `independent_value` against the RDP partners;
  - record the best value and method.
- **Output:** one `Curve` per trajectory.
- **Why partners as RDP:** a fixed, plausible guess at how partners will look. It still scores each
  trajectory on its own, but it rewards matching how partners will likely be simplified. This took
  J5 from 0.446 (`GATEORIG_a0`) to 0.506 (`GATE_a0`).

### `curve_methods(curves, budgets)` — internal

- **Input:** curves and per-trajectory budgets.
- **Computation:** `method[b − 2]` for each trajectory. Throws if a budget is off its curve.
- **Output:** `Methods`, the `GATE_a0` selection.

### `best_response(db, objective, candidates, method, seed)` — internal

- **Why:** this step uses the partners' **actual** choices instead of a guess. A trajectory's best
  method depends on what its join partners chose, so the teacher revisits trajectories until no
  single change helps.
- **Input:** candidates, a starting selection (`GATE_a0`), and the seed.
- **Computation:**
  1. `order` = the trajectories with any seen join link. Only they can gain, since the other terms
     don't depend on anyone else.
  2. Prepares candidate views, caches their separable scores, and builds the starting `JoinState`.
  3. For up to 10 passes, in a seeded random order:
     - for each trajectory i, and each other method c: gain = `separable(c) − separable(current)`
       + `join.propose(c).gain`. That's the exact change in J_T;
     - switches to the best c only if its gain beats 1e-12 (strictly positive, so ties can't
       cycle), and calls `join.apply`.
  4. Stops after a pass with no switches.
- **Output:** the improved `Methods`, which gives `BR_a0`. Every switch raises J_T, so it always
  stops; in practice that takes 5–8 passes.

### `to_simplification`, `wins`, `wants`, `seconds_since`, `is_arm` — helpers

- `to_simplification`: `KeptSets` → `core::Simplification`, the format the pipeline evaluates and
  exports.
- `wins`: a "RDP n, SED n, UNIFORM n" summary for the log.
- `wants`: whether any of the given arm names was requested.
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

## Results so far (Geolife, 0.25%, seed 42)

| Arm | J_T | J5 |
|---|---:|---:|
| RDP_a0 (baseline) | — | 0.492 |
| GATEORIG_a0 | 0.458 | 0.446 |
| GATE_a0 | 0.503 | 0.506 |
| BR_a0 | 0.548 | 0.544 |

Teacher run time: about 7 s for `GATEORIG_a0`, 16 s for the curves and 66 s for best response,
plus about 10 s to recompute the negatives when loading an old cache.

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
