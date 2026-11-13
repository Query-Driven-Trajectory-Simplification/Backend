# Teacher: next steps

Written 2026-11-13 on branch `teacher`. A plan for whoever picks this up next: what was changed,
what the first numbers show, and what to change and run next, in order.

## Where things stand

Two changes are in the working tree. **They are not committed yet**, so make sure you have them
before starting (`git status` should show about 22 modified files).

1. **J4 is fully held out.** With `join.mode: "HeldOut"` (now the Geolife config) the join term
   of J4 compares original trajectories from the query pool with the simplified database. The
   database self join, whose pairs the teacher trains on, is scored separately
   (`self_join_f1`, `norm_self_join`) and is not part of J4.
2. **The teacher's join term follows `join.mode`.** `ObjectiveConfig::join_mode` is new. In
   `HeldOut` mode every database trajectory also acts as an original query against its compressed
   partners. `teacher.join` in the config, or `--teacher-join Self|HeldOut` on `teacher_eval`,
   overrides it.

All 227 tests pass.

### Build, test, run

```bash
cmake --build build/clang-release
ctest --test-dir build/clang-release

./build/clang-release/engine/teacher_eval --config engine/configs/geolife.json \
    --seeds 42 --shares 0.0025 --arms GATEORIG_a0,GATE_a0,BR_a0 --out results/<new_dir>
python3 engine/apps/teacher_stats.py results/<new_dir>
```

- Always use a **new `--out` directory**. The CSVs are opened in append mode and the columns
  changed, so old and new rows must not share a file.
- One seed, one share, three arms takes roughly 10 to 15 minutes on 20 cores.
- A run may rewrite the tracked file `cache/teacher_cache_seed42.bin` if the committed cache does
  not match the current settings. Do not commit that by accident.

## What the first run shows

Seed 42, budget 0.25 %, teacher trained on the **self join** (`results/heldout_smoke`). All
values are normalised: 0 = endpoints only, 1 = original.

| Arm | J4 | kNN | range | where-at | held-out join | self join 150 m | self join 100 m | self join 300 m | clustering |
|---|---|---|---|---|---|---|---|---|---|
| `GATEORIG_a0` | 0.405 | 0.452 | 0.442 | 0.357 | 0.368 | 0.215 | 0.154 | 0.218 | 0.210 |
| `GATE_a0` | 0.393 | 0.456 | 0.414 | 0.346 | 0.355 | 0.466 | 0.328 | 0.363 | 0.424 |
| `BR_a0` | 0.396 | 0.465 | 0.412 | 0.338 | 0.366 | 0.726 | 0.446 | 0.414 | 0.616 |
| `RDP_a0` | 0.362 | 0.518 | 0.294 | 0.253 | 0.385 | 0.396 | 0.327 | 0.500 | 0.405 |
| `PARTNER_l0.5` | 0.371 | 0.535 | 0.290 | 0.241 | 0.416 | 0.478 | 0.447 | 0.492 | 0.493 |

What to take from it (one seed, one budget, so a direction only):

- **Coordination does not transfer to held-out queries.** Best response lifts the self join from
  0.47 to 0.73 but the held-out join stays at about 0.36. This is structural: a held-out query
  is an original trajectory, so there is no second compressed side to coordinate with.
- **The self-join gain is fitted to the trained threshold.** At 100 m `BR_a0` only ties the
  trivial `PARTNER` baseline, and at 300 m it is below plain RDP.
- **The teacher loses both shape-based external queries.** It is below RDP on kNN and on the
  held-out join. Its J4 lead comes entirely from range and where-at.
- **The best teacher arm on J4 is `GATEORIG_a0`**, the one that ignores compressed partners.

## What to do, in order

### Step 0. Teacher trained on the held-out join term

Done for seed 42 at 0.25 % (`results/heldout_teacher_smoke`, teacher arms only, compare with the
baselines in the table above):

| Arm | J4 | kNN | range | where-at | held-out join | self join 150 m | teacher's own join estimate |
|---|---|---|---|---|---|---|---|
| `GATEORIG_a0` = `GATE_a0` | 0.405 | 0.451 | 0.443 | 0.358 | 0.367 | 0.214 | 0.400 |
| `BR_a0` | 0.410 | 0.461 | 0.444 | 0.352 | 0.383 | 0.274 | 0.428 |

- **Best response now helps the held-out join, a little:** 0.367 to 0.383, and the teacher's own
  estimate moves the same way (0.400 to 0.428), so the new term tracks the test.
- **It only reaches RDP's level** (0.385) and stays below `PARTNER` (0.416).
- **J4 0.410 is the best so far**, but the margin over the self-join-trained arms (0.396 to
  0.405) is too small to call on one seed.
- **The self join is no longer optimised** (0.27 vs 0.73), as expected.

So training on the right term stops the join from being a loss against RDP, but does not make it
a win. Steps 1 and 3 are where a win would have to come from. To repeat on other seeds:

```bash
./build/clang-release/engine/teacher_eval --config engine/configs/geolife.json \
    --seeds 43,44 --shares 0.0025 --arms GATEORIG_a0,BR_a0 --out results/heldout_teacher
```

### Step 1. Sweep the shape term

The teacher gives up shape for range and where-at, which costs it kNN and the join. The objective
already has a shape term that is switched off.

```bash
for w in 0.1 0.25 0.5; do
  ./build/clang-release/engine/teacher_eval --config engine/configs/geolife.json \
      --seeds 42 --shares 0.0025 --arms GATEORIG_a0,BR_a0 --no-baselines \
      --shape-weight $w --out results/shape_$w
done
```

Look for a weight where kNN and the held-out join reach RDP's level without losing the range and
where-at lead. No code change needed.

### Step 2. Train the join on several thresholds

Today the join term uses one threshold (`join.eps_m`, 150 m) and the gain does not carry to 100 m
or 300 m.

- **Change:** let the teacher's join term be the mean over several thresholds, e.g. 100, 150 and
  300 m. The pairs and their truth depend on the threshold, so `TeacherWorkloads`
  (`engine/include/backend/engine/teacher/workloads.hpp`) needs positives and negatives per
  threshold, and `Objective` / `JoinState` (`engine/src/teacher/objective.cpp`) one set of pair
  lists, links and counts per threshold. The workload cache format changes, so bump its version.
- **Test on thresholds that were not trained on**, e.g. 125 and 200 m. `teacher_eval` currently
  hardcodes 100 and 300 m for `thresholds.csv`; make that list configurable.
- **Done when:** the self-join gain of `BR_a0` over `PARTNER` holds at the untrained thresholds.
  If it does not, the self-join result is threshold fitting and should be reported as such.

### Step 3. Make the kNN term coupled (the main bet)

This is the one change that could give a relational result inside the held-out headline.

- **Why:** the kNN proxy keeps competitors original (see the comment on `Objective` in
  `objective.hpp`), so it is separable. In the real test the whole database is compressed:
  whether trajectory i stays in a query's top k depends on how its neighbours' distances shift.
  That coupling exists for held-out queries too.
- **Change:** score each kNN query against the current compressed state. For query e, i is a hit
  if it is in e's true top k and still ranks in the top k among e's compressed candidates. This
  needs a tracker like `JoinState`: per query, the current distances to a candidate list (true
  top k plus the nearest non-members), updated when one trajectory changes.
- **Cost:** every proposal in best response recomputes Fréchet from each query that has i as a
  candidate. Keep the candidate list short (e.g. top 3k) to bound it.
- **Done when:** the tracked value equals a full recompute after random changes (copy the
  pattern of `ObjectiveIsZeroAtTheFloorAndTracksJoinExactly` in
  `engine/tests/teacher/teacher_test.cpp`), and `BR_a0` beats `GATE_a0` on held-out kNN recall.

### Step 4. Only then: more seeds and budgets

Once steps 0 to 3 settle which configuration to report, run it on at least 5 seeds and the full
budget list, with every arm and the baselines. Pick the configuration on one seed and report on
the others, so the reported numbers are not the ones that were tuned on.

## How to frame the paper if nothing else changes

- **External queries** (kNN, range, where-at, held-out join): the gain comes from being
  workload-aware, and so far only on range and where-at.
- **Internal queries** (self join, clustering): coordination gives a large gain at the trained
  threshold. Whether it is more than threshold fitting depends on step 2.

Step 3 is what would let the paper claim a relational effect on held-out queries. Without it,
the honest claim is the two-class split above.
