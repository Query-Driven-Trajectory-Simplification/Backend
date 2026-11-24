# Teacher: code walkthrough

How the teacher runs end to end, at the level of files, functions and data structures, followed
by what is currently wrong in it. Line numbers refer to `loop-trade` at `135c2a5`, which adds
budget trading, the 1 mm kNN ranking grid and the cache format V6 on top of `main` (`21f2368`).
On `main`, the trading arm, `KnnState::watch` and `propose_bounded` do not exist yet.

A visual version of the same walkthrough, without the code detail, is the "Teacher Pipeline"
artifact. The status of every incorrect item and missing step is on the "Teacher Fix List"
artifact.

## Files

| File | Contents |
|---|---|
| [workloads.hpp](engine/include/backend/engine/teacher/workloads.hpp), [workloads.cpp](engine/src/teacher/workloads.cpp) | Training workloads built from the database, and the on-disk cache |
| [objective.hpp](engine/include/backend/engine/teacher/objective.hpp), [objective.cpp](engine/src/teacher/objective.cpp) | `Objective` (J_T) and the trackers `RangeState`, `JoinState`, `KnnState` |
| [curves.hpp](engine/include/backend/engine/teacher/curves.hpp), [curves.cpp](engine/src/teacher/curves.cpp) | `Curve` and the budget allocator `allocate` |
| [greedy.hpp](engine/include/backend/engine/teacher/greedy.hpp), [greedy.cpp](engine/src/teacher/greedy.cpp) | `Greedy`, the query-aware point selector used by three of the six candidates |
| [teacher.hpp](engine/include/backend/engine/teacher/teacher.hpp), [teacher.cpp](engine/src/teacher/teacher.cpp) | `Teacher`, candidates, value curves, the loop (`Search`), budget trading, the arms |
| [trade_matching.hpp](engine/include/backend/engine/teacher/trade_matching.hpp) | `find_profitable_seller`, the buyer-seller matching used by trading |
| [query/knn.hpp](engine/include/backend/engine/query/knn.hpp), [query/knn.cpp](engine/src/query/knn.cpp) | `query::knn`, the 1 mm ranking grid and the shared tie rule `knn_rank_less` |
| [teacher_eval.cpp](engine/apps/teacher_eval.cpp) | The experiment runner: split, teacher, baselines, held-out scoring, CSVs |
| [teacher_stats.py](engine/apps/teacher_stats.py) | Summarises a `teacher_eval` output directory |
| [pipeline/config.cpp](engine/src/pipeline/config.cpp) | Turns the JSON config into `WorkloadConfig` and `ObjectiveConfig` |

## Call flow

```
teacher_eval main                                      teacher_eval.cpp
 per seed:
 ├─ split::split(all, config.split, seed)               database 85 %, query pool 15 %
 ├─ load_or_build_workloads(db, ...)                    workloads.cpp:173
 ├─ Teacher(db, workloads, objective_config, seed)      teacher.cpp, builds Objective once
 ├─ per share: Teacher::run(total_budget, arms)         teacher.cpp:774
 │   ├─ make_candidates(equal budgets)                  teacher.cpp:147
 │   ├─ EQUAL_CHOICE        independent_choice          teacher.cpp:210
 │   ├─ build_curves(partners = equal-budget RDP)       teacher.cpp:238
 │   ├─ EQUAL_CHOICE_RDP_PARTNERS, EQUAL_LOOP
 │   ├─ allocate(curves.rdp)  -> BUDGET_RDP             curves.cpp:43
 │   ├─ allocate(curves.best) -> BUDGET_CHOICE
 │   ├─ Search(...).run()     -> BUDGET_LOOP            teacher.cpp:294
 │   ├─ trade_budgets(BUDGET_LOOP) -> BUDGET_LOOP_TRADE teacher.cpp:457
 │   ├─ reallocation rounds   -> BUDGET_LOOP_REALLOCATE
 │   ├─ Search.run(update = false) -> BUDGET_LOOP_NO_UPDATE
 │   └─ shuffled_partners + best_response -> BUDGET_LOOP_SHUFFLED   teacher.cpp:672
 ├─ baseline_arms(db, pairs, shares)                    RDP/SED/UNI, RDP_GLOBAL, PARTNER
 ├─ Teacher::single_recipes                             RECIPE_*, teacher.cpp:937
 ├─ validate(arm, db) for every arm
 │  ---------- held-out boundary: nothing above touches the query pool ----------
 ├─ TestWorkloads(db, pool, config, self_knn)           held-out kNN, range, where-at, joins
 ├─ joins at --eval-eps (125, 175, 250 m), self and held-out
 └─ per arm: Evaluator(...).evaluate(), normalize_scores, objective.metrics -> CSVs
```

## 1. Training workloads

`load_or_build_workloads(db, config, seed, cache_dir)` in [workloads.cpp:173](engine/src/teacher/workloads.cpp#L173).

| Struct | Fields | Meaning |
|---|---|---|
| `WorkloadConfig` | `k`, `join_eps_m`, `extra_join_eps_m`, `frechet`, `range`, `where_at` | Built by `pipeline::teacher_workload_config` |
| `KnnEntry` | `valid`, `top`, `cutoff` | Trajectory e as a query: its true top k and the distance to its (k+1)-th neighbour |
| `JoinPair` | `i < j` | A pair of database trajectories |
| `ExtraJoin` | `eps_m`, `positives`, `negatives` | Pairs at one extra threshold |
| `TeacherWorkloads` | `knn`, `positives`, `negatives`, `extra_joins`, `range`, `where_at`, settings | Everything `Objective` needs |

- **kNN** (`build_knn`, line 62): for each e in parallel, `query::knn(db, e, k + 2)`, drop e, keep
  k as `top`, compute `cutoff` with `query::frechet_distance`. The slow part: about 140 s on a
  15 % sample, 25–45 min on full Geolife.
- **Join pairs** (`collect_pairs`, line 83): `query::generate_self_join(db, eps)`. A candidate in
  `reference` is a positive; any other candidate (endpoints within eps, originals not) is a
  negative. Done at `join_eps_m` (cached) and at each extra threshold (rebuilt every load).
- **Extra thresholds:** `teacher.extra_join_eps_m`, by default `join.score_eps_m`
  (`pipeline::teacher_extra_join_eps_m`).
- **Range and where-at:** always regenerated, seeds `seed + kRangeSeedOffset` (7001) and
  `seed + kWhereAtSeedOffset` (7002). Range uses `teacher.range_query_count`, 128,000 by default.
- **Continuous Fréchet ranking:** `query::knn` ranks continuous distances on a fixed 1 mm grid
  (`kKnnResolutionM`): `continuous_rank_distance` returns the first grid radius the decision
  predicate accepts, so the same pair always gets the same key. Equal keys go to the lower
  database index (`knn_rank_less`).
- **Cache:** `cache/teacher_cache_seed<seed>.bin`, format `kMagicV6` (V6 since the grid ranking;
  older caches are rebuilt). Invalidated by the seed,
  `k`, `join_eps_m`, the Fréchet variant, or the database fingerprint (FNV-1a over every point's
  time, lat/lon and projected x/y). `save` writes a `.tmp` file and renames it.

## 2. The objective, J_T

`Objective` in [objective.hpp:82](engine/include/backend/engine/teacher/objective.hpp#L82).
Built once in the `Teacher` constructor.

### Configuration

`ObjectiveConfig` ([objective.hpp:49](engine/include/backend/engine/teacher/objective.hpp#L49)), built
by `pipeline::teacher_objective_config`:

| Field | Config key | Default (geolife.json) |
|---|---|---|
| `min_tolerance_m`, `max_tolerance_m` | `where_at_scoring` | 25, 500 m |
| `join_mode` | `teacher.join`, else `join.mode` | `HeldOut` |
| `held_out_term` | `teacher.held_out_term` | false |
| `ranking_knn` | `teacher.ranking_knn` | true |
| `compressed_knn_query` | `teacher.knn_query` | compressed only with a `Self` join |
| `term_weights` | `teacher.term_weights` | 1, 1, 1, 1 |

### Terms

Each term is normalised `(x − floor) / (1 − floor)`, with `floor = metrics(endpoints_only(db))`.
`m_scale[t] = weight[t] / (weight sum · (1 − floor[t]))`, and
`total(kept) = Σ m_scale[t] · (metrics(kept)[t] − floor[t])`
([objective.cpp:487](engine/src/teacher/objective.cpp#L487)).

| Term | Exact metric (`metrics`, line 448) | Tracker | Used in curves and independent choice |
|---|---|---|---|
| kNN | `knn_recall`: recall@k of every valid database query, ranked with `query::knn` on the compressed database (`knn_recall`, line 468) | `KnnState` | per-trajectory check: `frechet_within(original e, candidate, cutoff(e))`, competitors original |
| Range | `RangeState::metric`: mean over scored boxes of `2h / (n + h)` | `RangeState` | `range_weights`: marginal F1 of i given the partners' coverage, `range_marginal(others, n, scored)` (line 67) |
| Where-at | `raw_separable`: per trajectory, mean of `clamp((500 − err) / 475, 0, 1)` over its probes, averaged over trajectories with probes | none needed | the same, exact |
| Join | `join_metric` → `join_f1s` → `join_value`: macro F1 per slot and term, each term normalised by its own endpoint floor, averaged over terms with true pairs (line 397) | `JoinState` | i's own F1 per term against fixed partners (`prepare_independent`) |

### Join slots

A slot is `slot(t, term) = t + term · n`: trajectory t's F1 at one join term. `m_truth[slot]` is
its number of positives. A slot counts when `found + truth > 0`.

- **Self** terms: a pair (i, j) is one outcome, counted for both i and j.
- **HeldOut** terms: each direction is its own outcome. Original i queries compressed j, and the
  result counts for i's slot. `JoinState::slot(i, link)` stores each direction separately.
- Terms: `join_eps_m`, then each extra threshold, then the held-out form of `join_eps_m` if
  `held_out_term` is set. At most `kMaxJoinTerms` (8).

### Trackers

All three follow the same pattern: `propose(i, candidate)` returns the change in J_T if
trajectory i switched to `candidate`, given everyone's current state, without changing anything;
`apply(proposal)` commits it.

- **`RangeState`** ([objective.hpp:207](engine/include/backend/engine/teacher/objective.hpp#L207)):
  per box, the count of trajectories with a kept point inside; per trajectory, which of its boxes
  it covers. A proposal rescores only i's boxes. `weights(i)` gives the marginal range weights
  that `build_curves` and `independent_choice` use.
- **`JoinState`** ([objective.hpp:233](engine/include/backend/engine/teacher/objective.hpp#L233)):
  per pair and direction, matched or not; per slot, hits and found; per term, the F1 sum and count.
  A proposal rescores only i's pairs and updates i's slots and its partners'.
- **`KnnState`** ([objective.hpp:274](engine/include/backend/engine/teacher/objective.hpp#L274)):
  each valid query e keeps a list: its true top k first, then the nearest `watch` others (3k) in
  the starting state and in every all-one-recipe state. `m_distance[e]` holds the current distance
  to each list member, and `hits(e, …)` ranks them with `ranked_hits`, using the same
  `query::knn_rank_less` as `query::knn`, to count how many true neighbours are in the top k. A
  trajectory outside every list it could enter is assumed never to enter it.
  - `watch(versions, state)` (line 855) adds, per query, the nearest others among extra versions
    (the states trading will offer), so the lists can see them. It can change the tracked metric
    of the current state; trading adds that change to its running total.
  - `propose_bounded(i, candidate)` (line 907) estimates i's kNN change without exact distances
    where i is clearly inside or outside the k-th place, by more than the grid error.

### Independent scoring

`prepare_independent` and `independent_value` (lines 291–395) score one trajectory alone:
separable terms, range with the given weights, join against `partners` (empty = original), and
the per-trajectory kNN check with early exit once even all remaining hits could not beat `beat`.
`pick` uses this to rank candidates, scoring RDP fully and the rest best bound first.

## 3. Candidates

`CandidateMaker` ([teacher.cpp:69](engine/src/teacher/teacher.cpp#L69)) makes candidate c of
trajectory i at budget b. `kCandidates` (line 41):

| c | Name | How |
|---|---|---|
| 0 | `RDP` | prefix of `selection::rdp::sequence(original, cap)` |
| 1 | `SED` | prefix of `selection::sed::sequence(original, cap)` |
| 2 | `UNIFORM` | `selection::simplify_trajectory(..., UNIFORM, b)` |
| 3 | `GREEDY` | `Greedy::sequence(cap, no seed, fill = SED order)`, prefix b |
| 4 | `GREEDY_SED2` | the same, seeded with half the interior budget from SED |
| 5 | `GREEDY_RDP2` | the same, seeded with half the interior budget from RDP |

- Sequences are computed once per trajectory up to the cap, so every budget is a prefix.
- `Greedy` ([greedy.cpp](engine/src/teacher/greedy.cpp)) keeps, one at a time, the point with the
  largest gain in range (each uncovered box's weight, with other trajectories original) plus
  where-at (the change in the probes' ramp scores on the segment it splits). Below `kMinGain`
  (1e-15) it fills from the SED order.
- `first_equal` (line 60) maps duplicate kept sets to the first candidate that produced them, so
  each distinct set is scored once.
- `make_candidates(objective, budgets, caps, selected)` (line 147) builds all six per trajectory,
  or only `selected[i]`.

## 4. Value curves and allocation

- **Caps** (`Teacher::run`): `cap = 2 + ceil(max(6, 4 · mean interior budget))`,
  `caps[i] = min(max(cap, equal_budgets[i]), size(i))`.
- **`build_curves(db, objective, partners, caps, rdp_only)`** (line 238): for each trajectory in
  parallel and each budget b from 2 to its cap, makes every candidate, drops duplicates, and calls
  `pick` against the partner views and `RangeState(partners).weights(i)`. It stores the best value
  and method (`curves.best`) and RDP's value (`curves.rdp`).
- **`allocate(curves, total_budget)`** ([curves.cpp:43](engine/src/teacher/curves.cpp#L43)):
  starts every trajectory at 2 points, then repeatedly takes the offer with the highest gain per
  point from a priority queue. `best_offer` looks at steps of any length up to the budget left,
  which picks the next vertex of the curve's upper concave envelope. Offers are versioned, so a
  stale one is skipped and remade.
- **`curve_methods(curves, budgets)`** (line 282) reads each trajectory's best method at its
  budget.

## 5. The loop

`Search` ([teacher.cpp:294](engine/src/teacher/teacher.cpp#L294)) is set up once per start state,
and `run(seed, update)` copies the trackers so several runs can share the setup.

- **Active trajectories:** those with range boxes, join links, or an active kNN role.
  Inactive ones keep their method.
- **Setup:** for each active trajectory and each distinct candidate, its view, its separable
  score, and (with original kNN queries) its distances to every query watching it. Then a
  `JoinState` and a `RangeState` on the starting state, and a `KnnState` if ranking kNN is on.
- **Each pass:** shuffle the active order (`seed + kOrderSeedOffset`). For trajectory i, every
  other distinct candidate c gets
  `gain = separable[i][c] − separable[i][current] + join.propose + range.propose + knn.propose`.
  The best gain above `kMinImprovement` (1e-12) is applied to the state and all trackers.
- **Stop:** a pass with no switches, or `kMaxResponsePasses` (10) passes. Each pass prints the
  switch count and the tracked join F1 and kNN recall.
- **`update = false`:** every trajectory picks against the starting state and nothing is applied
  (`BUDGET_LOOP_NO_UPDATE`).

## 5b. Budget trading

`trade_budgets(db, objective, caps, kept)` ([teacher.cpp:457](engine/src/teacher/teacher.cpp#L457)),
on `loop-trade` only. It starts from `BUDGET_LOOP`'s kept sets and moves points between
trajectories, scored by the trackers against the current state.

- **Setup:** a `JoinState`, `RangeState` and separable score on the start state, and a `KnnState`
  built once and kept current by the trades.
- **Each round** (at most `kMaxTradeRounds`, 3), for every trajectory i and each step
  δ ∈ `kTradeSteps` {1, 2, 4}:
  - `buy[d][i]`: i's best recipe at b_i + δ (up to its cap), with its gain;
  - `sell[d][i]`: i's best recipe at b_i − δ (at least 2 points), with its loss.

  After this sweep, `knn->watch` extends the kNN lists to the offered versions, and the change
  this makes to the tracked kNN metric is added to the running total ("kNN list correction" in
  the log).
- **Matching:** buyers sorted by gain, sellers by loss. For each buyer, `find_profitable_seller`
  ([trade_matching.hpp](engine/include/backend/engine/teacher/trade_matching.hpp)) returns the
  cheapest unused seller other than itself whose loss the gain outweighs by more than
  `kMinImprovement`. The cursor only moves past used sellers, so a buyer's own entry stays
  available to later buyers (the fix in `61cce4d`).
- **Applying:** each pair is rescored exactly, seller first and then buyer against the changed
  state, and kept only if the combined gain is still positive; otherwise the seller is reverted.
- **Stop:** a round with no kept pair, or 3 rounds.
- **Final check:** exact `objective.total` of the result; if it is below the start, the start is
  returned. The log prints tracked against exact J_T and each term.

## 6. Arms

`Teacher::run` ([teacher.cpp:774](engine/src/teacher/teacher.cpp#L774)) computes only what the
requested arms need, and `keep` stores each arm with its exact `objective.total`.

| Arm | Construction |
|---|---|
| `EQUAL_CHOICE` | `independent_choice(equal candidates, partners = original)` |
| `EQUAL_CHOICE_RDP_PARTNERS` | `curve_methods(curves.best, equal budgets)`, or `independent_choice` with RDP partners |
| `EQUAL_LOOP` | `best_response` from `EQUAL_CHOICE_RDP_PARTNERS` |
| `BUDGET_RDP` | `allocate(curves.rdp)`, RDP everywhere |
| `BUDGET_CHOICE` | `allocate(curves.best)`, `curve_methods` |
| `BUDGET_LOOP` | `Search(BUDGET_CHOICE).run(seed)` |
| `BUDGET_LOOP_TRADE` | `trade_budgets(BUDGET_LOOP)` (section 5b), on `loop-trade` only |
| `BUDGET_LOOP_REALLOCATE` | up to `kMaxAllocationRounds` (10) rounds of `build_curves(current)` → `allocate` → `make_candidates` → `best_response`; a round is kept only if exact J_T rises by more than 1e-12 |
| `BUDGET_LOOP_NO_UPDATE` | `Search(BUDGET_CHOICE).run(seed, false)` |
| `BUDGET_LOOP_SHUFFLED` | `best_response` on an `Objective` over `shuffled_partners(...)`, scored on the real objective |

`shuffled_partners` (line 672) buckets trajectories by the 1 km cells of their start and end
points, shuffles labels within each bucket, and relabels every join pair and kNN list.

## 7. Evaluation

[teacher_eval.cpp](engine/apps/teacher_eval.cpp), per seed:

- **Arms:** every requested teacher arm per share (all of `kArms` unless `--arms` is given, so
  reallocation and trading run by default), then `baseline_arms` (`RDP_GLOBAL`,
  `RDP_a0`, `RDP_a025`, `SED_a0`, `UNI_a0`, `PARTNER_l0.5`, `PARTNER_l1`) and
  `Teacher::single_recipes` (`RECIPE_*`). `validate` checks the endpoints and the total budget.
- **Held-out workloads**, built only after that: `pipeline::TestWorkloads` (held-out kNN, range,
  where-at, the held-out join at `join.eps_m` plus `join.score_eps_m`, the self join, self kNN
  from the teacher's cache), and both joins at each `--eval-eps` threshold (default 125, 175,
  250 m).
- **Scores:** `Evaluator(db, compressed, test).evaluate()`, normalised against the endpoints-only
  floor. J4 is the mean of normalised kNN recall, range F1, where-at average hit and the join grid
  mean. `objective.metrics(arm.kept)` gives the teacher's own training terms next to the test.

Output files in `--out`. They are appended to, so use a new directory per run.

| File | Contents |
|---|---|
| `results.csv` | One row per seed, share and arm: raw and normalised scores, J4, J_T, self join, J4-self |
| `thresholds.csv` | Both joins and route clustering at each `--eval-eps` threshold, with `teacher_threshold` |
| `terms.csv` | Teacher training terms next to test terms, per arm |
| `budgets.csv` | Points per trajectory per arm |
| `floors.csv` | Endpoints-only floors per seed |
| `timings.csv` | Seconds per step |

## 8. Constants

| Constant | Value | Where |
|---|---|---|
| `kMaxResponsePasses` | 10 | teacher.cpp:43 |
| `kKnnWatchFactor` | 3 (watch 3k per query) | teacher.cpp:44 |
| `kMaxAllocationRounds` | 10 | teacher.cpp:45 |
| `kTradeSteps` | 1, 2, 4 points | teacher.cpp:46 |
| `kMaxTradeRounds` | 3 | teacher.cpp:47 |
| `kMinImprovement` | 1e-12 | teacher.cpp:48 |
| `kCell` (shuffle buckets) | 1000 m | teacher.cpp:674 |
| `kKnnResolutionM` | 1 mm (continuous kNN ranking grid) | query/knn.hpp:12 |
| `kMaxJoinTerms` | 8 | objective.cpp:74 |
| `kBoundSlack` | 1e-15 | objective.hpp:18 |
| `kMinGain` (greedy) | 1e-15 | greedy.cpp:12 |
| `kRangeQueryCount` | 128,000 | workloads.hpp:22 |
| Seed offsets | range 7001, where-at 7002, visit order 7004, shuffle 7006 | workloads.hpp:17–20 |
| `kEvalEps` | 125, 175, 250 m | teacher_eval.cpp |
| `kPartnerLambdas` | 0.5, 1 | teacher_eval.cpp |

## 9. What is currently incorrect

Only places where the code does something wrong. Approximations made on purpose (the value
curves, the per-trajectory kNN check, partners fixed at RDP) are not listed.

| # | Item | Status |
|---|---|---|
| 1 | The loop can lower J_T unnoticed | open |
| 2 | `KnnState` ranks near-ties differently from `query::knn` | fixed on `loop-trade` (`ee1fa18`) |
| 3 | Shuffled control keeps the real range boxes | open |
| 4 | Shuffled kNN lists can be shorter than k | open |
| 5 | `teacher_eval` builds a fixed subset of the baselines | open |
| 6 | `teacher_stats.py` shows the wrong self join | merged into `main` (PR #57) |

### In the teacher

1. **The loop can lower J_T, and nothing checks it.**
   `KnnState` only tracks each query's watch list (true top k plus 3k others), so in a mixed state
   an unwatched trajectory can enter a query's top k without the tracker seeing it. A switch with a
   tracked gain above 0 can then lower the real J_T. `BUDGET_LOOP` is saved with whatever
   `total()` returns and is never compared with `BUDGET_CHOICE`.
   Trading already guards against the same risk: `trade_budgets` recomputes exact J_T at the end
   and falls back to its start if it lost. The loop has no such check.
   *Fix:* keep the better of the loop's start and end by exact J_T, as trading does, or rebuild
   `KnnState` from the current state between passes. (On the sample run the loop still ended
   above its start.)

2. **With continuous Fréchet, `KnnState` ranks differently from `query::knn`.** *Fixed on
   `loop-trade`; still open on `main` until it merges.*
   `query::knn` treats distances within 1 mm as ties ordered by index
   ([knn.cpp](engine/src/query/knn.cpp), `tolerance = 1e-3`). `KnnState::hits`
   ([objective.cpp:817](engine/src/teacher/objective.cpp#L817),
   [834](engine/src/teacher/objective.cpp#L834)) sorts bisected distances by strict value.
   Reproduced with 24 trajectories, every point kept and every trajectory watched: tracked recall
   0.9583, recomputed 1.0. The exported J_T is recomputed and correct; the loop's decisions, and
   the budget-trading code that also uses `KnnState`, optimise a slightly different ranking.
   `RankingKnnIsTrackedExactly` avoids it by switching to discrete Fréchet; `geolife.json` uses
   continuous.
   *Fixed* in `ee1fa18`: continuous distances are ranked on a fixed 1 mm grid, and `query::knn`
   and `KnnState::ranked_hits` share `query::knn_rank_less`. Tested by
   `ContinuousKnnNearTiesMatchFullScoringThroughSwitches`. `BUDGET_LOOP`'s J4 on the sample did not
   change (0.685 at 0.25 %).

3. **The shuffled control does not relabel range boxes.**
   `shuffled_partners` relabels join pairs and kNN lists only. Since range became macro F1 with
   `RangeState`, range is relational too, so `BUDGET_LOOP_SHUFFLED` still optimises the real range
   relations and no longer measures all of what reacting to relations adds.
   *Fix:* relabel each range box's reference list with the same `label` map.

4. **Shuffled kNN lists can be shorter than k.**
   Relabelling drops duplicate neighbours, but `m_knn_weight` still assumes k per valid query, so
   those queries can never reach full recall. Control arm only, small effect.
   *Fix:* fill to k distinct neighbours, or mark short queries invalid.

### In the evaluation

5. **`teacher_eval` compares the teacher against a fixed subset of the configured baselines.**
   It hard-codes 5 cheap baselines plus `PARTNER`, while `run` builds `algorithms` ×
   `allocation_alphas` (10 with `geolife.json`), and `teacher_stats.py` picks the best from a
   hard-coded `CHEAP` list. Scores are unaffected; the "best baseline" and the reported lead can
   be. On the 15 % sample, the best baseline at 2 % is SED α = 0.25 (0.936 J4), which
   `teacher_eval` does not build (its best there would be `SED_a0`, 0.930).
   *Fix:* build the baselines from the config in `baseline_arms` (at least α ∈ {0, 0.25}) and
   derive `CHEAP` from the arms in the CSV.

6. **`teacher_stats.py` on `main` shows the wrong self join.**
   It never loads `norm_self_join`, so its "150 m self join" column falls back to `norm_join`,
   the held-out join, and its 100 m and 300 m columns are always empty because `thresholds.csv`
   now holds 125, 175 and 250 m. The CSVs are correct.
   *Fixed* and merged into `main` through PR #57 (`21f2368`).

### Not a bug, but the main limit

The teacher's kNN and join gains do not carry over to held-out queries. On the sample at 0.25 %,
`BUDGET_LOOP`'s kNN drops from 0.821 on training queries to 0.755 on test queries (below plain
RDP's 0.786), and its join from 0.900 to 0.618, while range and where-at score the same on both.
Budget trading confirms it: +0.019 J_T but +0.003 held-out J4 at 0.25 %, trading held-out join
(−0.020) for held-out kNN (+0.029). See [TEACHER_SAMPLE_RUN_RESULTS.md](TEACHER_SAMPLE_RUN_RESULTS.md)
and [TEACHER_OPEN_PROBLEMS.md](TEACHER_OPEN_PROBLEMS.md).

## 10. Next steps

What still has to be done, item by item. 10.2 and 10.6 are done, and exact budget moves (10.8)
exist on `loop-trade`. Suggested order for the rest: 10.1 → 10.3 and 10.4 → 10.5 → drop
reallocation (10.8) → 10.7, starting with its measurement step → 10.9. The first four are each
small and make the numbers trustworthy; 10.7 and 10.9 are where held-out J4 can still move.

### 10.1 Check the loop against exact J_T (incorrect item 1)

1. Copy the final check in `trade_budgets`. In `Teacher::run`, after `search->run(m_seed)`, compute
   `objective.total(assemble(candidates, looped))` and compare it with `BUDGET_CHOICE`'s J_T,
   which `keep` already computes. If the loop's result is lower, keep `BUDGET_CHOICE`'s methods
   and print a warning with both values.
2. Do the same in `best_response` for `EQUAL_LOOP` and for the loop inside each reallocation
   round. Reallocation already compares whole rounds, but not the loop inside a round.
3. Optional: rebuild `KnnState` from the current state at the start of each pass, so its watch
   lists follow the state instead of the starting candidates. This costs one kNN query per valid
   query per pass, so do it only if step 1 shows the problem happens.
4. Test on a database large enough that 3k is a small fraction of it (for example 200
   trajectories with k = 3), asserting the loop never ends below its start.

Effort: small for steps 1, 2 and 4.

### 10.2 Make `KnnState` use `query::knn`'s tie rule (incorrect item 2) — done on `loop-trade`

Done in `ee1fa18`, with a fixed 1 mm grid instead of the bracket logic below, so the comparison is
transitive and can go into `partial_sort`. Kept for reference:

1. Move the tie rule into one shared function in `query`, with the 1 mm tolerance for continuous
   Fréchet, and call it from `query::knn` and from both `KnnState::hits` overloads.
2. A comparison with a tolerance is not transitive (a ≈ b and b ≈ c do not imply a ≈ c), so it
   cannot go straight into `partial_sort`. Follow `query::knn`'s bracket logic instead: sort by
   distance, merge runs within 1 mm, and order each run by database index.
3. Test: run `RankingKnnIsTrackedExactly`, or a copy of it, with continuous Fréchet and assert
   that tracked recall equals `knn_recall` after every move. The 24-trajectory reproduction
   (tracked 0.9583, recomputed 1.0) is a ready-made case.
4. Tell whoever owns the budget-trading code. It uses `KnnState`, so it gets the fix for free,
   but its tests should also run with continuous Fréchet.

Effort: medium. The bracket logic is the fiddly part.

### 10.3 Relabel range boxes in the shuffled control (incorrect item 3)

1. In `shuffled_partners`, after the kNN loop, map every index in
   `result.range.queries[b].reference` through `label`, then sort and remove duplicates. Reference
   lists must stay sorted, because the evaluator and `Objective` search and invert them.
2. Print how many box memberships stayed real, as it already does for pairs and neighbours.
3. Re-run the control. Expect `BUDGET_LOOP_SHUFFLED`'s range score to drop and its gap to
   `BUDGET_LOOP` to grow.

Effort: small.

### 10.4 Keep shuffled kNN lists at k (incorrect item 4)

In `shuffled_partners`, when a relabelled neighbour is a duplicate, fall back to the original
neighbour if it is not already in the list, or take the next true neighbour beyond k.
Alternatively, set `valid = false` on lists that end up shorter than k.

Effort: small.

### 10.5 Build the configured baselines in `teacher_eval` (incorrect item 5)

1. Make `baseline_arms` loop over `config.algorithms` × `config.allocation_alphas`, as `run`
   does in `pipeline.cpp`, adding `RDP_GLOBAL` once if it is listed. Keep the current names
   (`RDP_a0`, `RDP_a025`, `UNI_a0`; new ones become `RDP_a1`, `SED_a025`, and so on) and the
   `PARTNER` and `RECIPE_*` arms.
2. Require α = 0 to be present: the teacher's equal-budget arms, `RECIPE_RDP` and the steps in
   `teacher_stats.py` compare against `RDP_a0`.
3. Pass `config` instead of `config.budget_shares` at the call site, and delete `kRdpAlpha`.
4. Derive `CHEAP` in `teacher_stats.py` from the arms in the CSV: every arm that is not a teacher
   arm and does not start with `RECIPE_`.

The empty branch `teacher-eval-config-baselines` is set up for this. Cost: about 4 s per extra
baseline per share and seed on the 15 % sample.

### 10.6 Merge the `teacher_stats.py` fix (incorrect item 6) — done

Merged through PR #57. What is left: re-run `teacher_stats.py` on any old `HeldOut` runs whose
self-join table has been used.

### 10.7 Make kNN and join gains carry over to new queries

This is research rather than a fix. Possible steps, cheapest first:

1. **Measure where the loss comes from.** Split the held-out kNN loss into queries whose true
   neighbours the teacher compressed hard, and queries where a compressed outsider moved into the
   top k. `budgets.csv` and the evaluator's kNN answers hold what is needed. This shows whether
   the teacher is starving trajectories no training query needed.
2. **Floor the budgets.** If step 1 shows starved trajectories, give every trajectory a minimum
   interior budget during allocation, so that none drops to endpoints only because no training
   query touched it.
3. **Make the training workloads less specific.** Cheap options: perturbed copies of database
   trajectories as extra kNN queries; join pairs at more thresholds (already partly done with
   100, 200 and 300 m); weighting each kNN query by how close its k-th and (k+1)-th neighbours
   are, since near-ties are the fragile queries.
4. **Hold out part of the training workloads.** Split the teacher's own kNN queries and join
   pairs into a fit half and a check half, and use the check half to stop the loop early or to
   choose term weights. The removed seen/unseen experiment did this for the join as a training
   mode; it could come back as an early-stopping signal.
5. **Treat the loop as optional.** On the sample the loop adds 0.004–0.010 J4, and its J_T gains
   sit in the terms that do not carry over. If steps 1–4 do not change that, the paper's teacher
   could be `BUDGET_CHOICE` plus exact budget moves: cheaper and nearly as good. Trading now exists,
   so this can be tested directly: an arm that trades from `BUDGET_CHOICE`, against
   `BUDGET_LOOP_TRADE`.

### 10.8 Missing for the paper

- **Exact budget moves** (open item 5 in TEACHER_NEXT_STEPS.md). *Done on `loop-trade`* as
  `BUDGET_LOOP_TRADE` (section 5b). On the sample: +0.019 / +0.005 J_T, +0.003 / +0.000 held-out
  J4 at 0.25 % / 1 %, for 114 s / 207 s. Interleaving trading with the loop added nothing
  (experimental branch `loop-trade-interleaved`, not for merging).
- **Drop or demote reallocation.** It takes most of the run time and did not raise J4 on the
  sample; trading does the same job for a quarter of the time with higher J4. `teacher_eval`
  still runs it by default, since its default arms are all of `kArms`.
- **More seeds and the full dataset.** Every number here is one seed on a 15 % sample.
- **Published baselines** (open item 6), for example the `Adding-RL4QDTS` branch.

### 10.9 Widen the candidate set

Selection, the loop and trading choose among the same six recipes, and the greedy ones reward only
range and where-at. No candidate is built for kNN or join, and more search over these six has
stopped moving held-out J4.

1. Add a slack-constrained recipe: keep the points needed to hold the Fréchet error under a slack
   derived from the trajectory's kNN and join margins, via the triangle inequality.
2. Check how often selection picks it, and whether held-out kNN and join improve or only the
   training terms.

See [TEACHER_OPEN_PROBLEMS.md](TEACHER_OPEN_PROBLEMS.md).
