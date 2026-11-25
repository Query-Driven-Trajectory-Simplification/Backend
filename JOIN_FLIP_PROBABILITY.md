# Expected join outcomes: `flip_probability` and pair distances

Branch `join-flip-probability` (from `main` at `bb6ea56`). This is the groundwork for training the
teacher's join term on expected outcomes instead of exact ones. It adds the two building blocks,
with tests; nothing in the teacher uses them yet, so every current run gives exactly the same
results as before.

## Why

The teacher overfits the join. At `1%` (5 seeds, `TEACHER_RESULTS.md` on `origin/results`):

| Arm | Training join | Held-out join |
|---|---:|---:|
| `RDP_a0` | 0.786 | 0.781 |
| `BUDGET_RDP` | 0.957 | 0.835 |
| `BUDGET_LOOP_TRADE` | 0.952 | 0.791 |

The training join uses database trajectories as the queries. For each trajectory `i`, the teacher
knows every training pair's **slack** `s = |d(j, i) − ε|`, the room before the pair's answer
changes. It then gives `i` just enough points to keep its deviation `δ_i` under the smallest of
those slacks. A held-out query on the same route usually has a different, often smaller slack, and
its answer flips. The teacher is fitting its budgets to one particular set of queries rather than to
the queries it will meet.

The fix is to score each training pair by how likely a query **like it** is to keep the right
answer, given `i`'s deviation, rather than by whether that exact pair survived.

## `flip_probability(slack, deviation, bandwidth)`

Declared in [objective.hpp](engine/include/backend/engine/teacher/objective.hpp), defined in
[objective.cpp](engine/src/teacher/objective.cpp).

Two kinds of uncertainty:

1. **The error.** Simplifying `i` moves its distance to any query by at most `δ_i`
   (`|d(Q, i′) − d(Q, i)| ≤ δ_i`, with `δ_i` from `simplification_bound`). The direction is
   unknown, so the move is taken as uniform within `±δ_i`. This is the same assumption the kNN
   margin term already makes.
2. **The query.** A training pair stands for the queries near it. A nearby query's slack is taken
   as uniform within `±h` of this pair's, cut at 0.

A query with slack `x` flips when the move passes `x` in the wrong direction, which has probability
`max(0, δ − x) / (2δ)`. Averaging over `x` uniform in `[a, b] = [max(0, s − h), s + h]`:

```
P_flip = [G(min(b, δ)) − G(min(a, δ))] / (2δ (b − a)),     G(x) = δx − x²/2
```

A positive pair then counts as matched with probability `1 − P_flip`, and a negative one as wrongly
matched with probability `P_flip`.

### Properties (all covered by tests)

| Property | Meaning |
|---|---|
| `δ = 0` gives 0 | Full fidelity is still scored as perfect |
| Never above ½; exactly ½ for infinite `δ` | `δ` is a bound, so a large error is equally likely to move the distance either way |
| `h = 0` gives `(δ − s) / (2δ)` above the slack | Exactly the kNN margin term's rule ([objective.cpp](engine/src/teacher/objective.cpp), `raw_separable`) |
| Rises with `δ`, falls with `s` | More error or less room means more flips |
| With `h > 0` the charge starts at `δ = s − h`, not `δ = s` | The fix for fitting just under the training slack |
| Equals a numerical average of the definition | Checked on 200 random inputs, to 1e-7 |

### Worked example: slack 30 m

| `δ_i` (m) | `h = 0` (kNN margin rule) | `h = 15 m` | `h = 30 m` |
|---:|---:|---:|---:|
| 15 | 0.000 | 0.000 | 0.062 |
| 20 | 0.000 | 0.010 | 0.083 |
| 25 | 0.000 | 0.033 | 0.104 |
| **29** | **0.000** | **0.056** | **0.121** |
| 35 | 0.071 | 0.095 | 0.146 |
| 60 | 0.250 | 0.250 | 0.250 |
| 300 | 0.450 | 0.450 | 0.450 |

Without a bandwidth, `δ = 29 m` is free: the training pair survives, so the teacher spends nothing
on it. With `h = 15 m` it costs 0.056 per link, which is the actual flip rate of queries whose
slacks are spread between 15 and 45 m.

### The bandwidth `h`

`h` decides how far "a query like this one" reaches. Scale it with the threshold, `h = β · ε`, since
the four join thresholds run from 50 m to 200 m.

- **Small `β` (≈ 0.1):** close to the kNN margin rule. It removes the fitting right at the training
  slacks and otherwise stays faithful to the specific pairs.
- **Large `β` (≈ 0.5):** close to "charge `δ_i` in proportion to how many partners sit near `ε`",
  almost independent of individual pairs. Only a wide `h` helps routes whose training partners are
  all near-duplicates (slack ≈ `ε`) while held-out queries arrive at middle distances.
- **Choose it on the pseudo split** (`teacher_eval --pseudo-split`), not on the final seeds.
  Comparing the slacks of training pairs with those of held-out pairs gives a starting value.

## Pair distances

`flip_probability` needs each training pair's slack, so it needs `d(j, i)` between the
**originals**. The existing pair bracket isn't enough: its lower end is a Hausdorff-based bound and
its upper end the discrete Fréchet distance, and the two can be tens of metres apart.

New in `Objective`:

| API | What it does |
|---|---|
| `ObjectiveConfig::pair_distances` | Opt in. Default `false`, so nothing changes or costs more unless it is set |
| `pair_distance(pair)` | The originals' distance in the workloads' Fréchet variant. Throws `std::logic_error` if not enabled |
| `slack(link)` | `|pair_distance − ε|` at the link's threshold |
| `join_eps(term)` | The threshold of a join term |

How it is computed (in `bracket_pairs`, once per unique pair across all thresholds):

- **Continuous Fréchet:** `continuous_frechet_distance` searches within the existing bracket, to
  1 mm. For pairs above the largest threshold the discrete distance, now computed without a cutoff,
  serves as the upper end.
- **Discrete Fréchet:** the exact discrete distance.

**Cost:** about one decision per halving of the bracket down to 1 mm (around 15–20), against one
discrete pass for the bracket alone. The construction cost on full GeoLife is not measured yet.
Before using it in every run, measure it, and if it is significant, cache the distances with the
teacher workloads.

**Tests** ([teacher_test.cpp](engine/tests/teacher/teacher_test.cpp)):

- Every continuous distance is within 1 mm of the true one, checked with the decision procedure
  itself: within at the stored distance, not within 1 mm below it.
- Positives lie at most `ε` (+1 mm) apart and negatives more than `ε` apart, matching how the pairs
  were labelled.
- Discrete distances equal `query::frechet_distance` exactly.
- Turning the option on leaves every metric and floor of the objective bit-identical. Without it,
  `pair_distance` and `slack` throw.

All 290 engine tests pass.

## What comes next

1. **Step 0: diagnostics on the pseudo split.** Export per trajectory and arm: `δ_i`, the smallest
   training slack per threshold, and the held-out flips attributed to `i`. The prediction to check:
   for the teacher, `δ_i / s_min` piles up just below 1, while RDP spreads out. Also measure the
   real error `e = d(j, i′) − d(j, i)` on training pairs. If `e / δ` sits well inside `±1`, the
   uniform error model charges too much and should be replaced by the measured distribution.
2. **1a: a separable `JoinMargin` term.** For each link where `i` is the database side, add
   `1 − P_flip` (positive) or `P_flip` (negative), weighted like the kNN margin term. Value curves,
   the loop and trading handle separable terms already, so they need no changes. Sweep `β` in
   {0.1, 0.25, 0.5} and the term weight in {0.5, 1}.
3. **1b: replace the exact join term** with expected counts in `JoinState`, only if 1a leaves a
   large gap. Keep `Exact` as the default mode, so all exact-tracking tests run unchanged.
4. **Choose by a rule written down in advance:** the best validation `J4` among configurations whose
   validation join is at least `BUDGET_RDP`'s. Then one run on the 5 final seeds.

Whatever the mode, report the **exact** training join next to the held-out one, so the train/test
gap stays comparable with the table above.

## Limitations

- **Only `δ_i` matters, not which points are kept.** Two candidates with the same deviation get the
  same expected join. That stops the teacher exploiting lucky geometry for one partner, which is the
  aim, but it also stops it rewarding geometry that genuinely helps. The results doc found the join
  gap is a budget effect, not a shape effect, so little should be lost.
- **`δ_i` is an upper bound.** If it is loose, the term over-charges. The calibration in step 0
  checks this.
- **Trajectories with no training pairs still get no join value.** That case needs a prior on query
  density, not this term.

## Files changed

| File | Change |
|---|---|
| [objective.hpp](engine/include/backend/engine/teacher/objective.hpp) | `flip_probability`, `ObjectiveConfig::pair_distances`, `pair_distance`, `slack`, `join_eps` |
| [objective.cpp](engine/src/teacher/objective.cpp) | Pair distances in `bracket_pairs`; the three functions |
| [flip_probability_test.cpp](engine/tests/teacher/flip_probability_test.cpp) | 7 tests, new file |
| [teacher_test.cpp](engine/tests/teacher/teacher_test.cpp) | 3 pair-distance tests |
| [engine/CMakeLists.txt](engine/CMakeLists.txt) | Registers the new test file |
