# Teacher: status and next steps

Implementation update, 2026-11-20: the range objective and endpoint floor now use macro F1 like
J4, with exact incremental updates in the loop and F1 marginals for candidate and budget scoring.
`RDP_GLOBAL`, a database-wide top-down PED baseline, runs alongside the other baselines in
`teacher_eval` and is enabled in `geolife.json`. It is included in the best-cheap-baseline comparison.
Continuous kNN now shares a deterministic millimetre-grid ranking between the evaluator and
tracker; V6 caches rebuild old reference rankings. Limited watch lists remain approximate.
The numerical findings below predate these changes and need to be checked with fresh runs.

Updated 2026-11-17 on branch `teacher`. The teacher is described in
[engine/src/teacher/README.md](engine/src/teacher/README.md); this file says what is settled and
what is open.

## The teacher

`BUDGET_LOOP`: budgets searched on value curves, the best candidate recipe per trajectory, then the
selection loop, which re-picks each trajectory's recipe against its neighbours' current compressed
state until nothing switches. kNN is scored as a ranking among the compressed database
(`teacher.ranking_knn`, the default).

| Arm | Role |
|---|---|
| `EQUAL_CHOICE`, `EQUAL_CHOICE_RDP_PARTNERS`, `EQUAL_LOOP` | the same steps at equal budgets |
| `BUDGET_RDP` | searched budgets with RDP's points |
| `BUDGET_CHOICE` | searched budgets and recipe choice, no loop |
| `BUDGET_LOOP_REALLOCATE` | budgets re-searched after the loop, so they react to neighbours too |
| `BUDGET_LOOP_NO_UPDATE` | control: one look at the start, no reacting |
| `BUDGET_LOOP_SHUFFLED` | control: join partners and kNN neighbours relabelled |
| `RECIPE_<recipe>` (baselines in `teacher_eval`) | one recipe for every trajectory |

## Settled (Geolife, 0.25 % unless stated)

- **Budgets per trajectory:** `BUDGET_RDP` 0.553 J4 against `RDP_a0` 0.456, every seed.
- **Recipe per trajectory:** `EQUAL_CHOICE` beats the best single recipe (`RECIPE_GREEDY_RDP2`) by
  0.014 to 0.035 J4 at 0.25 % to 1 %, every seed.
- **Relations, database self join:** trained on the self join, the loop lifts the 150 m self join
  well above the start; the shuffled control falls far below it. Trained on several thresholds
  (`teacher.extra_join_eps_m`) the gain holds at thresholds not trained on, at small budgets.
- **Relations, kNN on held-out queries:** the ranking loop adds 0.01 to 0.05 held-out kNN over the
  same loop with the per-trajectory score, every seed and budget tested; shuffled neighbours turn
  it into a loss.
- **Overall:** `BUDGET_LOOP` J4 0.600 against `RDP_a0` 0.458 at 0.25 %, 0.876 against 0.795 at 1 %
  (seed 42).

## Open

1. **Held-out kNN and join at larger budgets.** Above 0.25 % the teacher trails `RDP_a0` on
   held-out kNN, and at 1 % on the held-out join. Its kNN losses carry over to new queries while
   most of its gains do not.
2. **Value curves still use the per-trajectory kNN check.** A ranking version against a fixed
   state that only rescored each query's true neighbours made no difference; one that also rescores
   intruders is far too slow as is.
3. **Term weights** (`teacher.term_weights`, `--term-weights`) trade range and where-at for kNN.
   Seeds 17, 42 and 43 at 0.25 %: with kNN weighted 1×, 2× or 4× `BUDGET_LOOP` beats `RDP_a0` on
   all four held-out terms on every seed; at 4× held-out kNN is 0.719 against 0.669, for 0.019
   J4. At 0.5 % (seeds 17 and 43) 4× brings kNN level with `RDP_a0` (0.827 against 0.828) for
   0.006 J4. At 1 % (seed 42) it barely moves kNN: there the gap comes from the candidates.
4. **Held-out join thresholds.** `teacher_eval` now scores both joins at every `--eval-eps`
   threshold. Trained at 150 m only, the held-out join lead at 0.25 % (seeds 42 and 17) peaks at
   150 m (+0.10 over the best baseline) and falls to −0.01 at 100 m. Trained on 100, 125, 150, 200
   and 300 m (`teacher.extra_join_eps_m`) it is +0.07 to +0.10 at every threshold from 100 to
   300 m, including 175 and 250 m, which neither trained on; J4 −0.006. `engine/configs/geolife.json` now trains on these thresholds, and
   `teacher_eval` scores both joins at 100 to 300 m by default. At 1 % it narrows the gap
   to the baselines from −0.16 to −0.08 at 100 m and from −0.03 to −0.01 at 300 m, J4 +0.002.
5. **Budget moves between trajectories scored exactly** with `KnnState` and `JoinState`, beyond
   what `BUDGET_LOOP_REALLOCATE` does.
6. **For a paper:** published baselines, a second dataset, the 2 % budget, more seeds.

## Evaluation

With `engine/configs/geolife.json`, J4's join term is the held-out join averaged over 100, 150, 200
and 300 m (`join.score_eps_m`), each normalised against its own floor, so a teacher cannot win it
by fitting 150 m. `results.csv` keeps the 150 m value alone as `norm_join_main`, and
`thresholds.csv` has both joins at every `--eval-eps` threshold, including 175 and 250 m, which
nothing trains on. Numbers above from before 2026-11-17 use the 150 m join alone.

## Run

```bash
cmake --build build-release && ctest --test-dir build-release
./build-release/engine/teacher_eval --config engine/configs/geolife.json \
    --seeds 42 --shares 0.0025 --out results/<new_dir>
python3 engine/apps/teacher_stats.py results/<new_dir>
```

Use a new `--out` directory each time: the CSVs are appended to. Runs may rewrite the cache files
in `cache/`, which are tracked although `.gitignore` lists them.
