# Teacher: status and next steps

Updated 2026-11-22. The teacher is described in
[engine/src/teacher/README.md](engine/src/teacher/README.md); this file says what is settled and
what is open. Findings from before 2026-11-22, including the removed arms and the self-trained
teacher, are in git history.

## The teacher

Trained for held-out queries only: database kNN queries and join queries are originals against
the simplified database, as in J4. J_T is J4's four terms on the database's own workloads plus two
query-free terms on each trajectory's Fréchet deviation bound (shape 1, margin 1, in
`engine/configs/geolife.json`).

| Arm | Role |
|---|---|
| `BUDGET_RDP` | budgets from value curves, RDP's points |
| `BUDGET_CHOICE` | budgets and recipe choice per trajectory |
| `BUDGET_LOOP` | then best response on the tracked J_T |
| `BUDGET_LOOP_TRADE` | then points traded between trajectories: the teacher |
| `RECIPE_<recipe>` (baselines in `teacher_eval`) | one recipe for every trajectory |

## Settled (Geolife, held-out J4, seeds 42, 7, 99, 123)

Research notes: `Teacher pseudo held out selection 2026-11-21`, `Teacher fresh seed confirmation
2026-11-22` in `~/code/qdts-research/research_notes/`.

| Share | Arm | kNN | join | range | where-at | J4 |
|---|---|---|---|---|---|---|
| 0.25 % | RDP_a0 | 0.669 | 0.433 | 0.401 | 0.340 | 0.461 |
| 0.25 % | BUDGET_LOOP_TRADE | 0.734 | 0.614 | 0.656 | 0.445 | 0.612 |
| 1 % | RDP_a0 | 0.915 | 0.875 | 0.705 | 0.698 | 0.798 |
| 1 % | BUDGET_LOOP_TRADE | 0.911 | 0.882 | 0.943 | 0.836 | 0.893 |

- Shape and margin weights were chosen on pseudo-held-out splits of the database (the real query
  pool never read), then confirmed once on four seeds.
- At 1 % the teacher trails RDP on held-out kNN by 0.001 to 0.006 per seed; the gap is near-tie
  swaps (recall with 5 % distance tolerance 0.993 against 0.991).
- Most of the J4 gain comes from allocation, then recipe choice; the loop and trades add little
  J4 but win back kNN and join.

## Open

1. **The student:** learn the frozen teacher's kept sets or budgets.
2. **Value curves still use the per-trajectory kNN check** (join scoring now uses partners' F1 slots via `JoinState::gain`).
3. **For a paper:** published baselines, a second dataset, the 2 % budget, more seeds. The loop
   controls (shuffled partners, no update) were removed; restore them from git history if a claim
   about the loop needs them.

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
