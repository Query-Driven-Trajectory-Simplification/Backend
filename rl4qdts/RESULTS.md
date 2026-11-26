# RL4QDTS on our data: results and causes

Status: 26 Nov 2026. Engine seed 42 unless stated, `engine/configs/geolife.json` (1 km range boxes, 256,000 practice queries).
All J4 values are the engine's combined, floor-normalised score (0 = endpoints only, 1 = perfect).

## TL;DR

- **RL4 scores low because its cube agent concentrates the budget.** It adds points only inside the cube it zooms into, and it always picks the same cube for the same cell. Most trajectories get no points.
- **Whether a trained model avoids this is luck.** RL4's reward gives the cube agent no reason to stop zooming early. In the authors' own code the stop action is never trained at all. So some training runs learn to stop and score well; most don't.
- **Same code, different seed:** over our first 12 training seeds, J4 at 2% ranges from **0.126 to 0.738** (mean 0.360). Of 28 seeds trained, 9 learn to stop, and those score 0.63–0.76. The seed picked on validation (0.704) approaches the simple baselines; the strongest baselines (0.83–0.85) stay ahead.
- **It happens in the authors' own setup too.** We re-ran their unchanged pipeline on their data, using the paper's own metric (their range F1, not our J4). Their shipped model reproduces the paper's result (0.735 vs 0.733), but retraining gives 0.51–0.78 over 12 seeds. The paper's error bar (± 0.018) only covers re-running one trained model, which varies about 7 times less than retraining.
- **It doesn't depend on our engine seed.** On a second split (engine seed 7) the validation-best RL4 scores about the same as on seed 42.
- **Our pipeline is correct.** Every step was checked against the engine. The bugs we found (the authors' and one of ours) are fixed.

## Results (J4)

| Method | 0.25% | 0.5% | 1% | 2% |
|---|---|---|---|---|
| Best baseline (RDP/SED, alpha 0–0.25) | 0.395 | 0.565 | 0.725 | 0.851 |
| RDP_GLOBAL | 0.237 | 0.396 | 0.640 | 0.818 |
| RDP alpha=1 | 0.190 | 0.333 | 0.530 | 0.736 |
| UNIFORM alpha=1 | 0.169 | 0.289 | 0.447 | 0.602 |
| **RL4, best of 28 seeds on validation (seed 5)** | **0.213** | **0.345** | **0.516** | **0.704** |
| RL4, mean ± std of seeds 1–12 (not selected) | 0.140 ± 0.053 | 0.206 ± 0.100 | 0.282 ± 0.164 | 0.360 ± 0.242 |
| RL4, worst / best of the 19 scored seeds | 0.083 / 0.221 | 0.101 / 0.363 | 0.116 / 0.547 | 0.126 / 0.762 |
| RL4, seed 1 (`RL4_fixed`) | 0.121 | 0.169 | 0.220 | 0.261 |
| *RL4 with the authors' shipped weights* | *0.196* | – | – | *0.631* |
| *RL4 trained with the authors' loop and queries, lucky seed 13 (1 of 16)* | *0.205* | *0.333* | *0.498* | *0.684* |
| *RL4 without the cube agent (ablation only, not RL4)* | *0.222* | – | – | *0.767* |
| *First model, trained with our since-fixed bug (don't use)* | *0.074* | *0.083* | *0.087* | *0.088* |

RL4 runs: trained on our data and our practice queries with `--fixed`, simplified with `rl4_simplify.py --fixed`, scored by `run.exe` on the test workloads. Seeds 1–12 and baselines in one run: `data/rl4/experiments/overnight/scores_all_seeds.md`; the scored seeds from 13–28: `data/rl4/experiments/seed_search/scores_seeds_13-28.md` (identical baselines).

The best seed on test (seed 19, 0.762 at 2%) is not the one validation picks (seed 5); reporting it would be picking on the test set.

### Second engine seed (7)

To check that seed 42's split isn't special, we repeated the pipeline on engine seed 7 (a different database, practice and test split): export, 3 training seeds, simplify the validation-best, score.

| Method (engine seed 7) | 0.25% | 0.5% | 1% | 2% |
|---|---|---|---|---|
| Best baseline (RDP/SED, alpha 0–0.25) | 0.387 | 0.558 | 0.720 | 0.848 |
| RDP alpha=1 | 0.186 | 0.332 | 0.529 | 0.734 |
| UNIFORM alpha=1 | 0.166 | 0.286 | 0.441 | 0.595 |
| **RL4, best of 3 seeds on validation (seed 2)** | **0.215** | **0.359** | **0.546** | **0.755** |

- **Baselines barely move** between engine seeds (within 0.01).
- **RL4 follows the same pattern.** The best rounds of the 3 training seeds stop at the coarse level 0%, 100% and 62% of the time. Validation picks the one that always stops, which scores like seed 42's best.
- **So the training seed matters, the engine seed doesn't.**

Scores: `data/rl4/experiments/overnight/scores_seed7.md`; files: `data/rl4/seed7/`.

## Cause 1: the cube agent concentrates points

RL4 adds each point in two steps:
1. **Agent-Cube** draws a coarse cell (4.4 km × 3.4 km × 14 days), weighted towards busy areas, and either stops or zooms into a smaller cube (down to 1.1 km × 0.85 km × 3.5 days).
2. **Agent-Point** inserts a point on a trajectory inside that cube.

Agents act deterministically when simplifying, so **every draw of a cell ends in the same cube**. When the agent always zooms, all of a cell's budget goes into 1/64 of it. Measured on the first model at 2%: 6,216 cells → exactly 6,216 cubes, up to 1,996 points in one cube.

| At 2% | Trajectories that get any point | J4 |
|---|---|---|
| UNIFORM alpha=1 | 100% | 0.602 |
| RL4 without the cube agent | 97% | 0.767 |
| RL4 seed 5 (stops 99%) | 95% | 0.704 |
| RL4 seed 1 (stops 73%) | 64% | 0.261 |
| RL4 seed 4 (stops 40%) | 44% | 0.126 |

Our scores judge every trajectory along its whole length (kNN, where-at, join, clustering, range), so trajectories without points lose everywhere. More budget only piles more points into the same cubes, which is why weak RL4 models barely improve from 0.25% to 2%.

Over all 19 scored seeds, how often the cube agent stops at the coarse level almost fully decides the ranking at 2% (rank correlation 0.98). Seeds that stop at least 85% of the time score 0.55–0.76; the rest score 0.13–0.29.

## Cause 2: nothing makes the agent learn to stop

- **Shared reward.** Every decision in a window gets the same reward (the change in range F1 after 50 points), so the agent can't tell that stopping spreads points better.
- **Zooming is favoured.** The reward is never negative (adding points can't lower F1). Q-learning values a zoom as *reward + 0.99 × next value* and a stop as *reward*, so zooming always looks at least as good.
- **The authors' code never trains stop.** The stop decision is never stored. We checked: after training, the stop weights are bit-for-bit their random initial values in all 16 runs. Whether their model stops is purely down to those random weights.
- **Our bug (fixed 25 Nov).** Our first fix stored a "stop" at the deepest level, which inflated zoom values. That model never stopped (J4 0.088 at 2%); after the fix, stopping is learned in some runs.

## Cause 3: so the result is luck

Training rarely passes through a stopping policy (about 6 of 60 rounds per run). Validation then picks one of those rounds, and how much it stops varies by seed:

| Training procedure | Runs | Learned to stop (≥ 90%) | Never stopped |
|---|---|---|---|
| Ours (`--fixed`, our queries) | 28 | 9 (seeds 5, 7, 11, 13, 15, 19, 21, 24, 26) | 3 (seeds 14, 16, 22) |
| Authors' loop and queries, on our data | 16 | 1 (seed 13) | 7 |

- **Seed 13 behaves and scores like the authors' shipped model.** It stops 91% of the time (shipped: 93%) and scores J4 0.205 / 0.333 / 0.498 / 0.684 (shipped: 0.196 at 0.25%, 0.631 at 2%). The authors'-loop seed 4 (stops 65%) gets only 0.097 / 0.128 / 0.157 / 0.179. So the shipped model was most likely one lucky run. The paper reports only a single training run ("the best model is chosen during training"), with error bars over 50 simplification runs of that one model.
- **Validation finds a stopping run, not the best one.** In both procedures the best seed on validation is a stopping one, and over seeds 1–12 the top 3 on validation (5, 11, 7) are also the top 3 on test (rank correlation 0.89). Among the 9 stopping seeds, though, validation F1 says little about J4: seed 5 (best on validation) scores 0.704 at 2%, while seeds 19 and 24, lower on validation, score 0.762 and 0.757. Validation F1 is RL4's own range F1 on 500 practice queries, while J4 also covers kNN, where-at, join and clustering.

### All training seeds (our setup)

Every model we trained with our setup: `rl4_train.py --fixed` on our practice queries, engine seed 42, seeds 1–28. Each row is the round with the best validation F1, which is the model that gets used. "Stops" is how often Agent-Cube stops at the coarse level, one level down or the smallest level for 5,000 sampled cells (rounded). J4 is available for seeds 1–12 and for the seeds from 13–28 that stop more than 80% of the time at the coarse level. Bold marks the best seed on validation (seed 5, the one we report) and the best on J4 (seed 19, found on the test set, so not a valid pick).

| Seed | Validation F1 | Stops: coarse / one down / smallest | J4 at 2% |
|---|---|---|---|
| 1 | 0.637 | 73% / 26% / 0% | 0.261 |
| 2 | 0.567 | 72% / 23% / 5% | 0.227 |
| 3 | 0.645 | 73% / 20% / 6% | 0.293 |
| 4 | 0.560 | 40% / 38% / 23% | 0.126 |
| **5 (best on validation)** | **0.655** | 99% / 1% / 0% | 0.704 |
| 6 | 0.613 | 69% / 28% / 3% | 0.190 |
| 7 | 0.650 | 93% / 6% / 1% | 0.694 |
| 8 | 0.562 | 59% / 33% / 8% | 0.152 |
| 9 | 0.578 | 56% / 39% / 5% | 0.154 |
| 10 | 0.609 | 85% / 11% / 4% | 0.572 |
| 11 | 0.652 | 98% / 2% / 0% | 0.738 |
| 12 | 0.585 | 60% / 27% / 13% | 0.205 |
| 13 | 0.631 | 100% / 0% / 0% | 0.754 |
| 14 | 0.566 | 0% / 0% / 100% | – |
| 15 | 0.649 | 99% / 1% / 0% | 0.733 |
| 16 | 0.539 | 0% / 0% / 100% | – |
| 17 | 0.624 | 88% / 11% / 1% | 0.552 |
| 18 | 0.561 | 10% / 34% / 56% | – |
| **19 (best J4)** | 0.622 | 100% / 0% / 0% | **0.762** |
| 20 | 0.564 | 12% / 38% / 50% | – |
| 21 | 0.631 | 95% / 5% / 0% | 0.633 |
| 22 | 0.593 | 0% / 0% / 100% | – |
| 23 | 0.578 | 59% / 33% / 8% | – |
| 24 | 0.608 | 100% / 0% / 0% | 0.757 |
| 25 | 0.563 | 31% / 41% / 28% | – |
| 26 | 0.636 | 100% / 0% / 0% | 0.743 |
| 27 | 0.551 | 22% / 44% / 34% | – |
| 28 | 0.598 | 73% / 27% / 0% | – |

- **9 of 28 seeds learn to stop** (≥ 90% at the coarse level), 3 always zoom to the smallest level, and the rest are in between. Same code and data; only the seed differs.
- **Stopping seeds validate better:** mean validation F1 0.637 (range 0.608–0.655) against 0.584 (0.539–0.645) for the rest.
- **Stopping decides J4:** all 9 stopping seeds score 0.63–0.76 at 2%; every scored seed that stops less than 85% of the time scores 0.13–0.29.
- **Validation isn't a perfect filter:** seed 3 (stops 73%) ranks 5th on validation but scores only 0.293 J4 at 2%. Its top 4 (seeds 5, 11, 7, 15) all stop ≥ 93%. Among the stopping seeds, validation F1 doesn't predict J4 (seed 24 is lowest on validation of them but scores 0.757).
- The 16 seeds of the authors' training loop on our data are in `data/rl4/experiments/seed_search/screen.csv` (setup "their").

### The same on the authors' own setup

We ran the authors' full pipeline unchanged: their `preprocess.py` on raw GeoLife, their `rl_main_by_data.py`, and their evaluation (`simplification.py` + `range_query.py`). The only changes were a fixed shuffle seed, their UTC+8 time zone, and a different model seed per run.

**What the score is.** The paper's own metric (Table II, Geolife, range queries on the data distribution):
- **Data:** their test set, 1,000 trajectories (files 7,000–7,999 of their preprocessing).
- **Budget:** 0.25% of all points, plus each trajectory's first and last point.
- **Queries:** 100 of their range queries, each one grid cell of 0.02° × 0.02° (about 2.2 × 1.7 km) × 1 week, chosen in proportion to how many trajectories pass through it.
- **Score per query:** F1 between the trajectories the box finds in the original data and in the simplified data (a query that finds none in either counts as 1.0).
- **The number:** the average over the 100 queries.

It is **not comparable to our J4 or engine range F1**, which use different queries, all 16,830 database trajectories and five query types. It only compares runs within the authors' setup.

| Model | Range F1 (authors' metric) | Agent-Cube stops at coarse level |
|---|---|---|
| Paper, Table II (RL4QDTS) | 0.733 ± 0.018 | not reported |
| Authors' shipped model, run by us | **0.735**; 0.734 ± 0.012 over 10 runs | 94% |
| 12 retrained seeds, their unchanged code | 0.512 to 0.780, mean 0.676 ± 0.094 | 0% to 100% |
| Best of the 12 on validation (seed 6) | 0.780; 0.769 ± 0.014 over 10 runs | 86% |

Per seed (range F1 / stops at coarse level): 0.542 / 40%, 0.774 / 97%, 0.643 / 65%, 0.668 / 74%, 0.512 / 0%, 0.780 / 86%, 0.575 / 8%, 0.711 / 77%, 0.779 / 100%, 0.656 / 74%, 0.743 / 98%, 0.730 / 83%. Details: `data/rl4/experiments/setup_B_original/results.md`.

- **The reproduction is faithful.** Their shipped model, in our run of their pipeline, scores 0.735 against the paper's 0.733, and 0.734 ± 0.012 over 10 simplification runs (paper: ± 0.018 over 50).
- **Retraining their unchanged code** gives anywhere from 0.512 to 0.780, depending only on the seed, and the score follows how often the cube agent stops.
- **The paper's error bar hides this.** Re-running one trained model varies by ± 0.012–0.014; retraining varies by ± 0.094, about 7 times more. The paper's ± covers only the first.
- **The paper's number is a good draw.** 4 of 12 retrained seeds reach 0.733; half land between 0.51 and 0.67.
- **Validation would find the good run here too:** the seed with the best validation F1 (seed 6) has the best test score (0.780).

## What we verified (so the numbers can be trusted)

- **Same database and queries as the engine.** RL4 uses the engine's database ids and practice queries (`rl4_export`); all 66,505 tested queries find exactly the engine's trajectories.
- **Test workloads never used in training.** Training uses practice queries (seed + 7001); scoring uses the test workloads (seed 42, or 7 for the second split).
- **Kept points complete and correct.** The engine checks every trajectory, both endpoints, the indices and the exact budget, and refuses otherwise.
- **Our code changes don't change results.** Old vs new code gives byte-identical weights and output.

## How to report RL4

- **Main result:** best of N seeds on validation, stating the procedure, plus the spread over seeds (mean ± std, worst and best). For seed 42: 0.213 / 0.345 / 0.516 / 0.704; the second split (seed 7) confirms it.
- **Explain the variance** with Causes 1–3. It is a property of RL4, not of our setup: the authors' own pipeline shows it too.
- **Optional context:** the shipped weights and the no-cube ablation, as context only.

## Where things are

- **Models:** `data/rl4/model_fixed` (seed 1) and `data/rl4/model_fixed_s2` … `_s28` (of seeds 13–28, only those stopping > 80% were simplified and scored: 13, 15, 17, 19, 21, 24, 26).
- **Kept points:** `data/rl4/kept`.
- **Engine seed 7:** `data/rl4/seed7/`.
- **All experiments behind this file:** setup B, the authors' loop on our data, the seed search, no-cube, shipped weights, the agent swap and the overnight run, each with models, logs and scores. Index: `data/rl4/experiments/README.md`.
- **Code and changes:** `rl4qdts/` (see `CHANGES.md` and `README.md`).
