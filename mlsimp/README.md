# MLSimp

Runs the pretrained GeoLife model from [MLSimp](https://github.com/yumengs-exp/MLSimp) on our
database and scores the result with our own evaluation, next to the teacher and the baselines.

The engine exports the database, `score_points.py` scores every point with MLSimp's model,
`select_points.py` picks the kept points, and the engine reads them back as arms.

## Setup

MLSimp is the `third_party/MLSimp` submodule and needs Git LFS.

```bash
git lfs install
git submodule update --init third_party/MLSimp

python -m venv mlsimp/.venv
mlsimp/.venv/Scripts/python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
mlsimp/.venv/Scripts/python -m pip install -r mlsimp/requirements.txt
```

If the submodule update fails with "active `post-checkout` hook found", prefix it with
`GIT_CLONE_PROTECTION_ACTIVE=false`.

## With teacher_eval

```bash
./build/rel/engine/teacher_eval.exe --seeds 7,99,123 --export-db data/mlsimp_teacher

for s in 7 99 123; do
  mlsimp/.venv/Scripts/python mlsimp/score_points.py --data data/mlsimp_teacher/seed$s
  mlsimp/.venv/Scripts/python mlsimp/select_points.py --data data/mlsimp_teacher/seed$s
done

./build/rel/engine/teacher_eval.exe --seeds 7,99,123 --no-teacher --no-baselines \
  --external-dir data/mlsimp_teacher \
  --external MLSimp
```

Use the same `--config`, `--cache`, `--shares` and `--out` as the teacher run. Scoring takes about
30 minutes per seed on 16 cores and resumes if stopped.

## With run.exe (Travis)

```bash
./build/rel/engine/export_db.exe engine/configs/geolife_mlsimp.json
mlsimp/.venv/Scripts/python mlsimp/score_points.py --data data/mlsimp
mlsimp/.venv/Scripts/python mlsimp/select_points.py --data data/mlsimp
./build/rel/engine/run.exe engine/configs/geolife_mlsimp.json
```

Rerun the export and both scripts when the seed, split or budget shares change.

## Selection

The arm is called `MLSimp`. As in MLSimp, each point's score is half the model score and half
whether it lies in one of the range boxes, and the kept points are drawn with probability
proportional to the score over the whole database. It keeps exactly the budget and both endpoints
of every trajectory.

## Differences from MLSimp's validation.py

- The range part uses 100 boxes like MLSimp, drawn from the teacher's training range queries
  (`teacher_eval`) or generated with a separate seed (`export_db`). MLSimp uses its test queries.
- Scores are normalized over the whole database only. MLSimp first divides each trajectory's
  scores by their L2 norm, which with our trajectory lengths (100 to 12,488 points) favors short
  trajectories.
- The budget is exact. MLSimp adds the endpoints on top of its budget.
- The model runs in eval mode, so the scores are deterministic.
- Trajectories of any length are scored, using MLSimp's 500-point windows with 100 points overlap.
