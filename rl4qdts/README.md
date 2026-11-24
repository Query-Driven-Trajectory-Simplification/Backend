# RL4QDTS

Runs [RL4QDTS](https://github.com/zhengwang125/Query-TS) (Wang et al., ICDE 2024) on our database and
scores the result with our own evaluation, next to the other methods.

The engine's `rl4_export` writes the database's trajectory ids and our practice range queries,
`export_rl4.py` exports those trajectories in RL4's format, `rl4_train.py` trains on our range
queries, `rl4_simplify.py` runs a trained model, and the engine reads the kept points back as an
external method and scores them on its test workloads. `CHANGES.md` lists every way we differ from
the authors' code.

## Layout

- `third_party/RL4QDTS`: the authors' code, a submodule, untouched.
- `rl4qdts/`: ours. `rl4_shims.install()` sets up their code and swaps in `rl4_agents.py` (numpy
  agents instead of their Keras `rl_brain.py`) and `rl4_patches.py` (faster functions in their
  `data_utils.py` and `rl_env.py`). Neither changes what RL4 computes.

## Setup

```bash
git submodule update --init third_party/RL4QDTS

python -m venv rl4qdts/.venv
rl4qdts/.venv/Scripts/python -m pip install -r rl4qdts/requirements.txt
```

No TensorFlow, Julia or PyTorch is needed: only RL4's kNN queries use the last two, and the shims
stand in for them.

## Run

```bash
# 1. The database's trajectory ids and our practice range queries, for the config's seed
#    (--seed N for another; it must match the seed the engine scores with)
./build/engine/rl4_export.exe engine/configs/geolife.json data/rl4/engine

# 2. Export those trajectories, in that order
rl4qdts/.venv/Scripts/python rl4qdts/export_rl4.py data/geolife.parquet data/rl4/db \
  --ids data/rl4/engine/database_ids.csv

# 3. Simplify with the authors' trained models (or your own, see Training)
rl4qdts/.venv/Scripts/python rl4qdts/rl4_simplify.py --db data/rl4/db --label data \
  --inb third_party/RL4QDTS/save/model_data_RL_INB_F1.h5 \
  --oub third_party/RL4QDTS/save/model_data_RL_OUB_F1.h5 \
  --shares 0.0025 0.005 0.01 0.02 --out data/rl4/kept
```

Use the config's `budget_shares` for `--shares`. Step 3 writes `kept_RL4_data_<pct>pct.parquet`
(`--seed 7` adds `_s7` to the name). To score them, add this to the config and rerun the engine:

```json
"external": { "dir": "data/rl4/kept", "methods": ["RL4_data"] }
```

`--fixed` uses the corrected tanh (see `CHANGES.md`); without it you get the authors' numerics.
Give the two variants different `--name`s if you score both. `--no-cube-agent` and
`--no-point-agent` are ablations.

## Training

```bash
rl4qdts/.venv/Scripts/python rl4qdts/export_rl4.py data/geolife.parquet data/rl4/train \
  --ids data/rl4/engine/database_ids.csv --shuffle-seed 1
rl4qdts/.venv/Scripts/python rl4qdts/rl4_train.py --db data/rl4/train --out data/rl4/model --seed 1 \
  --range-queries data/rl4/engine/range_practice.parquet
```

It trains on our practice range queries, never on the test workload the engine scores with. The
database must come from the same `rl4_export` run; training checks that every query finds the
engine's trajectories and stops if not. `--their-queries` trains on the authors' queries instead.
This is the authors' training loop; add `--fixed` for the corrected one (see `CHANGES.md`). Then
simplify with `--inb data/rl4/model/best_INB.h5 --oub data/rl4/model/best_OUB.h5`.
`rl4_train.py --help` lists the sizes and budgets.

## The authors' scripts

`rl4_run.py` runs one of their scripts with our changes, from their folder:

```bash
cd third_party/RL4QDTS
../../rl4qdts/.venv/Scripts/python ../../rl4qdts/rl4_run.py simplification.py
```

They expect the authors' data in `TrajData/` (see their README).
