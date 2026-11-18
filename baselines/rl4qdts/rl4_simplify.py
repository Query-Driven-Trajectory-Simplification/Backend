"""Runs a trained RL4QDTS model on a database exported by export_rl4.py and writes which
points it keeps, as (trajectory_id, point_index) rows our Backend can score.

The loop is simplification.py's run_dqn() line for line. Differences, all outside their code:
  * the interior budget is ours: round(share * (points - 2 * trajectories)), the same count
    the Backend's total_budget() gives RDP/SED/UNIFORM, so every method keeps equally many points;
  * a fresh TrajSam per budget (their script reuses one, so ID2Grids leaks between budgets);
  * --seed reseeds numpy/random after TrajSam is built, for repeated runs (the paper uses 50);
    without --seed the random stream is exactly theirs;
  * kept points are read from F_ward (its keys are the kept (trajID, pointID) pairs);
  * --no-cube-agent / --no-point-agent run the paper's ablations (Table II): the start cube is
    used as drawn (their "S = 9, E = 10"), and/or the candidate with the largest error is taken.

  python rl4_simplify.py --rl4 <Query-TS dir> --db <export dir> --label data \
      --inb save/model_data_RL_INB_F1.h5 --oub save/model_data_RL_OUB_F1.h5 \
      --shares 0.0025 0.005 0.01 0.02 --out <dir> [--seed 1] [--name RL4_shipped] [--numpy]

Output: <out>/kept_<method>_<pct>pct.parquet, method = <name>[_nocube][_nopoint][_s<seed>].
"""
import argparse
import csv
import math
import os
import time

import rl4_shims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl4", required=True, help="checkout of zhengwang125/Query-TS")
    ap.add_argument("--db", required=True, help="folder written by export_rl4.py")
    ap.add_argument("--label", choices=["data", "gau"], required=True)
    ap.add_argument("--inb", required=True, help="Agent-Point weights (.h5)")
    ap.add_argument("--oub", required=True, help="Agent-Cube weights (.h5)")
    ap.add_argument("--shares", type=float, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--name", help="method name in the output files (default RL4_<label>)")
    ap.add_argument("--numpy", action="store_true", help="run the agents without TensorFlow")
    ap.add_argument("--no-cube-agent", action="store_true", help="ablation: no Agent-Cube")
    ap.add_argument("--no-point-agent", action="store_true", help="ablation: no Agent-Point")
    args = ap.parse_args()

    rl4_shims.install(os.path.abspath(args.rl4), no_tensorflow=args.numpy)
    import random
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rl_env import TrajComp, TrajSam
    from rl_brain import AGENT_Sel, AGENT_Sam

    random.seed(0)  # as simplification.py
    K, LEN = 2, 2  # as simplification.py
    level_start = 9 if args.label == "data" else 10  # as simplification.py
    level_end = 12
    if args.no_cube_agent:
        level_end = level_start + 1  # paper: S = 9, E = 10, so the drawn start cube is used as is

    if args.numpy:
        RL_INB = rl4_shims.numpy_agent(AGENT_Sel, args.inb)
        RL_OUB = rl4_shims.numpy_agent(AGENT_Sam, args.oub)
    else:
        RL_INB, RL_OUB = AGENT_Sel(K * LEN, K), AGENT_Sam(2 * 8, 9)
        RL_INB.load(args.inb)
        RL_OUB.load(args.oub)

    with open(os.path.join(args.db, "index.csv")) as f:
        traj_ids = [int(r["trajectory_id"]) for r in csv.DictReader(f)]
    n = len(traj_ids)
    path = os.path.join(args.db, "")
    loaded = TrajComp(path, 0, n, K, LEN)
    ori_set, db_size = loaded.ori_traj_set, loaded.DB_size
    print(f"{n} trajectories, {db_size} points")
    os.makedirs(args.out, exist_ok=True)

    for share in args.shares:
        interior = int(math.floor(share * (db_size - 2 * n) + 0.5))
        t0 = time.time()
        env_sam = TrajSam(ori_set, level_start, args.label)
        if args.seed is not None:
            np.random.seed(args.seed)
            random.seed(args.seed)

        # --- simplification.py run_dqn(), unchanged apart from the budget ---
        sample_list = []
        for _ in range(interior):
            obs = env_sam.starting(level_start)
            while True:
                action = RL_OUB.fast_online_act(obs)
                if action == 8 or env_sam.level_cur == level_end:
                    break
                obs = env_sam.get_agent_oub_state(action)
            sample_list.append(env_sam.cubeID)
        env_sam.get_ID2Grids(sample_list)
        t1 = time.time()

        env = TrajComp(path, 0, n, K, LEN, '', env_sam.ID2Grids, 'V')
        budget = interior + 2 * n
        for traj in range(n):
            s, e = (traj, 0), (traj, len(env.ori_traj_set[traj]) - 1)
            env.read(s, e)
            env.add_buffer(s, e)
        step = 0
        while len(env.F_ward) < budget:
            obs = env.get_agent_inb_state(sample_list[step])
            env.update_inb(0 if args.no_point_agent else RL_INB.fast_online_act(obs))
            step += 1
        # ----------------------------------------------------------------------
        t2 = time.time()

        kept = sorted(env.F_ward)  # keys are the kept (trajID, pointID) pairs
        method = (args.name or f"RL4_{args.label}") + ("_nocube" if args.no_cube_agent else "") \
            + ("_nopoint" if args.no_point_agent else "") \
            + (f"_s{args.seed}" if args.seed is not None else "")
        name = f"{method}_{share * 100:g}pct"  # the Backend looks for kept_<method>_<pct>pct.parquet
        pq.write_table(pa.table({
            "trajectory_id": pa.array([traj_ids[i] for i, _ in kept], pa.int32()),
            "point_index": pa.array([p for _, p in kept], pa.int64()),
        }), os.path.join(args.out, f"kept_{name}.parquet"))
        print(f"{name}: kept {len(kept)} of {db_size} points (budget {budget}); "
              f"cube sampling {t1 - t0:.1f}s, point insertion {t2 - t1:.1f}s")


if __name__ == "__main__":
    main()
