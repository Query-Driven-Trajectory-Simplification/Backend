"""Trains RL4's agents on our exported data: rl_main_by_data.py's training loop, resized to our database

Differences from the authors' script:
- Agent-Cube's stop action (8) is stored as a terminal transition, so it gets trained. The authors'
  loop breaks out before storing it, so its weights never leave their random init.
  --keep-stop-bug restores the authors' behaviour.
- Database sizes, training budget and number of rounds are arguments instead of constants.
  Training cycles through --chunk sized slices of the first --train trajectories, as the authors
  cycle through 500-trajectory slices; the trajectories after that are the validation set.
- Validation builds a fresh cube sampler each time, so every checkpoint is scored on the same samples.

Writes one checkpoint pair per round, a log (train_log.csv) and best_INB.h5/best_OUB.h5 (best
validation F1). Simplify with: rl4_simplify.py --inb <out>/best_INB.h5 --oub <out>/best_OUB.h5
"""
import argparse
import csv
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl4_shims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl4", required=True, help="checkout of zhengwang125/Query-TS")
    ap.add_argument("--db", required=True, help="folder written by export_rl4.py (use --shuffle-seed)")
    ap.add_argument("--out", required=True, help="folder for checkpoints and the log")
    ap.add_argument("--train", type=int, default=1500, help="trajectories 0..train-1 are for training")
    ap.add_argument("--val", type=int, help="validation trajectories after --train (default: the rest)")
    ap.add_argument("--chunk", type=int, default=500, help="trajectories per training epoch")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=5, help="rounds per epoch, a checkpoint after each")
    ap.add_argument("--ratio", type=float, default=0.0025, help="training budget, share of points")
    ap.add_argument("--query-count", type=int, default=50, help="steps between rewards")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--keep-stop-bug", action="store_true",
                    help="authors' loop: never store the stop action (it stays untrained)")
    args = ap.parse_args()

    rl4_shims.install(os.path.abspath(args.rl4))
    import numpy as np
    import data_utils as F
    from rl_env import TrajComp, TrajSam
    from rl_brain import AGENT_Sel, AGENT_Sam

    random.seed(args.seed)
    np.random.seed(args.seed)
    K, LEN = 2, 2
    level_start, level_end = 9, 12
    batch_size = 32

    path = os.path.join(args.db, "")
    with open(os.path.join(args.db, "index.csv")) as f:
        n = sum(1 for _ in csv.DictReader(f))
    val_end = n if args.val is None else min(n, args.train + args.val)
    if not 0 < args.train < val_end:
        sys.exit(f"--train {args.train} leaves no validation data ({n} trajectories)")
    os.makedirs(args.out, exist_ok=True)

    RL_INB = AGENT_Sel(K * LEN, K)
    RL_OUB = AGENT_Sam(2 * 8, 9)

    # --- validation data, built once ---
    val_env = TrajComp(path, args.train, val_end, K, LEN)
    val_set = val_env.ori_traj_set
    rtree_ref_val, _ = F.build_Rtree(val_set)
    val_distri, _, _ = F.get_distribution_feature_data(val_set)
    _, q1, q2 = F.get_query_workload_data(val_distri)
    val_query = q1 + q2
    print(f"validation: trajectories {args.train}..{val_end - 1}, {val_env.DB_size} points")

    def validate():
        """rl_main_by_data.py run_dqn(); also returns the share of samples ended by the stop action"""
        env_sam = TrajSam(val_set, level_start, "data")
        samples, stops = [], 0
        for _ in range(int(val_env.DB_size * args.ratio)):
            obs = env_sam.starting(level_start)
            while True:
                action = RL_OUB.online_act(obs)
                if action == 8 or env_sam.level_cur == level_end:
                    stops += action == 8
                    break
                obs = env_sam.get_agent_oub_state(action)
            samples.append(env_sam.cubeID)
        env_sam.get_ID2Grids(samples)
        env = TrajComp(path, args.train, val_end, K, LEN, val_distri, env_sam.ID2Grids, "V")
        budget = int(args.ratio * env.DB_size) + 2 * len(val_set)
        for t in range(len(val_set)):
            s, e = (t, 0), (t, len(env.ori_traj_set[t]) - 1)
            env.read(s, e)
            env.add_buffer(s, e)
        step = 0
        while len(env.F_ward) < budget:
            env.update_inb(RL_INB.online_act(env.get_agent_inb_state(samples[step])))
            step += 1
        return env.submit_query(rtree_ref_val, val_query), stops / max(1, len(samples))

    chunks = [(s, min(s + args.chunk, args.train)) for s in range(0, args.train, args.chunk)]
    log_path = os.path.join(args.out, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "round", "first_traj", "last_traj", "val_f1",
                                "val_stop_share", "train_stop_share", "epsilon_cube", "seconds"])
    best = -1.0
    started = time.time()

    # --- rl_main_by_data.py Training(), resized ---
    for epoch in range(args.epochs):
        cs, ce = chunks[epoch % len(chunks)]
        env_sel = TrajComp(path, cs, ce, K, LEN)
        tra_set = env_sel.ori_traj_set
        tra_distri, _, _ = F.get_distribution_feature_data(tra_set)
        _, train_query, _ = F.get_query_workload_data(tra_distri)
        rtree_ref_tra, _ = F.build_Rtree(tra_set)
        buffer_train_size = int(args.ratio * env_sel.DB_size) + 2 * (ce - cs)

        for rnd in range(args.rounds):
            t0 = time.time()
            transitions_oub, samples = [], []
            sam_size = int(env_sel.DB_size * args.ratio)
            env_sam = TrajSam(tra_set, level_start, "data")
            train_stops = 0
            for _ in range(int(sam_size * 1.2)):
                obs_oub, act_oub, trans = [], [], []
                observation = env_sam.starting(level_start)
                obs_oub.append(observation)
                while True:
                    action = RL_OUB.act(observation)
                    act_oub.append(action)
                    if action == 8 or env_sam.level_cur == level_end:
                        if action == 8:
                            train_stops += 1
                            if not args.keep_stop_bug:  # the fix: store the stop decision too
                                trans.append([observation, 8, observation, True])
                        break
                    observation = env_sam.get_agent_oub_state(action)
                    obs_oub.append(observation)
                    if len(obs_oub) >= 2:
                        trans.append([obs_oub[-2], act_oub[-1], obs_oub[-1], False])
                if trans:
                    trans[-1][-1] = True
                transitions_oub.append(trans)
                samples.append(env_sam.cubeID)
            env_sam.get_ID2Grids(samples)

            env_sel = TrajComp(path, cs, ce, K, LEN, tra_distri, env_sam.ID2Grids, "T")
            for t in range(ce - cs):
                s, e = (t, 0), (t, len(env_sel.ori_traj_set[t]) - 1)
                env_sel.read(s, e)
                env_sel.add_buffer(s, e)

            obs_inb, act_inb, trans_inb, trans_oub_tmp = [], [], [], []
            step = 0
            while len(env_sel.F_ward) < buffer_train_size:
                trans_oub_tmp.append(transitions_oub[step])
                observation = env_sel.get_agent_inb_state(samples[step])
                obs_inb.append(observation)
                action = RL_INB.act(observation)
                env_sel.update_inb(action)
                done = len(env_sel.F_ward) == buffer_train_size
                if len(obs_inb) >= 2:
                    trans_inb.append([obs_inb[-2], act_inb[-1], obs_inb[-1], done])
                act_inb.append(action)

                if step % args.query_count == 0:
                    reward = env_sel.submit_query(rtree_ref_tra, train_query)
                    for tr in trans_inb:
                        RL_INB.remember(tr[0], tr[1], reward, tr[2], tr[3])
                    for trans in trans_oub_tmp:
                        for tr in trans:
                            RL_OUB.remember(tr[0], tr[1], reward, tr[2], tr[3])
                    trans_inb.clear()
                    trans_oub_tmp.clear()
                if step % args.query_count == 0 or done:
                    if len(RL_INB.memory) > batch_size:
                        RL_INB.replay(batch_size)
                    if len(RL_OUB.memory) > batch_size:
                        RL_OUB.replay(batch_size)
                step += 1

            RL_INB.update_target_model()
            RL_OUB.update_target_model()

            val_f1, val_stops = validate()
            tag = f"e{epoch:02d}_r{rnd}"
            RL_INB.save(os.path.join(args.out, f"ckpt_{tag}_INB.h5"))
            RL_OUB.save(os.path.join(args.out, f"ckpt_{tag}_OUB.h5"))
            if val_f1 > best:
                best = val_f1
                for agent_name in ("INB", "OUB"):
                    shutil.copyfile(os.path.join(args.out, f"ckpt_{tag}_{agent_name}.h5"),
                                    os.path.join(args.out, f"best_{agent_name}.h5"))
            seconds = time.time() - t0
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, rnd, cs, ce - 1, f"{val_f1:.4f}", f"{val_stops:.3f}",
                                        f"{train_stops / max(1, len(samples)):.3f}",
                                        f"{RL_OUB.epsilon:.3f}", f"{seconds:.0f}"])
            print(f"epoch {epoch} round {rnd} (trajectories {cs}..{ce - 1}): val F1 {val_f1:.4f}, "
                  f"stop share {val_stops:.1%}, best {best:.4f}, {seconds:.0f}s "
                  f"(total {(time.time() - started) / 60:.0f} min)", flush=True)

    print(f"done: best validation F1 {best:.4f}; models in {args.out}")


if __name__ == "__main__":
    main()
