"""Trains RL4's agents on our exported data: rl_main_by_data.py's training loop, resized to our database

Differences from the authors' script (the first three only with --fixed; without it, the authors' loop):
- Stop action: Agent-Cube's stop decision (action 8; action 9 in the paper) is stored as a terminal
  transition, so it is trained like the other 8 actions. The authors' loop breaks out before storing
  it, so its weights never leave their random init. At the deepest level there is no decision left,
  so none is asked for (and no stop stored); the path ends on its last zoom.
- First reward: the reward is the F1 *change* since the previous query (paper, Eq. 10). The authors
  start from pre = 0, so the first reward of every round is the whole F1 of the initial database
  (endpoints only), ~100x a normal reward.
- Last window: transitions after the last full window of --reward-every steps get their reward too.
  The authors drop them, so e.g. Agent-Point's terminal transition is never stored.
- Range queries: with --range-queries, the rewards and validation use our range queries and F1
  (the engine's practice workload, see rl4_queries.py) instead of the authors' grid cells. Each
  chunk gets --train-queries of the queries that find one of its trajectories, the same every epoch.
- Database sizes, training budget, rounds and seeds are arguments.
  Training cycles through --chunk sized slices of the first --train trajectories, as the authors
  cycle through 500-trajectory slices; the trajectories after that are the validation set.
- Each round reseeds numpy with (seed, epoch, round). The authors' TrajSam() resets numpy's global
  seed to 0 every round, so all seeds explored with the same random numbers.
- Validation builds a fresh cube sampler each time, so every checkpoint is scored the same way.

Writes one checkpoint pair per round, a log (train_log.csv) and best_INB.h5/best_OUB.h5 (best
validation F1). Simplify with: rl4_simplify.py --inb <out>/best_INB.h5 --oub <out>/best_OUB.h5
"""
import argparse
import collections
import csv
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl4_shims


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--db", required=True,
                    help="folder written by export_rl4.py --ids database_ids.csv --shuffle-seed N")
    ap.add_argument("--out", required=True, help="folder for checkpoints and the log")
    queries = ap.add_mutually_exclusive_group(required=True)
    queries.add_argument("--range-queries",
                         help="our practice range queries (range_practice.parquet from the engine's "
                              "rl4_export, the same run whose database_ids.csv --db was exported from)")
    queries.add_argument("--their-queries", action="store_true",
                         help="the authors' range queries (grid cells) instead of ours")
    ap.add_argument("--fixed", action="store_true",
                    help="fix the stop action, the first reward and the last window (see CHANGES.md)")
    ap.add_argument("--seed", type=int, default=1, help="agents' initial weights and exploration")
    # The authors' sizes: 12 epochs of 500 new trajectories each, 1,000 for validation
    ap.add_argument("--train", type=int, default=6000, help="trajectories 0..train-1 are for training")
    ap.add_argument("--val", type=int, default=1000, help="validation trajectories after --train")
    ap.add_argument("--chunk", type=int, default=500, help="training trajectories per epoch")
    ap.add_argument("--epochs", type=int, default=12,
                    help="one chunk per epoch; train/chunk epochs see every training trajectory once")
    ap.add_argument("--rounds", type=int, default=5, help="rounds per epoch, a checkpoint after each")
    ap.add_argument("--train-queries", type=int, default=200,
                    help="with --range-queries: queries per training chunk")
    ap.add_argument("--val-queries", type=int, default=500,
                    help="with --range-queries: queries for validation")
    ap.add_argument("--ratio", type=float, default=0.0025,
                    help="training budget, share of interior points (our smallest budget share)")
    ap.add_argument("--reward-every", type=int, default=50,
                    help="points added between two rewards (the authors' query_count)")
    # Rarely needed: the code folder and single-fix ablations (see CHANGES.md)
    ap.add_argument("--rl4", default=rl4_shims.UPSTREAM, help=argparse.SUPPRESS)
    ap.add_argument("--keep-stop-bug", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--keep-reward-bugs", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not args.fixed:
        args.keep_stop_bug = args.keep_reward_bugs = True

    rl4_shims.install(os.path.abspath(args.rl4))
    import numpy as np
    import data_utils as F
    from rl_env import TrajComp, TrajSam
    from rl4_agents import AGENT_Sel, AGENT_Sam

    random.seed(args.seed)
    np.random.seed(args.seed)
    K, LEN = 2, 2
    level_start, level_end = 9, 12
    batch_size = 32

    path = os.path.join(args.db, "")
    with open(os.path.join(args.db, "index.csv")) as f:
        trajectory_ids = [int(r["trajectory_id"]) for r in csv.DictReader(f)]
    n = len(trajectory_ids)
    if args.train <= 0 or args.val <= 0 or args.train + args.val > n:
        sys.exit(f"--train {args.train} + --val {args.val} must fit in the {n} trajectories of --db")
    val_end = args.train + args.val
    os.makedirs(args.out, exist_ok=True)

    practice = None
    if args.range_queries:
        import rl4_queries
        try:
            with open(os.path.join(args.db, "export.json")) as f:
                shift_hours = json.load(f)["shift_hours"]
        except FileNotFoundError:
            sys.exit(f"{args.db} has no export.json: export it again with export_rl4.py")
        practice = rl4_queries.PracticeQueries(args.range_queries, trajectory_ids, shift_hours * 3600)
        rl4_queries.use_our_range_queries(F)
        chunk_queries = {}  # first trajectory of a chunk -> its queries, the same every epoch

        def our_queries(s, e, count, traj_set):
            queries = practice.select(s, e, count, np.random.default_rng([args.seed, s]))
            if not queries:
                sys.exit(f"no practice range query finds a trajectory among {s}..{e - 1}")
            rl4_queries.check(F, traj_set, queries)
            return queries

    RL_INB = AGENT_Sel(K * LEN, K, seed=args.seed)
    RL_OUB = AGENT_Sam(2 * 8, 9, seed=args.seed + 1000)

    # --- validation data, built once ---
    val_env = TrajComp(path, args.train, val_end, K, LEN)
    val_set = val_env.ori_traj_set
    val_distri, _, _ = F.get_distribution_feature_data(val_set)
    if practice:
        rtree_ref_val = None  # our queries carry their reference
        val_query = our_queries(args.train, val_end, args.val_queries, val_set)
        source = f"our range queries ({len(val_query)} for validation, of {len(practice)})"
    else:
        rtree_ref_val, _ = F.build_Rtree(val_set)
        _, q1, q2 = F.get_query_workload_data(val_distri)
        val_query = q1 + q2
        source = "the authors' range queries"
    print(f"validation: trajectories {args.train}..{val_end - 1}, {val_env.DB_size} points; "
          f"{'fixed' if args.fixed else 'published'} training loop; {source}")

    def validate():
        """rl_main_by_data.py run_dqn(); also returns the levels of the chosen cubes"""
        env_sam = TrajSam(val_set, level_start, "data")
        samples, levels = [], collections.Counter()
        for _ in range(int(val_env.DB_size * args.ratio)):
            obs = env_sam.starting(level_start)
            while True:
                action = RL_OUB.online_act(obs)
                if action == 8 or env_sam.level_cur == level_end:
                    break
                obs = env_sam.get_agent_oub_state(action)
            samples.append(env_sam.cubeID)
            levels[env_sam.level_cur - 1] += 1  # level_cur is the level of the observed children
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
        return env.submit_query(rtree_ref_val, val_query), levels

    chunks = [(s, min(s + args.chunk, args.train)) for s in range(0, args.train, args.chunk)]
    used = sum(e - s for s, e in chunks[:args.epochs])
    print(f"training: trajectories 0..{args.train - 1} in {len(chunks)} chunks of {args.chunk}; "
          f"{args.epochs} epochs use {used} of them"
          + ("" if args.epochs >= len(chunks) else
             f" (--epochs {len(chunks)} would use all)")
          + f"; {n - val_end} of the {n} trajectories in --db are outside --train and --val",
          flush=True)
    log_path = os.path.join(args.out, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "round", "first_traj", "last_traj", "val_f1",
                                "val_level9", "val_level10", "val_level11", "train_stop_share",
                                "epsilon_cube", "seconds"])
    best = -1.0
    started = time.time()

    # --- rl_main_by_data.py Training(), resized ---
    for epoch in range(args.epochs):
        cs, ce = chunks[epoch % len(chunks)]
        env_sel = TrajComp(path, cs, ce, K, LEN)
        tra_set = env_sel.ori_traj_set
        tra_distri, _, _ = F.get_distribution_feature_data(tra_set)
        if practice:
            if cs not in chunk_queries:
                chunk_queries[cs] = our_queries(cs, ce, args.train_queries, tra_set)
            train_query, rtree_ref_tra = chunk_queries[cs], None
        else:
            _, train_query, _ = F.get_query_workload_data(tra_distri)
            rtree_ref_tra, _ = F.build_Rtree(tra_set)
        buffer_train_size = int(args.ratio * env_sel.DB_size) + 2 * (ce - cs)

        for rnd in range(args.rounds):
            t0 = time.time()
            transitions_oub, samples = [], []
            sam_size = int(env_sel.DB_size * args.ratio)
            env_sam = TrajSam(tra_set, level_start, "data")  # resets numpy's seed to 0 ...
            np.random.seed([args.seed, epoch, rnd])  # ... so reseed per seed and round
            train_stops = 0
            for _ in range(int(sam_size * 1.2)):
                obs_oub, act_oub, trans = [], [], []
                observation = env_sam.starting(level_start)
                obs_oub.append(observation)
                while True:
                    # At the deepest level there is nothing left to decide. The authors still ask
                    # and ignore the answer; a stop stored there would leave the last zoom
                    # non-terminal, bootstrapping from a state whose zooms are never trained.
                    if env_sam.level_cur == level_end and not args.keep_stop_bug:
                        break
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
            if not args.keep_reward_bugs:  # the fix: rewards are F1 changes from the start
                env_sel.submit_query(rtree_ref_tra, train_query)  # sets env_sel.pre

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

                if step % args.reward_every == 0 or (done and not args.keep_reward_bugs):
                    reward = env_sel.submit_query(rtree_ref_tra, train_query)
                    for tr in trans_inb:
                        RL_INB.remember(tr[0], tr[1], reward, tr[2], tr[3])
                    for trans in trans_oub_tmp:
                        for tr in trans:
                            RL_OUB.remember(tr[0], tr[1], reward, tr[2], tr[3])
                    trans_inb.clear()
                    trans_oub_tmp.clear()
                if step % args.reward_every == 0 or done:
                    if len(RL_INB.memory) > batch_size:
                        RL_INB.replay(batch_size)
                    if len(RL_OUB.memory) > batch_size:
                        RL_OUB.replay(batch_size)
                step += 1

            RL_INB.update_target_model()
            RL_OUB.update_target_model()

            val_f1, levels = validate()
            tag = f"e{epoch:02d}_r{rnd}"
            RL_INB.save(os.path.join(args.out, f"ckpt_{tag}_INB.h5"))
            RL_OUB.save(os.path.join(args.out, f"ckpt_{tag}_OUB.h5"))
            if val_f1 > best:
                best = val_f1
                for agent_name in ("INB", "OUB"):
                    shutil.copyfile(os.path.join(args.out, f"ckpt_{tag}_{agent_name}.h5"),
                                    os.path.join(args.out, f"best_{agent_name}.h5"))
            seconds = time.time() - t0
            total = max(1, sum(levels.values()))
            shares = [levels[lv] / total for lv in (9, 10, 11)]
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, rnd, cs, ce - 1, f"{val_f1:.4f}"]
                                       + [f"{s:.3f}" for s in shares]
                                       + [f"{train_stops / max(1, len(samples)):.3f}",
                                          f"{RL_OUB.epsilon:.3f}", f"{seconds:.0f}"])
            print(f"epoch {epoch} round {rnd} (trajectories {cs}..{ce - 1}): val F1 {val_f1:.4f}, "
                  f"cube levels 9/10/11 {shares[0]:.0%}/{shares[1]:.0%}/{shares[2]:.0%}, "
                  f"best {best:.4f}, {seconds:.0f}s (total {(time.time() - started) / 60:.1f} min)",
                  flush=True)

    print(f"done: best validation F1 {best:.4f}; models in {args.out}")


if __name__ == "__main__":
    main()
