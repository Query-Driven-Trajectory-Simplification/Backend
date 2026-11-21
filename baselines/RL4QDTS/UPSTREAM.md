# RL4QDTS: where this code comes from

- **Upstream:** https://github.com/zhengwang125/Query-TS, commit `cb0f695` ("Delete save/1"). The repository has no license file.
- **Paper:** Wang, Long, Cong, Jensen, "Collectively Simplifying Trajectories in a Database: A Query Accuracy Driven Approach", ICDE 2024 ([arXiv 2311.11204](https://arxiv.org/abs/2311.11204)).
- **Ours:** `scripts/` and this file. Everything else is the authors' code.
- **History:** the commit "Vendor RL4QDTS (Query-TS cb0f695) into baselines/RL4QDTS" holds the unmodified upstream code. Every later commit is one of our changes.

To see our edits to the authors' code only:

```powershell
git log --oneline -- baselines/RL4QDTS
git diff b3a4dd6 -- baselines/RL4QDTS ':!baselines/RL4QDTS/scripts' ':!baselines/RL4QDTS/UPSTREAM.md'
```

## Our changes to the authors' code

None of these changes what RL4 computes.

| File | Change | Check |
|---|---|---|
| `data_utils.py` | In-memory R-trees are bulk loaded. `range_query_operator` reads raw objects and caches the reference database's answers. `Eu`/`Et` compute their 2-D norms without numpy arrays, bit for bit equal to `np.linalg.norm`. | A training run and the shipped model's output are bit-identical. |
| `rl_env.py` | `TrajComp` caches parsed trajectory files. | Same as above |
| `rl_brain.py` | The two DQN agents in numpy instead of Keras/TensorFlow: same network, Huber loss, Adam settings and one-transition-at-a-time replay. Weights are read and written in the same Keras `.h5` layout. | Matches Keras to ~3e-7 per weight after 160 updates. About 1,800× faster per update. |
| `rl_brain.py` | `fast_online_act` uses `np.tanh`. The original tanh returns NaN for inputs above ~710, and the agent then always picks action 0. `rl_brain.LEGACY_TANH = True` (`rl4_simplify.py --legacy`) restores it. | For the shipped models on our data, 0 of 72,678 decisions change. |

## Our training changes

These are in `scripts/rl4_train.py`, which replaces `rl_main_by_data.py` for our data. `rl_main_by_data.py` is unchanged and kept as the authors' reference.

- **Stop action:** Agent-Cube's stop decision is stored and trained.
- **First reward:** rewards are F1 changes from the start (paper, Eq. 10).
- **Last window:** the last window of steps is rewarded.
- **Seeds:** each run reseeds per seed and round.
- **Authors' loop:** `--legacy` gives the authors' training loop.
