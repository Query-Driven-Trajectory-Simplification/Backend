# RL4QDTS: all changes to the published code

- **Upstream:** https://github.com/zhengwang125/Query-TS, commit `cb0f695` ("Delete save/1"). The repository has no license file.
- **Paper:** Wang, Long, Cong, Jensen, "Collectively Simplifying Trajectories in a Database: A Query Accuracy Driven Approach", ICDE 2024 ([arXiv 2311.11204](https://arxiv.org/abs/2311.11204)).
- **The authors' code** is the `third_party/RL4QDTS` submodule and stays untouched.
- **Ours** is this folder. All changes live here: `rl4_shims.install()` sets up the authors' code and swaps in our parts while running.
  - `rl4_agents.py` replaces their `rl_brain.py`. It is registered as `rl_brain`, so their scripts get it too.
  - `rl4_patches.py` replaces single functions in their `data_utils.py` and `rl_env.py`.
  - `rl4_train.py` replaces their `rl_main_by_data.py` for our data.

As in MLSimp, `--fixed` turns on the changes that alter results; without it you get the authors'
behaviour. The rest apply always and don't change results.

## Our changes to the authors' code

None of these changes what RL4 computes, except the tanh fix (`--fixed`).

| Their file | Change | Where | Check |
|---|---|---|---|
| `data_utils.py` | In-memory R-trees are bulk loaded. `range_query_operator` reads raw objects and caches the reference database's answers. `Eu`/`Et` compute their 2-D norms without numpy arrays, bit for bit equal to `np.linalg.norm`. | `rl4_patches.py` | A training run and the shipped model's output are bit-identical. |
| `rl_env.py` | `TrajComp._load` caches parsed trajectory files. | `rl4_patches.py` | Same as above |
| `rl_brain.py` | The two DQN agents in numpy instead of Keras/TensorFlow: same network, Huber loss, Adam settings and one-transition-at-a-time replay. Weights are read and written in the same Keras `.h5` layout. | `rl4_agents.py` | Matches Keras to ~3e-7 per weight after 160 updates. About 1,800× faster per update. |
| `rl_brain.py` | `--fixed`: `fast_online_act` uses `np.tanh`. The original tanh returns NaN for inputs above ~710, and the agent then always picks action 0. `rl4_simplify.py --fixed` sets `rl4_agents.FIXED_TANH = True`. | `rl4_agents.py` | For the shipped models on our data, 0 of 72,678 decisions change. |

## Our training changes

These are in `rl4_train.py`, which replaces `rl_main_by_data.py` for our data. `rl_main_by_data.py` is unchanged and kept as the authors' reference.

With `--fixed` (`--keep-stop-bug` and `--keep-reward-bugs` turn single fixes off again):

- **Stop action:** Agent-Cube's stop decision is stored and trained.
- **First reward:** rewards are F1 changes from the start (paper, Eq. 10).
- **Last window:** the last window of steps is rewarded.

Always:

- **Seeds:** each run reseeds per seed and round.
- **Validation:** a fresh cube sampler each time, so every checkpoint is scored the same way.
