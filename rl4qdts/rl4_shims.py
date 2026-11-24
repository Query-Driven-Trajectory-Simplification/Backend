"""Sets up the authors' RL4 code (third_party/RL4QDTS) with our changes

- Stand-ins so it can be imported without Julia and t2vec (only its kNN queries use them).
- rl4_agents.py, our numpy agents, replaces their rl_brain.py (Keras/TensorFlow).
- rl4_patches.py replaces some functions in their data_utils.py and rl_env.py.
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(os.path.dirname(HERE), "third_party", "RL4QDTS")  # the authors' code


class _Anything:
    """Accepts any attribute access or call and returns itself."""

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def install(rl4_dir=UPSTREAM):
    sys.dont_write_bytecode = True  # no __pycache__ inside the RL4 folders
    for d in (os.path.abspath(rl4_dir), HERE):
        if d not in sys.path:
            sys.path.insert(0, d)
    # data_utils: "from julia import Main; Main.include('call.jl')"
    _module("julia", Main=_Anything())
    # data_utils: "from t2vec_utils import args, model_init, submit" (needs torch + a checkpoint)
    _module("t2vec_utils", args=types.SimpleNamespace(), model_init=lambda a: None,
            submit=lambda *a, **k: None)
    # their scripts "from rl_brain import AGENT_Sel, AGENT_Sam" get our agents
    import rl4_agents
    sys.modules["rl_brain"] = rl4_agents
    import rl4_patches
    rl4_patches.apply()
