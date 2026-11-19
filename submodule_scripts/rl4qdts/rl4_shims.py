"""Stand-ins so RL4's code can be imported without Julia, t2vec or (optionally) TensorFlow"""
import sys
import types


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


def install(rl4_dir, no_tensorflow=False):
    if rl4_dir not in sys.path:
        sys.path.insert(0, rl4_dir)
    # data_utils: "from julia import Main; Main.include('call.jl')"
    _module("julia", Main=_Anything())
    # data_utils: "from t2vec_utils import args, model_init, submit" (needs torch + a checkpoint)
    _module("t2vec_utils", args=types.SimpleNamespace(), model_init=lambda a: None,
            submit=lambda *a, **k: None)
    if no_tensorflow:
        anything = _Anything()
        _module("tensorflow", set_random_seed=lambda s: None, where=anything)
        keras = _module("keras")
        for sub in ("models", "layers", "optimizers"):
            m = _module("keras." + sub, Sequential=anything, Dense=anything,
                        BatchNormalization=anything, Activation=anything, Adam=anything)
            setattr(keras, sub, m)
        keras.backend = anything
        sys.modules["keras.backend"] = anything


def numpy_agent(agent_class, h5_path):
    """An RL4QDTS agent whose fast_online_act works without Keras: weights read with h5py."""
    import h5py
    import numpy as np
    agent = agent_class.__new__(agent_class)  # skips _build_model (Keras)
    weights = []
    with h5py.File(h5_path, "r") as f:
        for layer in f.attrs["layer_names"]:
            group = f[layer]
            for name in group.attrs.get("weight_names", []):
                weights.append(np.array(group[name]))
    agent.w1, agent.b1, agent.w2, agent.b2 = weights
    return agent
