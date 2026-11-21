"""Stand-ins so RL4's code can be imported without Julia and t2vec (only its kNN queries use them)"""
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


def install(rl4_dir):
    sys.dont_write_bytecode = True  # no __pycache__ inside the RL4 folder
    if rl4_dir not in sys.path:
        sys.path.insert(0, rl4_dir)
    # data_utils: "from julia import Main; Main.include('call.jl')"
    _module("julia", Main=_Anything())
    # data_utils: "from t2vec_utils import args, model_init, submit" (needs torch + a checkpoint)
    _module("t2vec_utils", args=types.SimpleNamespace(), model_init=lambda a: None,
            submit=lambda *a, **k: None)
