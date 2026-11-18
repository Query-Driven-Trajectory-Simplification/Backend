"""Runs one of RL4QDTS's own scripts unchanged: training (rl_main_by_data.py,
rl_main_by_gau.py) or their simplification.py / range_query.py for the reproduction check.

  cd <their repo or a copy> && python /path/to/rl4_run.py <script.py>

The training script reads ./TrajData/Geolife_out/0..6999 (files 0-5999: 12 training databases of
500 trajectories; 6000-6999: validation) and writes ./save/<time>_<label>_RL_{INB,OUB}_F1_<val F1>.h5.
So: export >= 7000 training trajectories with export_rl4.py --shuffle-seed, point
<work>/TrajData/Geolife_out at that folder, and run this from <work> (a copy of their repo).

  cd <work> && python /path/to/rl4_run.py rl_main_by_data.py

Works with TF 1.15 + Keras 2.2/2.3 (what they used). With TF 2.x + Keras 2.x (<= 2.15) two
things are patched here: the missing TF1 call tf.set_random_seed, and Adam, because Keras
2.11+ silently drops Adam(lr=0.01) and trains with 0.001. Keras 3 does not work.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl4_shims

rl4_shims.install(os.getcwd())
import tensorflow as tf

if not hasattr(tf, "set_random_seed"):  # TF 2.x
    tf.set_random_seed = tf.compat.v1.set_random_seed
    import keras.optimizers
    if hasattr(keras.optimizers, "legacy"):  # Keras 2.11-2.15: the new Adam ignores lr=
        keras.optimizers.Adam = keras.optimizers.legacy.Adam

os.makedirs("save", exist_ok=True)
runpy.run_path(sys.argv[1], run_name="__main__")
