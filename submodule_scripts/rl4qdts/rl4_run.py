"""Runs one of RL4's own scripts unchanged (training, simplification, range query) on TF 2.15"""
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
