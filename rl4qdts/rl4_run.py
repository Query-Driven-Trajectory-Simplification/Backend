"""Runs one of RL4's own scripts (training, simplification, range query) from third_party/RL4QDTS"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl4_shims

rl4_shims.install(os.getcwd())
os.makedirs("save", exist_ok=True)
runpy.run_path(sys.argv[1], run_name="__main__")
