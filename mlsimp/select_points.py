"""Turns MLSimp's point scores into kept points for the engine."""

import argparse
from pathlib import Path

import numpy as np

AD_RATIO = 0.5


def mlsimp_scores(model, hits):
    model = model.astype(np.float64)
    model /= model.sum()
    adjust = hits.astype(np.float64)
    if adjust.sum() > 0:
        adjust /= adjust.sum()
    return (1 - AD_RATIO) * model + AD_RATIO * adjust


def sample(offsets, scores, count, rng):
    interior = np.ones(offsets[-1], dtype=bool)
    interior[offsets[:-1]] = False
    interior[offsets[1:] - 1] = False
    interior_index = np.flatnonzero(interior)
    weights = scores[interior_index]
    with np.errstate(divide="ignore"):
        keys = np.log(rng.random(len(weights))) / weights
    picked = np.argpartition(-keys, count - 1)[:count] if count else np.empty(0, dtype=np.int64)
    endpoints = np.concatenate((offsets[:-1], offsets[1:] - 1))
    return np.sort(np.concatenate((endpoints, interior_index[picked]))).astype(np.uint64)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True,
                        help="export folder (selection_dir, or <export-db>/seed<s>)")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    args = parser.parse_args()

    offsets = np.fromfile(args.data / "offsets.bin", dtype=np.uint64).astype(np.int64)
    n = len(offsets) - 1
    assert (np.diff(offsets) >= 2).all(), "every trajectory needs two endpoints"
    scores = np.load(args.data / "scores.npy")
    assert len(scores) == offsets[-1], "scores.npy doesn't match the database, rerun score_points"
    done = args.data / "scores_done.txt"
    assert done.exists() and int(done.read_text()) == n, \
        "score_points hasn't finished every trajectory yet"
    hits = np.fromfile(args.data / "range_hits.bin", dtype=np.uint8)
    assert len(hits) == offsets[-1], "range_hits.bin doesn't match the database, rerun the export"
    scores = mlsimp_scores(scores, hits)

    for budget_file in sorted(args.data.glob("budgets_*pct.bin")):
        share = budget_file.stem.removeprefix("budgets_")
        budgets = np.fromfile(budget_file, dtype=np.uint64).astype(np.int64)
        assert len(budgets) == n
        count = int(budgets.sum()) - 2 * n
        kept = sample(offsets, scores, count, np.random.default_rng(args.seed))
        assert len(kept) == budgets.sum()
        kept.tofile(args.data / f"MLSimp_{share}.bin")
        print(f"{share}: {len(kept)} points")


if __name__ == "__main__":
    main()
