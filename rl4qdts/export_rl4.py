"""Exports our data in RL4's format, with timestamps shifted -8 h to match RL4 by default"""
import argparse
import os

import numpy as np
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("out_dir")
    ap.add_argument("--ids", help="csv with a trajectory_id column; keeps that order")
    ap.add_argument("--shuffle-seed", type=int, help="shuffle trajectory order (training data)")
    ap.add_argument("--shift-hours", type=int, default=-8,
                    help="added to every timestamp; -8 = RL4's own convention (default)")
    args = ap.parse_args()

    table = pq.read_table(args.parquet, columns=["trajectory_id", "t", "lat", "lon"])
    tid = table["trajectory_id"].to_numpy()
    t = table["t"].to_numpy() + np.int64(args.shift_hours * 3600)
    lat = table["lat"].to_numpy()
    lon = table["lon"].to_numpy()

    # Rows of one trajectory are contiguous and in time order in our files.
    starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    ends = np.r_[starts[1:], len(tid)]
    by_id = {int(tid[s]): (s, e) for s, e in zip(starts, ends)}

    if args.ids:
        import csv
        with open(args.ids) as f:
            order = [int(r["trajectory_id"]) for r in csv.DictReader(f)]
    else:
        order = list(by_id)
    if args.shuffle_seed is not None:
        np.random.default_rng(args.shuffle_seed).shuffle(order)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "index.csv"), "w") as index:
        index.write("file_no,trajectory_id,n_points\n")
        for file_no, traj_id in enumerate(order):
            s, e = by_id[traj_id]
            with open(os.path.join(args.out_dir, str(file_no)), "w") as out:
                # Python floats: repr keeps every digit, so the text round trip is exact
                for la, lo, ti in zip(lat[s:e].tolist(), lon[s:e].tolist(), t[s:e].tolist()):
                    out.write(f"{la!r} {lo!r} {ti}\n")
            index.write(f"{file_no},{traj_id},{e - s}\n")
    print(f"wrote {len(order)} trajectories to {args.out_dir} (time shift {args.shift_hours} h)")


if __name__ == "__main__":
    main()
