import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

WEEK = 7 * 24 * 60 * 60
FIRST_MONDAY = 4 * 24 * 60 * 60

parser = argparse.ArgumentParser()
parser.add_argument("--parquet", default="data/geolife.parquet")
parser.add_argument("--out", default="output/geojson")
parser.add_argument("--cell", type=float, default=500.0, help="cell size in metres")
args = parser.parse_args()

table = pq.read_table(args.parquet, columns=["trajectory_id", "t", "lat", "lon"])
tid = table["trajectory_id"].to_numpy().astype(np.int64)
t = table["t"].to_numpy()
lat = table["lat"].to_numpy()
lon = table["lon"].to_numpy()

lat_min, lat_max = np.percentile(lat, [0.5, 99.5])
lon_min, lon_max = np.percentile(lon, [0.5, 99.5])
lat_step = args.cell / 111_320
lon_step = lat_step / math.cos(math.radians((lat_min + lat_max) / 2))

keep = (lat >= lat_min) & (lat < lat_max) & (lon >= lon_min) & (lon < lon_max)
tid, t = tid[keep], t[keep]
row = ((lat[keep] - lat_min) // lat_step).astype(np.int64)
col = ((lon[keep] - lon_min) // lon_step).astype(np.int64)
cols = int((lon_max - lon_min) // lon_step) + 1
cell = row * cols + col
week = (t - FIRST_MONDAY) // WEEK


def counts(key):
    keys, points = np.unique(key, return_counts=True)
    ids = tid.max() + 1
    traj_keys, trajectories = np.unique(np.unique(key * ids + tid) // ids, return_counts=True)
    return keys.tolist(), points.tolist(), trajectories.tolist()


def feature(c, properties):
    south = round(lat_min + (c // cols) * lat_step, 5)
    west = round(lon_min + (c % cols) * lon_step, 5)
    north = round(south + lat_step, 5)
    east = round(west + lon_step, 5)
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "Polygon", "coordinates": [[
            [west, south], [east, south], [east, north], [west, north], [west, south]]]},
    }


features = [feature(c, {"points": p, "trajectories": n}) for c, p, n in zip(*counts(cell))]
total = len(features)

weeks = int(week.max()) + 1
for key, p, n in zip(*counts(cell * weeks + week)):
    features.append(feature(key // weeks, {"week": int(key % weeks) * WEEK + FIRST_MONDAY,
                                           "points": p, "trajectories": n}))

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
with gzip.open(out / "heatmap.geojson.gz", "wt") as f:
    json.dump({"type": "FeatureCollection", "features": features}, f, separators=(",", ":"))
print(f"{total} cells in total and {len(features) - total} cell-weeks written to "
      f"{out / 'heatmap.geojson.gz'}")
