"""Our range queries for RL4's training, instead of the authors' grid cells

The boxes are the teacher's practice workload, written by the engine's rl4_export
(range_practice.parquet): sized by the config's range section (box_size_m in our local metres,
window_seconds), with the trajectory ids each box finds in the database. use_our_range_queries() switches RL4's R-trees to the same metres and
its range_query_operator to our F1, so a box finds exactly the trajectories the engine finds.
"""
import collections
import math

import numpy as np
import pyarrow.parquet as pq

# --- core/projection.hpp, the same operations in the same order, so the same doubles ---

ORIGIN_LAT, ORIGIN_LON = 39.95, 116.35


def _wgs84_coeffs(lat_deg):
    deg_to_rad = math.pi / 180.0
    semi_major_axis = 6378137.0
    eccentricity_sq = 0.00669437999014132
    rad = lat_deg * deg_to_rad
    sin_lat, cos_lat, term_s, term_c = 0.0, 0.0, rad, 1.0
    for i in range(1, 11):
        sin_lat += term_s
        cos_lat += term_c
        term_s *= -rad * rad / ((2 * i) * (2 * i + 1))
        term_c *= -rad * rad / ((2 * i - 1) * (2 * i))
    w2 = 1.0 - eccentricity_sq * sin_lat * sin_lat
    w = w2
    for _ in range(6):
        w = 0.5 * (w + w2 / w)
    M = semi_major_axis * (1.0 - eccentricity_sq) / (w2 * w)
    N = semi_major_axis / w
    deg_to_rad_sq = deg_to_rad * deg_to_rad
    return (M * deg_to_rad,                                                   # lat1
            N * cos_lat * deg_to_rad,                                         # lon1
            -M * sin_lat * deg_to_rad_sq,                                     # lon_lat_cross
            0.5 * N * sin_lat * cos_lat * deg_to_rad_sq,                      # lon2
            1.5 * M * eccentricity_sq * sin_lat * cos_lat / w2 * deg_to_rad_sq)  # lat2


_LAT1, _LON1, _LON_LAT_CROSS, _LON2, _LAT2 = _wgs84_coeffs(ORIGIN_LAT)


def to_local_metres(lat, lon):
    dlat = lat - ORIGIN_LAT
    dlon = lon - ORIGIN_LON
    return (dlon * (_LON1 + _LON_LAT_CROSS * dlat),
            dlat * (_LAT1 + _LAT2 * dlat) + _LON2 * (dlon * dlon))


def _to_metres(box):
    """An R-tree point entry (lat, lon, t, lat, lon, t) in metres: (x, y, t, x, y, t)"""
    x, y = to_local_metres(box[0], box[1])
    return (x, y, box[2], x, y, box[5])


# --- the queries ---

Query = collections.namedtuple("Query", "box reference")  # box in R-tree order; reference: local ids


class PracticeQueries:
    """The engine's practice range queries over an export_rl4.py database"""

    def __init__(self, path, trajectory_ids, shift_seconds):
        """trajectory_ids: our trajectory id of each RL4 file, from index.csv"""
        table = pq.read_table(path)
        col = {name: table[name].to_numpy() for name in
               ("x_min", "x_max", "y_min", "y_max", "t_min", "t_max")}
        # RL4's timestamps are ours + shift (export_rl4.py), so the windows move with them
        t_min = col["t_min"] + np.int64(shift_seconds)
        t_max = col["t_max"] + np.int64(shift_seconds)
        self.boxes = [(float(a), float(b), int(c), float(d), float(e), int(f)) for a, b, c, d, e, f
                      in zip(col["x_min"], col["y_min"], t_min, col["x_max"], col["y_max"], t_max)]

        reference = table["reference"].combine_chunks()
        ids = reference.values.to_numpy()
        lengths = np.diff(reference.offsets.to_numpy())
        query = np.repeat(np.arange(len(self.boxes)), lengths)
        file_no = {int(tid): i for i, tid in enumerate(trajectory_ids)}
        files = np.array([file_no.get(int(tid), -1) for tid in ids], dtype=np.int64)
        present = files >= 0
        # (query, file) pairs sorted by query: the trajectories each box finds, as RL4 file numbers
        self._query, self._file = query[present], files[present]

    def __len__(self):
        return len(self.boxes)

    def select(self, s, e, count, rng):
        """Up to count random queries that find a trajectory among files s..e-1, with their
        references in TrajComp's local ids (file - s)"""
        inside = (self._file >= s) & (self._file < e)
        q, f = self._query[inside], self._file[inside] - s
        candidates = np.unique(q)
        if len(candidates) > count:
            candidates = np.sort(rng.choice(candidates, count, replace=False))
        start = np.searchsorted(q, candidates, "left")
        end = np.searchsorted(q, candidates, "right")
        return [Query(self.boxes[c], frozenset(f[a:b].tolist()))
                for c, a, b in zip(candidates, start, end)]


# --- RL4 switched to them ---

def _f1(Rtree_ref, Rtree_sim, QUERY, verbose=False):
    """evaluator/range.cpp: mean F1 over the queries; the reference comes with each query"""
    total = 0.0
    for box, reference in QUERY:
        found = set(Rtree_sim.idx.intersection(box, objects="raw"))
        hits = len(found & reference)
        total += 2.0 * hits / (len(reference) + len(found))
    return total / len(QUERY)


def use_our_range_queries(F):
    """Makes data_utils' R-trees store points in our metres and its range_query_operator score our
    queries. Call after rl4_shims.install()."""
    Base = F.Rtree

    class MetricRtree(Base):
        def __init__(self, *args, stream=None):
            if stream is not None:
                stream = ((i, _to_metres(box), obj) for i, box, obj in stream)
            super().__init__(*args, stream=stream)

        def insert(self, id, data, obj):
            super().insert(id, _to_metres(data), obj)

        def delete(self, id, data):
            super().delete(id, _to_metres(data))

    F.Rtree = MetricRtree
    F.range_query_operator = _f1


def check(F, traj_set, queries):
    """Each query finds exactly its engine reference among traj_set, i.e. our projection, time
    shift and box edges match the engine's. Raises otherwise."""
    tree, _ = F.build_Rtree(traj_set)
    wrong = [q for q in queries
             if set(tree.idx.intersection(q.box, objects="raw")) != q.reference]
    if wrong:
        raise RuntimeError(f"{len(wrong)} of {len(queries)} range queries find other trajectories "
                           "than the engine: is the RL4 database exported from database_ids.csv "
                           "of the same rl4_export run, with export_rl4.py's time shift?")
