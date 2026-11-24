"""Our changes to the authors' data_utils.py and rl_env.py, swapped in by rl4_shims.install()

The authors' modules are imported from third_party/RL4QDTS and only the functions below are
replaced. Their code calls these through the module (F.Eu, F.build_Rtree, ...), so it uses ours.
None of them changes what RL4 computes; see CHANGES.md.
"""
import math

from rtree import index

_applied = False


# --- data_utils: Eu and Et without numpy arrays ---

_SPLIT = 134217729.0  # 2**27 + 1, Veltkamp splitting


def _norm2(a, b):
    """np.linalg.norm(np.array([a, b])) without numpy, bit for bit: numpy computes it as a BLAS dot,
    sqrt(fma(b, b, a*a)); the fma is emulated exactly with Dekker's two-product and two-sum"""
    c = a * a
    p = b * b
    t = _SPLIT * b
    hi = t - (t - b)
    lo = b - hi
    e = ((hi * hi - p) + 2.0 * hi * lo) + lo * lo  # b*b == p + e exactly
    s = p + c
    bb = s - p
    t2 = (p - (s - bb)) + (c - bb)  # p + c == s + t2 exactly
    return math.sqrt(s + (t2 + e))


def Eu(segment):
    ps = segment[0]
    pe = segment[-1]
    syn_time = segment[1][2]
    time_ratio = 1 if (pe[2]- ps[2]) == 0  else (syn_time-ps[2]) / (pe[2]-ps[2])
    syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
    syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
    e = _norm2(segment[1][0] - syn_x, segment[1][1] - syn_y)
    return e


def Et(segment):
    ps = segment[0]
    pm = segment[1]
    pe = segment[-1]
    A = pe[1] - ps[1]
    B = ps[0] - pe[0]
    C = pe[0] * ps[1] - ps[0] * pe[1]
    if A == 0 and B == 0:
        return 0.0
    else:
        x = (B*B*pm[0] - A*B*pm[1] - A*C)/(A*A + B*B)
        y = (-A*B*pm[0] + A*A*pm[1] - B*C)/(A*A + B*B)
        dt = pe[2]-ps[2]
        # numpy gave inf (with a warning) for equal timestamps; the time error is then |ps_t - pm_t|
        speed = _norm2(ps[0] - pe[0], ps[1] - pe[1])/dt if dt != 0 else math.inf
        return abs(ps[2] + _norm2(ps[0] - x, ps[1] - y)/speed - pm[2])


# --- data_utils: R-trees bulk loaded, reference answers cached ---

def _rtree_class(Original):
    class Rtree(Original):
        """The authors' Rtree, plus bulk loading and a cache of query answers"""

        def __init__(self, *args, stream=None):
            self.cache = {}  # query box -> result, kept by range_query_operator until the tree changes
            if stream is None:
                super().__init__(*args)
            else:  # bulk loading: ~20x faster than inserting points one by one
                self.p = index.Property()
                self.p.dimension = 3
                self.idx = index.Index(stream, properties=self.p)

        def insert(self, id, data, obj):
            self.cache.clear()
            super().insert(id, data, obj)

        def delete(self, id, data):
            self.cache.clear()
            super().delete(id, data)

    return Rtree


def _build_rtree(F, original):
    def build_Rtree(DB, filename=''):
        if filename=='' and any(len(traj) > 0 for traj in DB):  # in-memory tree: bulk load, same ids and objects
            def stream():
                c = 0
                for trajID in range(len(DB)):
                    for point in DB[trajID]:
                        yield (c, (point[0],point[1],point[2],point[0],point[1],point[2]), trajID)
                        c += 1
            Rtree_ = F.Rtree(stream=stream())
            delete_rec = {}
            c = 0
            for trajID in range(len(DB)):
                for pointID in range(len(DB[trajID])):
                    delete_rec[(trajID, pointID)] = c
                    c += 1
            return Rtree_, delete_rec
        return original(DB, filename)  # a tree saved to file: the authors' code

    return build_Rtree


def range_query_operator(Rtree_ref, Rtree_sim, QUERY, verbose=False):
    #query on sim_DB
    F1 = []
    x_length, y_length, t_length = 0.02, 0.02, 3600*24*7
    Xmin, Ymin, Tmin = 1.044024, -179.9695933, 1176341492
    ee, ef, fe, ff = 0, 0, 0, 0
    for i in range(len(QUERY)):
        (x_idx, y_idx, t_idx) = QUERY[i]
        x_center, y_center, t_center = Xmin+x_length*(0.5+x_idx), Ymin+y_length*(0.5+y_idx), Tmin+t_length*(0.5+t_idx)
        box = (x_center-x_length/2,
               y_center-y_length/2,
               t_center-t_length/2,
               x_center+x_length/2,
               y_center+y_length/2,
               t_center+t_length/2)
        # objects='raw' returns the stored trajIDs directly; the reference database does not change
        # between calls, so its answers are cached (Rtree.insert/delete clear the cache)
        A = Rtree_ref.cache.get(box)
        if A is None:
            A = Rtree_ref.cache[box] = set(Rtree_ref.idx.intersection(box, objects='raw'))
        B = set(Rtree_sim.idx.intersection(box, objects='raw'))

        if verbose:
            if A != B:
                print('A & B', A, B)
        if A == set() and B == set():
            ee += 1
            F1.append(1.0)
        if A == set() and B != set():
            ef += 1
            F1.append(0.0)
        if A != set() and B == set():
            fe += 1
            F1.append(0.0)
        if A != set() and B != set():
            ff += 1
            P = len(A&B)/len(B)
            R = len(A&B)/len(A)
            if (P+R) == 0:
                F1.append(0.0)
            else:
                F1.append((2*P*R)/(P+R))
    return sum(F1)/len(F1)


# --- rl_env: TrajComp caches parsed files ---

_TRAJ_CACHE = {}  # file path -> parsed trajectory


def _load(self, path, s, e):
    import data_utils as F
    self.ori_traj_set = []
    self.DB_size = 0
    for num in range(s, e):
        # parsed files are cached: training re-creates TrajComp several times per round and
        # nothing modifies the loaded trajectories
        traj = _TRAJ_CACHE.get(path + str(num))
        if traj is None:
            traj = _TRAJ_CACHE[path + str(num)] = F.to_traj(path + str(num))
        self.ori_traj_set.append(traj)
        self.DB_size += len(traj)


def apply():
    """Replaces the functions above in the authors' modules (once)"""
    global _applied
    if _applied:
        return
    import data_utils as F
    import rl_env

    F.Eu, F.Et = Eu, Et
    F.Rtree = _rtree_class(F.Rtree)
    F.build_Rtree = _build_rtree(F, F.build_Rtree)
    F.range_query_operator = range_query_operator
    rl_env.TrajComp._load = _load
    _applied = True
