"""Retrains every part of MLSimp on one seed's database, resumable."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

MLSIMP = Path(__file__).resolve().parents[1] / "third_party" / "MLSimp"


def load_database(folder):
    offsets = np.fromfile(folder / "offsets.bin", dtype=np.uint64).astype(np.int64)
    lat = np.fromfile(folder / "lat.bin", dtype=np.float64)
    lon = np.fromfile(folder / "lon.bin", dtype=np.float64)
    t = np.fromfile(folder / "t.bin", dtype=np.int64)
    assert offsets[-1] == len(lat) == len(lon) == len(t), "database files don't match"
    return offsets, lat, lon, t


def subset(db, picked):
    offsets = db[0]
    lengths = np.diff(offsets)[picked]
    new_offsets = np.concatenate([[0], np.cumsum(lengths)])
    index = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in picked])
    return (new_offsets,) + tuple(a[index] for a in db[1:])


def save_atomic(torch, obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


STAGES = ["grid", "node2vec", "T-BERT", "features", "GAT + Diff-TS"]


class Progress:
    def __init__(self):
        self.path = None
        self.data = {}
        self.last = 0.0

    def open(self, out, title):
        self.path = out / "progress.json"
        try:
            self.data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.data = {}
        self.data["title"] = title
        stages = self.data.setdefault("stages", {})
        for name in STAGES:
            stages.setdefault(name, {"status": "waiting"})
        self.write(True)

    def write(self, force=False):
        if self.path is None or (not force and time.time() - self.last < 2):
            return
        self.last = time.time()
        self.data["updated"] = time.time()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data))
        os.replace(tmp, self.path)

    def start(self, stage):
        info = self.data["stages"][stage]
        if info.get("status") != "running":
            info.update(status="running", started=time.time())
        self.data["current"] = {"stage": stage}
        self.write(True)

    def update(self, detail, done, total, stage_done=None, stage_total=None):
        current = self.data.setdefault("current", {})
        if current.get("detail") != detail:
            current.update(since=time.time(), done_at_since=done)
        if stage_done is not None and "stage_since" not in current:
            current.update(stage_since=time.time(), stage_done_at_since=stage_done)
        current.update(detail=detail, done=done, total=total, stage_done=stage_done, stage_total=stage_total)
        self.write()

    def loss(self, stage, epoch, value):
        self.data.setdefault("losses", []).append({"stage": stage, "epoch": epoch, "loss": float(value)})
        self.write(True)

    def finish(self, stage):
        info = self.data["stages"][stage]
        info.update(status="done", seconds=time.time() - info.get("started", time.time()))
        self.data.pop("current", None)
        self.write(True)


progress = Progress()


class Clock:
    def __init__(self, minutes):
        self.every = minutes * 60
        self.last = time.time()

    def due(self):
        if time.time() - self.last >= self.every:
            self.last = time.time()
            return True
        return False


def pick_device(torch, name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def windows(length, max_length, overlap):
    result = []
    start, end = 0, -1
    while end != length:
        end = min(start + max_length, length)
        valid_start = 0 if start == 0 else overlap
        valid_end = end - start if end == length else end - start - overlap
        result.append((start, end, valid_start, valid_end))
        start += max_length - 2 * overlap
    return result


def window_times(torch, times, fixed):
    if fixed:
        return torch.tensor(times - times[0], dtype=torch.float32)
    return torch.tensor(times, dtype=torch.float64).float()


class GridMap:
    def __init__(self, path):
        data = np.load(path)
        self.gridmap = data["gridmap"]
        self.grid_size = int(data["grid_size"])
        self.numx = int(data["numx"])
        self.minx = float(data["minx"])
        self.miny = float(data["miny"])
        self.step = float(data["step"])

    def tokens(self, lon, lat):
        from Utils.data_utils import lonlat2meters_np
        x, y = lonlat2meters_np(lon, lat)
        cx = (np.round(x - self.minx, 6) / self.step).astype(int)
        cy = (np.round(y - self.miny, 6) / self.step).astype(int)
        return self.gridmap[cy * self.numx + cx]


def build_grid(db, params, fixed, out):
    from sklearn.neighbors import KDTree
    from Utils.data_utils import lonlat2meters, lonlat2meters_np
    offsets, lat, lon, _ = db
    step = params["step"]
    minx, miny = lonlat2meters(params["minlon"], params["minlat"])
    maxx, maxy = lonlat2meters(params["maxlon"], params["maxlat"])
    numx = int(math.ceil(round(maxx - minx, 6) / step))
    numy = int(math.ceil(round(maxy - miny, 6) / step))

    use = np.ones(len(lat), dtype=bool)
    if not fixed:
        position = np.arange(len(lat)) - np.repeat(offsets[:-1], np.diff(offsets))
        use = position <= 1000
    inside = ((params["minlon"] <= lon) & (lon < params["maxlon"])
              & (params["minlat"] <= lat) & (lat < params["maxlat"]))
    assert inside.all(), "points outside MLSimp's area"
    x, y = lonlat2meters_np(lon[use], lat[use])
    cells = (np.floor(np.round(y - miny, 6) / step).astype(np.int64) * numx
             + np.floor(np.round(x - minx, 6) / step).astype(np.int64))
    counts = np.bincount(cells, minlength=numx * numy)
    order = np.argsort(-counts, kind="stable")
    hot = order[counts[order] > params["minfreq"]][:params["max_grid_size"]]
    grid_start = params["grid_start"]
    ids = np.arange(len(hot)) - 1 + grid_start

    centers = np.stack([minx + (hot % numx + 0.5) * step, miny + (hot // numx + 0.5) * step], 1)
    tree = KDTree(centers)
    gridmap = np.empty(numx * numy, dtype=np.int32)
    block = 1_000_000
    for begin in range(0, numx * numy, block):
        cell = np.arange(begin, min(begin + block, numx * numy))
        query = np.stack([minx + (cell % numx + 0.5) * step, miny + (cell // numx + 0.5) * step], 1)
        gridmap[begin:begin + len(cell)] = ids[tree.query(query, k=1)[1][:, 0]]
    gridmap[hot] = ids
    np.savez(out, gridmap=gridmap, grid_size=grid_start + len(hot), numx=numx, numy=numy,
             minx=minx, miny=miny, step=step)
    print(f"grid: {len(hot)} cells with own id, grid size {grid_start + len(hot)}", flush=True)


def graph_edges(db, grid, fixed):
    offsets, lat, lon, _ = db
    tokens = grid.tokens(lon, lat)
    pairs = []
    for i in range(len(offsets) - 1):
        seq = tokens[offsets[i]:offsets[i + 1]]
        if fixed:
            a, b = seq[:-1], seq[1:]
            keep = a != b
            pairs.append(np.stack([a[keep], b[keep]], 1))
        else:
            rest = seq[1:1001]
            pairs.append(np.stack([np.full(len(rest), seq[0]), rest], 1))
    pairs = np.concatenate(pairs)
    pairs = np.concatenate([pairs, pairs[:, ::-1]])
    pairs = np.unique(pairs, axis=0)
    return pairs


class Walks:
    def __init__(self, torch, edges, num_nodes, walk_length, context_size, walks_per_node):
        self.torch = torch
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges = edges[order]
        self.col = torch.tensor(edges[:, 1], dtype=torch.long)
        counts = np.bincount(edges[:, 0], minlength=num_nodes)
        self.rowptr = torch.tensor(np.concatenate([[0], np.cumsum(counts)]), dtype=torch.long)
        self.deg = torch.tensor(counts, dtype=torch.long)
        self.num_nodes = num_nodes
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node

    def _windows(self, rw):
        count = 1 + self.walk_length + 1 - self.context_size
        return self.torch.cat([rw[:, j:j + self.context_size] for j in range(count)], 0)

    def pos(self, batch, generator):
        torch = self.torch
        node = batch.repeat(self.walks_per_node)
        walk = [node]
        for _ in range(self.walk_length):
            deg = self.deg[node]
            pick = (torch.rand(len(node), generator=generator) * deg.clamp(min=1)).long()
            nxt = self.col[(self.rowptr[node] + pick).clamp(max=len(self.col) - 1)]
            node = torch.where(deg > 0, nxt, node)
            walk.append(node)
        return self._windows(torch.stack(walk, 1))

    def neg(self, batch, generator):
        torch = self.torch
        node = batch.repeat(self.walks_per_node)
        rw = torch.randint(self.num_nodes, (len(node), self.walk_length), generator=generator)
        return self._windows(torch.cat([node.view(-1, 1), rw], 1))


def node2vec_loss(torch, embedding, pos_rw, neg_rw):
    eps = 1e-15
    dim = embedding.size(1)

    def side(rw):
        start, rest = rw[:, 0], rw[:, 1:].contiguous()
        h_start = embedding[start].view(rw.size(0), 1, dim)
        h_rest = embedding[rest.view(-1)].view(rw.size(0), -1, dim)
        return (h_start * h_rest).sum(dim=-1).view(-1)

    pos_loss = -torch.log(torch.sigmoid(side(pos_rw)) + eps).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(side(neg_rw)) + eps).mean()
    return pos_loss + neg_loss


def train_node2vec(torch, db, grid, params, fixed, out, ckpt_path, device, args):
    edges = graph_edges(db, grid, fixed)
    walks = Walks(torch, edges, grid.grid_size, params["walk_length"], params["context_size"],
                  params["walks_per_node"])
    dim = params["gridemb_dim"]
    embedding = torch.nn.Embedding(grid.grid_size, dim).to(device)
    torch.nn.init.normal_(embedding.weight)
    optimizer = torch.optim.Adam(embedding.parameters(), lr=params["gridemb_lr"])
    state = {"epoch": 0, "cnt": 0, "last_loss": 0.0}
    if ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location=device)
        embedding.load_state_dict(saved["embedding"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        print(f"node2vec: resuming at epoch {state['epoch']}", flush=True)
    epochs = args.node2vec_epochs or params["gridemb_epochs"]
    patience = 200
    while state["epoch"] < epochs:
        generator = torch.Generator().manual_seed(args.seed * 1000003 + state["epoch"])
        perm = torch.randperm(grid.grid_size, generator=generator)
        total, batches = 0.0, 0
        for begin in range(0, grid.grid_size, 128):
            batch = perm[begin:begin + 128]
            pos_rw = walks.pos(batch, generator).to(device)
            neg_rw = walks.neg(batch, generator).to(device)
            optimizer.zero_grad()
            loss = node2vec_loss(torch, embedding.weight, pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        loss = total / batches
        state["epoch"] += 1
        progress.update("epoch", state["epoch"], epochs, state["epoch"], epochs)
        if abs(loss - state["last_loss"]) < 1e-3:
            state["cnt"] += 1
        else:
            state["cnt"] = 0
        state["last_loss"] = loss
        progress.loss("node2vec", state["epoch"], loss)
        print(f"node2vec epoch {state['epoch']}/{epochs} loss {loss:.4f}", flush=True)
        save_atomic(torch, {"embedding": embedding.state_dict(), "optimizer": optimizer.state_dict(),
                            "state": state}, ckpt_path)
        if state["cnt"] == patience:
            break
    save_atomic(torch, embedding.weight.detach().cpu(), out)


def token_windows(db, grid, params):
    offsets, lat, lon, t = db
    tokens = grid.tokens(lon, lat)
    result = []
    for i in range(len(offsets) - 1):
        begin, end = offsets[i], offsets[i + 1]
        for s, e, _, _ in windows(end - begin, params["max_length"], params["overlap"]):
            result.append((begin + s, begin + e))
    return tokens, t, result


def make_tbert(torch, grid_size, embed_size, grid_emb=None):
    from Model.tbert import TBERT, TBERTEmbedding, TemporalEncoding
    embedding = TBERTEmbedding(TemporalEncoding(embed_size), embed_size, grid_size, grid_emb)
    return TBERT(embedding, embed_size * 4, num_layers=4, num_heads=8, init_param=False, detach=False)


def train_tbert(torch, db, grid, params, fixed, grid_emb, out, ckpt_path, device, args):
    from Model.tbert import MaskedLM, gen_random_mask
    tokens, t, spans = token_windows(db, grid, params)
    embed_size = params["gridemb_dim"]
    model = make_tbert(torch, grid.grid_size, embed_size, grid_emb).to(device)
    objective = MaskedLM(embed_size, grid.grid_size).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(objective.parameters()), lr=1e-4)
    state = {"epoch": 0, "batch": 0}
    if ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(saved["model"])
        objective.load_state_dict(saved["objective"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        print(f"T-BERT: resuming at epoch {state['epoch']}, batch {state['batch']}", flush=True)
    epochs = args.tbert_epochs or (500 if fixed else params["Bert"]["embed_epoch"])
    batch_size = 64
    clock = Clock(args.save_minutes)

    def checkpoint():
        save_atomic(torch, {"model": model.state_dict(), "objective": objective.state_dict(),
                            "optimizer": optimizer.state_dict(), "state": dict(state)}, ckpt_path)

    model.train()
    objective.train()
    while state["epoch"] < epochs:
        order = np.random.default_rng([args.seed, 1, state["epoch"]]).permutation(len(spans))
        count = math.ceil(len(spans) / batch_size)
        total = 0.0
        while state["batch"] < count:
            pick = order[state["batch"] * batch_size:(state["batch"] + 1) * batch_size]
            lengths = [spans[k][1] - spans[k][0] for k in pick]
            width = max(lengths)
            src = np.zeros((len(pick), width), dtype=np.int64)
            ts = torch.zeros((len(pick), width), dtype=torch.float32)
            for row, k in enumerate(pick):
                s, e = spans[k]
                src[row, :e - s] = tokens[s:e]
                ts[row, :e - s] = window_times(torch, t[s:e], fixed)
            src = torch.tensor(src, device=device)
            ts = ts.to(device)
            torch.manual_seed(args.seed * 1000003 + state["epoch"] * 100003 + state["batch"])
            valid = torch.tensor(lengths, device=device)
            mask_index = gen_random_mask(valid, width, mask_prop=0.2)
            flat = src.reshape(-1)
            origin = flat[mask_index]
            masked = flat.index_fill(0, mask_index, 1).reshape(len(pick), -1)
            out_t = model(masked, timestamp=ts)
            loss = objective(out_t.reshape(-1, embed_size)[mask_index], origin_tokens=origin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            state["batch"] += 1
            progress.update(f"epoch {state['epoch'] + 1}/{epochs} | batch", state["batch"], count,
                            state["epoch"] * count + state["batch"], epochs * count)
            if clock.due():
                checkpoint()
                print(f"T-BERT epoch {state['epoch'] + 1}/{epochs} batch {state['batch']}/{count} "
                      f"loss {loss.item():.4f}", flush=True)
        state["epoch"] += 1
        state["batch"] = 0
        checkpoint()
        progress.loss("T-BERT", state["epoch"], total / count)
        print(f"T-BERT epoch {state['epoch']}/{epochs} mean loss {total / count:.4f} "
              f"at {time.strftime('%H:%M:%S')}", flush=True)
    save_atomic(torch, model.state_dict(), out)


def trajectory_features(torch, tbert, db, grid, params, fixed, device):
    import torch.nn.functional as F
    offsets, lat, lon, t = db
    tokens = grid.tokens(lon, lat)
    tbert.to(device)
    tbert.eval()
    items = []
    with torch.no_grad():
        for i in range(len(offsets) - 1):
            begin, end = offsets[i], offsets[i + 1]
            spans = windows(end - begin, params["max_length"], params["overlap"])
            width = max(e - s for s, e, _, _ in spans)
            src = torch.zeros((len(spans), width), dtype=torch.long)
            ts = torch.zeros((len(spans), width), dtype=torch.float32)
            for row, (s, e, _, _) in enumerate(spans):
                src[row, :e - s] = torch.tensor(tokens[begin + s:begin + e])
                ts[row, :e - s] = window_times(torch, t[begin + s:begin + e], fixed)
            out = tbert(src.to(device), timestamp=ts.to(device))
            parts = [out[j][vs:ve] for j, (_, _, vs, ve) in enumerate(spans)]
            point = torch.cat(parts, 0)
            seg = torch.stack([p.mean(0) for p in parts])
            unit = F.normalize(point, dim=1, eps=1e-8)
            m = point.size(0)
            neighbor = torch.empty((m, 10), dtype=torch.long, device=device)
            for r in range(0, m, 4096):
                neighbor[r:r + 4096] = torch.topk(unit[r:r + 4096] @ unit.T, min(11, m), dim=-1).indices[:, 1:]
            items.append({"feature": torch.cat([point, seg], 0).cpu(), "points": m,
                          "sizes": [ve - vs for _, _, vs, ve in spans],
                          "emb": point.mean(0).cpu(), "neighbor": neighbor.T.contiguous().cpu()})
            progress.update("trajectories", i + 1, len(offsets) - 1)
            if (i + 1) % 2000 == 0:
                print(f"features {i + 1}/{len(offsets) - 1}", flush=True)
    return items


def make_gat(torch, params, fixed):
    from Model.GraphSimp import GAT
    from torch_geometric.nn import GATConv
    gat = GAT(params["GAT"])
    if fixed:
        heads, out = params["GAT"]["num_heads"], params["GAT"]["out_features"]
        gat.conv2 = GATConv(out * heads, out, heads=heads, concat=False)
    return gat


def edge_index(torch, m, sizes, fixed, device):
    if not fixed:
        points = torch.arange(m, device=device)
        return torch.cartesian_prod(points, torch.arange(m, m + len(sizes), device=device)).t().contiguous()
    owner = m + torch.repeat_interleave(torch.arange(len(sizes), device=device),
                                        torch.as_tensor(sizes, device=device))
    points = torch.arange(m, device=device)
    return torch.stack([torch.cat([points, owner]), torch.cat([owner, points])])


def uniqueness(torch, z, neighbor, fixed):
    distance = [(z - z[neighbor[i]]).norm(dim=1) for i in range(neighbor.size(0))]
    if not fixed:
        distance = [d.pow(2) for d in distance]
    return sum(distance) / neighbor.size(0)


def globality(torch, z, fixed):
    from torch.utils.checkpoint import checkpoint
    n = z.size(0)

    def rows(block, full):
        near = torch.cdist(block, full).pow(2).mul(-2).exp()
        if fixed:
            return ((near.sum(dim=1) - 1) / max(n - 1, 1)).log()
        return near.mean(dim=1).log()

    if n <= 4096:
        return rows(z, z)
    return torch.cat([checkpoint(rows, z[r:r + 4096], z, use_reentrant=False) for r in range(0, n, 4096)])


def point_scores(torch, align, glob, fixed):
    return align * glob.exp() + 1e-6 if fixed else align * glob


def gat_terms(torch, z, neighbor, amply, fixed):
    align = uniqueness(torch, z, neighbor, fixed)
    glob = globality(torch, z, fixed)
    important = torch.softmax(point_scores(torch, align, glob, fixed), dim=0)
    mutual = torch.nn.functional.binary_cross_entropy(important, amply) if amply is not None else 0
    top = torch.sort(torch.sort(important[1:-1], descending=True).indices[:3]).values + 1
    simp = (z[top].sum(0) + z[0] + z[-1]) / 5
    return align.mean(), glob.mean(), simp, mutual


def gat_simp_indices(torch, z, neighbor, fixed, generator):
    align = uniqueness(torch, z, neighbor, fixed)
    important = torch.softmax(point_scores(torch, align, globality(torch, z, fixed), fixed), dim=0)[1:-1]
    if not fixed:
        k = int(0.4 * z.size(0))
        top = torch.sort(torch.sort(important, descending=True).indices[:k]).values
        return (top + 1).cpu().numpy()
    k = min(int(0.5 * z.size(0)), important.size(0))
    noise = torch.rand(important.size(0), generator=generator).to(important.device)
    keys = noise.clamp(min=1e-12).log() / important.clamp(min=1e-30)
    top = torch.sort(torch.topk(keys, k).indices).values
    return (top + 1).cpu().numpy()


def diffusion_inputs(db, params):
    offsets, lat, lon, _ = db
    feats = []
    for i in range(len(offsets) - 1):
        b, e = offsets[i], offsets[i + 1]
        x = (lon[b:e] - params["minlon"]) / (params["maxlon"] - params["minlon"])
        y = (lat[b:e] - params["minlat"]) / (params["maxlat"] - params["minlat"])
        src = np.stack([lon[b:e], lat[b:e]], 1)
        lens = np.hypot(np.diff(src[:, 0]), np.diff(src[:, 1]))
        f = np.zeros((e - b, 4), dtype=np.float32)
        f[:, 0], f[:, 1] = x, y
        if e - b > 2:
            prev, cur, nxt = src[:-2], src[1:-1], src[2:]
            radian = (math.pi - np.arctan2(prev[:, 0] - cur[:, 0], prev[:, 1] - cur[:, 1])
                      + np.arctan2(nxt[:, 0] - cur[:, 0], nxt[:, 1] - cur[:, 1]))
            f[1:-1, 2] = (lens[:-1] + lens[1:]) / 2
            f[1:-1, 3] = 1 - np.abs(radian) / math.pi
        feats.append(f)
    return feats


def length_batches(lengths, budget, cap, rng):
    order = np.argsort(lengths, kind="stable")
    batches, current, width = [], [], 0
    for k in order:
        width_next = max(width, lengths[k])
        if current and (len(current) == cap or width_next * (len(current) + 1) > budget):
            batches.append(current)
            current, width_next = [], lengths[k]
        current.append(k)
        width = width_next
    if current:
        batches.append(current)
    rng.shuffle(batches)
    return batches


def pad(torch, arrays, device):
    width = max(len(a) for a in arrays)
    out = torch.zeros((len(arrays), width, arrays[0].shape[1]), dtype=torch.float32)
    mask = torch.ones((len(arrays), width), dtype=torch.bool)
    for i, a in enumerate(arrays):
        out[i, :len(a)] = torch.as_tensor(a)
        mask[i, :len(a)] = False
    return out.to(device), mask.to(device)


def make_chunks(lengths, size):
    chunks = []
    for k, length in enumerate(lengths):
        bounds = np.linspace(0, length, -(-length // size) + 1).round().astype(np.int64)
        chunks.extend((k, int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]))
    return chunks


def chunk_inputs(feats, simp_idx, chunks, pick):
    trajs, idxs = [], []
    for c in pick:
        k, a, b = chunks[c]
        idx = np.asarray(simp_idx[k], dtype=np.int64)
        trajs.append(feats[k][a:b])
        idxs.append(idx[(idx > a) & (idx < b - 1)] - a)
    return trajs, idxs


def diffusion_batch(torch, trajs, idxs, device):
    simps, labels = [], []
    for traj, idx in zip(trajs, idxs):
        simps.append(traj[np.concatenate([[0], idx, [-1]]).astype(np.int64)])
        labels.append(np.concatenate([[0], idx, [len(traj) - 1]]))
    traj_pad, traj_mask = pad(torch, trajs, device)
    simp_pad, simp_mask = pad(torch, simps, device)
    label = torch.full((len(trajs), simp_pad.size(1)), -100, dtype=torch.long)
    for i, l in enumerate(labels):
        label[i, :len(l)] = torch.as_tensor(l)
    return traj_pad, traj_mask, simp_pad, simp_mask, label.to(device)


def masked_mean(torch, values, valid):
    weight = valid.float()
    per = (values * weight).sum(1) / weight.sum(1).clamp(min=1)
    return per.mean()


def diversity(torch, h, valid):
    from torch.utils.checkpoint import checkpoint
    weight = valid.float()

    def rows(block, block_valid, full):
        near = torch.cdist(block, full).pow(2).mul(-2).exp() * weight[:, None, :]
        others = (weight.sum(1)[:, None] - block_valid).clamp(min=1)
        return ((near.sum(-1) - block_valid) / others).clamp(min=1e-30).log() * block_valid

    chunk = 4096
    if h.size(1) <= chunk:
        part = rows(h, weight, h)
    else:
        part = torch.cat([checkpoint(rows, h[:, r:r + chunk], weight[:, r:r + chunk], h, use_reentrant=False)
                          for r in range(0, h.size(1), chunk)], 1)
    return (part.sum(1) / weight.sum(1).clamp(min=1)).mean()


def diffusion_loss(torch, model, diffusion, batch, t, diversity_weight=0.0):
    from DiffTeacher.GaussianDiffusion import _extract_into_tensor
    traj_pad, traj_mask, simp_pad, simp_mask, labels = batch
    length = traj_pad.size(1)
    pad_mask = torch.cat([traj_mask, simp_mask], 1)
    x_start_mean = model.get_embeds(torch.cat([traj_pad, simp_pad], 1), pad_mask)
    std = _extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod,
                               torch.tensor([0], device=x_start_mean.device), x_start_mean.shape)
    x_start = x_start_mean + std * torch.randn_like(x_start_mean)
    x_t = diffusion.q_sample(x_start, t, noise=torch.randn_like(x_start))
    x_t[:, :length] = x_start_mean[:, :length]
    out = model(x_t, pad_mask, diffusion._scale_timesteps(t))
    simp_valid = ~simp_mask
    mse = masked_mean(torch, ((x_start_mean[:, length:] * 10 - out[:, length:] * 10) ** 2).mean(-1), simp_valid)
    logits = torch.bmm(x_start_mean[:, length:], x_start_mean[:, :length].transpose(1, 2))
    logits = logits.masked_fill(traj_mask[:, None, :], float("-inf"))
    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, length), labels.reshape(-1), ignore_index=-100)
    t_last = torch.full((x_start.size(0),), diffusion.num_timesteps - 1, device=x_start.device, dtype=torch.long)
    out_mean, _, _ = diffusion.q_mean_variance(x_start, t_last)
    t_t = masked_mean(torch, (out_mean ** 2).mean(-1), ~pad_mask)
    loss = mse + t_t + 0.1 * nll
    if diversity_weight:
        loss = loss + diversity_weight * diversity(torch, out[:, length:], simp_valid)
    return loss


def diffusion_sample(torch, model, diffusion, batch, amplify_len, steps, fixed, device):
    traj_pad, traj_mask = batch[0], batch[1]
    length = traj_pad.size(1)
    x_start_mean = model.get_embeds(traj_pad, traj_mask)
    pad_mask = torch.cat([traj_mask, torch.zeros((traj_mask.size(0), amplify_len), dtype=torch.bool,
                                                   device=device)], 1)
    x = torch.randn((x_start_mean.size(0), length + amplify_len, x_start_mean.size(2)), device=device)
    x[:, :length] = x_start_mean
    wrapped = diffusion._wrap_model(model)
    for i in (reversed(range(steps)) if fixed else [0]):
        t = torch.full((x.size(0),), i, device=device, dtype=torch.long)
        out = diffusion.p_sample(wrapped, x, pad_mask, t, clip_denoised=False, denoised_fn=None, top_p=-1)
        sample = out["sample"]
        sample[:, :length] = x_start_mean
        if fixed:
            x = sample
    if not fixed:
        x = sample
    summary = x[:, length:]
    logits = torch.bmm(summary, x[:, :length].transpose(1, 2)).masked_fill(traj_mask[:, None, :], float("-inf"))
    candidates = torch.topk(torch.softmax(logits, -1), k=amplify_len, dim=-1).indices.cpu().numpy()
    result = []
    for row in candidates:
        chosen = []
        for options in row:
            for c in options:
                if c not in chosen:
                    chosen.append(int(c))
                    break
        result.append(np.array(chosen, dtype=np.int64))
    return result


def chunk_attention(torch, model, threshold=2048, chunk=1024):
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    def block(q, k, v, bias, p, training):
        weights = torch.softmax(q @ k.transpose(-2, -1) + bias, dim=-1)
        return F.dropout(weights, p, training) @ v

    class ChunkedAttention(torch.nn.MultiheadAttention):
        def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None,
                    average_attn_weights=True, is_causal=False):
            if query.size(1) <= threshold or need_weights or attn_mask is not None:
                return super().forward(query, key, value, key_padding_mask=key_padding_mask,
                                       need_weights=need_weights, attn_mask=attn_mask,
                                       average_attn_weights=average_attn_weights, is_causal=is_causal)
            batch, length, embed = query.shape
            heads = self.num_heads
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
            q, k, v = (x.reshape(batch, length, heads, -1).transpose(1, 2) for x in (q, k, v))
            q = q * (q.size(-1) ** -0.5)
            bias = torch.zeros((batch, 1, 1, length), dtype=q.dtype, device=q.device)
            if key_padding_mask is not None:
                bias = bias.masked_fill(key_padding_mask.bool()[:, None, None, :], float("-inf"))
            out = torch.cat([checkpoint(block, q[:, :, i:i + chunk], k, v, bias, self.dropout, self.training,
                                        use_reentrant=False) for i in range(0, length, chunk)], dim=2)
            return self.out_proj(out.transpose(1, 2).reshape(batch, length, embed)), None

    torch.backends.mha.set_fastpath_enabled(False)
    for encoder in (model.traj_embed, model.encoder):
        encoder.use_nested_tensor = False
        for layer in encoder.layers:
            layer.self_attn.__class__ = ChunkedAttention


def extend_positions(model, length):
    from DiffTeacher.PositionalEncoding import PositionalEncoding
    old = model.position_embeddings
    if old.pe.size(0) <= length:
        model.position_embeddings = PositionalEncoding(old.pe.size(2), max_len=length).to(old.pe.device)


def make_diffusion(params):
    from DiffTeacher.model_util import create_model_and_diffusion
    d = params["Diff"]
    return create_model_and_diffusion(
        d["in_dim"], d["encoder_n_head"], d["encoder_hidden_dim"], d["encoder_n_layer"],
        d["in_channels"], d["hidden_channels"], d["out_channels"], d["n_head"], d["n_layer"],
        d["trans_hidden_channels"], d["attn_dropout"], d["dropout"], d["learn_sigma"],
        d["sigma_small"], d["diffusion_steps"], d["noise_schedule"], d["timestep_respacing"],
        d["use_kl"], d["predict_xstart"], d["rescale_timesteps"], d["rescale_learned_sigmas"],
        d["training_mode"])


def train_mlsimp(torch, db, items, feats, params, fixed, out_dir, ckpt_path, device, args):
    gat_params = params["GAT"]
    diff_params = params["Diff"]
    rounds = args.rounds or (2 if fixed else params["mutual_epochs"])
    gat_epochs = args.gat_epochs or gat_params["num_epoch"]
    diff_epochs = args.diff_epochs or diff_params["epochs"]
    amplify_len = diff_params["amplify_len"]
    lengths = np.array([it["points"] for it in items])
    max_len = int(lengths.max())
    n = len(items)

    gat = make_gat(torch, params, fixed).to(device)
    diff_model, diffusion = make_diffusion(params)
    chunk_size = 1000
    chunks = make_chunks(lengths, chunk_size)
    chunk_len = np.array([b - a for _, a, b in chunks])
    chunk_amp = np.maximum(1, np.round(amplify_len * chunk_len / chunk_size)).astype(int)
    extend_positions(diff_model, int(chunk_len.max() * 1.5) + amplify_len + 4)
    chunk_attention(torch, diff_model)
    diff_model.to(device)
    gat_opt = torch.optim.Adam(gat.parameters(), lr=float(gat_params["lr"]), weight_decay=gat_params["wd"])
    diff_opt = torch.optim.AdamW(diff_model.parameters(), lr=float(diff_params["lr"]),
                                 weight_decay=diff_params["weight_decay"])
    state = {"round": 0, "phase": "gat", "epoch": 0, "batch": 0}
    simp_idx, amply_idx, sampled = None, None, {}
    if ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        gat.load_state_dict(saved["gat"])
        diff_model.load_state_dict(saved["diff"])
        gat_opt.load_state_dict(saved["gat_opt"])
        diff_opt.load_state_dict(saved["diff_opt"])
        state, simp_idx, amply_idx, sampled = saved["state"], saved["simp_idx"], saved["amply_idx"], saved["sampled"]
        print(f"MLSimp: resuming at round {state['round'] + 1}, {state['phase']}, "
              f"epoch {state['epoch']}, batch {state['batch']}", flush=True)
    clock = Clock(args.save_minutes)

    def checkpoint():
        save_atomic(torch, {"gat": gat.state_dict(), "diff": diff_model.state_dict(),
                            "gat_opt": gat_opt.state_dict(), "diff_opt": diff_opt.state_dict(),
                            "state": dict(state), "simp_idx": simp_idx, "amply_idx": amply_idx,
                            "sampled": sampled}, ckpt_path)

    def advance(phase):
        state.update(phase=phase, epoch=0, batch=0)
        checkpoint()

    while state["round"] < rounds:
        r = state["round"]
        if state["phase"] == "gat":
            if state["epoch"] == 0 and state["batch"] == 0:
                gat_opt = torch.optim.Adam(gat.parameters(), lr=float(gat_params["lr"]),
                                           weight_decay=gat_params["wd"])
            gat.train()
            while state["epoch"] < gat_epochs:
                order = np.random.default_rng([args.seed, 2, r, state["epoch"]]).permutation(n)
                count = math.ceil(n / gat_params["batch_size"])
                epoch_loss = []
                while state["batch"] < count:
                    pick = order[state["batch"] * gat_params["batch_size"]:(state["batch"] + 1) * gat_params["batch_size"]]
                    aligns, uniforms, simps, mutuals, embs = [], [], [], [], []
                    for k in pick:
                        it = items[k]
                        feature = it["feature"].to(device)
                        z = gat(feature, edge_index(torch, it["points"], it["sizes"], fixed, device))[:it["points"]]
                        amply = None
                        if amply_idx is not None:
                            amply = torch.zeros(it["points"], device=device)
                            amply[torch.as_tensor(amply_idx[k], device=device)] = 1
                        a, u, s, mu = gat_terms(torch, z, it["neighbor"].to(device), amply, fixed)
                        aligns.append(a); uniforms.append(u); simps.append(s); mutuals.append(mu)
                        embs.append(it["emb"].to(device))
                    simps = torch.nn.functional.normalize(torch.stack(simps), p=2, dim=1)
                    embs = torch.nn.functional.normalize(torch.stack(embs), p=2, dim=1)
                    mutual = sum(mutuals) / len(pick) if amply_idx is not None else 0
                    if fixed:
                        loss = torch.stack(aligns).mean() + 0.5 * torch.stack(uniforms).mean() + 0.5 * mutual
                    else:
                        batch_loss = torch.nn.functional.mse_loss(simps @ simps.T, embs @ embs.T)
                        loss = (torch.stack(aligns).mean() + gat_params["lambda1"] * torch.stack(uniforms).mean()
                                + gat_params["lambda2"] * batch_loss + gat_params["lambda3"] * mutual)
                    gat_opt.zero_grad()
                    loss.backward()
                    gat_opt.step()
                    epoch_loss.append(loss.item())
                    state["batch"] += 1
                    progress.update(f"round {r + 1}/{rounds} | GNN-TS epoch {state['epoch'] + 1}/{gat_epochs} | batch",
                                    state["batch"], count, r * 3 + (state["epoch"] + state["batch"] / count) / gat_epochs,
                                    rounds * 3)
                    if clock.due():
                        checkpoint()
                        print(f"round {r + 1}/{rounds} GAT epoch {state['epoch'] + 1}/{gat_epochs} "
                              f"batch {state['batch']}/{count} loss {loss.item():.4f}", flush=True)
                state["epoch"] += 1
                state["batch"] = 0
                checkpoint()
                if epoch_loss:
                    progress.loss(f"GNN-TS stage {r + 1}", state["epoch"], np.mean(epoch_loss))
                print(f"round {r + 1}/{rounds} GAT epoch {state['epoch']}/{gat_epochs} done at {time.strftime('%H:%M:%S')}", flush=True)
            gat.eval()
            with torch.no_grad():
                simp_idx = []
                generator = torch.Generator().manual_seed(args.seed * 1000003 + r)
                for it in items:
                    feature = it["feature"].to(device)
                    z = gat(feature, edge_index(torch, it["points"], it["sizes"], fixed, device))[:it["points"]]
                    simp_idx.append(gat_simp_indices(torch, z, it["neighbor"].to(device), fixed, generator))
            save_atomic(torch, gat.state_dict(), out_dir / "gat.pt")
            advance("diff")
        if state["phase"] == "diff":
            if r == rounds - 1:
                state["round"] += 1
                advance("gat")
                break
            if state["epoch"] == 0 and state["batch"] == 0:
                diff_opt = torch.optim.AdamW(diff_model.parameters(), lr=float(diff_params["lr"]),
                                             weight_decay=diff_params["weight_decay"])
            diff_model.train()
            budget = diff_params["batch_size"] * 1400
            while state["epoch"] < diff_epochs:
                rng = np.random.default_rng([args.seed, 3, r, state["epoch"]])
                trajs_all, idxs_all = chunk_inputs(feats, simp_idx, chunks, range(len(chunks)))
                batches = length_batches(chunk_len + np.array([len(i) for i in idxs_all]) + 2, budget,
                                         diff_params["batch_size"], rng)
                epoch_loss = []
                while state["batch"] < len(batches):
                    pick = batches[state["batch"]]
                    batch = diffusion_batch(torch, [trajs_all[c] for c in pick], [idxs_all[c] for c in pick],
                                            device)
                    t = torch.randint(0, diffusion.num_timesteps, (len(pick),), device=device)
                    loss = diffusion_loss(torch, diff_model, diffusion, batch, t,
                                          diff_params["lambda1"] if fixed else 0.0)
                    diff_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 3)
                    diff_opt.step()
                    epoch_loss.append(loss.item())
                    state["batch"] += 1
                    progress.update(f"round {r + 1}/{rounds} | Diff-TS epoch {state['epoch'] + 1}/{diff_epochs} | batch",
                                    state["batch"], len(batches),
                                    r * 3 + 1 + (state["epoch"] + state["batch"] / len(batches)) / diff_epochs, rounds * 3)
                    if clock.due():
                        checkpoint()
                        print(f"round {r + 1}/{rounds} diffusion epoch {state['epoch'] + 1}/{diff_epochs} "
                              f"batch {state['batch']}/{len(batches)} loss {loss.item():.4f}", flush=True)
                state["epoch"] += 1
                state["batch"] = 0
                checkpoint()
                if epoch_loss:
                    progress.loss("Diff-TS", state["epoch"], np.mean(epoch_loss))
                print(f"round {r + 1}/{rounds} diffusion epoch {state['epoch']}/{diff_epochs} done at {time.strftime('%H:%M:%S')}", flush=True)
            sampled = {}
            advance("sample")
        if state["phase"] == "sample":
            diff_model.eval()
            batches = []
            rng = np.random.default_rng([args.seed, 4, r])
            for amp in np.unique(chunk_amp):
                group = np.flatnonzero(chunk_amp == amp)
                for part in length_batches(chunk_len[group] + amp, diff_params["batch_size"] * 1400,
                                           diff_params["batch_size"], rng):
                    batches.append((int(amp), [int(c) for c in group[part]]))
            with torch.no_grad():
                for b, (amp, pick) in enumerate(batches):
                    if b in sampled:
                        continue
                    trajs, idxs = chunk_inputs(feats, simp_idx, chunks, pick)
                    batch = diffusion_batch(torch, trajs, idxs, device)
                    for c, chosen in zip(pick, diffusion_sample(torch, diff_model, diffusion, batch, amp,
                                                                diff_params["diff_step_eval"], fixed, device)):
                        sampled.setdefault(b, {})[c] = chosen + chunks[c][1]
                    progress.update(f"round {r + 1}/{rounds} | Diff-TS sampling | batch", len(sampled), len(batches),
                                    r * 3 + 2 + len(sampled) / len(batches), rounds * 3)
                    if clock.due():
                        checkpoint()
                        print(f"round {r + 1}/{rounds} sampling batch {b + 1}/{len(batches)}", flush=True)
            print(f"round {r + 1}/{rounds} sampling done at {time.strftime('%H:%M:%S')}", flush=True)
            gathered = [[] for _ in range(n)]
            for part in sampled.values():
                for c, chosen in part.items():
                    gathered[chunks[c][0]].append(chosen)
            amply_idx = [np.unique(np.concatenate(g)) for g in gathered]
            sampled = {}
            state["round"] += 1
            advance("gat")
            print(f"round {r + 1}/{rounds} done at {time.strftime('%H:%M:%S')}", flush=True)
    save_atomic(torch, gat.state_dict(), out_dir / "gat.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mlsimp", type=Path, default=MLSIMP,
                        help="MLSimp checkout (default: the third_party/MLSimp submodule)")
    parser.add_argument("--data", type=Path, required=True, help="export folder (<export-db>/seed<s>)")
    parser.add_argument("--out", type=Path, required=True, help="folder for the trained models")
    parser.add_argument("--fixed", action="store_true",
                        help="also fix the issues that change what MLSimp learns (see mlsimp/README.md)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="use only the first trajectories (to test)")
    parser.add_argument("--sample", type=int,
                        help="use this many random trajectories plus the longest one (to time a run)")
    parser.add_argument("--save-minutes", type=float, default=10, help="minutes between checkpoints")
    parser.add_argument("--node2vec-epochs", type=int, help="override (to test)")
    parser.add_argument("--tbert-epochs", type=int, help="override (to test)")
    parser.add_argument("--rounds", type=int, help="override the mutual rounds (to test)")
    parser.add_argument("--gat-epochs", type=int, help="override (to test)")
    parser.add_argument("--diff-epochs", type=int, help="override (to test)")
    args = parser.parse_args()

    data = args.data.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    db = load_database(data)
    if args.limit:
        db = subset(db, np.arange(min(args.limit, len(db[0]) - 1)))
    if args.sample:
        lengths = np.diff(db[0])
        picked = np.random.default_rng(args.seed).choice(len(lengths), args.sample, replace=False)
        db = subset(db, np.unique(np.append(picked, np.argmax(lengths))))

    os.chdir(args.mlsimp.resolve())
    sys.path.insert(0, str(Path.cwd()))
    import torch
    import yaml

    with open("Setting.yaml") as file:
        params = yaml.safe_load(file)["Geolife"]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(torch, args.device)
    config = {"fixed": args.fixed, "seed": args.seed, "data": str(data), "trajectories": len(db[0]) - 1,
              "points": int(db[0][-1])}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"device {device}, {config['trajectories']} trajectories, {config['points']} points, "
          f"{'fixed' if args.fixed else 'published'} MLSimp", flush=True)

    started = time.time()
    lap = [time.time()]
    progress.open(out, f"{data.name}/{out.name} | {'corrected' if args.fixed else 'published'} | "
                       f"{config['trajectories']} trajectories")

    def timed(name):
        print(f"time: {name} {time.time() - lap[0]:.0f} s", flush=True)
        lap[0] = time.time()
        progress.finish(name)

    if not (out / "grid.npz").exists():
        progress.start("grid")
        build_grid(db, params, args.fixed, out / "grid.npz")
        timed("grid")
    grid = GridMap(out / "grid.npz")
    if not (out / "node2vec.pt").exists():
        progress.start("node2vec")
        train_node2vec(torch, db, grid, params, args.fixed, out / "node2vec.pt", out / "node2vec_ckpt.pt", device, args)
        timed("node2vec")
    grid_emb = torch.load(out / "node2vec.pt")
    if not (out / "tbert.pt").exists():
        progress.start("T-BERT")
        train_tbert(torch, db, grid, params, args.fixed, grid_emb, out / "tbert.pt", out / "tbert_ckpt.pt", device, args)
        timed("T-BERT")
    if not (out / "done.txt").exists():
        progress.start("features")
        tbert = make_tbert(torch, grid.grid_size, params["gridemb_dim"])
        tbert.load_state_dict(torch.load(out / "tbert.pt", map_location="cpu"))
        items = trajectory_features(torch, tbert, db, grid, params, args.fixed, device)
        del tbert
        feats = diffusion_inputs(db, params)
        timed("features")
        progress.start("GAT + Diff-TS")
        train_mlsimp(torch, db, items, feats, params, args.fixed, out, out / "mlsimp_ckpt.pt", device, args)
        timed("GAT + Diff-TS")
        (out / "done.txt").write_text(f"{time.time() - started:.0f} s\n")
    print(f"done, models in {out}", flush=True)


if __name__ == "__main__":
    main()
