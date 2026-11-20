"""Scores every point of our database with MLSimp's pretrained GeoLife model."""

import argparse
import os
import sys
import time
from itertools import zip_longest
from pathlib import Path

import numpy as np


def load_database(folder):
    offsets = np.fromfile(folder / "offsets.bin", dtype=np.uint64).astype(np.int64)
    lat = np.fromfile(folder / "lat.bin", dtype=np.float64)
    lon = np.fromfile(folder / "lon.bin", dtype=np.float64)
    t = np.fromfile(folder / "t.bin", dtype=np.int64)
    assert offsets[-1] == len(lat) == len(lon) == len(t), "database files don't match"
    return offsets, lat, lon, t


def segments(length, max_length, overlap):
    result = []
    start, end = 0, -1
    while end != length:
        end = min(start + max_length, length)
        valid_start = 0 if start == 0 else overlap
        valid_end = end - start if end == length else end - start - overlap
        result.append((start, end, valid_start, valid_end))
        start += max_length - 2 * overlap
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mlsimp", type=Path,
                        default=Path(__file__).resolve().parents[1] / "third_party" / "MLSimp",
                        help="MLSimp checkout (default: the third_party/MLSimp submodule)")
    parser.add_argument("--data", type=Path, required=True, help="export folder (<export-db>/seed<s>)")
    parser.add_argument("--dataset", default="Geolife")
    parser.add_argument("--threads", type=int, help="CPU threads for torch (default: all cores)")
    parser.add_argument("--limit", type=int, help="stop after this many trajectories (to test)")
    parser.add_argument("--save-every", type=int, default=1000, help="trajectories per checkpoint")
    args = parser.parse_args()

    data = args.data.resolve()
    offsets, lat, lon, t = load_database(data)
    n = len(offsets) - 1
    stop = n if args.limit is None else min(n, args.limit)

    os.chdir(args.mlsimp.resolve())
    sys.path.insert(0, str(Path.cwd()))
    import torch
    import torch.nn.functional as F
    import yaml
    from Model.GraphSimp import GAT
    from Model.tbert import TBERT, TBERTEmbedding, TemporalEncoding
    from Utils.grid_utils import Grid

    with open("Setting.yaml") as file:
        params = yaml.safe_load(file)[args.dataset]
    torch.manual_seed(params["seed"])
    if args.threads:
        torch.set_num_threads(args.threads)
    bert_params = params["Bert"]
    device = "cpu"

    grid = Grid(params["minlon"], params["minlat"], params["maxlon"], params["maxlat"],
                params["step"], params["step"], params["dataset"], params["minfreq"],
                params["max_grid_size"], params["k"], params["grid_start"])
    grid_ids = np.array(grid.gridmap)

    embed_size = bert_params["embed_size"]
    tbert = TBERT(TBERTEmbedding(TemporalEncoding(embed_size), embed_size,
                                 bert_params["grid_size"]),
                  embed_size * 4, num_layers=bert_params["tbert_num_layers"],
                  num_heads=bert_params["tbert_num_heads"])
    tbert.load_state_dict(torch.load(bert_params["pretain_path"], map_location=device))
    tbert.eval()
    gat = GAT(params["GAT"])
    gat_state = torch.load(params["GAT"]["save_path"], map_location=device)
    for conv in ("conv1", "conv2"):
        source = gat_state.pop(f"{conv}.lin_src.weight", None)
        target = gat_state.pop(f"{conv}.lin_dst.weight", None)
        if source is not None:
            assert torch.equal(source, target)
            gat_state[f"{conv}.lin.weight"] = source
    gat.load_state_dict(gat_state)
    gat.eval()

    scores_path = data / "scores.npy"
    progress_path = data / "scores_done.txt"
    scores = np.zeros(len(lat), dtype=np.float32)
    first = 0
    if scores_path.exists() and progress_path.exists():
        scores = np.load(scores_path)
        first = int(progress_path.read_text())
        print(f"resuming at trajectory {first} of {n}")

    def tokens(begin, end):
        from Utils.data_utils import lonlat2meters_np
        x, y = lonlat2meters_np(lon[begin:end], lat[begin:end])
        cx = (np.round(x - grid.minx, 6) / grid.xstep).astype(int).clip(0, grid.numx - 1)
        cy = (np.round(y - grid.miny, 6) / grid.ystep).astype(int).clip(0, grid.numy - 1)
        return grid_ids[cy * grid.numx + cx]

    started = time.time()
    with torch.no_grad():
        for i in range(first, stop):
            begin, end = offsets[i], offsets[i + 1]
            seq = tokens(begin, end)
            times = t[begin:end].astype(np.float64)

            windows = segments(len(seq), params["max_length"], params["overlap"])
            seg_tokens = np.array(list(zip_longest(*[seq[s:e] for s, e, _, _ in windows],
                                                   fillvalue=0))).T
            seg_times = np.array(list(zip_longest(*[times[s:e] for s, e, _, _ in windows],
                                                  fillvalue=0))).T
            out = tbert(torch.tensor(seg_tokens).long(),
                        timestamp=torch.tensor(seg_times).float())
            point_parts = [out[j][vs:ve] for j, (_, _, vs, ve) in enumerate(windows)]
            point_emb = torch.cat(point_parts, dim=0)
            seg_emb = torch.stack([part.mean(dim=0) for part in point_parts])

            m = point_emb.size(0)
            neighbors = min(11, m)
            unit = F.normalize(point_emb, dim=1, eps=1e-8)
            sim = unit @ unit.T
            neighbor = torch.topk(sim, neighbors, dim=-1).indices[:, 1:].T
            del sim
            feature = torch.cat((point_emb, seg_emb), dim=0)
            point_nodes = torch.arange(m)
            seg_nodes = torch.arange(m, m + seg_emb.size(0))
            edge_index = torch.cartesian_prod(point_nodes, seg_nodes).t().contiguous()

            z = gat(feature, edge_index)[point_nodes]
            scores[begin:end] = gat.important_sigmoid(z, neighbor).numpy()

            done = i + 1
            if done % args.save_every == 0 or done == stop:
                np.save(scores_path, scores)
                progress_path.write_text(str(done))
                elapsed = time.time() - started
                rate = (done - first) / elapsed
                print(f"{done}/{n} trajectories, {elapsed / 60:.1f} min, "
                      f"~{(n - done) / rate / 60:.0f} min left", flush=True)

    print(f"wrote {scores_path}")


if __name__ == "__main__":
    main()
