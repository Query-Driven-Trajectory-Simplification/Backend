"""Measures how well an MLSimp model separates points, for the pretrained model or one from train.py."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

MLSIMP = Path(__file__).resolve().parents[1] / "third_party" / "MLSimp"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="export folder (<export-db>/seed<s>)")
    parser.add_argument("--model", type=Path, help="models from train.py (default: the authors' pretrained model)")
    parser.add_argument("--trajectories", type=int, default=300, help="random trajectories to measure")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, help="write the measurements to this JSON file")
    args = parser.parse_args()

    data = args.data.resolve()
    model_dir = args.model.resolve() if args.model else None
    out = args.out.resolve() if args.out else None
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import train
    offsets, lat, lon, t = train.load_database(data)
    lengths = np.diff(offsets)
    picked = np.sort(np.random.default_rng(args.seed).choice(len(lengths), args.trajectories, replace=False))

    os.chdir(MLSIMP)
    sys.path.insert(0, str(MLSIMP))
    import torch
    import torch.nn.functional as F
    import yaml
    params = yaml.safe_load(open("Setting.yaml"))["Geolife"]

    if model_dir:
        fixed = json.loads((model_dir / "config.json").read_text())["fixed"]
        grid = train.GridMap(model_dir / "grid.npz")
        tokens_of = grid.tokens
        grid_size = grid.grid_size
        tbert_path, gat_path = model_dir / "tbert.pt", model_dir / "gat.pt"
    else:
        from Utils.grid_utils import Grid
        fixed = False
        g = Grid(params["minlon"], params["minlat"], params["maxlon"], params["maxlat"], params["step"],
                 params["step"], params["dataset"], params["minfreq"], params["max_grid_size"], params["k"],
                 params["grid_start"])
        ids = np.array(g.gridmap)

        def tokens_of(lo, la):
            from Utils.data_utils import lonlat2meters_np
            x, y = lonlat2meters_np(lo, la)
            cx = (np.round(x - g.minx, 6) / g.xstep).astype(int).clip(0, g.numx - 1)
            cy = (np.round(y - g.miny, 6) / g.ystep).astype(int).clip(0, g.numy - 1)
            return ids[cy * g.numx + cx]

        grid_size = params["Bert"]["grid_size"]
        tbert_path, gat_path = params["Bert"]["pretain_path"], params["GAT"]["save_path"]

    tbert = train.make_tbert(torch, grid_size, params["gridemb_dim"])
    tbert.load_state_dict(torch.load(tbert_path, map_location="cpu"))
    tbert.eval()
    gat = train.make_gat(torch, params, fixed)
    state = torch.load(gat_path, map_location="cpu")
    for conv in ("conv1", "conv2"):
        source = state.pop(f"{conv}.lin_src.weight", None)
        state.pop(f"{conv}.lin_dst.weight", None)
        if source is not None:
            state[f"{conv}.lin.weight"] = source
    gat.load_state_dict(state)
    gat.eval()

    distinct, ties, corrs, points = [], 0, [], 0
    all_scores = []
    with torch.no_grad():
        for i in picked:
            b, e = offsets[i], offsets[i + 1]
            seq = tokens_of(lon[b:e], lat[b:e])
            spans = train.windows(e - b, params["max_length"], params["overlap"])
            width = max(s1 - s0 for s0, s1, _, _ in spans)
            src = torch.zeros((len(spans), width), dtype=torch.long)
            ts = torch.zeros((len(spans), width), dtype=torch.float32)
            for row, (s0, s1, _, _) in enumerate(spans):
                src[row, :s1 - s0] = torch.as_tensor(seq[s0:s1])
                ts[row, :s1 - s0] = train.window_times(torch, t[b + s0:b + s1], fixed)
            out_t = tbert(src, timestamp=ts)
            parts = [out_t[j][vs:ve] for j, (_, _, vs, ve) in enumerate(spans)]
            point = torch.cat(parts, 0)
            seg = torch.stack([p.mean(0) for p in parts])
            m = point.size(0)
            unit = F.normalize(point, dim=1, eps=1e-8)
            neighbor = torch.topk(unit @ unit.T, min(11, m), dim=-1).indices[:, 1:].T
            sizes = [ve - vs for _, _, vs, ve in spans]
            z = gat(torch.cat([point, seg], 0), train.edge_index(torch, m, sizes, fixed, "cpu"))[:m]
            align = train.uniqueness(torch, z, neighbor, fixed)
            glob = train.globality(torch, z, fixed)
            score = train.point_scores(torch, align, glob, fixed) if fixed else torch.sigmoid(align * glob)
            score = score.numpy()
            distinct.append(len(torch.unique(point, dim=0)) / m)
            values, counts = np.unique(score, return_counts=True)
            ties += counts.max() if counts.max() > 1 else 0
            points += m
            if align.std() > 0 and score.std() > 0:
                corrs.append(float(np.corrcoef(align.numpy(), score)[0, 1]))
            all_scores.append(score / score.max())

    result = {
        "model": str(model_dir) if model_dir else "pretrained",
        "fixed": fixed,
        "trajectories": int(len(picked)),
        "points": int(points),
        "distinct_embeddings": float(np.mean(distinct)),
        "tied_at_most_common_score": float(ties / points),
        "uniqueness_score_correlation": float(np.mean(corrs)),
        "score_histogram": np.histogram(np.concatenate(all_scores), bins=np.linspace(0, 1, 41))[0].tolist(),
        "score_histogram_edges": np.linspace(0, 1, 41).tolist(),
    }
    print(json.dumps({k: v for k, v in result.items() if k != "score_histogram"}, indent=2))
    if out:
        out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
