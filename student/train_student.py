from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ANCHORS = np.array([2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512], dtype=np.float32)
LOG_ANCHORS = np.log(ANCHORS)
MODEL_MAGIC = 0x53544431  # "STD1"
MODEL_VERSION = 1


def load_export(export_dir: Path):
    meta = json.loads((export_dir / "meta.json").read_text())
    n_trajs, n_feats = int(meta["num_trajs"]), int(meta["num_features"])
    X = np.fromfile(export_dir / "features.bin", dtype=np.float32).reshape(n_trajs, n_feats)
    lengths = np.fromfile(export_dir / "lengths.bin", dtype=np.uint32).astype(np.int64)
    shares = [float(s) for s in meta.get("shares", [0.0025, 0.01, 0.02])]
    simplex = np.array(json.loads((export_dir / "presets.json").read_text()), dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), lengths, shares, np.log(simplex)


def standardize(X: np.ndarray):
    mean = X.mean(axis=0).astype(np.float32)
    std = np.where(X.std(axis=0) < 1e-6, 1.0, X.std(axis=0)).astype(np.float32)
    return (X - mean) / std, mean, std


def build_mlp(dims: list[int], dropout_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i + 1 < len(dims) - 1:
            layers += [nn.LayerNorm(dims[i + 1]), nn.SiLU()]
            if i < dropout_layers:
                layers.append(nn.Dropout(0.05))
    return nn.Sequential(*layers)


class MonotoneMarginalCurveNet(nn.Module):
    """Predicts strictly decreasing log-marginal-utility curves ell_i(a_m) across log anchors."""

    def __init__(self, in_dim: int, hidden: int = 64, num_anchors: int = len(ANCHORS)):
        super().__init__()
        self.net = build_mlp([in_dim, hidden, hidden, num_anchors], dropout_layers=2)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        start = raw[:, :1]
        drops = F.softplus(raw[:, 1:]) + 1e-2  # strictly positive drops enforce monotonicity
        return torch.cat([start, start - torch.cumsum(drops, dim=1)], dim=1)


def eval_continuous_budget(ell: torch.Tensor, eta: torch.Tensor, lengths: torch.Tensor, log_anc: torch.Tensor):
    frac = torch.clamp((ell[:, :-1] - eta) / (ell[:, :-1] - ell[:, 1:]), 0.0, 1.0)
    log_b = log_anc[0] + (frac * (log_anc[1:] - log_anc[:-1]).unsqueeze(0)).sum(dim=1)
    last_slope = (ell[:, -2] - ell[:, -1]) / (log_anc[-1] - log_anc[-2])
    tail = torch.clamp((ell[:, -1] - eta) / (last_slope + 1e-3), min=0.0, max=4.0)
    return torch.minimum(torch.clamp(torch.exp(log_b + tail), min=2.0), lengths.float())


def differentiable_water_filling(ell: torch.Tensor, lengths: torch.Tensor, B: float, log_anc: torch.Tensor):
    """1D Lagrangian water-filling with exact Implicit Function Theorem (IFT) backward pass."""
    with torch.no_grad():
        lo, hi = ell.min() - 15.0, ell.max() + 2.0
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            if eval_continuous_budget(ell, mid, lengths, log_anc).sum() > B:
                lo = mid
            else:
                hi = mid
        eta_star = 0.5 * (lo + hi)

    # IFT: since F(ell, eta*) = sum(b_i) - B = 0, d(eta*)/d(ell) = -(dF/d(ell)) / (dF/d(eta))
    eta_var = eta_star.detach().requires_grad_(True)
    dF_deta = torch.autograd.grad(
        eval_continuous_budget(ell.detach(), eta_var, lengths, log_anc).sum(), eta_var
    )[0].detach().clamp(max=-1e-3)
    F_val = eval_continuous_budget(ell, eta_star.detach(), lengths, log_anc).sum() - B
    eta_ift = eta_star.detach() - F_val / dF_deta
    return eval_continuous_budget(ell, eta_ift, lengths, log_anc)


def train_stage1(X: np.ndarray, lengths: np.ndarray, budgets: dict[float, np.ndarray], epochs: int, seed: int, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xn, mean, std = standardize(X)
    x_t = torch.from_numpy(Xn).to(device)
    len_t = torch.from_numpy(lengths).to(device)
    log_anc = torch.from_numpy(LOG_ANCHORS).to(device)
    targets = {s: (torch.from_numpy(b.astype(np.float32)).to(device), float(b.sum())) for s, b in budgets.items()}

    model = MonotoneMarginalCurveNet(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        ell = model(x_t)
        loss = torch.tensor(0.0, device=device)
        for b_star, B in targets.values():
            b_hat = differentiable_water_filling(ell, len_t, B, log_anc)
            loss = loss + F.huber_loss(torch.log(b_hat), torch.log(b_star), delta=0.25)
            loss = loss + 0.15 * F.l1_loss(b_hat, b_star) / (b_star.mean() + 1e-6)
        loss = loss / len(targets) + 1e-4 * ((ell[:, :-1] - ell[:, 1:]) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
    model.eval()
    return model, mean, std


def train_stage2(
    train_dir: Path,
    X: np.ndarray,
    lengths: np.ndarray,
    budgets: dict[float, np.ndarray],
    num_presets: int,
    epochs: int,
    seed: int,
    device: str,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xs, Ps, Ws = [], [], []
    for s, b_gt in budgets.items():
        losses = np.fromfile(train_dir / f"preset_losses_{s:g}.bin", dtype=np.float32).reshape(
            len(lengths), num_presets
        )
        active = b_gt > 2
        b = np.maximum(2.0, b_gt[active].astype(np.float32))
        n = np.maximum(2.0, lengths[active].astype(np.float32))
        Xs.append(np.hstack([X[active], np.stack([np.log(b), b / n, np.log(n / b)], axis=1)]))
        # Boltzmann soft target over simplex presets (tau=0.25 focuses mass on best-matching presets)
        L_shift = losses[active] - losses[active].min(axis=1, keepdims=True)
        exp_t = np.exp(np.clip(-L_shift / 0.25, -30.0, 0.0))
        Ps.append(exp_t / exp_t.sum(axis=1, keepdims=True))
        Ws.append(np.log(b))

    Xn, mean, std = standardize(np.vstack(Xs))
    x_t = torch.from_numpy(Xn).to(device)
    p_t = torch.from_numpy(np.vstack(Ps).astype(np.float32)).to(device)
    w_t = torch.from_numpy(np.concatenate(Ws).astype(np.float32)).to(device)
    w_t = w_t / w_t.mean()

    model = build_mlp([Xn.shape[1], 64, 64, num_presets], dropout_layers=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    N = len(Xn)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        for st in range(0, N, 1024):
            idx = perm[st : st + 1024]
            loss = (-(p_t[idx] * F.log_softmax(model(x_t[idx]), dim=1)).sum(dim=1) * w_t[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model, mean, std


def save_model(out_path: Path, s1, s1_mean, s1_std, s2, s2_mean, s2_std, log_simplex: np.ndarray):
    def write_u32(f, val: int) -> None:
        f.write(struct.pack("<I", int(val)))

    def write_f32(f, arr: np.ndarray) -> None:
        f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())

    def write_seq(f, seq: nn.Sequential) -> None:
        mods, layers, i = list(seq), [], 0
        while i < len(mods):
            if isinstance(mods[i], nn.Linear):
                ln = mods[i + 1] if (i + 1 < len(mods) and isinstance(mods[i + 1], nn.LayerNorm)) else None
                layers.append((mods[i], ln))
                i += 2 if ln is not None else 1
            else:
                i += 1
        write_u32(f, len(layers))
        for lin, ln in layers:
            write_u32(f, lin.in_features)
            write_u32(f, lin.out_features)
            write_u32(f, 1 if ln is not None else 0)
            write_f32(f, lin.weight.detach().cpu().numpy())
            write_f32(f, lin.bias.detach().cpu().numpy())
            if ln is not None:
                write_f32(f, ln.weight.detach().cpu().numpy())
                write_f32(f, ln.bias.detach().cpu().numpy())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        write_u32(f, MODEL_MAGIC)
        write_u32(f, MODEL_VERSION)

        write_u32(f, len(s1_mean))
        write_u32(f, len(LOG_ANCHORS))
        write_f32(f, s1_mean)
        write_f32(f, s1_std)
        write_f32(f, LOG_ANCHORS)
        write_seq(f, s1.net)

        write_u32(f, len(s2_mean))
        write_u32(f, len(log_simplex))
        write_f32(f, s2_mean)
        write_f32(f, s2_std)
        write_f32(f, log_simplex)
        write_seq(f, s2)


def train_and_export(
    train_dir: Path,
    out_model: Path,
    epochs_s1: int = 350,
    epochs_s2: int = 60,
    seed: int = 42,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, lengths, shares, log_simplex = load_export(train_dir)
    budgets = {
        s: np.fromfile(train_dir / f"teacher_budgets_{s:g}.bin", dtype=np.uint32).astype(np.int64)
        for s in shares
    }
    t0 = time.perf_counter()
    s1, s1_mean, s1_std = train_stage1(X, lengths, budgets, epochs_s1, seed, device)
    t1 = time.perf_counter()
    s2, s2_mean, s2_std = train_stage2(train_dir, X, lengths, budgets, len(log_simplex), epochs_s2, seed, device)
    t2 = time.perf_counter()
    save_model(out_model, s1, s1_mean, s1_std, s2, s2_mean, s2_std, log_simplex)
    print(f"Saved {out_model} ({out_model.stat().st_size / 1024:.1f} KB) | S1 {t1-t0:.1f}s, S2 {t2-t1:.1f}s")
    return out_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--out-model", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train_and_export(args.train_dir, args.out_model, seed=args.seed)


if __name__ == "__main__":
    main()
