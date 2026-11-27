#!/usr/bin/env python3
"""
Offline Student Trainer (Stage 1 DWF-Alloc + Stage 2 Param-Submod).
Reads a C++ `student_tool export` + `fit-simplex` directory, trains both tiny MLPs,
and exports a single self-contained ~49 KB binary (`student_model.bin`) for pure-C++ inference.
"""

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
    num_trajs, num_features = int(meta["num_trajs"]), int(meta["num_features"])
    features = np.fromfile(export_dir / "features.bin", dtype=np.float32).reshape(num_trajs, num_features)
    lengths = np.fromfile(export_dir / "lengths.bin", dtype=np.uint32).astype(np.int64)
    shares = [float(share) for share in meta.get("shares", [0.0025, 0.01, 0.02])]
    simplex = np.array(json.loads((export_dir / "presets.json").read_text()), dtype=np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), lengths, shares, np.log(simplex)


def standardize(features: np.ndarray):
    mean = features.mean(axis=0).astype(np.float32)
    std = np.where(features.std(axis=0) < 1e-6, 1.0, features.std(axis=0)).astype(np.float32)
    return (features - mean) / std, mean, std


def build_mlp(dims: list[int], dropout_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for layer_idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[layer_idx], dims[layer_idx + 1]))
        if layer_idx + 1 < len(dims) - 1:
            layers += [nn.LayerNorm(dims[layer_idx + 1]), nn.SiLU()]
            if layer_idx < dropout_layers:
                layers.append(nn.Dropout(0.05))
    return nn.Sequential(*layers)


class MonotoneMarginalCurveNet(nn.Module):
    """Predicts strictly decreasing log-marginal-utility curves across log budget anchors."""

    def __init__(self, in_dim: int, hidden: int = 64, num_anchors: int = len(ANCHORS)):
        super().__init__()
        self.net = build_mlp([in_dim, hidden, hidden, num_anchors], dropout_layers=2)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        start = raw[:, :1]
        drops = F.softplus(raw[:, 1:]) + 1e-2  # strictly positive drops enforce monotonicity
        return torch.cat([start, start - torch.cumsum(drops, dim=1)], dim=1)


def eval_continuous_budget(
    marginal_curves: torch.Tensor, eta: torch.Tensor, lengths: torch.Tensor, log_anchors: torch.Tensor
):
    frac = torch.clamp(
        (marginal_curves[:, :-1] - eta) / (marginal_curves[:, :-1] - marginal_curves[:, 1:]), 0.0, 1.0
    )
    log_budget = log_anchors[0] + (frac * (log_anchors[1:] - log_anchors[:-1]).unsqueeze(0)).sum(dim=1)
    last_slope = (marginal_curves[:, -2] - marginal_curves[:, -1]) / (log_anchors[-1] - log_anchors[-2])
    tail = torch.clamp((marginal_curves[:, -1] - eta) / (last_slope + 1e-3), min=0.0, max=4.0)
    return torch.minimum(torch.clamp(torch.exp(log_budget + tail), min=2.0), lengths.float())


def differentiable_water_filling(
    marginal_curves: torch.Tensor, lengths: torch.Tensor, total_budget: float, log_anchors: torch.Tensor
):
    """1D Lagrangian water-filling with exact Implicit Function Theorem (IFT) backward pass."""
    with torch.no_grad():
        low_eta, high_eta = marginal_curves.min() - 15.0, marginal_curves.max() + 2.0
        for _ in range(36):
            mid_eta = 0.5 * (low_eta + high_eta)
            if eval_continuous_budget(marginal_curves, mid_eta, lengths, log_anchors).sum() > total_budget:
                low_eta = mid_eta
            else:
                high_eta = mid_eta
        eta_star = 0.5 * (low_eta + high_eta)

    # IFT: since F(curves, eta*) = sum(b_i) - B = 0, d(eta*)/d(curves) = -(dF/d(curves)) / (dF/d(eta))
    eta_var = eta_star.detach().requires_grad_(True)
    dF_deta = torch.autograd.grad(
        eval_continuous_budget(marginal_curves.detach(), eta_var, lengths, log_anchors).sum(), eta_var
    )[0].detach().clamp(max=-1e-3)
    constraint_residual = (
        eval_continuous_budget(marginal_curves, eta_star.detach(), lengths, log_anchors).sum() - total_budget
    )
    eta_ift = eta_star.detach() - constraint_residual / dF_deta
    return eval_continuous_budget(marginal_curves, eta_ift, lengths, log_anchors)


def train_stage1(
    features: np.ndarray,
    lengths: np.ndarray,
    budgets: dict[float, np.ndarray],
    epochs: int,
    seed: int,
    device: str,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    norm_features, mean, std = standardize(features)
    features_t = torch.from_numpy(norm_features).to(device)
    lengths_t = torch.from_numpy(lengths).to(device)
    log_anchors_t = torch.from_numpy(LOG_ANCHORS).to(device)
    targets = {
        share: (torch.from_numpy(budget.astype(np.float32)).to(device), float(budget.sum()))
        for share, budget in budgets.items()
    }

    model = MonotoneMarginalCurveNet(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        marginal_curves = model(features_t)
        loss = torch.tensor(0.0, device=device)
        for target_budgets, total_budget in targets.values():
            pred_budgets = differentiable_water_filling(marginal_curves, lengths_t, total_budget, log_anchors_t)
            loss = loss + F.huber_loss(torch.log(pred_budgets), torch.log(target_budgets), delta=0.25)
            loss = loss + 0.15 * F.l1_loss(pred_budgets, target_budgets) / (target_budgets.mean() + 1e-6)
        loss = loss / len(targets) + 1e-4 * ((marginal_curves[:, :-1] - marginal_curves[:, 1:]) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
    model.eval()
    return model, mean, std


def train_stage2(
    train_dir: Path,
    features: np.ndarray,
    lengths: np.ndarray,
    budgets: dict[float, np.ndarray],
    num_presets: int,
    epochs: int,
    seed: int,
    device: str,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    feature_blocks, target_probs, sample_weights = [], [], []
    for share, teacher_budgets in budgets.items():
        preset_losses = np.fromfile(train_dir / f"preset_losses_{share:g}.bin", dtype=np.float32).reshape(
            len(lengths), num_presets
        )
        active = teacher_budgets > 2
        budget_f = np.maximum(2.0, teacher_budgets[active].astype(np.float32))
        length_f = np.maximum(2.0, lengths[active].astype(np.float32))
        feature_blocks.append(
            np.hstack([features[active], np.stack([np.log(budget_f), budget_f / length_f, np.log(length_f / budget_f)], axis=1)])
        )
        # Boltzmann soft target over simplex presets (tau=0.25 focuses mass on best-matching presets)
        shifted_losses = preset_losses[active] - preset_losses[active].min(axis=1, keepdims=True)
        boltzmann_weights = np.exp(np.clip(-shifted_losses / 0.25, -30.0, 0.0))
        target_probs.append(boltzmann_weights / boltzmann_weights.sum(axis=1, keepdims=True))
        sample_weights.append(np.log(budget_f))

    norm_features, mean, std = standardize(np.vstack(feature_blocks))
    features_t = torch.from_numpy(norm_features).to(device)
    probs_t = torch.from_numpy(np.vstack(target_probs).astype(np.float32)).to(device)
    weights_t = torch.from_numpy(np.concatenate(sample_weights).astype(np.float32)).to(device)
    weights_t = weights_t / weights_t.mean()

    model = build_mlp([norm_features.shape[1], 64, 64, num_presets], dropout_layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    num_samples = len(norm_features)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(num_samples, device=device)
        for batch_start in range(0, num_samples, 1024):
            batch_idx = perm[batch_start : batch_start + 1024]
            loss = (
                -(probs_t[batch_idx] * F.log_softmax(model(features_t[batch_idx]), dim=1)).sum(dim=1)
                * weights_t[batch_idx]
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    return model, mean, std


def save_model(
    out_path: Path,
    budget_net,
    s1_mean,
    s1_std,
    weight_net,
    s2_mean,
    s2_std,
    log_simplex: np.ndarray,
):
    def write_u32(out_file, val: int) -> None:
        out_file.write(struct.pack("<I", int(val)))

    def write_f32(out_file, arr: np.ndarray) -> None:
        out_file.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())

    def write_seq(out_file, seq: nn.Sequential) -> None:
        modules, layers, idx = list(seq), [], 0
        while idx < len(modules):
            if isinstance(modules[idx], nn.Linear):
                layernorm = (
                    modules[idx + 1]
                    if (idx + 1 < len(modules) and isinstance(modules[idx + 1], nn.LayerNorm))
                    else None
                )
                layers.append((modules[idx], layernorm))
                idx += 2 if layernorm is not None else 1
            else:
                idx += 1
        write_u32(out_file, len(layers))
        for linear, layernorm in layers:
            write_u32(out_file, linear.in_features)
            write_u32(out_file, linear.out_features)
            write_u32(out_file, 1 if layernorm is not None else 0)
            write_f32(out_file, linear.weight.detach().cpu().numpy())
            write_f32(out_file, linear.bias.detach().cpu().numpy())
            if layernorm is not None:
                write_f32(out_file, layernorm.weight.detach().cpu().numpy())
                write_f32(out_file, layernorm.bias.detach().cpu().numpy())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as out_file:
        write_u32(out_file, MODEL_MAGIC)
        write_u32(out_file, MODEL_VERSION)

        write_u32(out_file, len(s1_mean))
        write_u32(out_file, len(LOG_ANCHORS))
        write_f32(out_file, s1_mean)
        write_f32(out_file, s1_std)
        write_f32(out_file, LOG_ANCHORS)
        write_seq(out_file, budget_net.net)

        write_u32(out_file, len(s2_mean))
        write_u32(out_file, len(log_simplex))
        write_f32(out_file, s2_mean)
        write_f32(out_file, s2_std)
        write_f32(out_file, log_simplex)
        write_seq(out_file, weight_net)


def train_and_export(
    train_dir: Path,
    out_model: Path,
    epochs_s1: int = 350,
    epochs_s2: int = 60,
    seed: int = 42,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features, lengths, shares, log_simplex = load_export(train_dir)
    budgets = {
        share: np.fromfile(train_dir / f"teacher_budgets_{share:g}.bin", dtype=np.uint32).astype(np.int64)
        for share in shares
    }
    t_start = time.perf_counter()
    budget_net, s1_mean, s1_std = train_stage1(features, lengths, budgets, epochs_s1, seed, device)
    t_stage1 = time.perf_counter()
    weight_net, s2_mean, s2_std = train_stage2(
        train_dir, features, lengths, budgets, len(log_simplex), epochs_s2, seed, device
    )
    t_stage2 = time.perf_counter()
    save_model(out_model, budget_net, s1_mean, s1_std, weight_net, s2_mean, s2_std, log_simplex)
    print(
        f"Saved {out_model} ({out_model.stat().st_size / 1024:.1f} KB) | "
        f"S1 {t_stage1 - t_start:.1f}s, S2 {t_stage2 - t_stage1:.1f}s"
    )
    return out_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_and_export(args.train_dir, args.out_model, seed=args.seed)


if __name__ == "__main__":
    main()
