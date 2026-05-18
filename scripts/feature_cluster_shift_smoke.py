#!/usr/bin/env python3
"""Real-feature proxy-environment continual shift smoke test.

Unlike shortcut_reversal_smoke.py, this script does not append a synthetic
shortcut dimension. It derives a proxy environment from real CONCH slide-level
embeddings, then constructs Task 1 and Task 2 by sampling opposite
label-environment correlations from real slides.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from shortcut_reversal_smoke import Logistic, acc, clone_from, load_cache, standardize


def pca_environment(x: torch.Tensor) -> torch.Tensor:
    x_centered = x - x.mean(dim=0, keepdim=True)
    # For 480 x 512 cached embeddings this is cheap. It keeps the script
    # dependency-free and avoids sklearn.
    _, _, vh = torch.linalg.svd(x_centered, full_matrices=False)
    score = x_centered @ vh[0]
    return (score > score.median()).long()


def split_cells(
    y: torch.Tensor,
    env: torch.Tensor,
    *,
    n_major_per_cell: int,
    n_minor_per_cell: int,
    n_test_per_cell: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    rng = random.Random(seed)
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (yy, ee) in enumerate(zip(y.tolist(), env.tolist())):
        cells[(int(yy), int(ee))].append(i)
    need = n_major_per_cell + n_minor_per_cell + n_test_per_cell
    for key, idxs in sorted(cells.items()):
        if len(idxs) < need:
            raise ValueError(f"Cell {key} has {len(idxs)} samples, need {need}")
        rng.shuffle(idxs)

    # Task 1: env mostly agrees with label, but each label still has both
    # environments so conditional environment-dependence is estimable.
    task1 = (
        cells[(0, 0)][:n_major_per_cell]
        + cells[(1, 1)][:n_major_per_cell]
        + cells[(0, 1)][:n_minor_per_cell]
        + cells[(1, 0)][:n_minor_per_cell]
    )
    # Task 2 reverses the correlation using disjoint examples.
    task2 = (
        cells[(0, 1)][n_minor_per_cell : n_minor_per_cell + n_major_per_cell]
        + cells[(1, 0)][n_minor_per_cell : n_minor_per_cell + n_major_per_cell]
        + cells[(0, 0)][n_major_per_cell : n_major_per_cell + n_major_per_cell + n_minor_per_cell]
        + cells[(1, 1)][n_major_per_cell : n_major_per_cell + n_major_per_cell + n_minor_per_cell]
    )
    test_bal = []
    old_corr = []
    reversed_corr = []
    for key in [(0, 0), (1, 1)]:
        start = n_major_per_cell + n_minor_per_cell
        old_corr.extend(cells[key][start : start + n_test_per_cell])
    for key in [(0, 1), (1, 0)]:
        start = n_major_per_cell + n_minor_per_cell
        reversed_corr.extend(cells[key][start : start + n_test_per_cell])
    test_bal.extend(old_corr)
    test_bal.extend(reversed_corr)
    rng.shuffle(task1)
    rng.shuffle(task2)
    rng.shuffle(test_bal)
    return {
        "task1": torch.tensor(task1),
        "task2": torch.tensor(task2),
        "test_balanced": torch.tensor(test_bal),
        "test_old_corr": torch.tensor(old_corr),
        "test_reversed": torch.tensor(reversed_corr),
    }


def residual_env_corr_scores(x: torch.Tensor, y: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
    resid = x.clone()
    for cls in [0, 1]:
        mask = y == cls
        resid[mask] = resid[mask] - resid[mask].mean(dim=0, keepdim=True)
    env_sign = env.float().mul(2).sub(1)
    env_centered = env_sign - env_sign.mean()
    resid_centered = resid - resid.mean(dim=0, keepdim=True)
    denom = resid_centered.norm(dim=0).clamp_min(1e-8) * env_centered.norm().clamp_min(1e-8)
    corr = (resid_centered * env_centered.unsqueeze(1)).sum(dim=0).abs() / denom
    return (1.0 - corr).clamp(0.0, 1.0)


def train_plain(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, epochs: int, lr: float) -> nn.Module:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def train_l2(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    old_state: dict[str, torch.Tensor],
    l2_lambda: float,
    weights: torch.Tensor | None = None,
    anti_threshold: float | None = None,
    anti_penalty: float = 0.0,
) -> nn.Module:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    old_weight = old_state["linear.weight"].detach().clone()
    old_bias = old_state["linear.bias"].detach().clone()
    if weights is None:
        weights = torch.ones(old_weight.shape[1])
    weights2d = weights.view(1, -1)
    for _ in range(epochs):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        diff = (model.linear.weight - old_weight).pow(2) * weights2d
        loss = loss + l2_lambda * diff.sum() / weights2d.sum().clamp_min(1e-8)
        loss = loss + 0.1 * l2_lambda * (model.linear.bias - old_bias).pow(2).mean()
        if anti_threshold is not None and anti_penalty > 0:
            low = (weights < anti_threshold).float().view(1, -1)
            anti = model.linear.weight.pow(2) * low
            loss = loss + anti_penalty * anti.sum() / low.sum().clamp_min(1e-8)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def env_pred_corr(model: nn.Module, x: torch.Tensor, env: torch.Tensor) -> float:
    prob = model(x).softmax(dim=1)[:, 1]
    env_sign = env.float().mul(2).sub(1)
    prob = prob - prob.mean()
    env_sign = env_sign - env_sign.mean()
    denom = prob.norm() * env_sign.norm()
    if float(denom) == 0.0:
        return 0.0
    return float((prob @ env_sign / denom).item())


@torch.no_grad()
def group_metrics(model: nn.Module, x: torch.Tensor, y: torch.Tensor, env: torch.Tensor) -> dict[str, float]:
    out: dict[str, float] = {}
    vals = []
    for yy in [0, 1]:
        for ee in [0, 1]:
            mask = (y == yy) & (env == ee)
            if int(mask.sum().item()) == 0:
                continue
            a = acc(model, x[mask], y[mask])
            out[f"acc_y{yy}_e{ee}"] = a
            vals.append(a)
    out["worst_group_acc"] = min(vals) if vals else 0.0
    return out


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, env: torch.Tensor, splits: dict[str, torch.Tensor]) -> dict[str, float]:
    out = {}
    for name in ["task1", "task2", "test_balanced", "test_old_corr", "test_reversed"]:
        idx = splits[name]
        out[f"{name}_acc"] = acc(model, x[idx], y[idx])
        out[f"{name}_env_pred_corr"] = env_pred_corr(model, x[idx], env[idx])
    out.update(group_metrics(model, x[splits["test_balanced"]], y[splits["test_balanced"]], env[splits["test_balanced"]]))
    out["old_minus_reversed_acc"] = out["test_old_corr_acc"] - out["test_reversed_acc"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="/data_2_4T/data_zjj/continual_wsi/smoke_multicancer/max60_seed7/mean_features_max60_seed7.pt")
    parser.add_argument("--out-dir", default="/data_2_4T/data_zjj/continual_wsi/feature_cluster_shift/seed7")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-major-per-cell", type=int, default=45)
    parser.add_argument("--n-minor-per-cell", type=int, default=15)
    parser.add_argument("--n-test-per-cell", type=int, default=15)
    parser.add_argument("--epochs-task1", type=int, default=300)
    parser.add_argument("--epochs-task2", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--l2-lambda", type=float, default=80.0)
    parser.add_argument("--score-power", type=float, default=4.0)
    parser.add_argument("--anti-threshold", type=float, default=0.2)
    parser.add_argument("--anti-penalty", type=float, default=500.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_raw, y_multi = load_cache(Path(args.cache))
    y = (y_multi >= 4).long()
    env = pca_environment(x_raw)
    splits = split_cells(
        y,
        env,
        n_major_per_cell=args.n_major_per_cell,
        n_minor_per_cell=args.n_minor_per_cell,
        n_test_per_cell=args.n_test_per_cell,
        seed=args.seed,
    )
    train_idx = torch.cat([splits["task1"], splits["task2"]])
    x = standardize(x_raw, train_idx)

    base = Logistic(x.shape[1])
    train_plain(base, x[splits["task1"]], y[splits["task1"]], epochs=args.epochs_task1, lr=args.lr)
    old_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    stability_raw = residual_env_corr_scores(x[splits["task1"]], y[splits["task1"]], env[splits["task1"]])
    stability = stability_raw.pow(args.score_power)
    random_scores = torch.rand(stability.shape, generator=torch.Generator().manual_seed(args.seed + 99))

    models = {
        "task1_only": base,
        "finetune": train_plain(clone_from(base), x[splits["task2"]], y[splits["task2"]], epochs=args.epochs_task2, lr=args.lr),
        "l2_all": train_l2(
            clone_from(base),
            x[splits["task2"]],
            y[splits["task2"]],
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
        ),
        "streaming_score_l2": train_l2(
            clone_from(base),
            x[splits["task2"]],
            y[splits["task2"]],
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
            weights=stability,
        ),
        "streaming_score_anti": train_l2(
            clone_from(base),
            x[splits["task2"]],
            y[splits["task2"]],
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
            weights=stability,
            anti_threshold=args.anti_threshold,
            anti_penalty=args.anti_penalty,
        ),
        "random_score_l2": train_l2(
            clone_from(base),
            x[splits["task2"]],
            y[splits["task2"]],
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
            weights=random_scores,
        ),
    }

    cell_counts = {}
    for yy in [0, 1]:
        for ee in [0, 1]:
            cell_counts[f"y{yy}_e{ee}"] = int(((y == yy) & (env == ee)).sum().item())

    rows = []
    result: dict[str, object] = {
        "seed": args.seed,
        "cache": args.cache,
        "cell_counts": cell_counts,
        "n_major_per_cell": args.n_major_per_cell,
        "n_minor_per_cell": args.n_minor_per_cell,
        "n_test_per_cell": args.n_test_per_cell,
        "l2_lambda": args.l2_lambda,
        "score_power": args.score_power,
        "anti_threshold": args.anti_threshold,
        "anti_penalty": args.anti_penalty,
        "stability_summary": {
            "raw_mean": float(stability_raw.mean().item()),
            "raw_min": float(stability_raw.min().item()),
            "raw_max": float(stability_raw.max().item()),
            "powered_mean": float(stability.mean().item()),
            "powered_min": float(stability.min().item()),
            "powered_max": float(stability.max().item()),
        },
        "models": {},
    }
    for name, model in models.items():
        metrics = evaluate(model, x, y, env, splits)
        metrics["weight_norm"] = float(model.linear.weight.norm().item())
        result["models"][name] = metrics
        rows.append({"model": name, **metrics})

    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

