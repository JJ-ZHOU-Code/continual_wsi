#!/usr/bin/env python3
"""Concept-probe proxy shift smoke test.

This is a bridge from dimension-wise CONCH scoring to concept/probe-space
scoring. We derive a proxy environment from real slide embeddings, project
slides into CAV-style contrast probes, score each probe activation for
environment stability, and run continual learning in that probe space.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from feature_cluster_shift_smoke import (
    Logistic,
    acc,
    clone_from,
    env_pred_corr,
    evaluate,
    pca_environment,
    residual_env_corr_scores,
    split_cells,
    standardize,
    train_l2,
    train_plain,
)
from shortcut_reversal_smoke import load_cache


def pca_projectors(x: torch.Tensor, k: int) -> torch.Tensor:
    x_centered = x - x.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(x_centered, full_matrices=False)
    return vh[:k].T.contiguous()


def normalize(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-8)


def cell_mean(x: torch.Tensor, y: torch.Tensor, env: torch.Tensor, yy: int, ee: int) -> torch.Tensor:
    mask = (y == yy) & (env == ee)
    if int(mask.sum().item()) == 0:
        raise ValueError(f"Missing cell y={yy}, env={ee}")
    return x[mask].mean(dim=0)


def orthogonal_append(columns: list[torch.Tensor], candidate: torch.Tensor, eps: float = 1e-6) -> bool:
    v = candidate.clone()
    for col in columns:
        v = v - (v @ col) * col
    norm = v.norm()
    if float(norm.item()) < eps:
        return False
    columns.append(v / norm)
    return True


def cav_projectors(
    x: torch.Tensor,
    y: torch.Tensor,
    env: torch.Tensor,
    *,
    num_probes: int,
) -> tuple[torch.Tensor, list[str]]:
    """Build a no-PCA bank of CAV-style Task-1 contrast probes.

    These are not final pathology concepts, but they are closer to TCAV/CBM
    practice than PCA: each direction is a supervised contrast in Task-1 cells.
    """
    m00 = cell_mean(x, y, env, 0, 0)
    m01 = cell_mean(x, y, env, 0, 1)
    m10 = cell_mean(x, y, env, 1, 0)
    m11 = cell_mean(x, y, env, 1, 1)
    global_mean = 0.25 * (m00 + m01 + m10 + m11)
    candidates = [
        ("label_cav", 0.5 * (m10 + m11) - 0.5 * (m00 + m01)),
        ("env_cav", 0.5 * (m01 + m11) - 0.5 * (m00 + m10)),
        ("label_cav_env0", m10 - m00),
        ("label_cav_env1", m11 - m01),
        ("env_cav_y0", m01 - m00),
        ("env_cav_y1", m11 - m10),
        ("interaction_cav", (m11 - m10) - (m01 - m00)),
        ("cell_y0e0", m00 - global_mean),
        ("cell_y0e1", m01 - global_mean),
        ("cell_y1e0", m10 - global_mean),
        ("cell_y1e1", m11 - global_mean),
    ]

    columns: list[torch.Tensor] = []
    names: list[str] = []
    for name, direction in candidates:
        if len(columns) >= num_probes:
            break
        if orthogonal_append(columns, normalize(direction)):
            names.append(name)
    return torch.stack(columns[:num_probes], dim=1).contiguous(), names[:num_probes]


def train_binary_cav(
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    epochs: int,
    lr: float,
) -> torch.Tensor | None:
    target = target.float()
    num_pos = float(target.sum().item())
    num_neg = float(target.numel() - num_pos)
    if num_pos < 2 or num_neg < 2:
        return None
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    opt = torch.optim.AdamW([weight, bias], lr=lr, weight_decay=1e-4)
    pos_weight = torch.tensor(num_neg / max(num_pos, 1.0))
    for _ in range(epochs):
        logits = x @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return normalize(weight.detach())


def learned_cav_projectors(
    x: torch.Tensor,
    y: torch.Tensor,
    env: torch.Tensor,
    *,
    num_probes: int,
    epochs: int,
    lr: float,
) -> tuple[torch.Tensor, list[str]]:
    """Train TCAV-style linear probes from Task-1 pseudo-concepts.

    The pseudo-concepts are intentionally simple and available in a streaming
    setting: label, environment, label within each environment, environment
    within each label, and cell membership. This avoids PCA while giving a
    richer concept bank than a few mean-difference CAVs.
    """
    tasks: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    all_mask = torch.ones_like(y, dtype=torch.bool)
    tasks.append(("probe_label_all", all_mask, y))
    tasks.append(("probe_env_all", all_mask, env))
    for ee in [0, 1]:
        mask = env == ee
        tasks.append((f"probe_label_env{ee}", mask, y[mask]))
    for yy in [0, 1]:
        mask = y == yy
        tasks.append((f"probe_env_y{yy}", mask, env[mask]))
    for yy in [0, 1]:
        for ee in [0, 1]:
            target = ((y == yy) & (env == ee)).long()
            tasks.append((f"probe_cell_y{yy}e{ee}", all_mask, target))

    columns: list[torch.Tensor] = []
    names: list[str] = []
    for name, mask, target in tasks:
        if len(columns) >= num_probes:
            break
        probe = train_binary_cav(x[mask], target, epochs=epochs, lr=lr)
        if probe is None:
            continue
        columns.append(probe)
        names.append(name)
    return torch.stack(columns[:num_probes], dim=1).contiguous(), names[:num_probes]


def project_and_standardize(x: torch.Tensor, projectors: torch.Tensor, train_idx: torch.Tensor) -> torch.Tensor:
    z = x @ projectors
    return standardize(z, train_idx)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="/data_2_4T/data_zjj/continual_wsi/smoke_multicancer/max60_seed7/mean_features_max60_seed7.pt")
    parser.add_argument("--out-dir", default="/data_2_4T/data_zjj/continual_wsi/concept_probe_shift/seed7")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--probe-type", choices=["learned_cav", "cav", "pca"], default="learned_cav")
    parser.add_argument("--num-probes", type=int, default=32)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--n-major-per-cell", type=int, default=45)
    parser.add_argument("--n-minor-per-cell", type=int, default=15)
    parser.add_argument("--n-test-per-cell", type=int, default=15)
    parser.add_argument("--epochs-task1", type=int, default=300)
    parser.add_argument("--epochs-task2", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--l2-lambda", type=float, default=20.0)
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
    x_std = standardize(x_raw, train_idx)
    if args.probe_type == "learned_cav":
        projectors, probe_names = learned_cav_projectors(
            x_std[splits["task1"]],
            y[splits["task1"]],
            env[splits["task1"]],
            num_probes=args.num_probes,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
        )
    elif args.probe_type == "cav":
        projectors, probe_names = cav_projectors(
            x_std[splits["task1"]],
            y[splits["task1"]],
            env[splits["task1"]],
            num_probes=args.num_probes,
        )
    else:
        projectors = pca_projectors(x_std[train_idx], args.num_probes)
        probe_names = [f"pca_{i}" for i in range(projectors.shape[1])]
    z = project_and_standardize(x_std, projectors, train_idx)

    base = Logistic(z.shape[1])
    train_plain(base, z[splits["task1"]], y[splits["task1"]], epochs=args.epochs_task1, lr=args.lr)
    old_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    stability_raw = residual_env_corr_scores(z[splits["task1"]], y[splits["task1"]], env[splits["task1"]])
    stability = stability_raw.pow(args.score_power)
    random_scores = torch.rand(stability.shape, generator=torch.Generator().manual_seed(args.seed + 99))

    models = {
        "task1_only": base,
        "finetune": train_plain(clone_from(base), z[splits["task2"]], y[splits["task2"]], epochs=args.epochs_task2, lr=args.lr),
        "l2_all": train_l2(
            clone_from(base),
            z[splits["task2"]],
            y[splits["task2"]],
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
        ),
        "streaming_score_l2": train_l2(
            clone_from(base),
            z[splits["task2"]],
            y[splits["task2"]],
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
            weights=stability,
        ),
        "streaming_score_anti": train_l2(
            clone_from(base),
            z[splits["task2"]],
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
            z[splits["task2"]],
            y[splits["task2"]],
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
            weights=random_scores,
        ),
    }

    rows = []
    result: dict[str, object] = {
        "seed": args.seed,
        "probe_type": args.probe_type,
        "probe_names": probe_names,
        "num_probes": args.num_probes,
        "probe_epochs": args.probe_epochs,
        "probe_lr": args.probe_lr,
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
        metrics = evaluate(model, z, y, env, splits)
        metrics["weight_norm"] = float(model.linear.weight.norm().item())
        metrics["probe_env_corr_abs_weighted"] = float((model.linear.weight.norm(dim=0) * (1.0 - stability_raw)).mean().item())
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

