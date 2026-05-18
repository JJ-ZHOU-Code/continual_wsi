#!/usr/bin/env python3
"""Track shortcut carryover during Task-2 training.

This is the reviewer-requested Figure-1 diagnostic: after training Task 1 with a
label-correlated shortcut, continue on Task 2 where the shortcut is reversed and
record how shortcut reliance changes over optimization time.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from shortcut_reversal_smoke import (
    Logistic,
    acc,
    add_shortcut,
    clone_from,
    load_cache,
    make_eval_sets,
    shortcut_weight,
    split_by_class,
    standardize,
    train_model,
)


def task2_step(
    model: Logistic,
    opt: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    old_weight: torch.Tensor | None,
    old_bias: torch.Tensor | None,
    l2_lambda: float,
    selective: bool,
    shortcut_aug: bool,
    shortcut_penalty: float,
) -> None:
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    if shortcut_aug:
        x_aug = x.clone()
        x_aug[:, -1] = 0.0
        loss = 0.5 * loss + 0.5 * F.cross_entropy(model(x_aug), y)
    if old_weight is not None and l2_lambda > 0:
        cur_w = model.linear.weight
        if selective:
            loss = loss + l2_lambda * (cur_w[:, :-1] - old_weight[:, :-1]).pow(2).mean()
        else:
            loss = loss + l2_lambda * (cur_w - old_weight).pow(2).mean()
        loss = loss + l2_lambda * 0.1 * (model.linear.bias - old_bias).pow(2).mean()
    if shortcut_penalty > 0:
        loss = loss + shortcut_penalty * model.linear.weight[:, -1].pow(2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


@torch.no_grad()
def shortcut_corr(model: Logistic, x: torch.Tensor) -> float:
    model.eval()
    prob = model(x).softmax(dim=1)[:, 1]
    shortcut = x[:, -1]
    prob = prob - prob.mean()
    shortcut = shortcut - shortcut.mean()
    denom = prob.norm() * shortcut.norm()
    if float(denom) == 0.0:
        return 0.0
    return float((prob @ shortcut / denom).item())


@torch.no_grad()
def shortcut_sensitivity(model: Logistic, x: torch.Tensor) -> float:
    model.eval()
    prob = model(x).softmax(dim=1)[:, 1]
    x_zero = x.clone()
    x_zero[:, -1] = 0.0
    prob_zero = model(x_zero).softmax(dim=1)[:, 1]
    return float((prob - prob_zero).abs().mean().item())


def record_metrics(
    rows: list[dict[str, object]],
    *,
    model_name: str,
    epoch: int,
    model: Logistic,
    x1: torch.Tensor,
    y1: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    eval_sets: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    row: dict[str, object] = {
        "model": model_name,
        "task2_epoch": epoch,
        "env1_train_acc": acc(model, x1, y1),
        "env2_train_acc": acc(model, x2, y2),
        "shortcut_weight_0": shortcut_weight(model)[0],
        "shortcut_weight_1": shortcut_weight(model)[1],
    }
    for eval_name, (xe, ye) in eval_sets.items():
        row[f"{eval_name}_acc"] = acc(model, xe, ye)
        row[f"{eval_name}_shortcut_corr"] = shortcut_corr(model, xe)
        row[f"{eval_name}_shortcut_sensitivity"] = shortcut_sensitivity(model, xe)
    row["old_minus_reversed_acc"] = row["old_corr_acc"] - row["reversed_acc"]
    row["neutral_random_mean_acc"] = 0.5 * (row["neutral_acc"] + row["random_acc"])
    rows.append(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="/data_2_4T/data_zjj/continual_wsi/smoke_multicancer/max60_seed7/mean_features_max60_seed7.pt")
    parser.add_argument("--out-dir", default="/data_2_4T/data_zjj/continual_wsi/shortcut_trajectory/seed7")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strength", type=float, default=6.0)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--epochs-task1", type=int, default=300)
    parser.add_argument("--epochs-task2", type=int, default=300)
    parser.add_argument("--checkpoints", default="0,1,5,10,25,50,100,200,300")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--l2-lambda", type=float, default=80.0)
    parser.add_argument("--shortcut-penalty", type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted({int(x) for x in args.checkpoints.split(",") if x.strip()})
    max_epoch = max(checkpoints)

    x, y_multi = load_cache(Path(args.cache))
    y = (y_multi >= 4).long()
    splits = split_by_class(y_multi, args.seed)
    train_idx = torch.cat([splits["env1"], splits["env2"]])
    x = standardize(x, train_idx)

    x1 = add_shortcut(x[splits["env1"]], y[splits["env1"]], +1.0, args.strength, args.noise, args.seed + 10)
    y1 = y[splits["env1"]]
    x2 = add_shortcut(x[splits["env2"]], y[splits["env2"]], -1.0, args.strength, args.noise, args.seed + 20)
    y2 = y[splits["env2"]]
    eval_sets = make_eval_sets(x, y, splits["test"], args.strength, args.noise, args.seed + 30)

    base = Logistic(x1.shape[1])
    train_model(base, x1, y1, epochs=args.epochs_task1, lr=args.lr)
    old_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    old_weight = old_state["linear.weight"]
    old_bias = old_state["linear.bias"]

    rows: list[dict[str, object]] = []
    record_metrics(
        rows,
        model_name="task1_only",
        epoch=0,
        model=base,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        eval_sets=eval_sets,
    )

    configs = {
        "finetune": {"l2_lambda": 0.0, "selective": False, "shortcut_aug": False, "shortcut_penalty": 0.0},
        "l2_all": {"l2_lambda": args.l2_lambda, "selective": False, "shortcut_aug": False, "shortcut_penalty": 0.0},
        "selective_l2": {"l2_lambda": args.l2_lambda, "selective": True, "shortcut_aug": False, "shortcut_penalty": 0.0},
        "csr_aug": {"l2_lambda": args.l2_lambda, "selective": True, "shortcut_aug": True, "shortcut_penalty": args.shortcut_penalty},
    }
    for name, cfg in configs.items():
        model = clone_from(base)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for epoch in range(max_epoch + 1):
            if epoch in checkpoints:
                record_metrics(
                    rows,
                    model_name=name,
                    epoch=epoch,
                    model=model,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    eval_sets=eval_sets,
                )
            if epoch == max_epoch:
                break
            task2_step(
                model,
                opt,
                x2,
                y2,
                old_weight=old_weight,
                old_bias=old_bias,
                l2_lambda=float(cfg["l2_lambda"]),
                selective=bool(cfg["selective"]),
                shortcut_aug=bool(cfg["shortcut_aug"]),
                shortcut_penalty=float(cfg["shortcut_penalty"]),
            )

    csv_path = out_dir / "trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "seed": args.seed,
        "strength": args.strength,
        "noise": args.noise,
        "l2_lambda": args.l2_lambda,
        "shortcut_penalty": args.shortcut_penalty,
        "checkpoints": checkpoints,
        "rows": rows,
    }
    (out_dir / "trajectory.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

