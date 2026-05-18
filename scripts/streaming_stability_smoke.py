#!/usr/bin/env python3
"""Streaming environment-invariant selective consolidation smoke test.

This script makes the score-cache assumption explicit. During Task 1, a proxy
environment label is available and correlated with the class label. A synthetic
style shortcut is generated from the environment, so it is predictive only
because environment and class are correlated. At the task boundary, we cache a
per-feature stability score computed from Task-1 data only. Task 2 then reverses
the environment-label correlation, and selective consolidation uses only the
cached stability scores, not old examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from shortcut_reversal_smoke import (
    Logistic,
    acc,
    clone_from,
    load_cache,
    split_by_class,
    standardize,
    train_model,
)


def assign_environment(y: torch.Tensor, corr: float, seed: int) -> torch.Tensor:
    """Assign binary environments with P(env == y) controlled by corr.

    corr=+0.9 makes environment mostly agree with label; corr=-0.9 makes it
    mostly disagree. corr=0.0 gives random environments.
    """
    gen = torch.Generator().manual_seed(seed)
    p_match = 0.5 + 0.5 * corr
    match = torch.rand(len(y), generator=gen) < p_match
    env = torch.where(match, y, 1 - y)
    return env.long()


def append_concepts(
    x: torch.Tensor,
    y: torch.Tensor,
    env: torch.Tensor,
    *,
    causal_strength: float,
    shortcut_strength: float,
    noise: float,
    seed: int,
    neutral_shortcut: bool = False,
    random_shortcut: bool = False,
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    y_sign = y.float().mul(2).sub(1).unsqueeze(1)
    env_sign = env.float().mul(2).sub(1).unsqueeze(1)
    causal = causal_strength * y_sign + noise * torch.randn((len(y), 1), generator=gen)
    if neutral_shortcut:
        shortcut = torch.zeros((len(y), 1))
    elif random_shortcut:
        shortcut = shortcut_strength * torch.randn((len(y), 1), generator=gen)
    else:
        shortcut = shortcut_strength * env_sign + noise * torch.randn((len(y), 1), generator=gen)
    return torch.cat([x, causal, shortcut], dim=1)


def make_eval_sets(
    x: torch.Tensor,
    y: torch.Tensor,
    idx: torch.Tensor,
    *,
    causal_strength: float,
    shortcut_strength: float,
    noise: float,
    seed: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    xt = x[idx]
    yt = y[idx]
    env_old = assign_environment(yt, +0.9, seed + 1)
    env_rev = assign_environment(yt, -0.9, seed + 2)
    env_rand = assign_environment(yt, 0.0, seed + 3)
    return {
        "old_corr": (
            append_concepts(xt, yt, env_old, causal_strength=causal_strength, shortcut_strength=shortcut_strength, noise=noise, seed=seed + 10),
            yt,
            env_old,
        ),
        "reversed": (
            append_concepts(xt, yt, env_rev, causal_strength=causal_strength, shortcut_strength=shortcut_strength, noise=noise, seed=seed + 20),
            yt,
            env_rev,
        ),
        "neutral": (
            append_concepts(xt, yt, env_rand, causal_strength=causal_strength, shortcut_strength=shortcut_strength, noise=noise, seed=seed + 30, neutral_shortcut=True),
            yt,
            env_rand,
        ),
        "random": (
            append_concepts(xt, yt, env_rand, causal_strength=causal_strength, shortcut_strength=shortcut_strength, noise=noise, seed=seed + 40, random_shortcut=True),
            yt,
            env_rand,
        ),
    }


def residual_env_corr_scores(x: torch.Tensor, y: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
    """Estimate per-dimension stability from Task-1 data only.

    We residualize each feature by class mean, then measure absolute correlation
    with the proxy environment. High residual environment correlation means the
    feature is environment-specific and should receive less consolidation.
    """
    resid = x.clone()
    for cls in [0, 1]:
        mask = y == cls
        resid[mask] = resid[mask] - resid[mask].mean(dim=0, keepdim=True)
    env_sign = env.float().mul(2).sub(1)
    env_centered = env_sign - env_sign.mean()
    resid_centered = resid - resid.mean(dim=0, keepdim=True)
    denom = resid_centered.norm(dim=0).clamp_min(1e-8) * env_centered.norm().clamp_min(1e-8)
    corr = (resid_centered * env_centered.unsqueeze(1)).sum(dim=0).abs() / denom
    stability = (1.0 - corr).clamp(0.0, 1.0)
    return stability


def train_with_weighted_l2(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    old_state: dict[str, torch.Tensor],
    feature_weights: torch.Tensor,
    l2_lambda: float,
    anti_shortcut: bool = False,
    shortcut_penalty: float = 0.0,
    anti_threshold: float | None = None,
) -> nn.Module:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    old_weight = old_state["linear.weight"].detach().clone()
    old_bias = old_state["linear.bias"].detach().clone()
    weights = feature_weights.view(1, -1)
    for _ in range(epochs):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        cur_w = model.linear.weight
        weighted_diff = (cur_w - old_weight).pow(2) * weights
        loss = loss + l2_lambda * weighted_diff.sum() / weights.sum().clamp_min(1e-8)
        loss = loss + l2_lambda * 0.1 * (model.linear.bias - old_bias).pow(2).mean()
        if anti_shortcut:
            if anti_threshold is None:
                low_weight = (1.0 - feature_weights).view(1, -1)
            else:
                low_weight = (feature_weights < anti_threshold).float().view(1, -1)
            anti = cur_w.pow(2) * low_weight
            loss = loss + shortcut_penalty * anti.sum() / low_weight.sum().clamp_min(1e-8)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def shortcut_sensitivity(model: nn.Module, x: torch.Tensor) -> float:
    model.eval()
    prob = model(x).softmax(dim=1)[:, 1]
    x_zero = x.clone()
    x_zero[:, -1] = 0.0
    prob_zero = model(x_zero).softmax(dim=1)[:, 1]
    return float((prob - prob_zero).abs().mean().item())


@torch.no_grad()
def env_prediction_corr(model: nn.Module, x: torch.Tensor, env: torch.Tensor) -> float:
    model.eval()
    prob = model(x).softmax(dim=1)[:, 1]
    env_sign = env.float().mul(2).sub(1)
    prob = prob - prob.mean()
    env_sign = env_sign - env_sign.mean()
    denom = prob.norm() * env_sign.norm()
    if float(denom) == 0.0:
        return 0.0
    return float((prob @ env_sign / denom).item())


def evaluate(
    model: nn.Module,
    eval_sets: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, (xe, ye, ee) in eval_sets.items():
        out[f"{name}_acc"] = acc(model, xe, ye)
        out[f"{name}_shortcut_sensitivity"] = shortcut_sensitivity(model, xe)
        out[f"{name}_env_pred_corr"] = env_prediction_corr(model, xe, ee)
    out["old_minus_reversed_acc"] = out["old_corr_acc"] - out["reversed_acc"]
    out["neutral_random_mean_acc"] = 0.5 * (out["neutral_acc"] + out["random_acc"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="/data_2_4T/data_zjj/continual_wsi/smoke_multicancer/max60_seed7/mean_features_max60_seed7.pt")
    parser.add_argument("--out-dir", default="/data_2_4T/data_zjj/continual_wsi/streaming_stability/seed7")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--env-corr-task1", type=float, default=0.9)
    parser.add_argument("--env-corr-task2", type=float, default=-0.9)
    parser.add_argument("--causal-strength", type=float, default=2.0)
    parser.add_argument("--shortcut-strength", type=float, default=6.0)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--epochs-task1", type=int, default=300)
    parser.add_argument("--epochs-task2", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--l2-lambda", type=float, default=80.0)
    parser.add_argument("--shortcut-penalty", type=float, default=0.5)
    parser.add_argument("--score-power", type=float, default=1.0)
    parser.add_argument("--anti-threshold", type=float, default=-1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y_multi = load_cache(Path(args.cache))
    y = (y_multi >= 4).long()
    splits = split_by_class(y_multi, args.seed)
    train_idx = torch.cat([splits["env1"], splits["env2"]])
    x = standardize(x, train_idx)

    y1 = y[splits["env1"]]
    y2 = y[splits["env2"]]
    env1 = assign_environment(y1, args.env_corr_task1, args.seed + 100)
    env2 = assign_environment(y2, args.env_corr_task2, args.seed + 200)
    x1 = append_concepts(
        x[splits["env1"]],
        y1,
        env1,
        causal_strength=args.causal_strength,
        shortcut_strength=args.shortcut_strength,
        noise=args.noise,
        seed=args.seed + 300,
    )
    x2 = append_concepts(
        x[splits["env2"]],
        y2,
        env2,
        causal_strength=args.causal_strength,
        shortcut_strength=args.shortcut_strength,
        noise=args.noise,
        seed=args.seed + 400,
    )
    eval_sets = make_eval_sets(
        x,
        y,
        splits["test"],
        causal_strength=args.causal_strength,
        shortcut_strength=args.shortcut_strength,
        noise=args.noise,
        seed=args.seed + 500,
    )

    base = Logistic(x1.shape[1])
    train_model(base, x1, y1, epochs=args.epochs_task1, lr=args.lr)
    old_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    raw_stability = residual_env_corr_scores(x1, y1, env1)
    stability = raw_stability.pow(args.score_power)
    oracle = torch.ones_like(stability)
    oracle[-1] = 0.0
    random_scores = torch.rand(stability.shape, generator=torch.Generator().manual_seed(args.seed + 999))

    models = {
        "task1_only": base,
        "finetune": train_model(clone_from(base), x2, y2, epochs=args.epochs_task2, lr=args.lr),
        "l2_all": train_model(
            clone_from(base),
            x2,
            y2,
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
            selective=False,
        ),
        "oracle_score_l2": train_with_weighted_l2(
            clone_from(base),
            x2,
            y2,
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            feature_weights=oracle,
            l2_lambda=args.l2_lambda,
        ),
        "streaming_score_l2": train_with_weighted_l2(
            clone_from(base),
            x2,
            y2,
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            feature_weights=stability,
            l2_lambda=args.l2_lambda,
        ),
        "streaming_score_anti": train_with_weighted_l2(
            clone_from(base),
            x2,
            y2,
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            feature_weights=stability,
            l2_lambda=args.l2_lambda,
            anti_shortcut=True,
            shortcut_penalty=args.shortcut_penalty,
            anti_threshold=args.anti_threshold if args.anti_threshold >= 0 else None,
        ),
        "random_score_l2": train_with_weighted_l2(
            clone_from(base),
            x2,
            y2,
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            feature_weights=random_scores,
            l2_lambda=args.l2_lambda,
        ),
    }

    results: dict[str, object] = {
        "seed": args.seed,
        "env_corr_task1": args.env_corr_task1,
        "env_corr_task2": args.env_corr_task2,
        "causal_strength": args.causal_strength,
        "shortcut_strength": args.shortcut_strength,
        "noise": args.noise,
        "l2_lambda": args.l2_lambda,
        "shortcut_penalty": args.shortcut_penalty,
        "score_power": args.score_power,
        "anti_threshold": args.anti_threshold,
        "stability_summary": {
            "raw_causal_dim": float(raw_stability[-2].item()),
            "raw_shortcut_dim": float(raw_stability[-1].item()),
            "mean": float(stability.mean().item()),
            "min": float(stability.min().item()),
            "max": float(stability.max().item()),
            "causal_dim": float(stability[-2].item()),
            "shortcut_dim": float(stability[-1].item()),
        },
        "models": {},
    }
    rows: list[dict[str, object]] = []
    for name, model in models.items():
        metrics = evaluate(model, eval_sets)
        metrics["env1_train_acc"] = acc(model, x1, y1)
        metrics["env2_train_acc"] = acc(model, x2, y2)
        metrics["causal_weight_norm"] = float(model.linear.weight[:, -2].norm().item())
        metrics["shortcut_weight_norm"] = float(model.linear.weight[:, -1].norm().item())
        results["models"][name] = metrics
        rows.append({"model": name, **metrics})

    (out_dir / "result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

