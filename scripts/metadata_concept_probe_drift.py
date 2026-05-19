#!/usr/bin/env python3
"""Metadata-environment concept-probe drift smoke test.

This experiment removes the PCA environment used by earlier proxy tests. It
derives a binary environment from TCGA tissue-source-site (TSS) metadata,
constructs fixed CAV-style concept probes from Task-1 real CONCH features, and
then evaluates continual concept-rationale drift in probe space.

The probes are still proxy concepts rather than expert pathology concepts, but
they are data-derived directions over real WSI features and obey the streaming
constraint: probe directions and anchor scores are computed at Task-1 boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from concept_probe_shift_smoke import cav_projectors, learned_cav_projectors
from feature_cluster_shift_smoke import acc, env_pred_corr, group_metrics, split_cells, standardize


class Logistic(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def clone_model(model: Logistic) -> Logistic:
    out = Logistic(model.linear.in_features)
    out.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
    return out


def load_cache_with_meta(path: Path) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, str]]]:
    payload = torch.load(path, map_location="cpu")
    return payload["x"].float(), payload["y"].long(), list(payload["meta"])


def tss_supergroup_env(meta: list[dict[str, str]], y: torch.Tensor, seed: int) -> tuple[torch.Tensor, dict[str, object]]:
    """Greedily partition TSS categories into two balanced supergroups.

    This uses metadata only. The objective balances both total count and binary
    label count across the two environment groups, so each label has examples in
    both environments.
    """
    by_tss: dict[str, Counter[int]] = defaultdict(Counter)
    for i, item in enumerate(meta):
        tss = item.get("tss", "") or "UNK"
        by_tss[tss][int(y[i].item())] += 1
    rng = random.Random(seed)
    items = list(by_tss.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: sum(kv[1].values()), reverse=True)

    group_counts = [Counter(), Counter()]
    assignment: dict[str, int] = {}

    def score(counts: Counter[int]) -> float:
        total0 = group_counts[0][0] + counts[0]
        total1 = group_counts[0][1] + counts[1]
        other0 = group_counts[1][0]
        other1 = group_counts[1][1]
        return abs(total0 - other0) + abs(total1 - other1) + 0.25 * abs((total0 + total1) - (other0 + other1))

    for tss, counts in items:
        s0 = score(counts)
        # Temporarily swap group roles to evaluate assigning to group 1.
        group_counts[0], group_counts[1] = group_counts[1], group_counts[0]
        s1 = score(counts)
        group_counts[0], group_counts[1] = group_counts[1], group_counts[0]
        target = 0 if s0 <= s1 else 1
        assignment[tss] = target
        group_counts[target].update(counts)

    env = torch.tensor([assignment[item.get("tss", "") or "UNK"] for item in meta], dtype=torch.long)
    cell_counts = {
        f"y{yy}_e{ee}": int(((y == yy) & (env == ee)).sum().item())
        for yy in [0, 1]
        for ee in [0, 1]
    }
    info = {
        "num_tss": len(by_tss),
        "group_counts": [{str(k): int(v) for k, v in sorted(group_counts[g].items())} for g in [0, 1]],
        "cell_counts": cell_counts,
        "top_tss": Counter(item.get("tss", "") or "UNK" for item in meta).most_common(20),
    }
    return env, info


def label_relevance_scores(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_sign = y.float().mul(2).sub(1)
    y_centered = y_sign - y_sign.mean()
    z_centered = z - z.mean(dim=0, keepdim=True)
    denom = z_centered.norm(dim=0).clamp_min(1e-8) * y_centered.norm().clamp_min(1e-8)
    return ((z_centered * y_centered.unsqueeze(1)).sum(dim=0).abs() / denom).clamp(0.0, 1.0)


def residual_env_corr_scores(z: torch.Tensor, y: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
    resid = z.clone()
    for cls in [0, 1]:
        mask = y == cls
        resid[mask] = resid[mask] - resid[mask].mean(dim=0, keepdim=True)
    env_sign = env.float().mul(2).sub(1)
    env_centered = env_sign - env_sign.mean()
    resid_centered = resid - resid.mean(dim=0, keepdim=True)
    denom = resid_centered.norm(dim=0).clamp_min(1e-8) * env_centered.norm().clamp_min(1e-8)
    corr = (resid_centered * env_centered.unsqueeze(1)).sum(dim=0).abs() / denom
    return (1.0 - corr).clamp(0.0, 1.0)


def train_model(
    model: Logistic,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    old_state: dict[str, torch.Tensor] | None = None,
    l2_lambda: float = 0.0,
    weights: torch.Tensor | None = None,
    anti_threshold: float | None = None,
    anti_penalty: float = 0.0,
    subspace_lambda: float = 0.0,
    stable_mask: torch.Tensor | None = None,
) -> Logistic:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    old_weight = old_state["linear.weight"].detach().clone() if old_state else None
    old_bias = old_state["linear.bias"].detach().clone() if old_state else None
    for _ in range(epochs):
        loss = F.cross_entropy(model(x), y)
        if old_weight is not None and l2_lambda > 0:
            if weights is None:
                loss = loss + l2_lambda * (model.linear.weight - old_weight).pow(2).mean()
            else:
                w = weights.view(1, -1)
                loss = loss + l2_lambda * ((model.linear.weight - old_weight).pow(2) * w).sum() / w.sum().clamp_min(1e-8)
            loss = loss + 0.1 * l2_lambda * (model.linear.bias - old_bias).pow(2).mean()
        if old_weight is not None and stable_mask is not None and subspace_lambda > 0:
            old_delta = old_weight[1] - old_weight[0]
            cur_delta = model.linear.weight[1] - model.linear.weight[0]
            old_stable = old_delta[stable_mask]
            cur_stable = cur_delta[stable_mask]
            if old_stable.numel() > 0:
                cos = torch.clamp((old_stable @ cur_stable) / (old_stable.norm().clamp_min(1e-8) * cur_stable.norm().clamp_min(1e-8)), -1.0, 1.0)
                loss = loss + subspace_lambda * (1.0 - cos)
        if weights is not None and anti_threshold is not None and anti_penalty > 0:
            low = (weights < anti_threshold).float().view(1, -1)
            loss = loss + anti_penalty * (model.linear.weight.pow(2) * low).sum() / low.sum().clamp_min(1e-8)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float)
    ranks[order] = torch.arange(len(values), dtype=torch.float)
    return ranks


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if len(a) < 2:
        return 0.0
    ra = rankdata(a.float())
    rb = rankdata(b.float())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    if float(denom) == 0.0:
        return 0.0
    return float((ra @ rb / denom).item())


@torch.no_grad()
def intervention_delta(model: Logistic, z: torch.Tensor, idx: int) -> float:
    p = model(z).softmax(dim=1)[:, 1]
    z_zero = z.clone()
    z_zero[:, idx] = 0.0
    p_zero = model(z_zero).softmax(dim=1)[:, 1]
    return float((p - p_zero).abs().mean().item())


@torch.no_grad()
def concept_metrics(
    old_model: Logistic,
    model: Logistic,
    z_eval: torch.Tensor,
    *,
    stability: torch.Tensor,
    top_k: int,
    threshold: float,
    probe_names: list[str],
) -> dict[str, object]:
    q_old = (old_model.linear.weight[1] - old_model.linear.weight[0]).detach().abs()
    q_new = (model.linear.weight[1] - model.linear.weight[0]).detach().abs()
    k = min(top_k, len(q_old))
    old_top = torch.topk(q_old, k=k).indices.tolist()
    new_top = torch.topk(q_new, k=k).indices.tolist()
    stable_mask = stability >= threshold
    if int(stable_mask.sum().item()) == 0:
        rotation = 90.0
    else:
        old_s = q_old[stable_mask]
        new_s = q_new[stable_mask]
        cos = torch.clamp((old_s @ new_s) / (old_s.norm().clamp_min(1e-8) * new_s.norm().clamp_min(1e-8)), -1.0, 1.0)
        rotation = float(torch.rad2deg(torch.arccos(cos)).item())
    old_delta = []
    new_delta = []
    for idx in torch.topk(q_old * stability, k=k).indices.tolist():
        old_delta.append(intervention_delta(old_model, z_eval, idx))
        new_delta.append(intervention_delta(model, z_eval, idx))
    return {
        "crc": spearman(q_old, q_new),
        "cdr": 1.0 - len(set(old_top) & set(new_top)) / max(k, 1),
        "scr": sum(1 for idx in new_top if float(stability[idx]) < threshold) / max(k, 1),
        "subspace_rotation_deg": rotation,
        "intervention_agreement": spearman(torch.tensor(old_delta), torch.tensor(new_delta)),
        "old_top": [probe_names[i] for i in old_top],
        "new_top": [probe_names[i] for i in new_top],
    }


@torch.no_grad()
def evaluate(
    old_model: Logistic,
    model: Logistic,
    z: torch.Tensor,
    y: torch.Tensor,
    env: torch.Tensor,
    splits: dict[str, torch.Tensor],
    *,
    stability: torch.Tensor,
    top_k: int,
    threshold: float,
    probe_names: list[str],
) -> dict[str, object]:
    out: dict[str, object] = {}
    for name in ["task1", "task2", "test_balanced", "test_old_corr", "test_reversed"]:
        idx = splits[name]
        out[f"{name}_acc"] = acc(model, z[idx], y[idx])
        out[f"{name}_env_pred_corr"] = env_pred_corr(model, z[idx], env[idx])
    out["old_minus_reversed_acc"] = float(out["test_old_corr_acc"]) - float(out["test_reversed_acc"])
    out.update(group_metrics(model, z[splits["test_balanced"]], y[splits["test_balanced"]], env[splits["test_balanced"]]))
    out.update(
        concept_metrics(
            old_model,
            model,
            z[splits["test_balanced"]],
            stability=stability,
            top_k=top_k,
            threshold=threshold,
            probe_names=probe_names,
        )
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="/data_2_4T/data_zjj/continual_wsi/smoke_multicancer/max60_seed7/mean_features_max60_seed7.pt")
    parser.add_argument("--out-dir", default="/data_2_4T/data_zjj/continual_wsi/metadata_concept_probe/seed7")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--probe-type", choices=["learned_cav", "cav"], default="learned_cav")
    parser.add_argument("--num-probes", type=int, default=16)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--n-major-per-cell", type=int, default=45)
    parser.add_argument("--n-minor-per-cell", type=int, default=15)
    parser.add_argument("--n-test-per-cell", type=int, default=15)
    parser.add_argument("--epochs-task1", type=int, default=250)
    parser.add_argument("--epochs-task2", type=int, default=250)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--l2-lambda", type=float, default=2.0)
    parser.add_argument("--score-power", type=float, default=4.0)
    parser.add_argument("--relevance-power", type=float, default=2.0)
    parser.add_argument("--anti-threshold", type=float, default=0.25)
    parser.add_argument("--anti-penalty", type=float, default=25.0)
    parser.add_argument("--subspace-lambda", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_raw, y_multi, meta = load_cache_with_meta(Path(args.cache))
    y = (y_multi >= 4).long()
    env, env_info = tss_supergroup_env(meta, y, args.seed)
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

    if args.probe_type == "learned_cav":
        projectors, probe_names = learned_cav_projectors(
            x[splits["task1"]],
            y[splits["task1"]],
            env[splits["task1"]],
            num_probes=args.num_probes,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
        )
    else:
        projectors, probe_names = cav_projectors(
            x[splits["task1"]],
            y[splits["task1"]],
            env[splits["task1"]],
            num_probes=args.num_probes,
        )
    z_raw = x @ projectors
    z = standardize(z_raw, train_idx)

    base = Logistic(z.shape[1])
    train_model(base, z[splits["task1"]], y[splits["task1"]], epochs=args.epochs_task1, lr=args.lr)
    old_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    env_stability = residual_env_corr_scores(z[splits["task1"]], y[splits["task1"]], env[splits["task1"]])
    relevance = label_relevance_scores(z[splits["task1"]], y[splits["task1"]])
    anchor = env_stability.pow(args.score_power) * relevance.pow(args.relevance_power)
    stable_mask = anchor >= args.anti_threshold
    random_scores = torch.rand(anchor.shape, generator=torch.Generator().manual_seed(args.seed + 1000))

    models = {
        "task1_only": base,
        "finetune": train_model(clone_model(base), z[splits["task2"]], y[splits["task2"]], epochs=args.epochs_task2, lr=args.lr),
        "l2_all": train_model(
            clone_model(base),
            z[splits["task2"]],
            y[splits["task2"]],
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
        ),
        "naive_probe_l2": train_model(
            clone_model(base),
            z[splits["task2"]],
            y[splits["task2"]],
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
            weights=torch.ones_like(anchor),
        ),
        "random_score_l2": train_model(
            clone_model(base),
            z[splits["task2"]],
            y[splits["task2"]],
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
            weights=random_scores,
        ),
        "metadata_cca": train_model(
            clone_model(base),
            z[splits["task2"]],
            y[splits["task2"]],
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
            weights=anchor,
            anti_threshold=args.anti_threshold,
            anti_penalty=args.anti_penalty,
            subspace_lambda=args.subspace_lambda,
            stable_mask=stable_mask,
        ),
    }

    rows = []
    result: dict[str, object] = {
        "seed": args.seed,
        "config": vars(args),
        "env_info": env_info,
        "probe_names": probe_names,
        "anchor_summary": {
            "env_stability_mean": float(env_stability.mean().item()),
            "relevance_mean": float(relevance.mean().item()),
            "anchor_mean": float(anchor.mean().item()),
            "anchor_min": float(anchor.min().item()),
            "anchor_max": float(anchor.max().item()),
            "stable_count": int(stable_mask.sum().item()),
        },
        "probe_scores": [
            {
                "name": probe_names[i],
                "env_stability": float(env_stability[i].item()),
                "relevance": float(relevance[i].item()),
                "anchor": float(anchor[i].item()),
            }
            for i in range(len(probe_names))
        ],
        "models": {},
    }
    for name, model in models.items():
        metrics = evaluate(
            base,
            model,
            z,
            y,
            env,
            splits,
            stability=anchor,
            top_k=args.top_k,
            threshold=args.anti_threshold,
            probe_names=probe_names,
        )
        result["models"][name] = metrics
        rows.append({"model": name, **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}})

    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
