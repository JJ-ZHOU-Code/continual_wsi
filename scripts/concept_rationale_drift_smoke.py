#!/usr/bin/env python3
"""Concept-rationale drift smoke test on cached TCGA/CONCH slide features.

This is a bridge experiment for the paper idea, not a final WSI benchmark. It
uses real slide-level CONCH mean embeddings as the base representation, then
adds a small fixed concept-evidence bank with stable disease concepts,
environment shortcut concepts, and irrelevant concepts. Task 1 has correlated
label/environment structure; Task 2 reverses it. The script checks whether
continual updates preserve the old concept rationale and whether cached
environment-stability scores help.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from shortcut_reversal_smoke import acc, load_cache, split_by_class, standardize


class LinearConceptMIL(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def clone_model(model: LinearConceptMIL) -> LinearConceptMIL:
    out = LinearConceptMIL(model.linear.in_features)
    out.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
    return out


def assign_environment(y: torch.Tensor, corr: float, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    p_match = 0.5 + 0.5 * corr
    match = torch.rand(len(y), generator=gen) < p_match
    return torch.where(match, y, 1 - y).long()


def append_concept_bank(
    x: torch.Tensor,
    y: torch.Tensor,
    env: torch.Tensor,
    *,
    stable_concepts: int,
    shortcut_concepts: int,
    noise_concepts: int,
    stable_strength: float,
    shortcut_strength: float,
    noise: float,
    seed: int,
    neutral_shortcut: bool = False,
    random_shortcut: bool = False,
) -> tuple[torch.Tensor, dict[str, object]]:
    gen = torch.Generator().manual_seed(seed)
    y_sign = y.float().mul(2).sub(1).unsqueeze(1)
    env_sign = env.float().mul(2).sub(1).unsqueeze(1)
    stable = stable_strength * y_sign + noise * torch.randn((len(y), stable_concepts), generator=gen)
    if neutral_shortcut:
        shortcut = noise * torch.randn((len(y), shortcut_concepts), generator=gen)
    elif random_shortcut:
        shortcut = shortcut_strength * torch.randn((len(y), shortcut_concepts), generator=gen)
    else:
        shortcut = shortcut_strength * env_sign + noise * torch.randn((len(y), shortcut_concepts), generator=gen)
    nuisance = noise * torch.randn((len(y), noise_concepts), generator=gen)
    concept = torch.cat([stable, shortcut, nuisance], dim=1)
    names = (
        [f"stable_{i}" for i in range(stable_concepts)]
        + [f"shortcut_{i}" for i in range(shortcut_concepts)]
        + [f"noise_{i}" for i in range(noise_concepts)]
    )
    groups = (
        ["stable"] * stable_concepts
        + ["shortcut"] * shortcut_concepts
        + ["noise"] * noise_concepts
    )
    meta = {"names": names, "groups": groups, "start": x.shape[1], "dim": concept.shape[1]}
    return torch.cat([x, concept], dim=1), meta


def make_eval_sets(
    x: torch.Tensor,
    y: torch.Tensor,
    idx: torch.Tensor,
    *,
    concept_cfg: dict[str, float | int],
    seed: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    xt = x[idx]
    yt = y[idx]
    env_old = assign_environment(yt, +0.9, seed + 1)
    env_rev = assign_environment(yt, -0.9, seed + 2)
    env_rand = assign_environment(yt, 0.0, seed + 3)
    return {
        "old_corr": (*append_concept_bank(xt, yt, env_old, seed=seed + 10, **concept_cfg)[0:1], yt, env_old),
        "reversed": (*append_concept_bank(xt, yt, env_rev, seed=seed + 20, **concept_cfg)[0:1], yt, env_rev),
        "neutral": (*append_concept_bank(xt, yt, env_rand, seed=seed + 30, neutral_shortcut=True, **concept_cfg)[0:1], yt, env_rand),
        "random": (*append_concept_bank(xt, yt, env_rand, seed=seed + 40, random_shortcut=True, **concept_cfg)[0:1], yt, env_rand),
    }


def residual_env_corr_scores(concepts: torch.Tensor, y: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
    resid = concepts.clone()
    for cls in [0, 1]:
        mask = y == cls
        resid[mask] = resid[mask] - resid[mask].mean(dim=0, keepdim=True)
    env_sign = env.float().mul(2).sub(1)
    env_centered = env_sign - env_sign.mean()
    resid_centered = resid - resid.mean(dim=0, keepdim=True)
    denom = resid_centered.norm(dim=0).clamp_min(1e-8) * env_centered.norm().clamp_min(1e-8)
    corr = (resid_centered * env_centered.unsqueeze(1)).sum(dim=0).abs() / denom
    return (1.0 - corr).clamp(0.0, 1.0)


def label_relevance_scores(concepts: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Absolute concept-label correlation on Task-1 data.

    Environment stability alone is insufficient: irrelevant noise concepts can
    be stable. The anchor score should prefer concepts that are both stable and
    label-relevant.
    """
    y_sign = y.float().mul(2).sub(1)
    y_centered = y_sign - y_sign.mean()
    x_centered = concepts - concepts.mean(dim=0, keepdim=True)
    denom = x_centered.norm(dim=0).clamp_min(1e-8) * y_centered.norm().clamp_min(1e-8)
    corr = (x_centered * y_centered.unsqueeze(1)).sum(dim=0).abs() / denom
    return corr.clamp(0.0, 1.0)


def train_model(
    model: LinearConceptMIL,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    old_state: dict[str, torch.Tensor] | None = None,
    concept_slice: slice | None = None,
    l2_lambda: float = 0.0,
    l2_weights: torch.Tensor | None = None,
    anti_threshold: float | None = None,
    anti_penalty: float = 0.0,
    subspace_lambda: float = 0.0,
    stable_mask: torch.Tensor | None = None,
) -> LinearConceptMIL:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    old_weight = old_state["linear.weight"].detach().clone() if old_state else None
    old_bias = old_state["linear.bias"].detach().clone() if old_state else None
    if concept_slice is None:
        concept_slice = slice(0, model.linear.in_features)
    for _ in range(epochs):
        loss = F.cross_entropy(model(x), y)
        if old_weight is not None and l2_lambda > 0:
            cur = model.linear.weight
            if l2_weights is None:
                diff = (cur - old_weight).pow(2)
                loss = loss + l2_lambda * diff.mean()
            else:
                cur_c = cur[:, concept_slice]
                old_c = old_weight[:, concept_slice]
                w = l2_weights.view(1, -1)
                loss = loss + l2_lambda * ((cur_c - old_c).pow(2) * w).sum() / w.sum().clamp_min(1e-8)
            loss = loss + 0.1 * l2_lambda * (model.linear.bias - old_bias).pow(2).mean()
        if old_weight is not None and stable_mask is not None and subspace_lambda > 0:
            old_delta = old_weight[1, concept_slice] - old_weight[0, concept_slice]
            cur_delta = model.linear.weight[1, concept_slice] - model.linear.weight[0, concept_slice]
            old_stable = old_delta[stable_mask]
            cur_stable = cur_delta[stable_mask]
            denom = old_stable.norm().clamp_min(1e-8) * cur_stable.norm().clamp_min(1e-8)
            cos = torch.clamp((old_stable @ cur_stable) / denom, -1.0, 1.0)
            loss = loss + subspace_lambda * (1.0 - cos)
        if l2_weights is not None and anti_threshold is not None and anti_penalty > 0:
            low = (l2_weights < anti_threshold).float().view(1, -1)
            anti = model.linear.weight[:, concept_slice].pow(2) * low
            loss = loss + anti_penalty * anti.sum() / low.sum().clamp_min(1e-8)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def prediction_delta(model: LinearConceptMIL, x: torch.Tensor, concept_offset: int, local_idx: int) -> float:
    p = model(x).softmax(dim=1)[:, 1]
    x_zero = x.clone()
    x_zero[:, concept_offset + local_idx] = 0.0
    p_zero = model(x_zero).softmax(dim=1)[:, 1]
    return float((p - p_zero).abs().mean().item())


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


def concept_rationale(model: LinearConceptMIL, concept_slice: slice) -> torch.Tensor:
    delta = model.linear.weight[1, concept_slice] - model.linear.weight[0, concept_slice]
    return delta.detach().abs()


def concept_metrics(
    old_model: LinearConceptMIL,
    model: LinearConceptMIL,
    eval_x: torch.Tensor,
    *,
    concept_slice: slice,
    concept_names: list[str],
    concept_groups: list[str],
    stability: torch.Tensor,
    top_k: int,
    spurious_threshold: float,
) -> dict[str, object]:
    q_old = concept_rationale(old_model, concept_slice)
    q_new = concept_rationale(model, concept_slice)
    k = min(top_k, len(q_old))
    old_top = torch.topk(q_old, k=k).indices.tolist()
    new_top = torch.topk(q_new, k=k).indices.tolist()
    overlap = len(set(old_top) & set(new_top)) / max(k, 1)
    stable_mask = stability >= spurious_threshold
    old_stable = q_old[stable_mask]
    new_stable = q_new[stable_mask]
    if len(old_stable) == 0 or len(new_stable) == 0:
        angle = 90.0
    else:
        cos = torch.clamp((old_stable @ new_stable) / (old_stable.norm().clamp_min(1e-8) * new_stable.norm().clamp_min(1e-8)), -1.0, 1.0)
        angle = float(torch.rad2deg(torch.arccos(cos)).item())
    old_deltas = []
    new_deltas = []
    for idx in torch.topk(stability * q_old, k=k).indices.tolist():
        old_deltas.append(prediction_delta(old_model, eval_x, concept_slice.start, idx))
        new_deltas.append(prediction_delta(model, eval_x, concept_slice.start, idx))
    intervention_agreement = spearman(torch.tensor(old_deltas), torch.tensor(new_deltas))
    top_groups = [concept_groups[i] for i in new_top]
    return {
        "crc": spearman(q_old, q_new),
        "cdr": 1.0 - overlap,
        "subspace_rotation_deg": angle,
        "scr": sum(1 for i in new_top if float(stability[i]) < spurious_threshold) / max(k, 1),
        "intervention_agreement": intervention_agreement,
        "old_top": [concept_names[i] for i in old_top],
        "new_top": [concept_names[i] for i in new_top],
        "new_top_groups": top_groups,
    }


@torch.no_grad()
def evaluate(
    old_model: LinearConceptMIL,
    model: LinearConceptMIL,
    eval_sets: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    concept_slice: slice,
    concept_names: list[str],
    concept_groups: list[str],
    stability: torch.Tensor,
    top_k: int,
    spurious_threshold: float,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for name, (xe, ye, _ee) in eval_sets.items():
        out[f"{name}_acc"] = acc(model, xe, ye)
    out["neutral_random_mean_acc"] = 0.5 * (float(out["neutral_acc"]) + float(out["random_acc"]))
    out["old_minus_reversed_acc"] = float(out["old_corr_acc"]) - float(out["reversed_acc"])
    out.update(
        concept_metrics(
            old_model,
            model,
            eval_sets["neutral"][0],
            concept_slice=concept_slice,
            concept_names=concept_names,
            concept_groups=concept_groups,
            stability=stability,
            top_k=top_k,
            spurious_threshold=spurious_threshold,
        )
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="/data_2_4T/data_zjj/continual_wsi/smoke_multicancer/max60_seed7/mean_features_max60_seed7.pt")
    parser.add_argument("--out-dir", default="/data_2_4T/data_zjj/continual_wsi/concept_rationale_drift/seed7")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--env-corr-task1", type=float, default=0.9)
    parser.add_argument("--env-corr-task2", type=float, default=-0.9)
    parser.add_argument("--stable-concepts", type=int, default=6)
    parser.add_argument("--shortcut-concepts", type=int, default=6)
    parser.add_argument("--noise-concepts", type=int, default=8)
    parser.add_argument("--stable-strength", type=float, default=2.0)
    parser.add_argument("--shortcut-strength", type=float, default=5.0)
    parser.add_argument("--noise", type=float, default=0.6)
    parser.add_argument("--epochs-task1", type=int, default=300)
    parser.add_argument("--epochs-task2", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--l2-lambda", type=float, default=80.0)
    parser.add_argument("--score-power", type=float, default=4.0)
    parser.add_argument("--relevance-power", type=float, default=2.0)
    parser.add_argument("--anti-threshold", type=float, default=0.25)
    parser.add_argument("--anti-penalty", type=float, default=500.0)
    parser.add_argument("--subspace-lambda", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_raw, y_multi = load_cache(Path(args.cache))
    y = (y_multi >= 4).long()
    splits = split_by_class(y_multi, args.seed)
    train_idx = torch.cat([splits["env1"], splits["env2"]])
    x = standardize(x_raw, train_idx)

    concept_cfg = {
        "stable_concepts": args.stable_concepts,
        "shortcut_concepts": args.shortcut_concepts,
        "noise_concepts": args.noise_concepts,
        "stable_strength": args.stable_strength,
        "shortcut_strength": args.shortcut_strength,
        "noise": args.noise,
    }
    y1 = y[splits["env1"]]
    y2 = y[splits["env2"]]
    env1 = assign_environment(y1, args.env_corr_task1, args.seed + 100)
    env2 = assign_environment(y2, args.env_corr_task2, args.seed + 200)
    x1, meta = append_concept_bank(x[splits["env1"]], y1, env1, seed=args.seed + 300, **concept_cfg)
    x2, _ = append_concept_bank(x[splits["env2"]], y2, env2, seed=args.seed + 400, **concept_cfg)
    eval_sets = make_eval_sets(x, y, splits["test"], concept_cfg=concept_cfg, seed=args.seed + 500)

    concept_slice = slice(int(meta["start"]), int(meta["start"]) + int(meta["dim"]))
    concept_names = list(meta["names"])
    concept_groups = list(meta["groups"])

    base = LinearConceptMIL(x1.shape[1])
    train_model(base, x1, y1, epochs=args.epochs_task1, lr=args.lr)
    old_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    raw_stability = residual_env_corr_scores(x1[:, concept_slice], y1, env1)
    label_relevance = label_relevance_scores(x1[:, concept_slice], y1)
    stability = raw_stability.pow(args.score_power) * label_relevance.pow(args.relevance_power)
    random_scores = torch.rand(stability.shape, generator=torch.Generator().manual_seed(args.seed + 999))
    stable_mask = stability >= args.anti_threshold

    models = {
        "task1_only": base,
        "finetune": train_model(clone_model(base), x2, y2, epochs=args.epochs_task2, lr=args.lr),
        "l2_all": train_model(
            clone_model(base),
            x2,
            y2,
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
        ),
        "naive_concept_l2": train_model(
            clone_model(base),
            x2,
            y2,
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            concept_slice=concept_slice,
            l2_lambda=args.l2_lambda,
            l2_weights=torch.ones_like(stability),
        ),
        "random_score_l2": train_model(
            clone_model(base),
            x2,
            y2,
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            concept_slice=concept_slice,
            l2_lambda=args.l2_lambda,
            l2_weights=random_scores,
        ),
        "cca_weighted": train_model(
            clone_model(base),
            x2,
            y2,
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            concept_slice=concept_slice,
            l2_lambda=args.l2_lambda,
            l2_weights=stability,
            subspace_lambda=args.subspace_lambda,
            stable_mask=stable_mask,
        ),
        "cca_anti_subspace": train_model(
            clone_model(base),
            x2,
            y2,
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            concept_slice=concept_slice,
            l2_lambda=args.l2_lambda,
            l2_weights=stability,
            anti_threshold=args.anti_threshold,
            anti_penalty=args.anti_penalty,
            subspace_lambda=args.subspace_lambda,
            stable_mask=stable_mask,
        ),
    }

    rows = []
    result: dict[str, object] = {
        "seed": args.seed,
        "cache": args.cache,
        "config": vars(args),
        "concept_names": concept_names,
        "concept_groups": concept_groups,
        "stability": {
            concept_names[i]: float(stability[i].item()) for i in range(len(concept_names))
        },
        "label_relevance_summary": {},
        "raw_stability_summary": {},
        "anchor_score_summary": {},
        "memory_bytes_estimate": int((base.linear.weight.numel() + base.linear.bias.numel() + stability.numel()) * 4),
        "models": {},
    }
    group_slices = {
        "stable": slice(0, args.stable_concepts),
        "shortcut": slice(args.stable_concepts, args.stable_concepts + args.shortcut_concepts),
        "noise": slice(args.stable_concepts + args.shortcut_concepts, len(concept_names)),
    }
    for group, sl in group_slices.items():
        if sl.start == sl.stop:
            continue
        result["label_relevance_summary"][f"{group}_mean"] = float(label_relevance[sl].mean().item())
        result["raw_stability_summary"][f"{group}_mean"] = float(raw_stability[sl].mean().item())
        result["anchor_score_summary"][f"{group}_mean"] = float(stability[sl].mean().item())
    result["anchor_score_summary"]["stable_count"] = int(stable_mask.sum().item())
    for name, model in models.items():
        metrics = evaluate(
            base,
            model,
            eval_sets,
            concept_slice=concept_slice,
            concept_names=concept_names,
            concept_groups=concept_groups,
            stability=stability,
            top_k=args.top_k,
            spurious_threshold=args.anti_threshold,
        )
        result["models"][name] = metrics
        flat = {k: v for k, v in metrics.items() if not isinstance(v, list)}
        rows.append({"model": name, **flat})

    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
