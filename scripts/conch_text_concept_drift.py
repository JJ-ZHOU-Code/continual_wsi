#!/usr/bin/env python3
"""CONCH text-concept continual rationale drift experiment.

This script uses fixed external pathology concept text embeddings from the local
CONCH text encoder. It projects cached CONCH slide features into a curated
morphology concept bank, then runs a continual binary classification stream with
TCGA TSS metadata as the environment proxy.

Compared with prior bridge experiments, this removes synthetic concept evidence
and avoids PCA environments. The remaining limitation is that the current smoke
label is a coarse multicancer binary grouping, so this is still a bridge
diagnostic rather than a final pathology benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


TRAIN_CONCEPTS = [
    "tumor necrosis",
    "lymphocytic infiltration",
    "gland formation",
    "keratinization",
    "mucin production",
    "nuclear pleomorphism",
    "high mitotic activity",
    "stromal desmoplasia",
    "papillary architecture",
    "solid tumor nests",
    "acinar growth pattern",
    "squamous differentiation",
    "clear cell morphology",
    "hemorrhage",
    "fibrosis",
    "vascular invasion",
    "small round blue cells",
    "microvascular proliferation",
    "tumor infiltrating lymphocytes",
    "poor differentiation",
    "well differentiated glands",
    "spindle cell morphology",
    "extracellular mucin",
    "cribriform architecture",
]

EVAL_CONCEPTS = [
    "coagulative necrosis",
    "dense immune infiltrate",
    "malignant glands",
    "keratin pearl",
    "mucinous stroma",
    "atypical nuclei",
    "frequent mitoses",
    "reactive stroma",
    "papillary fronds",
    "solid sheets of tumor",
    "alveolar growth pattern",
    "squamous morphology",
    "clear cytoplasm",
    "blood filled spaces",
    "collagen deposition",
    "vascular tumor thrombus",
    "hypercellular tumor",
    "glomeruloid microvasculature",
    "peritumoral lymphocytes",
    "undifferentiated carcinoma",
    "well formed glandular structures",
    "sarcomatoid morphology",
    "mucin pools",
    "cribriform glands",
]


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


def load_cache(path: Path) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, str]]]:
    payload = torch.load(path, map_location="cpu")
    return payload["x"].float(), payload["y"].long(), list(payload["meta"])


def standardize(x: torch.Tensor, train_idx: torch.Tensor) -> torch.Tensor:
    mu = x[train_idx].mean(dim=0, keepdim=True)
    sd = x[train_idx].std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x - mu) / sd


def acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float((model(x).argmax(dim=1) == y).float().mean().item())


def tss_supergroup_env(meta: list[dict[str, str]], y: torch.Tensor, seed: int) -> tuple[torch.Tensor, dict[str, object]]:
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

    def objective(c0: Counter[int], c1: Counter[int]) -> float:
        return abs(c0[0] - c1[0]) + abs(c0[1] - c1[1]) + 0.25 * abs(sum(c0.values()) - sum(c1.values()))

    for tss, counts in items:
        cand0 = group_counts[0].copy()
        cand1 = group_counts[1].copy()
        cand0.update(counts)
        score0 = objective(cand0, cand1)
        cand0 = group_counts[0].copy()
        cand1 = group_counts[1].copy()
        cand1.update(counts)
        score1 = objective(cand0, cand1)
        target = 0 if score0 <= score1 else 1
        assignment[tss] = target
        group_counts[target].update(counts)
    env = torch.tensor([assignment[item.get("tss", "") or "UNK"] for item in meta], dtype=torch.long)
    info = {
        "num_tss": len(by_tss),
        "group_counts": [{str(k): int(v) for k, v in sorted(group_counts[g].items())} for g in [0, 1]],
        "cell_counts": {
            f"y{yy}_e{ee}": int(((y == yy) & (env == ee)).sum().item())
            for yy in [0, 1]
            for ee in [0, 1]
        },
        "top_tss": Counter(item.get("tss", "") or "UNK" for item in meta).most_common(20),
    }
    return env, info


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
    task1 = (
        cells[(0, 0)][:n_major_per_cell]
        + cells[(1, 1)][:n_major_per_cell]
        + cells[(0, 1)][:n_minor_per_cell]
        + cells[(1, 0)][:n_minor_per_cell]
    )
    task2 = (
        cells[(0, 1)][n_minor_per_cell : n_minor_per_cell + n_major_per_cell]
        + cells[(1, 0)][n_minor_per_cell : n_minor_per_cell + n_major_per_cell]
        + cells[(0, 0)][n_major_per_cell : n_major_per_cell + n_major_per_cell + n_minor_per_cell]
        + cells[(1, 1)][n_major_per_cell : n_major_per_cell + n_major_per_cell + n_minor_per_cell]
    )
    old_corr = []
    reversed_corr = []
    for key in [(0, 0), (1, 1)]:
        start = n_major_per_cell + n_minor_per_cell
        old_corr.extend(cells[key][start : start + n_test_per_cell])
    for key in [(0, 1), (1, 0)]:
        start = n_major_per_cell + n_minor_per_cell
        reversed_corr.extend(cells[key][start : start + n_test_per_cell])
    test_balanced = old_corr + reversed_corr
    rng.shuffle(task1)
    rng.shuffle(task2)
    rng.shuffle(test_balanced)
    return {
        "task1": torch.tensor(task1),
        "task2": torch.tensor(task2),
        "test_balanced": torch.tensor(test_balanced),
        "test_old_corr": torch.tensor(old_corr),
        "test_reversed": torch.tensor(reversed_corr),
    }


def load_conch_text_embeddings(texts: list[str], *, device: str) -> torch.Tensor:
    # Import the copied CONCH package directly. Importing `model.conch` would
    # execute VLSA/model/__init__.py and pull unrelated MIL dependencies.
    sys.path.insert(0, "/home/zjj/code/VLSA/model")
    from conch import create_model_from_pretrained
    from conch.custom_tokenizer import get_tokenizer, tokenize

    ckpt = "/home/zjj/.cache/huggingface/hub/conch_ViT-B-16/pytorch_model.bin"
    model = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=ckpt, device=device, return_transform=False)
    model.eval()
    tokenizer = get_tokenizer()
    tokens = tokenize(tokenizer, texts).to(device)
    with torch.no_grad():
        z = model.encode_text(tokens).detach().cpu().float()
    return F.normalize(z, dim=1)


def load_or_create_text_cache(cache_path: Path, *, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return payload["train_text"].float(), payload["eval_text"].float()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    train_text = load_conch_text_embeddings(TRAIN_CONCEPTS, device=device)
    eval_text = load_conch_text_embeddings(EVAL_CONCEPTS, device=device)
    torch.save(
        {
            "train_concepts": TRAIN_CONCEPTS,
            "eval_concepts": EVAL_CONCEPTS,
            "train_text": train_text,
            "eval_text": eval_text,
        },
        cache_path,
    )
    return train_text, eval_text


def concept_evidence(x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    x_norm = F.normalize(x.float(), dim=1)
    return x_norm @ F.normalize(text_emb.float(), dim=1).T


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
        if old_weight is not None and stable_mask is not None and subspace_lambda > 0 and int(stable_mask.sum().item()) > 0:
            old_delta = old_weight[1] - old_weight[0]
            cur_delta = model.linear.weight[1] - model.linear.weight[0]
            old_s = old_delta[stable_mask]
            cur_s = cur_delta[stable_mask]
            cos = torch.clamp((old_s @ cur_s) / (old_s.norm().clamp_min(1e-8) * cur_s.norm().clamp_min(1e-8)), -1.0, 1.0)
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
    concept_names: list[str],
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
        "old_top": [concept_names[i] for i in old_top],
        "new_top": [concept_names[i] for i in new_top],
    }


@torch.no_grad()
def evaluate(
    old_model: Logistic,
    model: Logistic,
    z_train: torch.Tensor,
    z_eval: torch.Tensor,
    y: torch.Tensor,
    env: torch.Tensor,
    splits: dict[str, torch.Tensor],
    *,
    stability: torch.Tensor,
    top_k: int,
    threshold: float,
    train_concepts: list[str],
    eval_concepts: list[str],
) -> dict[str, object]:
    out: dict[str, object] = {}
    for name in ["task1", "task2", "test_balanced", "test_old_corr", "test_reversed"]:
        idx = splits[name]
        out[f"{name}_acc"] = acc(model, z_train[idx], y[idx])
        out[f"{name}_env_pred_corr"] = env_pred_corr(model, z_train[idx], env[idx])
    out["old_minus_reversed_acc"] = float(out["test_old_corr_acc"]) - float(out["test_reversed_acc"])
    out.update(
        concept_metrics(
            old_model,
            model,
            z_train[splits["test_balanced"]],
            stability=stability,
            top_k=top_k,
            threshold=threshold,
            concept_names=train_concepts,
        )
    )
    # Held-out concept bank is evaluation-only: project the learned linear
    # decision function through correlations between train and eval concept
    # evidence on the balanced test split.
    corr = torch.corrcoef(torch.cat([z_train[splits["test_balanced"]].T, z_eval[splits["test_balanced"]].T], dim=0))
    cross = corr[: len(train_concepts), len(train_concepts) :]
    old_q_train = (old_model.linear.weight[1] - old_model.linear.weight[0]).detach().abs()
    new_q_train = (model.linear.weight[1] - model.linear.weight[0]).detach().abs()
    old_q_eval = torch.nan_to_num(old_q_train @ cross, nan=0.0).abs()
    new_q_eval = torch.nan_to_num(new_q_train @ cross, nan=0.0).abs()
    out["heldout_crc"] = spearman(old_q_eval, new_q_eval)
    k = min(top_k, len(eval_concepts))
    old_top = torch.topk(old_q_eval, k=k).indices.tolist()
    new_top = torch.topk(new_q_eval, k=k).indices.tolist()
    out["heldout_cdr"] = 1.0 - len(set(old_top) & set(new_top)) / max(k, 1)
    out["heldout_old_top"] = [eval_concepts[i] for i in old_top]
    out["heldout_new_top"] = [eval_concepts[i] for i in new_top]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="/data_2_4T/data_zjj/continual_wsi/smoke_multicancer/max60_seed7/mean_features_max60_seed7.pt")
    parser.add_argument("--out-dir", default="/data_2_4T/data_zjj/continual_wsi/conch_text_concepts/seed7")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-major-per-cell", type=int, default=45)
    parser.add_argument("--n-minor-per-cell", type=int, default=15)
    parser.add_argument("--n-test-per-cell", type=int, default=15)
    parser.add_argument("--epochs-task1", type=int, default=250)
    parser.add_argument("--epochs-task2", type=int, default=250)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--l2-lambda", type=float, default=2.0)
    parser.add_argument("--score-power", type=float, default=4.0)
    parser.add_argument("--relevance-power", type=float, default=2.0)
    parser.add_argument("--anti-threshold", type=float, default=0.05)
    parser.add_argument("--anti-penalty", type=float, default=1.0)
    parser.add_argument("--subspace-lambda", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--text-cache", default="/data_2_4T/data_zjj/continual_wsi/conch_text_concepts/conch_text_concept_embeddings.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_raw, y_multi, meta = load_cache(Path(args.cache))
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

    train_text, eval_text = load_or_create_text_cache(Path(args.text_cache), device=args.text_device)
    z_train_raw = concept_evidence(x, train_text)
    z_eval_raw = concept_evidence(x, eval_text)
    z_train = standardize(z_train_raw, train_idx)
    z_eval = standardize(z_eval_raw, train_idx)

    base = Logistic(z_train.shape[1])
    train_model(base, z_train[splits["task1"]], y[splits["task1"]], epochs=args.epochs_task1, lr=args.lr)
    old_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    env_stability = residual_env_corr_scores(z_train[splits["task1"]], y[splits["task1"]], env[splits["task1"]])
    relevance = label_relevance_scores(z_train[splits["task1"]], y[splits["task1"]])
    anchor = env_stability.pow(args.score_power) * relevance.pow(args.relevance_power)
    stable_mask = anchor >= args.anti_threshold
    random_scores = torch.rand(anchor.shape, generator=torch.Generator().manual_seed(args.seed + 1000))

    models = {
        "task1_only": base,
        "finetune": train_model(clone_model(base), z_train[splits["task2"]], y[splits["task2"]], epochs=args.epochs_task2, lr=args.lr),
        "l2_all": train_model(
            clone_model(base),
            z_train[splits["task2"]],
            y[splits["task2"]],
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
        ),
        "naive_concept_l2": train_model(
            clone_model(base),
            z_train[splits["task2"]],
            y[splits["task2"]],
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
            weights=torch.ones_like(anchor),
        ),
        "random_score_l2": train_model(
            clone_model(base),
            z_train[splits["task2"]],
            y[splits["task2"]],
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
            weights=random_scores,
        ),
        "conch_cca": train_model(
            clone_model(base),
            z_train[splits["task2"]],
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

    result: dict[str, object] = {
        "seed": args.seed,
        "config": vars(args),
        "env_info": env_info,
        "train_concepts": TRAIN_CONCEPTS,
        "eval_concepts": EVAL_CONCEPTS,
        "anchor_summary": {
            "env_stability_mean": float(env_stability.mean().item()),
            "relevance_mean": float(relevance.mean().item()),
            "anchor_mean": float(anchor.mean().item()),
            "anchor_min": float(anchor.min().item()),
            "anchor_max": float(anchor.max().item()),
            "stable_count": int(stable_mask.sum().item()),
        },
        "concept_scores": [
            {
                "name": TRAIN_CONCEPTS[i],
                "env_stability": float(env_stability[i].item()),
                "relevance": float(relevance[i].item()),
                "anchor": float(anchor[i].item()),
            }
            for i in range(len(TRAIN_CONCEPTS))
        ],
        "models": {},
    }
    rows = []
    for name, model in models.items():
        metrics = evaluate(
            base,
            model,
            z_train,
            z_eval,
            y,
            env,
            splits,
            stability=anchor,
            top_k=args.top_k,
            threshold=args.anti_threshold,
            train_concepts=TRAIN_CONCEPTS,
            eval_concepts=EVAL_CONCEPTS,
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
