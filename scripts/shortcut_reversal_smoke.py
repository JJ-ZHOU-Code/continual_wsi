#!/usr/bin/env python3
"""Controlled shortcut-reversal smoke test for spurious consolidation.

We use cached slide-level CONCH embeddings, create a binary task, append a
synthetic shortcut feature, and train sequentially across two environments:

- Environment 1: shortcut is positively correlated with the label.
- Environment 2: shortcut correlation is reversed.

The goal is not to claim a final method. It is a go/no-go diagnostic:
does standard consolidation preserve the old shortcut strongly enough to hurt
adaptation, and can a causal-selective objective avoid that carryover?
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def load_cache(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    x = payload["x"].float()
    y_multi = payload["y"].long()
    return x, y_multi


def split_by_class(y_multi: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    rng = random.Random(seed)
    env1, env2, test = [], [], []
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(y_multi.tolist()):
        by_class[c].append(i)
    for _, idxs in sorted(by_class.items()):
        rng.shuffle(idxs)
        n = len(idxs)
        a = n // 3
        b = 2 * n // 3
        env1.extend(idxs[:a])
        env2.extend(idxs[a:b])
        test.extend(idxs[b:])
    return {
        "env1": torch.tensor(env1),
        "env2": torch.tensor(env2),
        "test": torch.tensor(test),
    }


def standardize(x: torch.Tensor, train_idx: torch.Tensor) -> torch.Tensor:
    mu = x[train_idx].mean(dim=0, keepdim=True)
    sd = x[train_idx].std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x - mu) / sd


def add_shortcut(
    x: torch.Tensor,
    y: torch.Tensor,
    corr: float,
    strength: float,
    noise: float,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    y_sign = y.float().mul(2).sub(1).unsqueeze(1)
    shortcut = corr * strength * y_sign + noise * torch.randn((len(y), 1), generator=gen)
    return torch.cat([x, shortcut], dim=1)


def make_eval_sets(
    x: torch.Tensor,
    y: torch.Tensor,
    idx: torch.Tensor,
    strength: float,
    noise: float,
    seed: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    xt = x[idx]
    yt = y[idx]
    return {
        "old_corr": (add_shortcut(xt, yt, +1.0, strength, noise, seed + 1), yt),
        "reversed": (add_shortcut(xt, yt, -1.0, strength, noise, seed + 2), yt),
        "neutral": (torch.cat([xt, torch.zeros((len(yt), 1))], dim=1), yt),
        "random": (torch.cat([xt, strength * torch.randn((len(yt), 1), generator=torch.Generator().manual_seed(seed + 3))], dim=1), yt),
    }


class Logistic(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    old_state: dict[str, torch.Tensor] | None = None,
    l2_lambda: float = 0.0,
    selective: bool = False,
    shortcut_aug: bool = False,
    shortcut_penalty: float = 0.0,
) -> nn.Module:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    old_weight = old_state["linear.weight"].detach().clone() if old_state else None
    old_bias = old_state["linear.bias"].detach().clone() if old_state else None
    for _ in range(epochs):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        if shortcut_aug:
            x_aug = x.clone()
            x_aug[:, -1] = 0.0
            loss = 0.5 * loss + 0.5 * F.cross_entropy(model(x_aug), y)
        if old_weight is not None and l2_lambda > 0:
            cur_w = model.linear.weight
            if selective:
                # Do not consolidate the explicit shortcut dimension.
                loss = loss + l2_lambda * (cur_w[:, :-1] - old_weight[:, :-1]).pow(2).mean()
            else:
                loss = loss + l2_lambda * (cur_w - old_weight).pow(2).mean()
            loss = loss + l2_lambda * 0.1 * (model.linear.bias - old_bias).pow(2).mean()
        if shortcut_penalty > 0:
            loss = loss + shortcut_penalty * model.linear.weight[:, -1].pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float((model(x).argmax(dim=1) == y).float().mean().item())


def shortcut_weight(model: nn.Module) -> list[float]:
    return [float(v) for v in model.linear.weight[:, -1].detach().cpu().tolist()]


def clone_from(model: nn.Module) -> nn.Module:
    new = Logistic(model.linear.in_features)
    new.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
    return new


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="/data_2_4T/data_zjj/continual_wsi/smoke_multicancer/max60_seed7/mean_features_max60_seed7.pt")
    parser.add_argument("--out-dir", default="/data_2_4T/data_zjj/continual_wsi/shortcut_reversal/max60_seed7")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strength", type=float, default=6.0)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--epochs-task1", type=int, default=300)
    parser.add_argument("--epochs-task2", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--l2-lambda", type=float, default=80.0)
    parser.add_argument("--shortcut-penalty", type=float, default=0.1)
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

    x1 = add_shortcut(x[splits["env1"]], y[splits["env1"]], +1.0, args.strength, args.noise, args.seed + 10)
    y1 = y[splits["env1"]]
    x2 = add_shortcut(x[splits["env2"]], y[splits["env2"]], -1.0, args.strength, args.noise, args.seed + 20)
    y2 = y[splits["env2"]]
    eval_sets = make_eval_sets(x, y, splits["test"], args.strength, args.noise, args.seed + 30)

    base = Logistic(x1.shape[1])
    train_model(base, x1, y1, epochs=args.epochs_task1, lr=args.lr)
    old_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

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
        "selective_l2": train_model(
            clone_from(base),
            x2,
            y2,
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
            selective=True,
        ),
        "csr_aug": train_model(
            clone_from(base),
            x2,
            y2,
            epochs=args.epochs_task2,
            lr=args.lr,
            old_state=old_state,
            l2_lambda=args.l2_lambda,
            selective=True,
            shortcut_aug=True,
            shortcut_penalty=args.shortcut_penalty,
        ),
    }

    results: dict[str, object] = {
        "cache": args.cache,
        "seed": args.seed,
        "num_env1": int(len(y1)),
        "num_env2": int(len(y2)),
        "num_test": int(len(splits["test"])),
        "shortcut_strength": args.strength,
        "shortcut_noise": args.noise,
        "l2_lambda": args.l2_lambda,
        "shortcut_penalty": args.shortcut_penalty,
        "models": {},
    }
    for name, model in models.items():
        metrics = {
            "env1_train_acc": acc(model, x1, y1),
            "env2_train_acc": acc(model, x2, y2),
            "shortcut_weight": shortcut_weight(model),
        }
        for eval_name, (xe, ye) in eval_sets.items():
            metrics[f"test_{eval_name}_acc"] = acc(model, xe, ye)
        results["models"][name] = metrics

    out_path = out_dir / "result.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

