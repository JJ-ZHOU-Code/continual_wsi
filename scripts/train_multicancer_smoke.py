#!/usr/bin/env python3
"""Smoke-test multicancer WSI classification from pre-extracted CONCH bags.

This script is intentionally small: it reads a CSV index whose rows point to
`.pt` tensors shaped [num_patches, dim], mean-pools each slide, and trains a
lightweight classifier. It validates that labels, feature files, PyTorch, and
the basic training loop are usable before we build continual-learning code.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def read_index(path: Path, max_per_class: int, seed: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_label: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for label, group in sorted(by_label.items()):
        rng.shuffle(group)
        selected.extend(group[:max_per_class] if max_per_class > 0 else group)
    rng.shuffle(selected)
    return selected


def slide_embedding(path: str) -> torch.Tensor:
    bag = torch.load(path, map_location="cpu")
    if isinstance(bag, dict):
        for key in ("features", "feats", "x", "embeddings"):
            if key in bag:
                bag = bag[key]
                break
    if not torch.is_tensor(bag):
        raise TypeError(f"Unsupported feature object in {path}: {type(bag)}")
    if bag.ndim != 2:
        raise ValueError(f"Expected [patches, dim] tensor in {path}, got {tuple(bag.shape)}")
    return bag.float().mean(dim=0)


def build_cache(rows: list[dict[str, str]], cache_path: Path) -> dict[str, object]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    xs = []
    ys = []
    metas = []
    for i, row in enumerate(rows, start=1):
        xs.append(slide_embedding(row["feature_path"]))
        ys.append(int(row["label"]))
        metas.append({k: row.get(k, "") for k in ("cancer", "case_id", "tss", "slide_id")})
        if i % 50 == 0:
            print(f"pooled {i}/{len(rows)} slides", flush=True)
    payload = {
        "x": torch.stack(xs),
        "y": torch.tensor(ys, dtype=torch.long),
        "meta": metas,
    }
    torch.save(payload, cache_path)
    return payload


def stratified_split(y: torch.Tensor, train_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(y.tolist()):
        by_label[label].append(idx)
    for _, idxs in sorted(by_label.items()):
        rng.shuffle(idxs)
        n_train = max(1, int(round(len(idxs) * train_frac)))
        n_train = min(n_train, len(idxs) - 1) if len(idxs) > 1 else len(idxs)
        train_idx.extend(idxs[:n_train])
        test_idx.extend(idxs[n_train:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return torch.tensor(train_idx), torch.tensor(test_idx)


class Classifier(nn.Module):
    def __init__(self, dim: int, num_classes: int, hidden: int = 0) -> None:
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, num_classes),
            )
        else:
            self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in DataLoader(TensorDataset(x, y), batch_size=batch_size):
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb).argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="/data_2_4T/data_zjj/continual_wsi/indices/multicancer_conch_s1024.csv")
    parser.add_argument("--out-dir", default="/data_2_4T/data_zjj/continual_wsi/smoke_multicancer")
    parser.add_argument("--max-per-class", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"mean_features_max{args.max_per_class}_seed{args.seed}.pt"

    rows = read_index(Path(args.index), args.max_per_class, args.seed)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        print(f"loaded cache {cache_path}", flush=True)
    else:
        payload = build_cache(rows, cache_path)

    x: torch.Tensor = payload["x"]
    y: torch.Tensor = payload["y"]
    train_idx, test_idx = stratified_split(y, 0.8, args.seed)
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    num_classes = int(y.max().item()) + 1
    model = Classifier(x.shape[1], num_classes, args.hidden).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    best = {"epoch": 0, "test_acc": 0.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * int(yb.numel())
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            train_acc = evaluate(model, x_train, y_train, args.batch_size, args.device)
            test_acc = evaluate(model, x_test, y_test, args.batch_size, args.device)
            if test_acc > best["test_acc"]:
                best = {"epoch": epoch, "test_acc": test_acc}
            print(
                f"epoch={epoch:03d} loss={total_loss / len(y_train):.4f} "
                f"train_acc={train_acc:.3f} test_acc={test_acc:.3f}",
                flush=True,
            )

    counts = Counter(y.tolist())
    result = {
        "index": args.index,
        "cache": str(cache_path),
        "num_slides": int(y.numel()),
        "num_classes": num_classes,
        "class_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "train_size": int(y_train.numel()),
        "test_size": int(y_test.numel()),
        "best": best,
        "final_train_acc": evaluate(model, x_train, y_train, args.batch_size, args.device),
        "final_test_acc": evaluate(model, x_test, y_test, args.batch_size, args.device),
        "device": args.device,
    }
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

