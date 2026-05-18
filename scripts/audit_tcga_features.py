#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path


def count_files(root: Path) -> Counter:
    counts = Counter()
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            suffix = Path(name).suffix.lower() or '<none>'
            counts[suffix] += 1
    return counts


def conch_counts(root: Path) -> list[tuple[str, int]]:
    rows = []
    for path in root.glob('*/ExpData/feats-l0-s*-CONCH/pt_files'):
        if path.is_dir():
            rows.append((str(path), len(list(path.glob('*.pt')))))
    return sorted(rows, key=lambda x: (-x[1], x[0]))


def first_lines(path: Path, n: int = 3) -> list[str]:
    try:
        with path.open('r', encoding='utf-8', errors='replace') as f:
            return [f.readline().rstrip('\n') for _ in range(n)]
    except OSError:
        return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/data_1_16T/data_tcga')
    parser.add_argument('--out', default='/data_2_4T/data_zjj/continual_wsi/audits/tcga_feature_audit.md')
    args = parser.parse_args()

    root = Path(args.data_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    counts = count_files(root)
    conch = conch_counts(root)
    label_files = []
    for pattern in ['*/download/gdc_sample_sheet.tsv', '*/download/*subtype*.csv', '*/download/metadata.cart.json']:
        label_files.extend(sorted(root.glob(pattern)))

    lines = []
    lines.append('# TCGA Feature Audit')
    lines.append('')
    lines.append(f'Data root: {root}')
    lines.append('')
    lines.append('## File Type Counts')
    lines.append('')
    for suffix, count in counts.most_common(40):
        lines.append(f'- {suffix}: {count}')
    lines.append('')
    lines.append('## CONCH Feature Counts')
    lines.append('')
    for path, count in conch:
        lines.append(f'- {path}: {count}')
    lines.append('')
    lines.append('## Candidate Label / Metadata Files')
    lines.append('')
    for path in label_files[:80]:
        lines.append(f'### {path}')
        for line in first_lines(path, 3):
            lines.append(f'    {line}')
        lines.append('')
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
