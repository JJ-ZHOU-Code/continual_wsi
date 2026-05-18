#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-csv', default='/data_1_16T/data_tcga/rcc/download/TCGA_RCC_path_subtype.csv')
    parser.add_argument('--feat-dir', default='/data_1_16T/data_tcga/rcc/ExpData/feats-l0-s1024-CONCH/pt_files')
    parser.add_argument('--out', default='/data_2_4T/data_zjj/continual_wsi/indices/rcc_subtype_conch_s1024.csv')
    args = parser.parse_args()

    label_csv = Path(args.label_csv)
    feat_dir = Path(args.feat_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    feat_by_stem = {p.stem: p for p in feat_dir.glob('*.pt')}
    rows = []
    missing = []
    with label_csv.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slide_id = row.get('pathology_id') or row.get('slide_id')
            if not slide_id:
                continue
            feat = feat_by_stem.get(slide_id)
            if feat is None:
                missing.append(slide_id)
                continue
            rows.append({
                'case_id': row.get('patient_id') or row.get('case_id') or '',
                'slide_id': slide_id,
                'subtype': row.get('subtype') or row.get('label') or '',
                'label': row.get('label') if row.get('label') is not None else '',
                'feature_path': str(feat),
            })

    with out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'slide_id', 'subtype', 'label', 'feature_path'])
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(r['subtype'] for r in rows)
    print(f'wrote={out}')
    print(f'matched={len(rows)} missing={len(missing)} features={len(feat_by_stem)}')
    print('subtype_counts=' + ', '.join(f'{k}:{v}' for k, v in sorted(counts.items())))
    if missing[:5]:
        print('missing_examples=' + ', '.join(missing[:5]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
