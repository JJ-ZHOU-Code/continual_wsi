#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/data_1_16T/data_tcga')
    parser.add_argument('--feature-name', default='feats-l0-s1024-CONCH')
    parser.add_argument('--out', default='/data_1_16T/data_zjj/continual_wsi/indices/multicancer_conch_s1024.csv')
    args = parser.parse_args()

    root = Path(args.data_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for feat_dir in sorted(root.glob(f'*/ExpData/{args.feature_name}/pt_files')):
        cancer = feat_dir.parts[-4]
        for feat in sorted(feat_dir.glob('*.pt')):
            slide_id = feat.stem
            case_id = '-'.join(slide_id.split('-')[:3]) if slide_id.startswith('TCGA-') else ''
            # TCGA tissue source site (TSS) is the second barcode field; useful as a weak site proxy.
            tss = slide_id.split('-')[1] if slide_id.startswith('TCGA-') and len(slide_id.split('-')) > 1 else ''
            rows.append({
                'cancer': cancer,
                'case_id': case_id,
                'tss': tss,
                'slide_id': slide_id,
                'feature_path': str(feat),
            })

    label_to_id = {name: i for i, name in enumerate(sorted({r['cancer'] for r in rows}))}
    for row in rows:
        row['label'] = label_to_id[row['cancer']]

    with out.open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['cancer', 'label', 'case_id', 'tss', 'slide_id', 'feature_path']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(r['cancer'] for r in rows)
    tss_counts = Counter(r['tss'] for r in rows if r['tss'])
    print(f'wrote={out}')
    print(f'total={len(rows)} cancers={len(counts)}')
    print('cancer_counts=' + ', '.join(f'{k}:{v}' for k, v in sorted(counts.items())))
    print(f'tss_unique={len(tss_counts)}')
    print('top_tss=' + ', '.join(f'{k}:{v}' for k, v in tss_counts.most_common(10)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
