#!/usr/bin/env python3
"""
按"来源 + 分组"生成防泄漏的留一来源划分。替代 02_make_manifests.py 里 polypgen_loco 的"每中心随机 9:1"。

输入: 一个 CSV,列 image(路径), mask(路径), source(来源名,如 HMUH/100H/UUID 或真 PolypGen 的 C1..C6)
      可选列 group(患者或检查 ID)。没有 group 列时,用 common.infer_group 从文件名推断,
      推断不出来的(UUID 等)每张图单独一组 —— 这时请务必向数据方要患者表,否则 val 仍可能泄漏。
输出: out/holdout_<S>/seed<k>/{train.csv,val.csv,test_<dataset>.csv}
      train/val 之间按 group 划分(整组进同一边);domain_id = 训练来源在本折内的序号;
      留出来源整体作为测试;并写 split_report.json(每折各划分的图数/组数/来源)。
并在末尾自检: train/val/test 之间不共享 group。

用法:
  python 04_make_group_split.py --table all_images.csv --out manifests/loso --seeds 0 1 2 --val_frac 0.1
"""
import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path
import pandas as pd
from common import infer_group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--min_test', type=int, default=100, help='少于此数的来源不单独作留出折,而是只作训练(避免 n=15 的折)')
    a = ap.parse_args()

    T = pd.read_csv(a.table)
    T['dataset'] = 'Source_' + T['source'].astype(str)
    if 'group' not in T:
        T['group'] = [f"{s}:{infer_group(Path(p).stem)}" for s, p in zip(T['source'], T['image'])]
    else:
        T['group'] = T['source'].astype(str) + ':' + T['group'].astype(str)
    sizes = T.groupby('source').size()
    folds = [s for s in sizes.index if sizes[s] >= a.min_test]
    small = [s for s in sizes.index if sizes[s] < a.min_test]
    print('留出折:', folds, '| 仅训练(太小):', small)
    out = Path(a.out)
    report = {}
    for seed in a.seeds:
        for hold in folds:
            test = T[T.source == hold]
            pool = T[T.source != hold]
            rng = random.Random(seed)
            val_idx = set()
            for s, g in pool.groupby('source'):
                gs = sorted(g.group.unique()); rng.shuffle(gs)
                n_val = max(1, round(len(gs) * a.val_frac))
                chosen = set(gs[:n_val])
                val_idx |= set(g.index[g.group.isin(chosen)])
            val = pool.loc[sorted(val_idx)]
            train = pool.drop(index=val.index)
            order = {s: i for i, s in enumerate(sorted(train.source.unique()))}
            d = out / f'holdout_{hold}' / f'seed{seed}'
            d.mkdir(parents=True, exist_ok=True)

            def dump(df, name, dom):
                df = df.copy()
                df['domain_id'] = df['source'].map(order).fillna(-1).astype(int) if dom else -1
                df['split'] = name
                df[['image', 'mask', 'dataset', 'domain_id', 'split']].to_csv(d / f'{name}.csv', index=False)
            dump(train, 'train', True)
            dump(val, 'val', True)
            dump(test, f'test_Source_{hold}', False)
            # 自检
            tg, vg, eg = set(train.group), set(val.group), set(test.group)
            assert not (tg & vg) and not (tg & eg) and not (vg & eg), '分组泄漏'
            assert not set(train.image) & set(test.image)
            report[f'{hold}/seed{seed}'] = dict(train=len(train), val=len(val), test=len(test), K=len(order),
                                              train_groups=len(tg), val_groups=len(vg), test_groups=len(eg))
    json.dump(report, open(out / 'split_report.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print('自检通过: train/val/test 之间无共享 group')


if __name__ == '__main__':
    main()
