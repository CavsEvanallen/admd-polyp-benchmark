#!/usr/bin/env python3
"""任意两个(结果根目录:方法)的配对比较,表 4/5/6 都用它。

目录约定(与 ADMD提交4 相同):
  <root>/loso/holdout_Ck/<method>/<TestSet>/<prompt>/seedS/metrics.csv      (需有列 image, dice, gt_area)

用法:
  # 先看某个根目录里到底有哪些方法/测试集/提示/种子(不确定命名时先跑这个)
  python3 10_pairwise.py --base . --list results_routing

  # B/A 配对:  A − B
  python3 10_pairwise.py --base . --a results_pat30:admd --b results:sam_ft --label "ADMD p30 vs standard p15" --out out_pat30.md

  # 一次比较多对:重复 --pair "A根:方法,B根:方法,标签"
  python3 10_pairwise.py --base . --pair "results_pat30:admd,results:sam_ft,ADMD-p30 vs std-p15" --pair "results_pat30:sam_ft,results:sam_ft,std-p30 vs std-p15"

口径(与论文一致):
  - 只用 gt_area>0 的帧(真值为空的帧不进任何统计)
  - 每张图先对种子、折取平均,再做图像级 bootstrap(3000 次,固定种子)得 95% CI
  - PolypGen 留出中心 C1–C5 报告每个中心的 Δ 与 CI,另报"中心均值"和"更优的中心数"
  - 公共集(CVC-ColonDB / ETIS-LaribPolypDB / CVC-300)若 A、B 都有则同样报告
  - 等效:Δ 的 90% CI ⊂ ±0.01 Dice
  - 不做 p 值;不做任何行删除(没有 drop_collided)
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
NB = 3000
MARGIN = 0.01


def load(base, root, method):
    rows = []
    for f in glob.glob(str(Path(base) / root / 'loso' / 'holdout_*' / method / '*' / '*' / 'seed*' / 'metrics.csv')):
        p = Path(f).relative_to(Path(base) / root / 'loso').parts
        d = pd.read_csv(f)
        d['hold'], d['ds'], d['prompt'], d['seed'] = p[0], p[2], p[3], p[4]
        rows.append(d)
    if not rows:
        sys.exit(f'没有找到 {root}:{method} 的 metrics.csv(先用 --list {root} 看实际命名)')
    d = pd.concat(rows)
    n0 = len(d)
    d = d[d.gt_area > 0]
    print(f'  {root}:{method}: {n0} 行,去掉空真值帧后 {len(d)} 行', file=sys.stderr)
    return d


def per_image(d, ds, prompt):
    s = d[(d.ds == ds) & (d.prompt == prompt)]
    return s.groupby('image').dice.mean()


def boot_ci(x, q):
    idx = RNG.integers(0, len(x), (NB, len(x)))
    return np.percentile(x[idx].mean(1), q)


def compare(dA, dB, label):
    out = [f'\n## {label}\n']
    pubs = ['Source_CVC-ColonDB', 'Source_ETIS-LaribPolypDB', 'Source_CVC-300']
    for pr in ['1pt', 'bbox']:
        out.append(f'\n**{"Point" if pr == "1pt" else "Box"} prompt (Dice, Δ = A − B)**\n')
        out.append('| Test set | n | A | B | Δ | 95% CI | equiv. (90% CI ⊂ ±0.01) |')
        out.append('|---|---|---|---|---|---|---|')
        deltas = []
        for c in ['Source_C1', 'Source_C2', 'Source_C3', 'Source_C4', 'Source_C5'] + pubs:
            if c not in set(dA.ds) or c not in set(dB.ds):
                continue
            a, b = per_image(dA, c, pr), per_image(dB, c, pr)
            ix = a.index.intersection(b.index)
            if len(ix) == 0:
                continue
            x = (a[ix] - b[ix]).values
            lo, hi = boot_ci(x, [2.5, 97.5]); l90, h90 = boot_ci(x, [5, 95])
            eq = '是' if (l90 > -MARGIN and h90 < MARGIN) else '否'
            out.append(f'| {c.replace("Source_", "")} | {len(ix)} | {a[ix].mean():.4f} | {b[ix].mean():.4f} | {x.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | {eq} |')
            if c.startswith('Source_C'):
                deltas.append(x.mean())
        if deltas:
            dl = np.array(deltas)
            out.append(f'| **中心均值 (C1–C5 中有数据的 {len(dl)} 个)** | | | | {dl.mean():+.4f} | A 更优的中心 {int((dl > 0).sum())}/{len(dl)} | |')
    return '\n'.join(out)


def list_root(base, root):
    rows = []
    for f in glob.glob(str(Path(base) / root / 'loso' / 'holdout_*' / '*' / '*' / '*' / 'seed*' / 'metrics.csv')):
        p = Path(f).relative_to(Path(base) / root / 'loso').parts
        rows.append(p[:5])
    if not rows:
        sys.exit(f'{root}/loso 下没有 metrics.csv;目录结构不同的话把 `find {root} -maxdepth 6 | head -30` 贴给我')
    df = pd.DataFrame(rows, columns=['hold', 'method', 'ds', 'prompt', 'seed'])
    print(df.groupby(['method', 'prompt']).agg(folds=('hold', 'nunique'), sets=('ds', 'nunique'), seeds=('seed', 'nunique'), files=('seed', 'size')).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default='.')
    ap.add_argument('--list')
    ap.add_argument('--a'); ap.add_argument('--b'); ap.add_argument('--label', default='A vs B')
    ap.add_argument('--pair', action='append', default=[])
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.list:
        list_root(a.base, a.list); sys.exit()
    pairs = [(a.a, a.b, a.label)] if a.a else []
    for s in a.pair:
        x, y, l = s.split(',', 2); pairs.append((x, y, l))
    md = ['# 配对比较(Δ = A − B;口径见脚本说明)']
    cache = {}
    for x, y, l in pairs:
        for s in (x, y):
            if s not in cache:
                cache[s] = load(a.base, *s.split(':'))
        md.append(compare(cache[x], cache[y], l))
    txt = '\n'.join(md)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt, encoding='utf-8')
