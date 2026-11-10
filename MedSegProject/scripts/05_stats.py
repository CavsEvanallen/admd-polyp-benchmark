#!/usr/bin/env python3
"""
配对统计：把"本文方法"与若干基线在同一测试集上逐图配对比较。

输入：每个方法一个（或多个种子的多个）metrics.csv，按 image 列对齐。
输出：每个指标的 mean±std、中位数、配对差值均值及 95% bootstrap CI、Wilcoxon 符号秩 p、Holm 校正 p、胜率。

用法：
  python 05_stats.py --ours "results/thesis/mmseg_full/CVC-ColonDB/1pt+bbox/seed*/metrics.csv" \
                     --baseline mmseg_noadmd="results/thesis/mmseg_noadmd/CVC-ColonDB/1pt+bbox/seed*/metrics.csv" \
                     --baseline edgesam="results/thesis/edgesam/CVC-ColonDB/1pt+bbox/seed0/metrics.csv" \
                     --metrics dice iou hd95 assd --out results/thesis/stats_CVC-ColonDB_1pt+bbox.md
多个种子：同一方法多个 csv 会先按图取跨种子平均，再做配对检验。
"""
import argparse, glob
import numpy as np
import pandas as pd
from scipy import stats

LOWER_BETTER = {'hd95', 'assd'}


def load_method(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    dfs = [pd.read_csv(f).set_index('image') for f in files]
    common = set.intersection(*[set(d.index) for d in dfs])
    dfs = [d.loc[sorted(common)] for d in dfs]
    num = [d.select_dtypes('number') for d in dfs]
    mean_df = sum(num) / len(num)
    per_seed_means = pd.DataFrame([d.mean(numeric_only=True) for d in num])
    return mean_df, per_seed_means, len(files)


def boot_ci(diff, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', required=True)
    ap.add_argument('--baseline', action='append', default=[], help='name=glob')
    ap.add_argument('--metrics', nargs='+', default=['dice', 'iou', 'hd95', 'assd'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    ours, ours_seed, n_ours = load_method(a.ours)
    lines = [f'# Paired comparison (ours: {a.ours}, seeds={n_ours}, n_images={len(ours)})', '']
    for m in a.metrics:
        sd = f'{ours_seed[m].std(ddof=0)*(100 if m in ("dice","iou") else 1):.2f}' if n_ours > 1 else 'n/a'
        scale = 100 if m in ('dice', 'iou') else 1
        lines.append(f'## {m}  ours mean={ours[m].mean()*scale:.2f} (seed-std {sd}), median={ours[m].median()*scale:.2f}')
        lines.append('')
        lines.append('| baseline | n | baseline mean | diff (ours-base) | 95% CI | Wilcoxon p | Holm p | win rate |')
        lines.append('|---|---|---|---|---|---|---|---|')
        rows, ps = [], []
        for spec in a.baseline:
            name, pat = spec.split('=', 1)
            base, _, _ = load_method(pat)
            common = ours.index.intersection(base.index)
            x, y = ours.loc[common, m].to_numpy(), base.loc[common, m].to_numpy()
            ok = ~(np.isnan(x) | np.isnan(y))
            x, y = x[ok], y[ok]
            diff = x - y
            if np.allclose(diff, 0):
                p = 1.0
            else:
                p = stats.wilcoxon(x, y, zero_method='wilcox', alternative='two-sided').pvalue
            lo, hi = boot_ci(diff)
            better = (x < y) if m in LOWER_BETTER else (x > y)
            win = better.mean()
            rows.append([name, len(x), y.mean() * scale, diff.mean() * scale, lo * scale, hi * scale, p, win])
            ps.append(p)
        adj = holm(np.array(ps)) if ps else []
        for r, pa in zip(rows, adj):
            lines.append(f'| {r[0]} | {r[1]} | {r[2]:.2f} | {r[3]:+.2f} | [{r[4]:+.2f}, {r[5]:+.2f}] | {r[6]:.2e} | {pa:.2e} | {r[7]*100:.1f}% |')
        lines.append('')
    with open(a.out, 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
