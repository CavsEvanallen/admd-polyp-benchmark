#!/usr/bin/env python3
"""
替代原 05_stats.py(原脚本因路径少一层 1pt/bbox 而报错,汇总全为 NaN)。

与原脚本的区别:
  * 路径按 holdout_*/<方法>/<数据集>/<提示>/seed*/ 读取
  * 公共测试集(ColonDB/ETIS/CVC-300)每折都会测一遍,先对折与种子取平均,避免把同一张图重复计数
  * 检验在"分组"层面做(同一次检查/序列的帧先平均),避免帧间相关造成的伪显著
  * 报告 分层 bootstrap 95%CI(先抽折,再抽分组)、折级符号计数、Wilcoxon(仅供参考)、
    以及等效性判断:90% bootstrap CI 是否完全落在 ±margin(默认 Dice/IoU ±0.01)内。
    无增益时可以正面写"等效",而不是"不显著"
  * --drop_collided: 丢掉预测掩码尺寸与 metrics 中 h,w 不一致、或与较小数据集逐项重复的图(ID 冲突造成的错配图,见反馈文档 §2)

用法:
  python 01_stats_fixed.py --results ADMD提交/results/polypgen_loco --out tables/stats.md \
      --ours admd --base sam_ft --drop_collided
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from common import load_results

METRICS = ['dice', 'iou', 'hd95', 'assd']


def boot_ci(per_fold, n_boot, rng):
    """per_fold: {fold: Series(group -> diff)}。两级 bootstrap:折 -> 折内分组。返回 (均值, lo, hi)"""
    folds = list(per_fold)
    vals = {f: per_fold[f].to_numpy() for f in folds}
    means = []
    for _ in range(n_boot):
        ff = rng.choice(len(folds), len(folds))
        acc = []
        for i in ff:
            v = vals[folds[i]]
            acc.append(rng.choice(v, len(v)).mean() if len(v) else np.nan)
        means.append(np.nanmean(acc))
    allv = np.nanmean([v.mean() for v in vals.values() if len(v)])
    return allv, np.percentile(means, 2.5), np.percentile(means, 97.5), np.percentile(means, 5), np.percentile(means, 95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ours', default='admd')
    ap.add_argument('--base', default='sam_ft')
    ap.add_argument('--heldout_prefix', default='PolypGen_')
    ap.add_argument('--drop_collided', action='store_true')
    ap.add_argument('--margin', type=float, default=0.01, help='Dice 等效边界(IoU 同值;HD95/ASSD 用相对 5%%)')
    ap.add_argument('--keep_empty', action='store_true', help='默认剔除 flag=both_empty/empty_pred 的图(真值为空,Dice 被强制为 1,会虚高并稀释差异)')
    ap.add_argument('--n_boot', type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    D = load_results(a.results, a.heldout_prefix, check_mask_size=a.drop_collided)
    n0 = len(D)
    if not a.keep_empty:
        D = D[D['gt_area'] > 0]
        print(f'剔除真值为空的图: {n0 - len(D)} 行')
        n0 = len(D)
    if a.drop_collided:
        bad = ~D['size_ok']
        print(f'丢弃尺寸错配的行: {bad.sum()} / {n0}')
        print(D[bad].groupby(['ds']).size().to_string())
        D = D[~bad]
    out = [f'# 配对比较(均值=折平均;留出集的 Wilcoxon 因 UUID 帧无法并组而偏乐观,以 CI 为准) {a.ours} − {a.base}\n',
           f'行数 {len(D)}(原 {n0});drop_collided={a.drop_collided};种子数 {D.seed.nunique()}\n']
    for prompt in sorted(D['prompt'].unique()):
        for dsg in ['HELDOUT'] + sorted(x for x in D['dsg'].unique() if x != 'HELDOUT'):
            S = D[(D['prompt'] == prompt) & (D['dsg'] == dsg)]
            if S.empty:
                continue
            out.append(f'\n## prompt={prompt}  测试集={dsg}\n')
            out.append('| 指标 | ours均值 | base均值 | Δ(ours−base) | 95% CI(分层bootstrap) | ours更优的折 | Wilcoxon p(仅参考) | 等效(90%CI⊂±δ) | 分组胜率 | 分组数 |')
            out.append('|---|---|---|---|---|---|---|---|---|---|')
            for met in METRICS:
                # 先对种子平均;公共集再对折平均 -> 每张图一个值;留出集按折保留
                key = ['hold', 'ds', 'image', 'group', 'method']
                P = S.groupby(key)[met].mean().unstack('method')
                if a.ours not in P or a.base not in P:
                    continue
                P = P.dropna(subset=[a.ours, a.base]).reset_index()
                if dsg != 'HELDOUT':
                    P = P.groupby(['ds', 'image', 'group'])[[a.ours, a.base]].mean().reset_index()
                    P['hold'] = 'all'
                P['d'] = P[a.ours] - P[a.base]
                # 分组层:同折同组的帧取平均
                G3 = P.groupby(['hold', 'group'])[[a.ours, a.base, 'd']].mean()
                G = G3['d']
                per_fold = {h: g.droplevel(0) for h, g in G.groupby(level=0)}
                mean, lo, hi, lo90, hi90 = boot_ci(per_fold, a.n_boot, rng)
                fold_mean = pd.Series({h: g.mean() for h, g in per_fold.items()})
                sign = (fold_mean > 0).sum() if met in ('dice', 'iou') else (fold_mean < 0).sum()
                gv = G.to_numpy()
                try:
                    p = wilcoxon(gv).pvalue
                except ValueError:
                    p = np.nan
                m = a.margin if met in ('dice', 'iou') else 0.05 * P[a.base].mean()
                equiv = '是' if (lo90 > -m and hi90 < m) else '否'
                fm = G3.groupby(level=0)[[a.ours, a.base]].mean().mean()      # 折平均,与 Δ 口径一致
                win = ((gv > 0) if met in ('dice', 'iou') else (gv < 0)).mean()
                out.append(f'| {met} | {fm[a.ours]:.4f} | {fm[a.base]:.4f} | {mean:+.4f} | [{lo:+.4f}, {hi:+.4f}] | '
                           f'{sign}/{len(fold_mean)} | {p:.2e} | {equiv} | {win:.0%} | {len(gv)} |')
            if dsg == 'HELDOUT':
                out.append('\n折级 Dice(ours / base / Δ):')
                for h, g in S[S.method.isin([a.ours, a.base])].groupby('hold'):
                    mm = g.groupby('method')['dice'].mean()
                    out.append(f'- {h}: n={g.image.nunique()}  {mm.get(a.ours, np.nan):.4f} / {mm.get(a.base, np.nan):.4f} / {mm.get(a.ours, np.nan) - mm.get(a.base, np.nan):+.4f}')
    open(a.out, 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
