#!/usr/bin/env python3
"""
诊断"跨数据集同名图互相覆盖"(2026-10-22 在 ADMD提交 里发现的 bug)。

症状:CVC-ColonDB 的 id 1–148 Dice≈0,ETIS 的 id 149–196 Dice≈0;
      ColonDB 的 id 149–208 的指标与 CVC-300(id 149–208)逐项相同;
      但同一批 bbox 提示下 ColonDB id≥149、ETIS id≤148 的 Dice≈0.9。
证据:ColonDB 1–148 的预测掩码尺寸 1225×966(ETIS 的尺寸),而 metrics 里 h×w=500×574。
原因:cache_embeddings.py / data.py / 03_prompts.py / 04_metrics.py 都以 Path(image).stem 作键,
      不同数据集同名("1","2",…)的图共用一个缓存文件和一条提示。
本脚本不需要原始数据,只读 results/,输出:
  1. 每个 (方法,数据集,提示) 下 预测掩码尺寸 ≠ metrics (h,w) 的图数
  2. 这些图的 Dice 均值 vs 其余图
  3. 不同数据集之间图像名重叠的数目(重叠 = 有碰撞风险)
退出码 0 = 未发现问题;1 = 发现碰撞。
"""
import argparse, sys
import pandas as pd
from common import load_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--heldout_prefix', default='PolypGen_')
    ap.add_argument('--csv_out', default=None)
    a = ap.parse_args()
    D = load_results(a.results, a.heldout_prefix, check_mask_size=True)
    one = D[(D.method == D.method.iloc[0]) & (D.prompt == D.prompt.iloc[0]) & (D.hold == D.hold.iloc[0])]
    ids = one.groupby('ds')['image'].apply(set)
    print('== 同一折内,不同数据集的图像名重叠 ==')
    names = list(ids.index)
    any_overlap = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            k = len(ids.iloc[i] & ids.iloc[j])
            if k:
                any_overlap = True
                print(f'  {names[i]} ∩ {names[j]}: {k} 个同名')
    if not any_overlap:
        print('  无')
    print('\n== 掩码尺寸与 metrics(h,w) 不一致,或与较小数据集逐项相同(同尺寸碰撞)的图 ==')
    g = D.groupby(['hold', 'method', 'ds', 'prompt']).agg(n=('image', 'size'), bad=('size_ok', lambda s: int((~s).sum())))
    g = g[g.bad > 0]
    print(g.to_string() if len(g) else '  无')
    bad = D[~D.size_ok]
    if len(bad):
        print('\n== 错配图 vs 其余图 的 Dice 均值 ==')
        print(D.groupby(['ds', 'size_ok'])['dice'].agg(['mean', 'count']).round(3).to_string())
        print('\n错配图 id 范围:')
        for ds, s in bad.groupby('ds'):
            try:
                ii = s.image.str.split('__').str[-1].astype(int)
                print(f'  {ds}: {ii.min()}–{ii.max()}  ({len(set(ii))} 张)')
            except ValueError:
                print(f'  {ds}: {s.image.nunique()} 张(非数字名)')
        if a.csv_out:
            bad[['hold', 'method', 'ds', 'prompt', 'image']].drop_duplicates().to_csv(a.csv_out, index=False)
    sys.exit(1 if len(bad) else 0)


if __name__ == '__main__':
    main()
