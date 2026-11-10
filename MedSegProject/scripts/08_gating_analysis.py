#!/usr/bin/env python3
"""
ADMD 门控行为分析（论文 5.5 节 / 图：Gating behaviour）。

需要模型在推理时按 10_mmseg_interface.md 的格式导出 gating.csv：
  image, domain_true, d_0..d_{K-1}, alpha, w_0..w_{K-1}
  domain_true: 源域测试集填真实域序号；未见目标域填 -1
再加同一目录下的 metrics.csv（04 生成）。

输出（--out 目录）：
  summary.md         域分类准确率（仅 domain_true>=0 的集合）、max-prob / 熵 / alpha 的均值、与 Dice 的 Spearman 相关、平均融合权重向量
  hist_maxprob.png   各测试集 max_k d_k 直方图
  hist_alpha.png     各测试集 alpha 直方图
  scatter_conf_dice.png  置信度 vs Dice 散点
用法：
  python 08_gating_analysis.py --runs results/polypgen_loco/holdout_C3/mmseg_full/*/bbox/seed0 --out figs/gating_C3
  （--runs 可给多个目录，每个目录 = 一个测试集；目录名倒数第三级当作测试集名）
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_run(d):
    d = Path(d)
    g = pd.read_csv(d / 'gating.csv').set_index('image')
    m = pd.read_csv(d / 'metrics.csv').set_index('image')
    df = g.join(m[['dice', 'hd95']], how='inner')
    dcols = sorted([c for c in df.columns if c.startswith('d_')], key=lambda s: int(s[2:]))
    wcols = sorted([c for c in df.columns if c.startswith('w_')], key=lambda s: int(s[2:]))
    P = df[dcols].to_numpy()
    df['max_prob'] = P.max(1)
    df['pred_domain'] = P.argmax(1)
    df['entropy'] = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)
    return df, dcols, wcols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for r in a.runs:
        name = Path(r).parts[-3]
        runs[name] = load_run(r)

    lines = ['# Gating behaviour summary', '',
             '| test set | n | domain acc (if known) | mean max-prob | mean entropy | mean alpha | Spearman(max-prob, Dice) | Spearman(alpha, Dice) | mean fusion weights |',
             '|---|---|---|---|---|---|---|---|---|']
    for name, (df, dcols, wcols) in runs.items():
        known = df[df['domain_true'] >= 0]
        acc = f'{(known["pred_domain"] == known["domain_true"]).mean()*100:.1f}%' if len(known) else 'n/a (unseen)'
        rho1 = stats.spearmanr(df['max_prob'], df['dice']).correlation
        rho2 = stats.spearmanr(df['alpha'], df['dice']).correlation if 'alpha' in df else float('nan')
        w = df[wcols].mean().to_numpy() if wcols else []
        lines.append(f'| {name} | {len(df)} | {acc} | {df["max_prob"].mean():.3f} | {df["entropy"].mean():.3f} | '
                     f'{df["alpha"].mean() if "alpha" in df else float("nan"):.3f} | {rho1:.3f} | {rho2:.3f} | '
                     f'{np.array2string(np.round(w, 3), separator=", ")} |')
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    for col, fname in (('max_prob', 'hist_maxprob.png'), ('alpha', 'hist_alpha.png')):
        plt.figure(figsize=(6, 4))
        for name, (df, _, _) in runs.items():
            if col in df:
                plt.hist(df[col], bins=30, alpha=0.5, label=name, density=True)
        plt.xlabel(col)
        plt.ylabel('density')
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / fname, dpi=200)
        plt.close()

    plt.figure(figsize=(6, 4))
    for name, (df, _, _) in runs.items():
        plt.scatter(df['max_prob'], df['dice'], s=6, alpha=0.5, label=name)
    plt.xlabel('max domain probability')
    plt.ylabel('Dice')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / 'scatter_conf_dice.png', dpi=200)
    plt.close()
    print(f'figures written to {out}')


if __name__ == '__main__':
    main()
