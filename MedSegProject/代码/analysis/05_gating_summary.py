#!/usr/bin/env python3
"""
汇总各折门控行为(只读 gating.csv + metrics.csv),给出论文 §5.5 需要的数:
  max-prob / 熵 / 与均匀分布的距离 / 融合权重塌缩程度(有效分支数 exp(H(w))) / 置信度与 Dice 的 Spearman
并给出"门控是否有信息"的判据:有效分支数是否明显小于 K、Spearman 的符号是否跨折一致。
"""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--prompt', default='bbox')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rows = []
    for g in sorted(glob.glob(f'{a.results}/holdout_*/admd/*/{a.prompt}/seed*/gating.csv')):
        p = Path(g)
        hold, ds, seed = p.parts[-6], p.parts[-4], p.parts[-2]
        G = pd.read_csv(g)
        M = pd.read_csv(p.parent / 'metrics.csv')
        G['image'] = G['image'].astype(str); M['image'] = M['image'].astype(str)
        J = G.merge(M[['image', 'dice', 'gt_area']], on='image')
        J = J[J.gt_area > 0]          # 真值为空的帧 Dice 被强制记 1,必须剔除,否则会污染 Spearman 与各项均值
        d = J[[c for c in J if c.startswith('d_')]].to_numpy()
        w = J[[c for c in J if c.startswith('w_')]].to_numpy()
        K = d.shape[1]
        ent = lambda x: -(x * np.log(x + 1e-12)).sum(1)
        rho = spearmanr(d.max(1), J.dice).statistic if len(J) > 5 else np.nan
        rows.append(dict(hold=hold, ds=ds, seed=seed, n=len(J), K=K, maxprob=d.max(1).mean(), uniform=1 / K,
                         eff_branches_d=np.exp(ent(d)).mean(), eff_branches_w=np.exp(ent(w)).mean(),
                         top_w=w.mean(0).max(), spearman_conf_dice=rho))
    R = pd.DataFrame(rows)
    txt = R.round(3).to_string(index=False)
    print(txt)
    pub = R[~R.ds.str.startswith('PolypGen_') & ~R.ds.str.startswith('Source_')]
    print('\n公共测试集上 Spearman(置信度, Dice) 的符号一致性:')
    print(pub.groupby('ds').spearman_conf_dice.agg(['mean', lambda s: (s > 0).mean()]).round(3).rename(columns={'<lambda_0>': '为正的折比例'}).to_string())
    if a.out:
        open(a.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
