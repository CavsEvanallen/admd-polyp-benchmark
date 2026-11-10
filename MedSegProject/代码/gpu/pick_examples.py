#!/usr/bin/env python3
"""为每个留出中心挑 3 张示例图,复制 原图/真值/各方法预测掩码 到 out/<centre>/<k>_<name>/,并写 index.csv。"""
import argparse, shutil
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--work', required=True); ap.add_argument('--results', default='results_masks')
ap.add_argument('--out', required=True); ap.add_argument('--seed0', default='0'); ap.add_argument('--prompt', default='bbox')
a = ap.parse_args()
W = Path(a.work); R = W / a.results / 'loso'; out = Path(a.out); methods = ['sam_ft', 'admd', 'wide_sam']
rows = []
for hold in sorted(R.glob('holdout_*')):
    c = hold.name.split('_')[1]
    ds = f'Source_{c}'
    man = pd.read_csv(W / 'manifests/loso' / hold.name / f'seed{a.seed0}' / f'test_{ds}.csv')
    man['key'] = man['dataset'] + '__' + man['image'].map(lambda p: Path(p).stem)
    m = {k: pd.read_csv(hold / k / ds / a.prompt / f'seed{a.seed0}' / 'metrics.csv').set_index('image') for k in methods}
    d = pd.DataFrame({k: v['dice'] for k, v in m.items()}).dropna()
    d = d[m['sam_ft'].loc[d.index, 'gt_area'] > 0]
    picks = {'median': (d.admd - d.sam_ft).abs().sub((d.admd - d.sam_ft).abs().median()).abs().idxmin(),
             'admd_worst': (d.admd - d.sam_ft).idxmin(), 'adapter_best': (d.wide_sam - d.sam_ft).idxmax()}
    for name, key in picks.items():
        r = man[man.key == key].iloc[0]; dst = out / c / f'{name}_{key}'; dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(W / r['image'], dst / ('image' + Path(r['image']).suffix))
        shutil.copy(W / r['mask'], dst / ('gt' + Path(r['mask']).suffix))
        for k in methods:
            p = hold / k / ds / a.prompt / f'seed{a.seed0}'
            gp = p / 'gating.csv'
            for f in (p / 'masks').glob(f'{key}.*'):
                shutil.copy(f, dst / f'pred_{k}{f.suffix}')
            if k == 'admd' and gp.exists():
                g = pd.read_csv(gp); g[g.image == key].to_csv(dst / 'admd_gating.csv', index=False)
        rows.append(dict(centre=c, pick=name, image=key, **{k: float(d.loc[key, k]) for k in methods}))
pd.DataFrame(rows).to_csv(out / 'index.csv', index=False); print(pd.DataFrame(rows).to_string(index=False))
