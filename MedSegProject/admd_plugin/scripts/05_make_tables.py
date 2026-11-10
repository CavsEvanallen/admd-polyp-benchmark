import glob, pandas as pd, numpy as np
from pathlib import Path
R = Path('/tmp/h4/ADMD提交4')
rng = np.random.default_rng(0)
M = ['sam_ft', 'admd', 'uniform_ens', 'wide_sam']


def load(root):
    rows = []
    for f in glob.glob(str(R / root / 'loso/holdout_*/*/*/*/seed*/metrics.csv')):
        p = Path(f).relative_to(R / root / 'loso').parts
        d = pd.read_csv(f)
        d['hold'], d['method'], d['ds'], d['prompt'] = p[0], p[1], p[2], p[3]
        rows.append(d)
    d = pd.concat(rows)
    return d[d.gt_area > 0]


def ci(x):
    idx = rng.integers(0, len(x), (3000, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


out = []
for root, enc in [('results', 'MobileSAM'), ('results_sam_b', 'SAM ViT-B')]:
    d = load(root)
    out.append(f'\n### {enc}\n')
    for pr in ['1pt', 'bbox']:
        s = d[d.prompt == pr]
        cen = s[s.ds.str.fullmatch(r'Source_C\d')]
        groups = {}
        # centre-average: mean over the 5 centre means (each centre mean over images)
        piv = cen.pivot_table(index=['ds', 'image'], columns='method', values='dice', aggfunc='mean')
        cm = piv.groupby(level=0).mean()
        groups['PolypGen C1–C5 (mean of centres)'] = cm.mean()
        # per-centre-mean difference signs
        for g in ['Source_CVC-ColonDB', 'Source_ETIS-LaribPolypDB', 'Source_CVC-300']:
            pv = s[s.ds == g].pivot_table(index='image', columns='method', values='dice', aggfunc='mean')
            groups[g.replace('Source_', '')] = pv.mean()
        out.append(f'\n**{ "Point" if pr=="1pt" else "Box"} prompt, Dice**\n')
        out.append('| Test set | Standard | ADMD | Uniform ens. | Pixel adapter | ADMD−Std | Adapter−Std |')
        out.append('|---|---|---|---|---|---|---|')
        for g, v in groups.items():
            if g.startswith('PolypGen'):
                dA = (cm['admd'] - cm['sam_ft']); dW = (cm['wide_sam'] - cm['sam_ft'])
                a = f'{dA.mean():+.4f} ({(dA>0).sum()}/5 better)'
                w = f'{dW.mean():+.4f} ({(dW>0).sum()}/5 better)'
            else:
                pv = s[s.ds == 'Source_' + g].pivot_table(index='image', columns='method', values='dice', aggfunc='mean')
                a = ''; w = ''
                for nm, col in [('a', 'admd'), ('w', 'wide_sam')]:
                    x = (pv[col] - pv['sam_ft']).values; lo, hi = ci(x)
                    t = f'{x.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]'
                    if nm == 'a': a = t
                    else: w = t
            out.append(f'| {g} | {v["sam_ft"]:.4f} | {v["admd"]:.4f} | {v["uniform_ens"]:.4f} | {v["wide_sam"]:.4f} | {a} | {w} |')
print('\n'.join(out))
