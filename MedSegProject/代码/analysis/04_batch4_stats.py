import glob, pandas as pd, numpy as np
from pathlib import Path
R = Path('/tmp/h4/ADMD提交4')
rng = np.random.default_rng(0)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 40)


def load(root):
    rows = []
    for f in glob.glob(str(R / root / 'loso/holdout_*/*/*/*/seed*/metrics.csv')):
        p = Path(f).relative_to(R / root / 'loso').parts
        d = pd.read_csv(f)
        d['hold'], d['method'], d['ds'], d['prompt'], d['seed'] = p[0], p[1], p[2], p[3], p[4]
        rows.append(d)
    d = pd.concat(rows)
    return d[d.gt_area > 0]


dd = pd.read_csv(R / 'out_check/dedup_report.txt')
dup = set((Path(x).parent.parent.name, Path(x).stem) for x in dd.test_file)


def boot(x, n=3000):
    m = len(x)
    idx = rng.integers(0, m, (n, m))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


def run(root, pairs, excl, metric='dice'):
    d = load(root)
    out = []
    for ds in sorted(d.ds.unique()):
        for pr in ['1pt', 'bbox']:
            s = d[(d.ds == ds) & (d.prompt == pr)]
            piv = s.pivot_table(index='image', columns='method', values=metric, aggfunc='mean')
            if excl:
                dsn = ds.replace('Source_', '')
                piv = piv[[(dsn, i.split('__')[-1]) not in dup for i in piv.index]]
            row = {'ds': ds, 'prompt': pr, 'n': len(piv)}
            for m in piv.columns:
                row[m] = round(piv[m].mean(), 4)
            for a, b in pairs:
                if a in piv and b in piv:
                    x = (piv[a] - piv[b]).dropna().values
                    lo, hi = boot(x)
                    row[f'{a}-{b}'] = f'{x.mean():+.4f} [{lo:+.4f},{hi:+.4f}]'
            out.append(row)
    return pd.DataFrame(out)


pairs = [('admd', 'sam_ft'), ('wide_sam', 'sam_ft'), ('admd', 'uniform_ens'), ('admd', 'wide_sam')]
for root in ['results', 'results_sam_b']:
    for ex in [False, True]:
        print('=====', root, 'dedup-excluded' if ex else 'all')
        print(run(root, pairs, ex).to_string(index=False))
print('===== pat30 (held-out centres)')
d = load('results_pat30')
base = load('results')
print(d.groupby(['method', 'prompt']).agg(n=('image', 'size'), dice=('dice', 'mean'), hd95=('hd95', 'mean')))
b = base[base.ds.str.match(r'^Source_C\d$')]
print(b.groupby(['method', 'prompt']).agg(n=('image', 'size'), dice=('dice', 'mean'), hd95=('hd95', 'mean')))
print('dup stems:', len(dup), {k: sum(1 for x in dup if x[0] == k) for k in set(x[0] for x in dup)})
