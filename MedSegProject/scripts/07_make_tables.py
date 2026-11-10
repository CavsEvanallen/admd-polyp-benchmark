#!/usr/bin/env python3
"""
把结果目录汇总成论文表格。

结果目录规范（必须遵守，否则脚本认不出来）：
  results/{protocol}/{method}/{target}/{prompt}/seed{n}/metrics.csv
  例： results/thesis/mmseg_full/CVC-ColonDB/1pt+bbox/seed0/metrics.csv
       results/polypgen_loco/holdout_C3/mmseg_noadmd/PolypGen_C3/bbox/seed0/metrics.csv  (protocol 带折号也行)

每个 (protocol, target) 输出一张表：行 = method，列 = prompt × metric，单元格 = 跨种子 mean±std。
用法：
  python 07_make_tables.py --results $WORK/results --protocol thesis --metrics dice iou hd95 assd --out tables/
"""
import argparse, glob, re
from pathlib import Path
import numpy as np
import pandas as pd

LOWER = {'hd95', 'assd'}
PROMPT_ORDER = ['1pt', 'bbox', '1pt+bbox', 'randpt', 'bbox_jit5', 'bbox_jit10', 'bbox_jit20', 'noprompt']


def collect(results, protocol):
    recs = []
    for f in glob.glob(str(Path(results) / protocol / '**' / 'metrics.csv'), recursive=True):
        rel = Path(f).relative_to(Path(results) / protocol).parts
        # 允许 protocol 下再有一层折目录（holdout_Cx）
        if len(rel) == 5:
            method, target, prompt, seed, _ = rel
            fold = ''
        elif len(rel) == 6:
            fold, method, target, prompt, seed, _ = rel
        else:
            print(f'  [skip] 路径层级不符: {f}')
            continue
        df = pd.read_csv(f)
        miss = (df['flag'] == 'missing_pred').sum() if 'flag' in df else 0
        if miss:
            print(f'  [warn] {f}: {miss} missing predictions')
        row = dict(fold=fold, method=method, target=target, prompt=prompt, seed=seed, n=len(df))
        for m in df.select_dtypes('number').columns:
            row[m] = df[m].mean()
        recs.append(row)
    return pd.DataFrame(recs)


def fmt(vals, metric):
    vals = np.array(vals, dtype=float)
    scale = 100 if metric in ('dice', 'iou', 'precision', 'recall') else 1
    m, s = np.nanmean(vals) * scale, (np.nanstd(vals, ddof=0) * scale if len(vals) > 1 else None)
    return f'{m:.2f}' if s is None else f'{m:.2f}±{s:.2f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--metrics', nargs='+', default=['dice', 'iou', 'hd95', 'assd'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    df = collect(a.results, a.protocol)
    if df.empty:
        raise SystemExit('没有找到 metrics.csv，检查目录规范')
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for (fold, target), g in df.groupby(['fold', 'target']):
        prompts = [p for p in PROMPT_ORDER if p in set(g['prompt'])] + sorted(set(g['prompt']) - set(PROMPT_ORDER))
        methods = sorted(set(g['method']))
        header = ['Method'] + [f'{p} {m}' for p in prompts for m in a.metrics]
        md = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
        tex = ['\\begin{tabular}{l' + 'c' * (len(header) - 1) + '}', '\\toprule',
               ' & '.join(header).replace('_', '\\_').replace('±', '$\\pm$') + ' \\\\', '\\midrule']
        best = {}
        for p in prompts:
            for m in a.metrics:
                sub = g[g['prompt'] == p].groupby('method')[m].mean()
                if len(sub):
                    best[(p, m)] = sub.idxmin() if m in LOWER else sub.idxmax()
        for meth in methods:
            cells = [meth]
            for p in prompts:
                for m in a.metrics:
                    vals = g[(g['method'] == meth) & (g['prompt'] == p)][m].tolist()
                    cell = fmt(vals, m) if vals else '—'
                    if best.get((p, m)) == meth and vals:
                        cell = f'**{cell}**'
                    cells.append(cell)
            md.append('| ' + ' | '.join(cells) + ' |')
            tex_cells = []
            for c in cells:
                if c.startswith('**') and c.endswith('**'):
                    c = '\\textbf{' + c[2:-2] + '}'
                tex_cells.append(c.replace('±', '$\\pm$').replace('_', '\\_'))
            tex.append(' & '.join(tex_cells) + ' \\\\')
        tex += ['\\bottomrule', '\\end{tabular}']
        name = f'{a.protocol}{"_" + fold if fold else ""}_{target}'
        seeds = g.groupby(['method', 'prompt'])['seed'].nunique().min()
        note = f'\n\nmean±std over seeds (min seeds per cell = {seeds}); n_images = {int(g["n"].max())}. Bold = best per column.\n'
        (Path(a.out) / f'{name}.md').write_text(f'### {name}\n\n' + '\n'.join(md) + note)
        (Path(a.out) / f'{name}.tex').write_text('\n'.join(tex) + '\n')
        print(f'wrote {name}.md / .tex  ({len(methods)} methods × {len(prompts)} prompts)')


if __name__ == '__main__':
    main()
