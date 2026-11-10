#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('--report', required=True); ap.add_argument('--work', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=6); a = ap.parse_args()
rows = [r for r in csv.DictReader(open(a.report))]
rows.sort(key=lambda r: (r['kind'] != 'md5', int(r['hamming'])))
rows = rows[:a.n]
if not rows: print('no pairs at this threshold'); raise SystemExit
H = 256; canvas = Image.new('RGB', (2 * H * 4 // 3 + 10, (H + 6) * len(rows)), 'white')
for i, r in enumerate(rows):
    for j, k in enumerate(['test_file', 'train_file']):
        im = Image.open(Path(a.work) / r[k]).convert('RGB'); im = im.resize((H * 4 // 3, H))
        canvas.paste(im, (j * (H * 4 // 3 + 10), i * (H + 6)))
canvas.save(a.out); print(a.out, 'left = test image, right = matching training image')
