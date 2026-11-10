#!/usr/bin/env python3
"""
从真值掩码生成提示，写成 JSON（坐标为原始掩码像素坐标，x 向右、y 向下）。

每张图输出：
  point_centroid : 最大连通域的质心；若质心不在前景内，改用距离变换最大点（保证点在病灶内）
  point_random   : 最大连通域内均匀随机一点（--seed 控制，可复现）
  bbox           : 全部前景的最小外接框 [x0, y0, x1, y1]（含端点）
  bbox_jit{5,10,20}: 每条边独立按 ±p% 框宽/高随机扰动后的框（--seed 控制）
  n_components, fg_area, image_h, image_w

用法：
  python 03_prompts.py --manifest $WORK/manifests/thesis/seed0/test_CVC-ColonDB.csv \
                       --out $WORK/prompts/thesis/test_CVC-ColonDB.json --seed 0
"""
import argparse, csv, json, random
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def read_mask(path):
    m = np.array(Image.open(path).convert('L'))
    return m > 127


def largest_component(mask):
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask, n
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1), n


def centroid_inside(comp):
    cy, cx = ndi.center_of_mass(comp)
    cy, cx = int(round(cy)), int(round(cx))
    if 0 <= cy < comp.shape[0] and 0 <= cx < comp.shape[1] and comp[cy, cx]:
        return [cx, cy]
    dt = ndi.distance_transform_edt(comp)
    cy, cx = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return [int(cx), int(cy)]


def random_inside(comp, rng):
    ys, xs = np.nonzero(comp)
    i = rng.randrange(len(xs))
    return [int(xs[i]), int(ys[i])]


def tight_bbox(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def jitter_bbox(b, p, rng, h, w):
    x0, y0, x1, y1 = b
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    j = lambda s: rng.uniform(-p, p) * s
    nx0 = int(round(min(max(x0 + j(bw), 0), w - 2)))
    ny0 = int(round(min(max(y0 + j(bh), 0), h - 2)))
    nx1 = int(round(min(max(x1 + j(bw), nx0 + 1), w - 1)))
    ny1 = int(round(min(max(y1 + j(bh), ny0 + 1), h - 1)))
    return [nx0, ny0, nx1, ny1]


def uid(r):
    return f"{r.get('dataset', '')}__{Path(r['image']).stem}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = {}
    empty = 0
    with open(a.manifest) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        key = uid(r)
        mask = read_mask(r['mask'])
        h, w = mask.shape
        rec = {'image': r['image'], 'mask': r['mask'], 'image_h': h, 'image_w': w,
               'fg_area': int(mask.sum())}
        if mask.sum() == 0:
            empty += 1
            rec.update(point_centroid=None, point_random=None, bbox=None, n_components=0)
            out[key] = rec
            continue
        comp, n = largest_component(mask)
        b = tight_bbox(mask)
        rec.update(n_components=int(n),
                   point_centroid=centroid_inside(comp),
                   point_random=random_inside(comp, rng),
                   bbox=b)
        for p in (5, 10, 20):
            rec[f'bbox_jit{p}'] = jitter_bbox(b, p / 100.0, rng, h, w)
        out[key] = rec
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {a.out}: {len(out)} images, {empty} with empty mask')


if __name__ == '__main__':
    main()
