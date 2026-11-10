#!/usr/bin/env python3
"""ColonDB 与 CVC-300 的 55 张"指标逐位相同"的图,到底是不是同一批像素?
用法: python3 A4b_ColonDB与CVC300逐像素核对.py --colondb data/CVC-ColonDB/images --cvc300 data/CVC-300/images --out out_check/colondb_cvc300
输出: pairs.csv(每张 ColonDB 图的最近 CVC-300 图及平均像素差),montage_*.png(前 6 对并排图)。
判读: 平均灰度差 < 8(0–255,真实内镜图;随机噪声图缩放后会到 4–5)视为同一张图(缩放/压缩差异);> 20 视为不同图。"""
import argparse, glob, os
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--colondb', required=True); ap.add_argument('--cvc300', required=True)
ap.add_argument('--colondb_masks'); ap.add_argument('--cvc300_masks')   # 可选:同时核对真值掩码
ap.add_argument('--out', required=True); ap.add_argument('--size', type=int, default=64)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def small(p):
    return np.asarray(Image.open(p).convert('L').resize((a.size, a.size), Image.BILINEAR), dtype=np.float32)


A = sorted(glob.glob(os.path.join(a.colondb, '*.*')))
B = sorted(glob.glob(os.path.join(a.cvc300, '*.*')))
fa = np.stack([small(p) for p in A]).reshape(len(A), -1)
fb = np.stack([small(p) for p in B]).reshape(len(B), -1)
rows = []
for i, p in enumerate(A):
    d = np.abs(fb - fa[i]).mean(1)
    j = int(d.argmin())
    rows.append((os.path.basename(p), os.path.basename(B[j]), float(d[j])))
rows.sort(key=lambda r: r[2])
with open(os.path.join(a.out, 'pairs.csv'), 'w') as f:
    f.write('colondb,cvc300,mean_abs_diff\n')
    for r in rows:
        f.write(f'{r[0]},{r[1]},{r[2]:.2f}\n')
same = [r for r in rows if r[2] < 8]
print(f'ColonDB {len(A)} 张, CVC-300 {len(B)} 张; 平均像素差<8 的有 {len(same)} 张(论文里是 55); 差>20 的 {sum(r[2] > 20 for r in rows)} 张')
for k, r in enumerate(rows[:6]):
    x = Image.open(os.path.join(a.colondb, r[0])).convert('RGB').resize((384, 300))
    y = Image.open(os.path.join(a.cvc300, r[1])).convert('RGB').resize((384, 300))
    m = Image.new('RGB', (768, 300)); m.paste(x, (0, 0)); m.paste(y, (384, 0))
    m.save(os.path.join(a.out, f'montage_{k+1}_{r[0]}_{r[1]}.png'))

if a.colondb_masks and a.cvc300_masks:
    def mask(d, n):
        return np.asarray(Image.open(os.path.join(d, n)).convert('L').resize((a.size * 4, a.size * 4), Image.NEAREST)) > 127
    with open(os.path.join(a.out, 'mask_check.csv'), 'w') as f:
        f.write('colondb,cvc300,mask_iou\n')
        for r in same:
            try:
                x, y = mask(a.colondb_masks, r[0]), mask(a.cvc300_masks, r[1])
                iou = (x & y).sum() / max((x | y).sum(), 1)
                f.write(f'{r[0]},{r[1]},{iou:.3f}\n')
            except FileNotFoundError:
                f.write(f'{r[0]},{r[1]},missing\n')
    print('掩码 IoU 写入 mask_check.csv;IoU<0.99 的就是"图相同但真值不同"的那几对')
