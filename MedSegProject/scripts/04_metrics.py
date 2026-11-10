#!/usr/bin/env python3
"""
逐图评测：Dice, IoU, Precision, Recall, HD95(px), ASSD(px)，在**真值原始分辨率**下计算。

输入：
  --manifest  清单 CSV（取 image/mask 列）
  --pred_dir  预测掩码目录，文件名 = 图像 stem + .png（0/255 二值或 0..255 概率图，>127 视为前景）
  --out       输出 metrics.csv（每行一张图）
边界情况：
  真值空且预测空 -> dice=iou=1, hd95=assd=0
  一方为空       -> dice=iou=0, hd95=assd=图像对角线长度，并在 flag 列标 'empty_pred' / 'empty_gt'
  缺预测文件     -> 整行 NaN，flag='missing_pred'（汇总时会报数量，不能静默忽略）

用法：
  python 04_metrics.py --manifest .../test_CVC-ColonDB.csv --pred_dir results/.../masks --out results/.../metrics.csv
"""
import argparse, csv, math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def load_bin(path, size=None):
    im = Image.open(path).convert('L')
    if size is not None and im.size != size:
        im = im.resize(size, Image.NEAREST)
    return np.array(im) > 127


def surface(mask):
    er = ndi.binary_erosion(mask, structure=np.ones((3, 3), bool), border_value=0)
    return mask & ~er


def surface_distances(a, b):
    """a 表面到 b 表面的距离数组（像素）。"""
    sa, sb = surface(a), surface(b)
    if sa.sum() == 0 or sb.sum() == 0:
        return None
    dt_b = ndi.distance_transform_edt(~sb)
    return dt_b[sa]


def metrics_one(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    diag = math.hypot(*gt.shape)
    flag = ''
    if gt.sum() == 0 and pred.sum() == 0:
        return dict(dice=1.0, iou=1.0, precision=1.0, recall=1.0, hd95=0.0, assd=0.0, flag='both_empty')
    if pred.sum() == 0:
        flag = 'empty_pred'
    elif gt.sum() == 0:
        flag = 'empty_gt'
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if flag:
        return dict(dice=dice, iou=iou, precision=prec, recall=rec, hd95=diag, assd=diag, flag=flag)
    d_pg = surface_distances(pred, gt)
    d_gp = surface_distances(gt, pred)
    hd95 = max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))
    assd = (d_pg.sum() + d_gp.sum()) / (len(d_pg) + len(d_gp))
    return dict(dice=dice, iou=iou, precision=prec, recall=rec, hd95=float(hd95), assd=float(assd), flag=flag)


def uid(r):
    return f"{r.get('dataset', '')}__{Path(r['image']).stem}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--pred_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pred_suffix', default='.png')
    a = ap.parse_args()
    pred_dir = Path(a.pred_dir)
    rows_out, missing = [], 0
    with open(a.manifest) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        stem = uid(r)
        gt = load_bin(r['mask'])
        p = pred_dir / f'{stem}{a.pred_suffix}'
        rec = dict(image=stem, dataset=r.get('dataset', ''), gt_area=int(gt.sum()), h=gt.shape[0], w=gt.shape[1])
        if not p.exists():
            missing += 1
            rec.update(dice=float('nan'), iou=float('nan'), precision=float('nan'), recall=float('nan'),
                       hd95=float('nan'), assd=float('nan'), pred_area=float('nan'), flag='missing_pred')
        else:
            pred = load_bin(p, size=(gt.shape[1], gt.shape[0]))
            rec.update(metrics_one(pred, gt))
            rec['pred_area'] = int(pred.sum())
        rows_out.append(rec)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cols = ['image', 'dataset', 'dice', 'iou', 'precision', 'recall', 'hd95', 'assd', 'gt_area', 'pred_area', 'h', 'w', 'flag']
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_out)
    ok = [r for r in rows_out if r['flag'] != 'missing_pred']
    mean = lambda k: np.nanmean([r[k] for r in ok]) if ok else float('nan')
    print(f'{a.out}: n={len(rows_out)} missing={missing} '
          f'Dice={mean("dice")*100:.2f} IoU={mean("iou")*100:.2f} HD95={mean("hd95"):.2f}px ASSD={mean("assd"):.2f}px')
    if missing:
        print(f'  [warn] {missing} predictions missing. Fix before reporting.')


if __name__ == '__main__':
    main()
