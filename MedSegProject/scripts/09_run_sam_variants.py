#!/usr/bin/env python3
"""
SAM 系基线统一推理（SamPredictor 兼容接口）：SAM-B、MobileSAM、EdgeSAM、RepViT-SAM、MedSAM。
EfficientSAM、SAM-Med2D 的接口不同，用各自仓库的推理脚本，但**提示 JSON 与输出格式要与本脚本一致**。

提示模式（--prompt）：1pt | bbox | 1pt+bbox | randpt | bbox_jit5 | bbox_jit10 | bbox_jit20
输出：--out/masks/{stem}.png（0/255，原图分辨率）、--out/timing.csv

用法：
  python 09_run_sam_variants.py --model sam_b --ckpt $WORK/ckpt/sam_vit_b_01ec64.pth \
      --manifest $WORK/manifests/thesis/seed0/test_CVC-ColonDB.csv \
      --prompts  $WORK/prompts/thesis/test_CVC-ColonDB.json --prompt 1pt+bbox \
      --out $WORK/results/thesis/sam_b/CVC-ColonDB/1pt+bbox/seed0
然后：python 04_metrics.py --manifest ... --pred_dir $WORK/results/.../seed0/masks --out $WORK/results/.../seed0/metrics.csv
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def build(model, ckpt):
    if model == 'sam_b':
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry['vit_b'](checkpoint=ckpt)
    elif model == 'medsam':   # MedSAM 是 ViT-B 结构的微调权重；官方推理只支持框提示，点提示结果仅供参考
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry['vit_b'](checkpoint=ckpt)
    elif model == 'mobilesam':
        from mobile_sam import sam_model_registry, SamPredictor
        sam = sam_model_registry['vit_t'](checkpoint=ckpt)
    elif model == 'edgesam':
        from edge_sam import sam_model_registry, SamPredictor
        sam = sam_model_registry['edge_sam'](checkpoint=ckpt)
    elif model == 'repvit_sam':
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'third_party' / 'RepViT' / 'sam'))
        from repvit_sam import sam_model_registry, SamPredictor
        sam = sam_model_registry['repvit'](checkpoint=ckpt)
    else:
        raise ValueError(model)
    sam.to('cuda').eval()
    return SamPredictor(sam)


def make_inputs(rec, mode):
    pts, lbls, box = None, None, None
    if mode in ('1pt', '1pt+bbox'):
        pts, lbls = np.array([rec['point_centroid']]), np.array([1])
    if mode == 'randpt':
        pts, lbls = np.array([rec['point_random']]), np.array([1])
    if mode in ('bbox', '1pt+bbox'):
        box = np.array(rec['bbox'])
    if mode.startswith('bbox_jit'):
        box = np.array(rec[mode])
    return pts, lbls, box


def uid(r):
    return f"{r.get('dataset', '')}__{Path(r['image']).stem}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['sam_b', 'medsam', 'mobilesam', 'edgesam', 'repvit_sam'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--multimask', action='store_true', help='单点提示时启用多掩码并按预测 IoU 选最高（SAM 默认行为）')
    a = ap.parse_args()

    predictor = build(a.model, a.ckpt)
    prompts = json.load(open(a.prompts))
    out = Path(a.out)
    (out / 'masks').mkdir(parents=True, exist_ok=True)
    timing = []
    with open(a.manifest) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        stem = uid(r)
        rec = prompts[stem]
        if rec.get('bbox') is None:   # 空真值：输出全黑
            Image.fromarray(np.zeros((rec['image_h'], rec['image_w']), np.uint8)).save(out / 'masks' / f'{stem}.png')
            continue
        img = np.array(Image.open(r['image']).convert('RGB'))
        pts, lbls, box = make_inputs(rec, a.prompt)
        t0 = time.time()
        with torch.no_grad():
            predictor.set_image(img)
            multimask = a.multimask and (box is None)
            masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbls, box=box, multimask_output=multimask)
        torch.cuda.synchronize()
        timing.append([stem, time.time() - t0])
        m = masks[int(np.argmax(scores))] if multimask else masks[0]
        Image.fromarray((m.astype(np.uint8) * 255)).save(out / 'masks' / f'{stem}.png')
    with open(out / 'timing.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'seconds'])
        w.writerows(timing)
    print(f'{a.model} {a.prompt}: {len(rows)} images, mean {np.mean([t[1] for t in timing])*1000:.1f} ms/img -> {out}')


if __name__ == '__main__':
    main()
