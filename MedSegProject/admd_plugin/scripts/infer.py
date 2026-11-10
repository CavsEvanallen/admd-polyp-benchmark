#!/usr/bin/env python3
"""
Inference with a trained decoder on cached embeddings. Writes, in the layout the analysis scripts expect:
  {out}/masks/{stem}.png        binary 0/255 at the ORIGINAL image resolution
  {out}/gating.csv              image, domain_true, d_0..d_{K-1}, alpha, w_0..w_{K-1}
  {out}/timing.csv              decoder-side milliseconds per image (encoder time is reported by profile_cost.py)

  python scripts/infer.py --run $WORK/runs/.../seed0 --manifest .../test_CVC-ColonDB.csv \
      --prompts $WORK/prompts/.../test_CVC-ColonDB.json --prompt bbox --cache_dir $WORK/cache \
      --ckpt $WORK/ckpt/mobile_sam.pt --out $WORK/results/.../CVC-ColonDB/bbox/seed0 [--fusion hard]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from admd.encoders import load_sam                                        # noqa: E402
from admd.decoder import build_decoder                                    # noqa: E402
from admd.data import CachedPromptDataset, collate, make_prompt_inputs, IMG_SIZE   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='training output folder containing best.pt')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fusion', default=None, choices=[None, 'soft', 'hard', 'oracle', 'uniform', 'random'])
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--repvit_root', default=None)
    a = ap.parse_args()
    ck = torch.load(Path(a.run) / 'best.pt', map_location='cpu')

    cfg = ck['config']
    sam = load_sam(cfg['encoder'], a.ckpt, a.device, a.repvit_root)
    if ck.get('prompt_encoder'):
        sam.prompt_encoder.load_state_dict(ck['prompt_encoder'])
    _has_wide = any('wide_mlp' in k for k in ck['decoder'].keys())
    dec = build_decoder(sam, cfg['decoder'], cfg['K'], cfg['admd_query'] == 'on', cfg['admd_align'] == 'on', cfg['fusion'], wide=_has_wide).to(a.device)
    dec.load_state_dict(ck['decoder']); dec.eval()
    ds = CachedPromptDataset(a.manifest, a.cache_dir, cfg['encoder'], prompts_json=a.prompts, train=False)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=4, collate_fn=collate)
    out = Path(a.out); (out / 'masks').mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device=a.device); gen.manual_seed(a.seed)
    pe = sam.prompt_encoder.get_dense_pe()
    K = cfg['K']
    gating, timing = [], []
    with torch.no_grad():
        for batch in dl:
            emb = batch['emb'].to(a.device)
            t0 = time.time()
            pts, boxes = make_prompt_inputs(batch, a.prompt, a.device)
            sparse, dense = sam.prompt_encoder(points=pts, boxes=boxes, masks=None)
            o = dec(emb, pe, sparse, dense, domain_id=batch['domain_id'].to(a.device), fusion=a.fusion, generator=gen)
            up = F.interpolate(o['mask'].float(), size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
            if a.device.startswith('cuda'):
                torch.cuda.synchronize()
            dt = (time.time() - t0) * 1000 / emb.shape[0]
            for i, stem in enumerate(batch['stem']):
                h, w = batch['res_hw'][i].tolist(); H, W = batch['orig_hw'][i].tolist()
                m = F.interpolate(up[i:i + 1, :, :h, :w], size=(H, W), mode='bilinear', align_corners=False)[0, 0] > 0
                if not batch['has_fg'][i]:
                    m = torch.zeros_like(m)          # empty GT -> no prompt available -> empty prediction (flagged by metrics)
                Image.fromarray(m.cpu().numpy().astype(np.uint8) * 255).save(out / 'masks' / f'{stem}.png')
                gating.append([stem, int(batch['domain_id'][i])] + o['d'][i].tolist() + [float(o['alpha'][i])] + o['w'][i].tolist())
                timing.append([stem, dt])
    with open(out / 'gating.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['image', 'domain_true'] + [f'd_{k}' for k in range(K)] + ['alpha'] + [f'w_{k}' for k in range(K)])
        wr.writerows(gating)
    with open(out / 'timing.csv', 'w', newline='') as f:
        wr = csv.writer(f); wr.writerow(['image', 'decoder_ms']); wr.writerows(timing)
    json.dump(dict(run=a.run, prompt=a.prompt, fusion=a.fusion or cfg['fusion'], n=len(ds)), open(out / 'infer_config.json', 'w'), indent=1)
    print(f'{len(ds)} images -> {out}  (decoder {np.mean([t[1] for t in timing]):.1f} ms/img)')


if __name__ == '__main__':
    main()
