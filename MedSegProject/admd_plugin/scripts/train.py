#!/usr/bin/env python3
"""
Train the ADMD plug-in decoder (or the plain SAM decoder baseline) on cached embeddings.

  python scripts/train.py --manifest_dir $WORK/manifests/polypgen_loco/holdout_C1/seed0 \
      --encoder mobilesam --ckpt $WORK/ckpt/mobile_sam.pt --cache_dir $WORK/cache \
      --decoder admd --seed 0 --out $WORK/runs/polypgen_loco/holdout_C1/mmseg_full/seed0

Baseline (identical pipeline, plain SAM decoder):   --decoder sam
ADMD component ablations:                           --admd_query off / --admd_align off / --domain_loss_w 0
K is read from the training manifest (number of distinct domain_id >= 0).
Outputs: {out}/best.pt (decoder weights + config), {out}/config.json, {out}/train_log.csv
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from admd.encoders import load_sam                                        # noqa: E402
from admd.decoder import build_decoder                                    # noqa: E402
from admd.data import CachedPromptDataset, collate, make_prompt_inputs    # noqa: E402
from admd.losses import seg_loss, dice_coef_lowres                        # noqa: E402


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(dec, sam, loader, device, modes=('1pt', 'bbox')):
    dec.eval()
    pe = sam.prompt_encoder.get_dense_pe()
    scores = {m: [] for m in modes}
    for batch in loader:
        emb = batch['emb'].to(device)
        for m in modes:
            pts, boxes = make_prompt_inputs(batch, m, device)
            sparse, dense = sam.prompt_encoder(points=pts, boxes=boxes, masks=None)
            out = dec(emb, pe, sparse, dense, domain_id=batch['domain_id'].to(device))
            d = dice_coef_lowres(out['mask'], batch['gt_low'].to(device), batch['valid'].to(device))
            scores[m] += d.tolist()
    dec.train()
    return {m: float(np.mean(v)) for m, v in scores.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest_dir', required=True, help='folder with train.csv and val.csv')
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--decoder', default='admd', choices=['admd', 'sam'])
    ap.add_argument('--admd_query', default='on', choices=['on', 'off'])
    ap.add_argument('--admd_align', default='on', choices=['on', 'off'])
    ap.add_argument('--fusion', default='soft', choices=['soft', 'hard', 'uniform'])
    ap.add_argument('--domain_loss_w', type=float, default=0.1)
    ap.add_argument('--branch_loss_w', type=float, default=1.0, help='seg loss on the branch of the true domain')
    ap.add_argument('--train_modes', nargs='+', default=['1pt', 'bbox', '1pt+bbox'])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--p_flip', type=float, default=0.5)
    ap.add_argument('--box_jitter', type=float, default=0.05)
    ap.add_argument('--train_prompt_encoder', action='store_true')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--wide', action='store_true', help='wide single decoder')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--repvit_root', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    set_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sam = load_sam(a.encoder, a.ckpt, a.device, a.repvit_root)
    md = Path(a.manifest_dir)
    tr = CachedPromptDataset(md / 'train.csv', a.cache_dir, a.encoder, train=True, p_flip=a.p_flip,
                             box_jitter=a.box_jitter, seed=a.seed)
    va = CachedPromptDataset(md / 'val.csv', a.cache_dir, a.encoder, train=False, seed=a.seed)
    K = tr.K
    dec = build_decoder(sam, a.decoder, K, a.admd_query == 'on', a.admd_align == 'on', a.fusion, a.wide).to(a.device)
    params = list(dec.parameters())
    if a.train_prompt_encoder:
        for p in sam.prompt_encoder.parameters():
            p.requires_grad_(True)
        params += list(sam.prompt_encoder.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    dev_type = 'cuda' if a.device.startswith('cuda') else 'cpu'
    scaler = torch.amp.GradScaler(dev_type, enabled=a.amp and dev_type == 'cuda')
    g = torch.Generator(); g.manual_seed(a.seed)
    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True, generator=g)
    vl = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate)
    cfg = dict(vars(a), K=K, n_train=len(tr), n_val=len(va), decoder_params=dec.n_params())
    json.dump(cfg, open(out / 'config.json', 'w'), indent=1)
    print(f'K={K}  decoder={a.decoder}  params={dec.n_params()/1e6:.2f}M  train={len(tr)}  val={len(va)}')

    pe = sam.prompt_encoder.get_dense_pe()
    best, bad, log = -1.0, 0, []
    rng = random.Random(a.seed)
    for ep in range(a.epochs):
        t0 = time.time(); tot = n = 0
        for batch in tl:
            mode = rng.choice(a.train_modes)
            emb = batch['emb'].to(a.device)
            gt, valid = batch['gt_low'].to(a.device), batch['valid'].to(a.device)
            dom = batch['domain_id'].to(a.device)
            has_fg = torch.tensor(batch['has_fg'], device=a.device)
            with torch.autocast(device_type=dev_type, enabled=scaler.is_enabled()):
                pts, boxes = make_prompt_inputs(batch, mode, a.device)
                sparse, dense = sam.prompt_encoder(points=pts, boxes=boxes, masks=None)
                o = dec(emb, pe, sparse, dense, domain_id=dom)
                w_fg = has_fg.float().view(-1, 1, 1, 1)          # empty-GT samples get no seg loss on fake prompts
                loss = seg_loss(o['mask'].float(), gt, valid * w_fg)
                if a.decoder == 'admd' and K > 1:
                    known = dom >= 0
                    if a.domain_loss_w > 0 and known.any():
                        loss = loss + a.domain_loss_w * F.cross_entropy(torch.log(o['d'][known] + 1e-8), dom[known])
                    if a.branch_loss_w > 0 and known.any():
                        idx = dom.clamp_min(0)
                        bm = o['branch_masks'].gather(1, idx.view(-1, 1, 1, 1).expand(-1, 1, 256, 256)).float()
                        loss = loss + a.branch_loss_w * seg_loss(bm, gt, valid * w_fg * known.float().view(-1, 1, 1, 1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item() * emb.shape[0]; n += emb.shape[0]
        sched.step()
        val = evaluate(dec, sam, vl, a.device)
        score = float(np.mean(list(val.values())))
        log.append(dict(epoch=ep, train_loss=tot / max(n, 1), **{f'val_{k}': v for k, v in val.items()}, val_mean=score, sec=time.time() - t0))
        print(f'ep {ep:3d}  loss {tot/max(n,1):.4f}  ' + '  '.join(f'val_{k} {v:.4f}' for k, v in val.items()) + f'  ({time.time()-t0:.0f}s)')
        if score > best:
            best, bad = score, 0
            torch.save(dict(decoder=dec.state_dict(), prompt_encoder=sam.prompt_encoder.state_dict() if a.train_prompt_encoder else None,
                            config=cfg, epoch=ep, val=val), out / 'best.pt')
        else:
            bad += 1
            if bad >= a.patience:
                print(f'early stop at epoch {ep} (best val_mean {best:.4f})')
                break
    with open(out / 'train_log.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys())); w.writeheader(); w.writerows(log)
    print(f'best val_mean {best:.4f} -> {out/"best.pt"}')


if __name__ == '__main__':
    main()
