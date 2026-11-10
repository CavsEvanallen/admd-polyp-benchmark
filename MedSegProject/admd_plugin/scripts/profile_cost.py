#!/usr/bin/env python3
"""
Parameters / FLOPs / latency of the decoder as a function of K, plus encoder latency (Table 8).
  python scripts/profile_cost.py --encoder mobilesam --ckpt $WORK/ckpt/mobile_sam.pt --K 1 2 3 5 8 --out results/profile.csv
FLOPs use fvcore (pip install fvcore); the decoder has no custom CUDA ops so the count is complete.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from admd.encoders import load_sam          # noqa: E402
from admd.decoder import build_decoder      # noqa: E402


@torch.no_grad()
def latency(fn, iters=50, warm=10, device='cuda'):
    for _ in range(warm):
        fn()
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    return (time.time() - t) / iters * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--K', nargs='+', type=int, default=[1, 2, 3, 5, 8])
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    sam = load_sam(a.encoder, a.ckpt, a.device)
    x = torch.zeros(1, 3, 1024, 1024, device=a.device)
    enc_ms = latency(lambda: sam.image_encoder(x), device=a.device)
    enc_params = sum(p.numel() for p in sam.image_encoder.parameters()) / 1e6
    try:
        from fvcore.nn import FlopCountAnalysis
        enc_gflops = FlopCountAnalysis(sam.image_encoder, x).unsupported_ops_warnings(False).total() / 1e9
    except Exception as e:  # noqa: BLE001
        print('fvcore unavailable:', e); FlopCountAnalysis = None; enc_gflops = float('nan')
    rows = [dict(component=f'encoder:{a.encoder}', K='', params_M=round(enc_params, 2), gflops=round(enc_gflops, 2), ms=round(enc_ms, 2))]
    emb = torch.zeros(1, 256, 64, 64, device=a.device)
    pe = sam.prompt_encoder.get_dense_pe()
    sparse, dense = sam.prompt_encoder(points=None, boxes=torch.tensor([[100., 100., 500., 500.]], device=a.device), masks=None)
    for K in a.K:
        dec = build_decoder(sam, 'admd' if K > 1 else 'sam', K).to(a.device).eval()
        ms = latency(lambda: dec(emb, pe, sparse, dense), device=a.device)
        gfl = float('nan')
        if FlopCountAnalysis is not None:
            class W(torch.nn.Module):
                def __init__(s, d): super().__init__(); s.d = d
                def forward(s, e): return s.d(e, pe, sparse, dense)['mask']
            gfl = FlopCountAnalysis(W(dec), emb).unsupported_ops_warnings(False).total() / 1e9
        rows.append(dict(component='decoder', K=K, params_M=round(dec.n_params() / 1e6, 2), gflops=round(gfl, 2), ms=round(ms, 2)))
        print(rows[-1])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'-> {a.out}')


if __name__ == '__main__':
    main()
