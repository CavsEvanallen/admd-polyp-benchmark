#!/usr/bin/env python3
"""
Run a frozen SAM-family image encoder once over every image in the given manifests and cache the
256x64x64 embedding (fp16). Optionally also cache the horizontally flipped image (--flip) for augmentation.

  python scripts/cache_embeddings.py --encoder mobilesam --ckpt $WORK/ckpt/mobile_sam.pt \
      --manifests $WORK/manifests/**/*.csv --cache_dir $WORK/cache --flip --batch 8
Cost: MobileSAM ~15 ms/img, SAM-B ~150 ms/img on a 4090. Storage: ~2 MB per image per view.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from admd.encoders import load_sam, embed_images          # noqa: E402
from admd.data import read_manifest, uid                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--manifests', nargs='+', required=True, help='CSV files or globs')
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--flip', action='store_true')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--repvit_root', default=None)
    a = ap.parse_args()

    from segment_anything.utils.transforms import ResizeLongestSide
    sam = load_sam(a.encoder, a.ckpt, a.device, a.repvit_root)
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    out = Path(a.cache_dir) / a.encoder
    out.mkdir(parents=True, exist_ok=True)

    files = []
    for pat in a.manifests:
        files += glob.glob(pat, recursive=True)
    todo = {}
    for f in files:
        for r in read_manifest(f):
            stem = uid(r)
            if not (out / f'{stem}.pt').exists():
                todo[stem] = r['image']
    print(f'{len(todo)} images to embed with {a.encoder} (already cached are skipped)')
    items = list(todo.items())
    for i in tqdm(range(0, len(items), a.batch)):
        chunk = items[i:i + a.batch]
        ims = [np.array(Image.open(p).convert('RGB')) for _, p in chunk]
        emb, res, orig = embed_images(sam, ims, transform, a.device)
        if a.flip:
            emb_f, _, _ = embed_images(sam, [im[:, ::-1].copy() for im in ims], transform, a.device)
        for j, (stem, _) in enumerate(chunk):
            rec = dict(emb=emb[j].cpu(), resized_hw=res[j], orig_hw=orig[j], encoder=a.encoder)
            if a.flip:
                rec['emb_flip'] = emb_f[j].cpu()
            torch.save(rec, out / f'{stem}.pt')
    print(f'done -> {out}')


if __name__ == '__main__':
    main()
