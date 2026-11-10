"""
Dataset over cached image embeddings + manifest CSV (+ optional prompt JSON for evaluation).

Cache file per image: {cache_dir}/{encoder}/{stem}.pt  ->  dict(emb fp16 [256,64,64], emb_flip (optional),
                                                            resized_hw (h',w'), orig_hw (H,W))
Coordinates handed to the model are in the 1024 "resized+padded" frame used by SAM.

Training prompts are generated on the fly from the ground-truth mask (centroid / random point / tight box,
with optional box jitter); evaluation prompts come from the JSON written by scripts/03_prompts.py so that
every method sees exactly the same prompts.
"""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from torch.utils.data import Dataset

IMG_SIZE = 1024
LOW = 256


def uid(r):
    """全局唯一键 = 数据集名__文件名。修复:不同数据集同名图(如 ColonDB 的 1.png 与 ETIS 的 1.png)互相覆盖缓存/提示/预测。"""
    return f"{r.get('dataset', '')}__{Path(r['image']).stem}"


def read_manifest(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def resized_shape(H, W, long=IMG_SIZE):
    s = long / max(H, W)
    return int(H * s + 0.5), int(W * s + 0.5)


def scale_coords(xy, orig_hw, res_hw):
    """original-pixel (x, y) -> resized frame."""
    H, W = orig_hw
    h, w = res_hw
    return [xy[0] * (w / W), xy[1] * (h / H)]


def largest_component(mask):
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def centroid_inside(comp):
    cy, cx = ndi.center_of_mass(comp)
    cy, cx = int(round(cy)), int(round(cx))
    if 0 <= cy < comp.shape[0] and 0 <= cx < comp.shape[1] and comp[cy, cx]:
        return [float(cx), float(cy)]
    dt = ndi.distance_transform_edt(comp)
    cy, cx = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return [float(cx), float(cy)]


class CachedPromptDataset(Dataset):
    def __init__(self, manifest, cache_dir, encoder, prompts_json=None, train=False,
                 p_flip=0.5, box_jitter=0.05, seed=0):
        self.rows = read_manifest(manifest)
        self.cache = Path(cache_dir) / encoder
        self.train = train
        self.p_flip = p_flip
        self.box_jitter = box_jitter
        self.prompts = json.load(open(prompts_json)) if prompts_json else None
        self.rng = random.Random(seed)
        missing = [r for r in self.rows if not (self.cache / f'{uid(r)}.pt').exists()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} embeddings missing in {self.cache}, e.g. {missing[0]["image"]}; '
                                    f'run scripts/cache_embeddings.py first')
        doms = sorted({int(r['domain_id']) for r in self.rows if int(r['domain_id']) >= 0})
        self.K = (max(doms) + 1) if doms else 1

    def __len__(self):
        return len(self.rows)

    def _mask_resized(self, path, res_hw):
        m = Image.open(path).convert('L').resize((res_hw[1], res_hw[0]), Image.NEAREST)
        return np.array(m) > 127

    @staticmethod
    def _pad_lowres(mask_res):
        """resized mask (h',w') -> padded 1024 -> 256x256 float, plus valid region mask."""
        h, w = mask_res.shape
        full = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
        full[:h, :w] = mask_res
        t = torch.from_numpy(full)[None, None]
        low = torch.nn.functional.interpolate(t, size=(LOW, LOW), mode='area')[0]
        valid = torch.zeros(1, LOW, LOW)
        valid[:, :max(1, int(round(h * LOW / IMG_SIZE))), :max(1, int(round(w * LOW / IMG_SIZE)))] = 1
        return (low > 0.5).float(), valid

    def __getitem__(self, i):
        r = self.rows[i]
        stem = uid(r)
        c = torch.load(self.cache / f'{stem}.pt')
        res_hw, orig_hw = tuple(c['resized_hw']), tuple(c['orig_hw'])
        emb = c['emb']
        mask = self._mask_resized(r['mask'], res_hw)
        flip = False
        if self.train and 'emb_flip' in c and self.rng.random() < self.p_flip:
            emb, mask, flip = c['emb_flip'], mask[:, ::-1].copy(), True
        emb = emb.float()
        h, w = res_hw
        fg = mask.sum() > 0
        if self.prompts is not None:                                  # evaluation: fixed prompts
            p = self.prompts[stem]
            if p.get('bbox') is None:
                fg = False
            else:
                pc = scale_coords(p['point_centroid'], orig_hw, res_hw)
                pr = scale_coords(p['point_random'], orig_hw, res_hw)
                b0 = scale_coords(p['bbox'][:2], orig_hw, res_hw)
                b1 = scale_coords(p['bbox'][2:], orig_hw, res_hw)
                jit = {k: scale_coords(v[:2], orig_hw, res_hw) + scale_coords(v[2:], orig_hw, res_hw)
                       for k, v in p.items() if k.startswith('bbox_jit')}
        elif fg:                                                      # training: prompts from GT
            comp = largest_component(mask)
            pc = centroid_inside(comp)
            ys, xs = np.nonzero(comp)
            j = self.rng.randrange(len(xs))
            pr = [float(xs[j]), float(ys[j])]
            ys, xs = np.nonzero(mask)
            b0, b1 = [float(xs.min()), float(ys.min())], [float(xs.max()), float(ys.max())]
            if self.box_jitter > 0:
                bw, bh = max(b1[0] - b0[0], 1), max(b1[1] - b0[1], 1)
                jx = lambda: self.rng.uniform(-self.box_jitter, self.box_jitter) * bw
                jy = lambda: self.rng.uniform(-self.box_jitter, self.box_jitter) * bh
                b0 = [min(max(b0[0] + jx(), 0), w - 2), min(max(b0[1] + jy(), 0), h - 2)]
                b1 = [min(max(b1[0] + jx(), b0[0] + 1), w - 1), min(max(b1[1] + jy(), b0[1] + 1), h - 1)]
            jit = {}
        if not fg:                                                    # empty GT: dummy prompts, flagged
            pc = pr = [w / 2.0, h / 2.0]
            b0, b1 = [0.0, 0.0], [float(w - 1), float(h - 1)]
            # evaluation batches must carry the same keys for every item, so give the jitter boxes a dummy too
            jit = {k: b0 + b1 for k in ('bbox_jit5', 'bbox_jit10', 'bbox_jit20')} if self.prompts is not None else {}
        low, valid = self._pad_lowres(mask)
        item = dict(emb=emb, gt_low=low, valid=valid, point_c=torch.tensor(pc), point_r=torch.tensor(pr),
                    box=torch.tensor(b0 + b1), domain_id=int(r['domain_id']), stem=stem, has_fg=bool(fg),
                    res_hw=torch.tensor(res_hw), orig_hw=torch.tensor(orig_hw), flip=flip)
        for k, v in jit.items():
            item[k] = torch.tensor(v)
        return item


def collate(batch):
    out = {}
    for k in batch[0]:
        v = [b[k] for b in batch]
        if torch.is_tensor(v[0]):
            out[k] = torch.stack(v)
        else:
            out[k] = v
    out['domain_id'] = torch.tensor(out['domain_id'])
    return out


def make_prompt_inputs(batch, mode, device):
    """-> (points, boxes) in the format expected by SAM's PromptEncoder, or None."""
    pts = boxes = None
    if mode in ('1pt', '1pt+bbox'):
        pts = (batch['point_c'][:, None, :].to(device), torch.ones(len(batch['point_c']), 1, device=device, dtype=torch.long))
    elif mode == 'randpt':
        pts = (batch['point_r'][:, None, :].to(device), torch.ones(len(batch['point_r']), 1, device=device, dtype=torch.long))
    if mode in ('bbox', '1pt+bbox'):
        boxes = batch['box'].to(device)
    elif mode.startswith('bbox_jit'):
        boxes = batch[mode].to(device)
    if pts is None and boxes is None and mode != 'noprompt':
        raise ValueError(mode)
    return pts, boxes
