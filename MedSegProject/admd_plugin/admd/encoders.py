"""
Frozen SAM-family models. Every entry exposes the standard `Sam` object
(image_encoder / prompt_encoder / mask_decoder / preprocess / postprocess_masks),
and every image encoder here outputs a 256 x 64 x 64 embedding for a 1024 x 1024 input.

name        package            registry key   install
sam_b       segment_anything   vit_b          pip install git+https://github.com/facebookresearch/segment-anything
sam_l/sam_h segment_anything   vit_l / vit_h  same
mobilesam   mobile_sam         vit_t          pip install git+https://github.com/ChaoningZhang/MobileSAM
edgesam     edge_sam           edge_sam       pip install -e EdgeSAM (git clone https://github.com/chongzhou96/EdgeSAM)
repvit_sam  repvit_sam         repvit         clone https://github.com/THU-MIG/RepViT and add RepViT/sam to PYTHONPATH
"""
import importlib
import sys
from pathlib import Path

import torch

ENCODERS = {
    'sam_b': ('segment_anything', 'vit_b'),
    'sam_l': ('segment_anything', 'vit_l'),
    'sam_h': ('segment_anything', 'vit_h'),
    'mobilesam': ('mobile_sam', 'vit_t'),
    'edgesam': ('edge_sam', 'edge_sam'),
    'repvit_sam': ('repvit_sam', 'repvit'),
}


class _FakeSam(torch.nn.Module):
    """Randomly initialised prompt encoder + mask decoder, no image encoder. For dry runs and unit tests only."""

    def __init__(self):
        super().__init__()
        from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
        self.prompt_encoder = PromptEncoder(embed_dim=256, image_embedding_size=(64, 64),
                                            input_image_size=(1024, 1024), mask_in_chans=16)
        self.mask_decoder = MaskDecoder(num_multimask_outputs=3,
                                        transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
                                        transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256)
        self.image_encoder = torch.nn.Identity()
        self.image_encoder.img_size = 1024


def load_sam(name, ckpt, device='cuda', repvit_root=None):
    if name == 'fake':
        sam = _FakeSam().to(device).eval()
        for p in sam.parameters():
            p.requires_grad_(False)
        return sam
    if name not in ENCODERS:
        raise KeyError(f'unknown encoder {name}; choose from {list(ENCODERS)}')
    pkg, key = ENCODERS[name]
    if name == 'repvit_sam' and repvit_root:
        sys.path.insert(0, str(Path(repvit_root) / 'sam'))
    mod = importlib.import_module(pkg)
    sam = mod.sam_model_registry[key](checkpoint=ckpt)
    sam.to(device).eval()
    for p in sam.parameters():
        p.requires_grad_(False)
    return sam


@torch.no_grad()
def embed_images(sam, images_uint8, transform, device='cuda'):
    """images_uint8: list of HxWx3 uint8 arrays (RGB). Returns (emb[B,256,64,64] fp16, [(h',w')], [(H,W)])."""
    batch, resized, orig = [], [], []
    for im in images_uint8:
        H, W = im.shape[:2]
        x = transform.apply_image(im)                     # longest side -> 1024, uint8 HWC
        resized.append(x.shape[:2])
        orig.append((H, W))
        t = torch.as_tensor(x, device=device).permute(2, 0, 1).contiguous()[None].float()
        batch.append(sam.preprocess(t))                   # normalize + pad to 1024x1024
    x = torch.cat(batch, 0)
    emb = sam.image_encoder(x)
    return emb.half(), resized, orig
