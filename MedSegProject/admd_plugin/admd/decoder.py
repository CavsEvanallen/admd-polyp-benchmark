"""
ADMD: Automatic Domain-Matching Decoder, re-implemented as a plug-in on top of the SAM mask decoder.

Components (numbers refer to the equations in paper_draft_v0_en.md):
  (1) domain recognizer       d = softmax(W_d · meanpool(F) + b_d)                     [B, K]
  (2) query modulation        Q~ = Q ⊙ (1 + tanh(W_q d))      (W_q zero-init -> identity at start)
  (3) confidence-gated align  h = F + alpha · Attn(q=F, k=Q~, v=Q~),  alpha = gamma · max_k d_k
                              (output projection zero-init -> identity at start)
  branches                    K copies of the SAM two-way-transformer mask decoder, independent weights
  (4) fusion                  w = softmax(W_a · meanpool(h)),  M = sum_k w_k M_k   (logits are fused)

Fusion modes at inference: soft | hard | oracle | uniform | random.
`--decoder sam` in train.py == ADMDDecoder(K=1, use_query=False, use_align=False), i.e. plain SAM decoder fine-tuning.

Implementation notes vs. the published description (Mo et al., JIIM 2026 / thesis §3.3):
  * the thesis writes CrossAttn(Q_domain, K_img, V_img) added to F_img, which is dimensionally undefined
    (N_q tokens vs N image tokens). We attend from image tokens to domain-aware queries so the residual
    has the shape of F. State this in the paper.
  * the thesis writes d ⊙ Q_prompt without saying how a K-vector meets a 256-channel token; we use a
    learned projection W_q with a (1 + tanh) gate.
  * alpha's functional form is not given in the thesis; we use a learnable scalar times max_k d_k.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F_


class BranchDecoder(nn.Module):
    """A SAM MaskDecoder with a batch-aware forward (the stock predict_masks assumes batch size 1)."""

    def __init__(self, sam_mask_decoder):
        super().__init__()
        md = copy.deepcopy(sam_mask_decoder)
        self.transformer = md.transformer
        self.iou_token = md.iou_token
        self.mask_tokens = md.mask_tokens
        self.num_mask_tokens = md.num_mask_tokens
        self.output_upscaling = md.output_upscaling
        self.output_hypernetworks_mlps = md.output_hypernetworks_mlps
        self.iou_prediction_head = md.iou_prediction_head
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(self, image_embeddings, image_pe, sparse, dense):
        """image_embeddings [B,C,H,W] (or [1,C,H,W]); image_pe [1,C,H,W]; sparse [B,Nq,C]; dense [B,C,H,W]."""
        B = sparse.shape[0]
        out_tok = torch.cat([self.iou_token.weight, self.mask_tokens.weight], 0).unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([out_tok, sparse], 1)
        src = image_embeddings.expand(B, -1, -1, -1) if image_embeddings.shape[0] == 1 else image_embeddings
        src = src + dense
        pos = image_pe.expand(B, -1, -1, -1)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos, tokens)
        iou_tok = hs[:, 0, :]
        mask_tok = hs[:, 1:1 + self.num_mask_tokens, :]
        src = src.transpose(1, 2).reshape(b, c, h, w)
        up = self.output_upscaling(src)
        hyper = torch.stack([self.output_hypernetworks_mlps[i](mask_tok[:, i, :]) for i in range(self.num_mask_tokens)], 1)
        b, c, h, w = up.shape
        masks = (hyper @ up.reshape(b, c, h * w)).reshape(b, -1, h, w)      # [B, num_mask_tokens, 256, 256]
        iou = self.iou_prediction_head(iou_tok)                                # [B, num_mask_tokens]
        return masks[:, :1], iou[:, :1]                                        # single-mask output (SAM multimask=False)


class ADMDDecoder(nn.Module):
    def __init__(self, sam_mask_decoder, K, use_query=True, use_align=True, fusion='soft',
                 embed_dim=256, n_heads=8, wide=False):
        super().__init__()
        assert K >= 1
        self.K, self.use_query, self.use_align, self.fusion = K, use_query, use_align, fusion
        self.recognizer = nn.Linear(embed_dim, K)                                   # (1)
        self.branches = nn.ModuleList([BranchDecoder(sam_mask_decoder) for _ in range(K)])
        if use_query:
            self.q_gate = nn.Linear(K, embed_dim)                                   # (2)
            nn.init.zeros_(self.q_gate.weight); nn.init.zeros_(self.q_gate.bias)
        if use_align:
            self.align = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)  # (3)
            nn.init.zeros_(self.align.out_proj.weight); nn.init.zeros_(self.align.out_proj.bias)
            self.gamma = nn.Parameter(torch.tensor(1.0))
        self.router = nn.Linear(embed_dim, K)                                       # (4)
        self.wide = wide
        if wide:
            self.wide_mlp = nn.Sequential(
                nn.Linear(embed_dim, 2048), nn.ReLU(),
                nn.Linear(2048, 2048), nn.ReLU(),
                nn.Linear(2048, 2048), nn.ReLU(),
                nn.Linear(2048, embed_dim),
            )

    @staticmethod
    def _pool(x):
        return x.mean(dim=(2, 3))

    def fusion_weights(self, logits, mode, domain_id=None, generator=None):
        B, K = logits.shape
        if K == 1:
            return torch.ones_like(logits)
        if mode == 'soft':
            return logits.softmax(-1)
        if mode == 'hard':
            return F_.one_hot(logits.argmax(-1), K).to(logits.dtype)
        if mode == 'oracle':
            if domain_id is None or (domain_id < 0).any():
                raise ValueError('oracle fusion needs a valid domain_id for every sample')
            return F_.one_hot(domain_id.long(), K).to(logits.dtype)
        if mode == 'uniform':
            return torch.full_like(logits, 1.0 / K)
        if mode == 'random':
            idx = torch.randint(0, K, (B,), device=logits.device, generator=generator)
            return F_.one_hot(idx, K).to(logits.dtype)
        raise ValueError(mode)

    def forward(self, F, image_pe, sparse, dense, domain_id=None, fusion=None, generator=None):
        """
        F [B,C,64,64] image embedding; image_pe [1,C,64,64]; sparse [B,Nq,C]; dense [B,C,64,64].
        Returns dict: mask [B,1,256,256] fused logits, branch_masks [B,K,256,256], iou [B,K],
                      d [B,K], alpha [B], w [B,K]
        """
        fusion = fusion or self.fusion
        B = F.shape[0]
        d = self.recognizer(self._pool(F)).softmax(-1)                              # (1)
        q = sparse
        if self.use_query:
            q = q * (1.0 + torch.tanh(self.q_gate(d))).unsqueeze(1)                  # (2)
        h = F
        alpha = d.max(-1).values
        if self.use_align:
            tok = F.flatten(2).transpose(1, 2)                                      # [B, HW, C]
            upd, _ = self.align(tok, q, q, need_weights=False)
            upd = upd.transpose(1, 2).reshape_as(F)
            h = F + (self.gamma * alpha).view(B, 1, 1, 1) * upd                     # (3)
        if self.wide:
            b, c, hh, ww = h.shape
            tok = h.flatten(2).transpose(1, 2)
            tok = self.wide_mlp(tok)
            h = tok.transpose(1, 2).reshape(b, c, hh, ww)
        masks, ious = [], []
        for br in self.branches:
            m, i = br(h, image_pe, q, dense)
            masks.append(m); ious.append(i)
        branch_masks = torch.cat(masks, 1)                                           # [B,K,256,256]
        iou = torch.cat(ious, 1)                                                     # [B,K]
        w = self.fusion_weights(self.router(self._pool(h)), fusion, domain_id, generator)   # (4)
        mask = (w.view(B, self.K, 1, 1) * branch_masks).sum(1, keepdim=True)
        return dict(mask=mask, branch_masks=branch_masks, iou=iou, d=d, alpha=alpha, w=w)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def build_decoder(sam, decoder, K, use_query=True, use_align=True, fusion='soft', wide=False):
    if decoder == 'sam':
        return ADMDDecoder(sam.mask_decoder, K=1, use_query=False, use_align=False, fusion='uniform', wide=wide)
    if decoder == 'admd':
        return ADMDDecoder(sam.mask_decoder, K=K, use_query=use_query, use_align=use_align, fusion=fusion, wide=wide)
    raise ValueError(decoder)
