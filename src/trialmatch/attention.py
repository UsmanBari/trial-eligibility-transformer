"""Task 4: reference scaled dot-product attention and a tiled online-softmax version.

Shapes: q (B, H, Lq, D), k and v (B, H, Lk, D). With causal=True, query row i sits at
absolute position (Lk - Lq + i), so it may attend to keys 0..Lk-Lq+i. This covers both
prefill (Lq == Lk) and cached decoding (Lq < Lk, e.g. Lq = 1).

reference_attention
    S = (q k^T) / sqrt(D); masked entries set to -inf; stable softmax (subtract the row
    max before exp); O = softmax(S) v. Materialises the full Lq x Lk score matrix.

tiled_attention (the FlashAttention idea, in plain PyTorch)
    Process queries in blocks of Bq and keys/values in blocks of Bk. For each query row
    keep three running quantities over the key blocks seen so far:
        m  = running max of the scores
        l  = running sum of exp(score - m)
        acc= running sum of exp(score - m) * v
    For a new key block with scores s:
        m_new = max(m, rowmax(s))
        p     = exp(s - m_new)
        alpha = exp(m - m_new)            # rescales everything accumulated so far
        l     = alpha * l + rowsum(p)
        acc   = alpha * acc + p @ v_block
        m     = m_new
    and finally O = acc / l. This is exactly softmax(S) v, but only a Bq x Bk score tile
    exists at any time, so extra memory is O(Bq*Bk) instead of O(Lq*Lk). With a causal
    mask, key blocks that lie entirely in the future of a query block are skipped.
All arithmetic is done in fp32 and cast back, like fused kernels do.
"""
from __future__ import annotations

import math

import torch


def _mask_from_sdpa(attn_mask, Lq, Lk, device):
    """SDPA mask convention -> additive fp32 mask (or None). bool: True = keep."""
    if attn_mask is None:
        return None
    if attn_mask.dtype == torch.bool:
        return torch.zeros(attn_mask.shape, device=device).masked_fill(~attn_mask, float("-inf"))
    return attn_mask.float()


def reference_attention(q, k, v, causal=True, scale=None, attn_mask=None):
    D = q.shape[-1]
    Lq, Lk = q.shape[-2], k.shape[-2]
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale          # (B,H,Lq,Lk)
    add = _mask_from_sdpa(attn_mask, Lq, Lk, q.device)
    if add is not None:
        s = s + add
    if causal:
        qpos = torch.arange(Lq, device=q.device)[:, None] + (Lk - Lq)
        kpos = torch.arange(Lk, device=q.device)[None, :]
        s = s.masked_fill(kpos > qpos, float("-inf"))
    m = s.amax(dim=-1, keepdim=True)
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)                   # fully-masked rows
    p = torch.exp(s - m)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return torch.matmul(p, v.float()).to(q.dtype)


def tiled_attention(q, k, v, causal=True, scale=None, block_q=64, block_k=64, attn_mask=None,
                    return_stats=False):
    B, H, Lq, D = q.shape
    Lk = k.shape[-2]
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    add = _mask_from_sdpa(attn_mask, Lq, Lk, q.device)
    out = torch.empty((B, H, Lq, D), dtype=torch.float32, device=q.device)
    offset = Lk - Lq
    n_tiles = n_skipped = 0
    kf, vf = k.float(), v.float()

    for qs in range(0, Lq, block_q):
        qe = min(qs + block_q, Lq)
        qb = q[:, :, qs:qe].float() * scale
        nq = qe - qs
        m = torch.full((B, H, nq), float("-inf"), device=q.device)
        l = torch.zeros((B, H, nq), device=q.device)
        acc = torch.zeros((B, H, nq, D), device=q.device)
        qpos = torch.arange(qs, qe, device=q.device) + offset                  # absolute positions
        for ks in range(0, Lk, block_k):
            ke = min(ks + block_k, Lk)
            if causal and ks > int(qpos[-1]):                                  # whole block in the future
                n_skipped += math.ceil((Lk - ks) / block_k)
                break
            n_tiles += 1
            s = torch.matmul(qb, kf[:, :, ks:ke].transpose(-1, -2))            # (B,H,nq,nk) tile
            if add is not None:
                s = s + add[..., qs:qe, ks:ke]
            if causal and (ke - 1) > int(qpos[0]):                             # tile crosses the diagonal
                kpos = torch.arange(ks, ke, device=q.device)
                s = s.masked_fill(kpos[None, :] > qpos[:, None], float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1))
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            p = torch.exp(s - m_safe[..., None])
            alpha = torch.exp(m - m_safe)                                      # exp(-inf)=0 on first tile
            l = alpha * l + p.sum(dim=-1)
            acc = alpha[..., None] * acc + torch.matmul(p, vf[:, :, ks:ke])
            m = m_new
        out[:, :, qs:qe] = acc / l.clamp_min(1e-30)[..., None]
    out = out.to(q.dtype)
    if return_stats:
        return out, {"tiles_computed": n_tiles, "tiles_skipped_causal": n_skipped}
    return out


# --------------------------------------------------------------------------- model integration
class use_attention_impl:
    """Context manager: route the model's scaled_dot_product_attention calls through our
    tiled (or reference) implementation, so a full Qwen2 forward pass runs on it.

    transformers 4.46 Qwen2 "sdpa" attention calls torch.nn.functional.scaled_dot_product_attention
    with k/v already expanded for grouped-query attention; batch 1 without padding passes
    attn_mask=None and is_causal=(q_len > 1)."""

    def __init__(self, impl="tiled", block_q=64, block_k=64):
        self.impl, self.bq, self.bk = impl, block_q, block_k
        self.calls = 0

    def __enter__(self):
        self._orig = torch.nn.functional.scaled_dot_product_attention

        def patched(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
            self.calls += 1
            if self.impl == "reference":
                return reference_attention(query, key, value, causal=is_causal, scale=scale, attn_mask=attn_mask)
            return tiled_attention(query, key, value, causal=is_causal, scale=scale,
                                   block_q=self.bq, block_k=self.bk, attn_mask=attn_mask)

        torch.nn.functional.scaled_dot_product_attention = patched
        return self

    def __exit__(self, *exc):
        torch.nn.functional.scaled_dot_product_attention = self._orig
        return False
