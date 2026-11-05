"""Greedy autoregressive decoding written out by hand: a no-cache reference and a KV-cached path.

Both functions call the model's forward pass directly (no model.generate), so every
detail of the decoding loop is visible:

no-cache reference (O(T^2) work over the whole run)
    step t: feed the FULL sequence [prompt + generated so far], take argmax of the
    last position's logits. Nothing is reused between steps.

KV-cached path (O(T) new work per step)
    prefill: feed the prompt once (positions 0..L-1); every layer stores its keys and
             values in a DynamicCache -> shape per layer (batch, n_kv_heads, L, head_dim).
    decode:  feed ONLY the newest token at position p = L+t-1. The layer computes q,k,v
             for that one token, APPENDS k,v to the cache (cache grows by exactly one),
             and attends the single query over all p+1 cached keys.

Positional handling: Qwen2 uses rotary embeddings (RoPE). Keys are rotated with their
absolute position *before* they are cached, so cached keys never need re-rotation. The
new token must therefore be given its true absolute position (position_ids=[[p]] and
cache_position=[p]); feeding it at position 0 would silently corrupt the output.

Masking: in prefill the causal mask stops token i from seeing j>i. In decode the single
query is the last token, so all cached keys are in its past and the causal mask reduces
to "attend to everything"; attention_mask is all ones with length p+1 (no padding, batch 1).

Numerical differences: with the cache, attention for the new token is computed as a
1 x (p+1) product instead of as one row of a (p+1) x (p+1) product. Same math, different
kernel / reduction order, so fp16 logits can differ in the last bits. Greedy outputs stay
identical unless the top-2 logits are nearly tied; bench_kv.py measures exactly this.
"""
from __future__ import annotations

import time

import torch
from transformers import DynamicCache


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def eos_ids_of(model, tok=None):
    e = model.generation_config.eos_token_id
    ids = set(e if isinstance(e, (list, tuple)) else [e]) if e is not None else set()
    if tok is not None and tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


def kv_cache_bytes_per_token(model) -> int:
    """2 (K and V) x layers x kv_heads x head_dim x bytes per element."""
    c = model.config
    head_dim = c.hidden_size // c.num_attention_heads
    nbytes = next(model.parameters()).element_size()
    return 2 * c.num_hidden_layers * c.num_key_value_heads * head_dim * nbytes


@torch.no_grad()
def greedy_no_cache(model, input_ids, max_new_tokens, eos_ids=(), stop_at_eos=True):
    """Reference: full recomputation of the whole sequence for every new token."""
    device = input_ids.device
    ids = input_ids
    out_tokens = []
    _sync(device); t0 = time.perf_counter()
    t_first = None
    for step in range(max_new_tokens):
        L = ids.shape[1]
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       position_ids=torch.arange(L, device=device)[None],
                       use_cache=False, num_logits_to_keep=1).logits
        nxt = logits[:, -1].argmax(-1)
        if step == 0:
            _sync(device); t_first = time.perf_counter()
        tok = int(nxt)
        out_tokens.append(tok)
        if stop_at_eos and tok in eos_ids:
            break
        ids = torch.cat([ids, nxt[:, None]], dim=1)
    _sync(device); t_end = time.perf_counter()
    return {"tokens": out_tokens, "prefill_s": t_first - t0, "decode_s": t_end - t_first,
            "total_s": t_end - t0}


@torch.no_grad()
def greedy_kv(model, input_ids, max_new_tokens, eos_ids=(), stop_at_eos=True, trace_cache=False):
    """KV-cached greedy decoding with explicit cache reuse, positions and mask."""
    device = input_ids.device
    L = input_ids.shape[1]
    cache = DynamicCache()
    trace = []
    _sync(device); t0 = time.perf_counter()

    # ---- prefill: whole prompt once, cache now holds L positions
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                position_ids=torch.arange(L, device=device)[None],
                cache_position=torch.arange(L, device=device),
                past_key_values=cache, use_cache=True, num_logits_to_keep=1)
    cache = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1)
    _sync(device); t_first = time.perf_counter()
    out_tokens = [int(nxt)]
    if trace_cache:
        trace.append(_cache_shape(cache))

    # ---- decode: one token per step at its true absolute position p
    for t in range(1, max_new_tokens):
        if stop_at_eos and out_tokens[-1] in eos_ids:
            break
        p = L + t - 1
        assert cache.get_seq_length() == p, "cache length must equal the new token's position"
        out = model(input_ids=nxt[:, None],
                    attention_mask=torch.ones((1, p + 1), dtype=torch.long, device=device),
                    position_ids=torch.tensor([[p]], device=device),
                    cache_position=torch.tensor([p], device=device),
                    past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
        out_tokens.append(int(nxt))
        if trace_cache:
            trace.append(_cache_shape(cache))
    _sync(device); t_end = time.perf_counter()
    res = {"tokens": out_tokens, "prefill_s": t_first - t0, "decode_s": t_end - t_first,
           "total_s": t_end - t0, "final_cache_len": cache.get_seq_length()}
    if trace_cache:
        res["cache_trace"] = trace
    return res


def _cache_shape(cache):
    k = cache.key_cache[0]
    return {"seq_len": cache.get_seq_length(), "layer0_key_shape": list(k.shape),
            "n_layers": len(cache.key_cache)}


@torch.no_grad()
def last_logits_both_ways(model, prefix_ids):
    """Logits for the token after `prefix_ids`, computed (a) without cache and (b) by
    prefilling prefix[:-1] then feeding the last token with the cache. Used to explain
    any token mismatch between the two decoders."""
    device = prefix_ids.device
    L = prefix_ids.shape[1]
    a = model(input_ids=prefix_ids, position_ids=torch.arange(L, device=device)[None],
              use_cache=False, num_logits_to_keep=1).logits[0, -1].float()
    cache = DynamicCache()
    model(input_ids=prefix_ids[:, :-1], position_ids=torch.arange(L - 1, device=device)[None],
          cache_position=torch.arange(L - 1, device=device), past_key_values=cache, use_cache=True)
    b = model(input_ids=prefix_ids[:, -1:], attention_mask=torch.ones((1, L), dtype=torch.long, device=device),
              position_ids=torch.tensor([[L - 1]], device=device), cache_position=torch.tensor([L - 1], device=device),
              past_key_values=cache, use_cache=True).logits[0, -1].float()
    top_a = torch.topk(a, 2)
    top_b = torch.topk(b, 2)
    return {"max_abs_logit_diff": float((a - b).abs().max()),
            "argmax_nocache": int(top_a.indices[0]), "argmax_kv": int(top_b.indices[0]),
            "top2_margin_nocache": float(top_a.values[0] - top_a.values[1]),
            "top2_margin_kv": float(top_b.values[0] - top_b.values[1])}
