"""Task 5: greedy speculative decoding with a small draft model, written from scratch.

State kept across iterations
    S        committed tokens (prompt + accepted output), a Python list
    t_cache  target KV cache;  invariant after every iteration: holds S[:-1]
    d_cache  draft KV cache;   holds some prefix S[:d_len] (may lag behind S)

One iteration (k = number of draft tokens)
  1. Draft: feed the draft every committed token it has not seen yet (S[d_len:], usually
     1-2 tokens) in ONE forward pass, take argmax -> d1; then k-1 single-token steps
     -> d2..dk. Cheap: the draft is ~3x smaller.
  2. Verify: feed the target [S[-1], d1, ..., dk] in ONE forward pass at positions
     len(S)-1 .. len(S)+k-1. Its logits at those k+1 positions are the target's own
     greedy choices t1..t(k+1) (t1 = what the target would emit after S, t2 = after
     S+d1, ...). Thanks to the causal mask, t_i only depends on d1..d(i-1).
  3. Accept the longest prefix where d_i == t_i (n tokens), then append t(n+1): the
     target's correction if n < k, or a free "bonus" token if all k matched.
     So every iteration commits n+1 tokens for one target forward pass.
  4. Cache rollback: the target cache now contains positions for all k draft tokens,
     but only n of them were accepted -> crop it back to len(S)-1. The draft cache
     contains S_old + d1..d(k-1) -> crop it to the part that is still a prefix of S.

Because every committed token is the target's argmax given the committed prefix, the
output equals plain greedy decoding of the target (up to floating-point ties: the
target scores k+1 tokens in one pass instead of one at a time, see bench_spec.py).
"""
from __future__ import annotations

import time

import torch
from transformers import DynamicCache


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _forward(model, cache, cache_len, tokens, device):
    """Feed `tokens` at absolute positions cache_len.. ; returns logits (n, V), appends to cache."""
    n = len(tokens)
    ids = torch.tensor([tokens], device=device)
    pos = torch.arange(cache_len, cache_len + n, device=device)
    out = model(input_ids=ids, attention_mask=torch.ones((1, cache_len + n), dtype=torch.long, device=device),
                position_ids=pos[None], cache_position=pos, past_key_values=cache, use_cache=True)
    return out.logits[0]


def check_draft_compatible(tok_target, tok_draft, target, draft) -> dict:
    same_vocab = tok_target.get_vocab() == tok_draft.get_vocab()
    return {"same_vocab": same_vocab, "target_vocab_size": len(tok_target), "draft_vocab_size": len(tok_draft),
            "target_embedding_rows": target.get_input_embeddings().weight.shape[0],
            "draft_embedding_rows": draft.get_input_embeddings().weight.shape[0],
            "same_special_tokens": tok_target.all_special_tokens == tok_draft.all_special_tokens,
            "compatible": same_vocab and target.get_input_embeddings().weight.shape[0]
            == draft.get_input_embeddings().weight.shape[0]}


@torch.no_grad()
def speculative_greedy(target, draft, input_ids, max_new_tokens, k=4, eos_ids=(), stop_at_eos=True):
    device = input_ids.device
    S = input_ids[0].tolist()
    P = len(S)
    t_cache, d_cache = DynamicCache(), DynamicCache()
    d_len = 0
    stats = {"iterations": 0, "target_forwards": 0, "draft_forwards": 0, "drafted": 0, "accepted": 0,
             "accepted_per_iter": []}
    _sync(device); t0 = time.perf_counter()

    # target prefill on the prompt gives the first new token (same as normal greedy)
    logits = _forward(target, t_cache, 0, S, device)
    stats["target_forwards"] += 1
    S.append(int(logits[-1].argmax()))
    _sync(device); t_first = time.perf_counter()

    def done():
        n_new = len(S) - P
        return n_new >= max_new_tokens or (stop_at_eos and S[-1] in eos_ids)

    while not done():
        remaining = max_new_tokens - (len(S) - P)
        k_eff = min(k, remaining - 1)          # we commit at most k_eff + 1 tokens
        S_old_len = len(S)
        if k_eff <= 0:                         # last token: plain target step
            logits = _forward(target, t_cache, S_old_len - 1, [S[-1]], device)
            stats["target_forwards"] += 1
            S.append(int(logits[-1].argmax()))
            continue

        # 1) draft k_eff tokens (first call catches the draft up on all unseen tokens)
        d = []
        logits = _forward(draft, d_cache, d_len, S[d_len:], device)
        d_len = len(S)
        d.append(int(logits[-1].argmax()))
        stats["draft_forwards"] += 1
        for _ in range(k_eff - 1):
            logits = _forward(draft, d_cache, d_len, [d[-1]], device)
            d_len += 1
            d.append(int(logits[-1].argmax()))
            stats["draft_forwards"] += 1

        # 2) verify all drafts with ONE target forward over [last committed, d1..dk]
        logits = _forward(target, t_cache, S_old_len - 1, [S[-1]] + d, device)
        stats["target_forwards"] += 1
        tgt = logits.argmax(-1).tolist()       # tgt[i] = target's choice after (S, d1..di)

        # 3) accept the matching prefix + one target token
        n = 0
        while n < k_eff and d[n] == tgt[n]:
            n += 1
        new = d[:n] + [tgt[n]]
        if stop_at_eos:
            for j, t in enumerate(new):
                if t in eos_ids:
                    new = new[: j + 1]
                    break
        S.extend(new)
        stats["iterations"] += 1
        stats["drafted"] += k_eff
        stats["accepted"] += min(n, len(new))
        stats["accepted_per_iter"].append(n)

        # 4) roll caches back to what is still valid
        t_cache.crop(len(S) - 1)               # target holds S[:-1]
        keep = min(S_old_len + n, d_len)       # draft prefix that matches S
        d_cache.crop(keep)
        d_len = keep
        assert t_cache.get_seq_length() == len(S) - 1

    _sync(device); t_end = time.perf_counter()
    tokens = S[P:][:max_new_tokens]
    stats["acceptance_rate"] = stats["accepted"] / stats["drafted"] if stats["drafted"] else None
    stats["tokens_per_target_forward"] = len(tokens) / stats["target_forwards"]
    return {"tokens": tokens, "prefill_s": t_first - t0, "decode_s": t_end - t_first,
            "total_s": t_end - t0, "stats": stats}
