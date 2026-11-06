"""Task 5: greedy speculative decoding, written from scratch, with two drafters.

Drafters
  * draft model : a small LM with the SAME tokenizer (Qwen2.5-0.5B-Instruct, optionally
                  LoRA-tuned on the train split). Proposes k tokens autoregressively.
  * prompt lookup ("prompt_lookup"): no model at all. Take the last n committed tokens
                  (n = 3, then 2, then 1), find their most recent earlier occurrence in
                  prompt + output, and propose the k tokens that followed it. Costs almost
                  nothing, and works well when the answer copies the prompt (label names,
                  the JSON keys, note phrases), which is exactly our task.

State kept across iterations
    S        committed tokens (prompt + accepted output), a Python list
    t_cache  target KV cache;  invariant after every iteration: holds S[:-1]
    d_cache  draft KV cache;   holds some prefix S[:d_len] (may lag behind S)

One iteration
  1. Draft k tokens d1..dk (draft model: first call catches the draft up on all unseen
     committed tokens in ONE forward pass, then k-1 single-token steps; the proposals stay
     on the GPU and are copied to the CPU once, so there is one sync per draft, not k).
  2. Verify: feed the target [S[-1], d1, ..., dk] in ONE forward pass at positions
     len(S)-1 .. len(S)+k-1. Its argmax at those k+1 positions are the target's own greedy
     choices t1..t(k+1) (t_i depends only on S and d1..d(i-1) thanks to the causal mask).
  3. Accept the longest prefix with d_i == t_i (n tokens) and append t(n+1): the target's
     correction if n < k, or a free bonus token if all k matched -> n+1 tokens per
     target forward.
  4. Roll back: crop the target cache to len(S)-1 and the draft cache to the part that is
     still a prefix of S.

Every committed token is the target's argmax given the committed prefix, so the output
equals plain greedy decoding of the target (up to floating-point ties, see bench_spec.py).

When does it pay off?  With acceptance a per drafted token, the expected tokens per
iteration are E = (1 - a^(k+1)) / (1 - a). In units of one target decode step, an
iteration costs  v_k + k*c  (v_k = cost of verifying k+1 tokens at once, c = one draft
step / one target step). Speedup ~ E / (v_k + k*c). If c is close to 1 (a draft with
almost as many layers as the target, run at batch size 1 where every step is dominated by
kernel-launch overhead), there is no speedup even at high acceptance; with prompt lookup
c ~ 0 and the speedup is ~ E / v_k.
"""
from __future__ import annotations

import time

import torch
from transformers import DynamicCache

PROMPT_LOOKUP = "prompt_lookup"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _forward(model, cache, cache_len, ids, device):
    """Feed ids (list or (1,n) tensor) at absolute positions cache_len.. ; returns logits (n, V)."""
    if not torch.is_tensor(ids):
        ids = torch.tensor([ids], device=device)
    n = ids.shape[1]
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


def prompt_lookup_draft(S: list[int], k: int, max_ngram: int = 3) -> list[int]:
    """Propose up to k tokens by matching the last n tokens (n = max_ngram..1) earlier in S."""
    L = len(S)
    for n in range(min(max_ngram, L - 1), 0, -1):
        pattern = S[L - n:]
        for start in range(L - n - 1, -1, -1):           # most recent earlier occurrence first
            if S[start:start + n] == pattern:
                cont = S[start + n:start + n + k]
                if cont:
                    return cont
    return []


@torch.no_grad()
def speculative_greedy(target, draft, input_ids, max_new_tokens, k=4, eos_ids=(), stop_at_eos=True,
                       max_ngram=3):
    """draft: a causal LM with the target's tokenizer, or the string "prompt_lookup"."""
    device = input_ids.device
    lookup = isinstance(draft, str)
    if lookup and draft != PROMPT_LOOKUP:
        raise ValueError(f"unknown drafter {draft!r}")
    S = input_ids[0].tolist()
    P = len(S)
    t_cache = DynamicCache()
    d_cache = None if lookup else DynamicCache()
    d_len = 0
    stats = {"drafter": PROMPT_LOOKUP if lookup else "model", "iterations": 0, "target_forwards": 0,
             "draft_forwards": 0, "drafted": 0, "accepted": 0, "accepted_per_iter": [], "empty_drafts": 0}
    _sync(device); t0 = time.perf_counter()

    # target prefill on the prompt gives the first new token (same as normal greedy)
    logits = _forward(target, t_cache, 0, S, device)
    stats["target_forwards"] += 1
    S.append(int(logits[-1].argmax()))
    _sync(device); t_first = time.perf_counter()

    def done():
        return len(S) - P >= max_new_tokens or (stop_at_eos and S[-1] in eos_ids)

    while not done():
        remaining = max_new_tokens - (len(S) - P)
        k_eff = min(k, remaining - 1)                 # we commit at most k_eff + 1 tokens
        S_old_len = len(S)

        # 1) draft
        if k_eff <= 0:
            d = []
        elif lookup:
            d = prompt_lookup_draft(S, k_eff, max_ngram)
        else:
            logits = _forward(draft, d_cache, d_len, S[d_len:], device)   # catch up in one pass
            d_len = len(S)
            nxt = logits[-1:].argmax(-1)
            props = [nxt]
            for _ in range(k_eff - 1):
                logits = _forward(draft, d_cache, d_len, nxt[None], device)
                d_len += 1
                nxt = logits[-1:].argmax(-1)
                props.append(nxt)
            d = torch.cat(props).tolist()             # single GPU->CPU copy for the whole draft
            stats["draft_forwards"] += k_eff

        if not d:                                     # nothing to verify: plain target step
            stats["empty_drafts"] += int(k_eff > 0)
            logits = _forward(target, t_cache, S_old_len - 1, [S[-1]], device)
            stats["target_forwards"] += 1
            S.append(int(logits[-1].argmax()))
            continue

        # 2) verify all drafts with ONE target forward over [last committed, d1..dk]
        logits = _forward(target, t_cache, S_old_len - 1, [S[-1]] + d, device)
        stats["target_forwards"] += 1
        tgt = logits.argmax(-1).tolist()              # tgt[i] = target's choice after (S, d1..di)

        # 3) accept the matching prefix + one target token
        n = 0
        while n < len(d) and d[n] == tgt[n]:
            n += 1
        new = d[:n] + [tgt[n]]
        if stop_at_eos:
            for j, t in enumerate(new):
                if t in eos_ids:
                    new = new[: j + 1]
                    break
        S.extend(new)
        stats["iterations"] += 1
        stats["drafted"] += len(d)
        stats["accepted"] += min(n, len(new))
        stats["accepted_per_iter"].append(n)

        # 4) roll caches back to what is still valid
        t_cache.crop(len(S) - 1)                      # target holds S[:-1]
        if not lookup:
            keep = min(S_old_len + n, d_len)          # draft prefix that matches S
            d_cache.crop(keep)
            d_len = keep
        assert t_cache.get_seq_length() == len(S) - 1

    _sync(device); t_end = time.perf_counter()
    tokens = S[P:][:max_new_tokens]
    stats["acceptance_rate"] = stats["accepted"] / stats["drafted"] if stats["drafted"] else None
    stats["tokens_per_target_forward"] = len(tokens) / stats["target_forwards"]
    return {"tokens": tokens, "prefill_s": t_first - t0, "decode_s": t_end - t_first,
            "total_s": t_end - t0, "stats": stats}


# --------------------------------------------------------------------------- cost model
@torch.no_grad()
def step_costs_ms(model, ctx_len, n_tokens_list=(1,), repeats=15, vocab_hi=1000):
    """Median latency of a forward that adds n tokens on top of a cached context of ctx_len."""
    device = next(model.parameters()).device
    vocab_hi = min(vocab_hi, model.config.vocab_size)
    ctx = torch.randint(1, vocab_hi, (1, ctx_len), device=device)
    out = {}
    for n in n_tokens_list:
        ts = []
        for r in range(repeats + 2):
            cache = DynamicCache()
            _forward(model, cache, 0, ctx, device)
            new = torch.randint(1, vocab_hi, (1, n), device=device)
            _sync(device); t0 = time.perf_counter()
            _forward(model, cache, ctx_len, new, device).argmax(-1).tolist()
            _sync(device)
            if r >= 2:                                 # first two are warm-up
                ts.append(time.perf_counter() - t0)
        ts.sort()
        out[n] = 1e3 * ts[len(ts) // 2]
    return out


def expected_tokens_per_iter(alpha, k):
    if alpha is None:
        return 1.0
    if alpha >= 1:
        return k + 1.0
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def predicted_speedup(alpha, k, c, v_k):
    """E(alpha,k) / (v_k + k*c) with costs in units of one target decode step."""
    return expected_tokens_per_iter(alpha, k) / (v_k + k * c)
