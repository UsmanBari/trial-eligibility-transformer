"""CPU tests for Tasks 3-5 with tiny randomly initialised Qwen2 models (no download).

fp32 on CPU: KV-cached greedy, no-cache greedy and speculative decoding must give
exactly the same tokens; tiled attention must match the reference within 1e-5.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trialmatch.attention import reference_attention, tiled_attention, use_attention_impl  # noqa: E402
from trialmatch.kv_decode import greedy_kv, greedy_no_cache  # noqa: E402
from trialmatch.speculative import speculative_greedy  # noqa: E402


def tiny_qwen(seed, layers=2, hidden=64, heads=4, kv_heads=2, vocab=97):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=kv_heads, max_position_embeddings=512,
                      attn_implementation="sdpa")
    return Qwen2ForCausalLM(cfg).eval()


@pytest.mark.parametrize("L,N", [(5, 12), (30, 25), (64, 40)])
def test_kv_equals_no_cache(L, N):
    m = tiny_qwen(0)
    ids = torch.randint(0, 97, (1, L), generator=torch.Generator().manual_seed(L))
    a = greedy_no_cache(m, ids, N, stop_at_eos=False)["tokens"]
    b = greedy_kv(m, ids, N, stop_at_eos=False, trace_cache=True)
    assert a == b["tokens"]
    assert b["final_cache_len"] == L + N - 1                 # last token is never fed back
    assert [t["seq_len"] for t in b["cache_trace"]] == list(range(L, L + N))


def test_kv_equals_hf_generate():
    m = tiny_qwen(1)
    ids = torch.randint(0, 97, (1, 20), generator=torch.Generator().manual_seed(3))
    hf = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=15, do_sample=False,
                    eos_token_id=None, pad_token_id=0)[0, 20:].tolist()
    assert greedy_kv(m, ids, 15, stop_at_eos=False)["tokens"] == hf


def test_wrong_positions_change_the_logits():
    """Positional handling matters: feeding the new token at position 0 instead of its true
    position p changes the logits (RoPE rotates q/k by absolute position), so a decoder
    that got positions wrong would not match the reference."""
    from transformers import DynamicCache
    m = tiny_qwen(0)
    ids = torch.randint(0, 97, (1, 30), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        def step(pos):
            cache = DynamicCache()
            m(input_ids=ids[:, :-1], past_key_values=cache, use_cache=True)
            return m(input_ids=ids[:, -1:], position_ids=torch.tensor([[pos]]),
                     cache_position=torch.tensor([29]), past_key_values=cache, use_cache=True).logits[0, -1]
        full = m(input_ids=ids).logits[0, -1]
    assert (step(29) - full).abs().max() < 1e-4          # correct position == full recompute
    assert (step(0) - full).abs().max() > 1e-3           # wrong position -> different logits


@pytest.mark.parametrize("k", [1, 2, 4, 7])
def test_speculative_equals_greedy(k):
    target, draft = tiny_qwen(0, layers=3), tiny_qwen(5, layers=1)
    for seed in range(3):
        ids = torch.randint(0, 97, (1, 15 + seed), generator=torch.Generator().manual_seed(seed))
        ref = greedy_kv(target, ids, 30, stop_at_eos=False)["tokens"]
        r = speculative_greedy(target, draft, ids, 30, k=k, stop_at_eos=False)
        assert r["tokens"] == ref
        assert 0 <= r["stats"]["acceptance_rate"] <= 1


def test_speculative_with_perfect_draft_accepts_everything():
    target = tiny_qwen(0)
    ids = torch.randint(0, 97, (1, 10), generator=torch.Generator().manual_seed(0))
    r = speculative_greedy(target, target, ids, 25, k=4, stop_at_eos=False)
    assert r["tokens"] == greedy_kv(target, ids, 25, stop_at_eos=False)["tokens"]
    assert r["stats"]["acceptance_rate"] == 1.0
    assert r["stats"]["tokens_per_target_forward"] > 3


def test_speculative_stops_at_eos():
    target, draft = tiny_qwen(0), tiny_qwen(2)
    ids = torch.randint(0, 97, (1, 12), generator=torch.Generator().manual_seed(4))
    ref = greedy_kv(target, ids, 40, stop_at_eos=False)["tokens"]
    eos = {ref[10]}
    stop = ref.index(ref[10]) + 1
    r = speculative_greedy(target, draft, ids, 40, k=3, eos_ids=eos)
    assert r["tokens"] == ref[:stop]


@pytest.mark.parametrize("shape", [(1, 1, 8, 8, 16), (2, 3, 37, 37, 32), (1, 2, 1, 50, 16), (2, 2, 20, 45, 8)])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("blocks", [(4, 4), (16, 8), (64, 64)])
def test_tiled_matches_reference(shape, causal, blocks):
    B, H, Lq, Lk, D = shape
    g = torch.Generator().manual_seed(0)
    q, k, v = torch.randn(B, H, Lq, D, generator=g), torch.randn(B, H, Lk, D, generator=g), torch.randn(B, H, Lk, D, generator=g)
    ref = reference_attention(q, k, v, causal=causal)
    til = tiled_attention(q, k, v, causal=causal, block_q=blocks[0], block_k=blocks[1])
    assert (ref - til).abs().max() < 1e-5
    if Lq == Lk:
        sd = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
        assert (ref - sd).abs().max() < 1e-5


def test_reference_is_stable_for_huge_scores():
    q = torch.full((1, 1, 4, 8), 300.0)
    k = torch.full((1, 1, 4, 8), 300.0)
    v = torch.randn(1, 1, 4, 8)
    for f in (reference_attention, lambda *a, **kw: tiled_attention(*a, block_q=2, block_k=2, **kw)):
        o = f(q, k, v, causal=True)
        assert torch.isfinite(o).all()


def test_model_runs_on_tiled_attention():
    m = tiny_qwen(0)
    ids = torch.randint(0, 97, (1, 40), generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        base = m(input_ids=ids).logits
        with use_attention_impl("tiled", 8, 8) as ctx:
            til = m(input_ids=ids).logits
            toks = greedy_kv(m, ids, 15, stop_at_eos=False)["tokens"]
    assert ctx.calls > 0 and (base - til).abs().max() < 1e-4
    assert toks == greedy_kv(m, ids, 15, stop_at_eos=False)["tokens"]


@pytest.mark.parametrize("k", [1, 3, 6])
def test_prompt_lookup_speculative_equals_greedy(k):
    target = tiny_qwen(0, layers=3)
    for seed in range(3):
        base = torch.randint(0, 97, (1, 12), generator=torch.Generator().manual_seed(seed))
        ids = torch.cat([base, base, base[:, :5]], dim=1)          # repetitive prompt -> lookups hit
        ref = greedy_kv(target, ids, 30, stop_at_eos=False)["tokens"]
        r = speculative_greedy(target, "prompt_lookup", ids, 30, k=k, stop_at_eos=False)
        assert r["tokens"] == ref
        assert r["stats"]["drafter"] == "prompt_lookup"


def test_prompt_lookup_draft_proposals():
    from trialmatch.speculative import prompt_lookup_draft
    S = [1, 2, 3, 4, 5, 9, 2, 3]
    assert prompt_lookup_draft(S, 3) == [4, 5, 9]                  # "2 3" seen before -> "4 5 9"
    assert prompt_lookup_draft([7, 8, 9], 2) == []                 # no earlier match


def test_prefix_cache_reuse_equals_full_prefill():
    from trialmatch.kv_decode import prefill_prefix
    m = tiny_qwen(0)
    prefix = torch.randint(0, 97, (1, 40), generator=torch.Generator().manual_seed(2))
    cache = prefill_prefix(m, prefix)
    for seed in range(3):
        suffix = torch.randint(0, 97, (1, 5 + seed), generator=torch.Generator().manual_seed(10 + seed))
        ids = torch.cat([prefix, suffix], dim=1)
        full = greedy_kv(m, ids, 20, stop_at_eos=False)["tokens"]
        reused = greedy_kv(m, ids, 20, stop_at_eos=False, prefix_cache=cache, prefix_len=40)
        assert reused["tokens"] == full and reused["prefilled_tokens"] == 5 + seed
        assert cache.get_seq_length() == 40                         # handed back unchanged


def test_cost_model_formula():
    from trialmatch.speculative import expected_tokens_per_iter, predicted_speedup
    assert expected_tokens_per_iter(1.0, 4) == 5 and abs(expected_tokens_per_iter(0.0, 4) - 1) < 1e-12
    assert predicted_speedup(0.85, 4, 0.75, 1.0) < 1 < predicted_speedup(0.85, 4, 0.0, 1.1)
