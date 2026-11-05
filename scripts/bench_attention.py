"""Task 4: numerical validation and speed/memory comparison of attention implementations.

Correctness: many shapes (incl. lengths not divisible by the block size, Lq < Lk decode-style,
causal and non-causal), several block sizes, fp32 and fp16; tiled vs reference and both vs
PyTorch's fused SDPA. Reports max / mean absolute error against tolerances.
End-to-end: the full fine-tuned model run with every attention call routed through our
tiled kernel; logits and greedy tokens compared with the normal model.
Performance: reference vs tiled vs torch SDPA across sequence lengths (median of repeats,
synchronised), plus peak extra memory.

Usage:  python scripts/bench_attention.py [--adapter <adapter>] [--skip-model]
Writes: results/bench/attention.json
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trialmatch.attention import reference_attention, tiled_attention, use_attention_impl  # noqa: E402

TOL = {torch.float32: 1e-4, torch.float16: 5e-3}


def sdpa(q, k, v, causal):
    Lq, Lk = q.shape[-2], k.shape[-2]
    if causal and Lq != Lk:     # SDPA's is_causal is top-left aligned; build bottom-right mask
        keep = torch.arange(Lk, device=q.device)[None] <= (torch.arange(Lq, device=q.device)[:, None] + Lk - Lq)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def err(a, b):
    d = (a.float() - b.float()).abs()
    return float(d.max()), float(d.mean())


def correctness(device):
    shapes = [(1, 1, 8, 8, 16), (2, 4, 37, 37, 64), (1, 12, 128, 128, 128), (1, 12, 1, 300, 128),
              (2, 4, 100, 250, 64), (1, 12, 513, 513, 128)]
    blocks = [(16, 16), (64, 64), (128, 32)]
    dtypes = [torch.float32] + ([torch.float16] if device.type == "cuda" else [])
    rows = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for (B, H, Lq, Lk, D) in shapes:
        for dt in dtypes:
            q = torch.randn(B, H, Lq, D, generator=g).to(device, dt)
            k = torch.randn(B, H, Lk, D, generator=g).to(device, dt)
            v = torch.randn(B, H, Lk, D, generator=g).to(device, dt)
            for causal in (True, False):
                ref = reference_attention(q, k, v, causal=causal)
                fused = sdpa(q, k, v, causal)
                for bq, bk in blocks:
                    til, stats = tiled_attention(q, k, v, causal=causal, block_q=bq, block_k=bk, return_stats=True)
                    e_ref, e_fused, e_ref_fused = err(til, ref), err(til, fused), err(ref, fused)
                    rows.append({"shape_BHLqLkD": [B, H, Lq, Lk, D], "dtype": str(dt).split(".")[-1],
                                 "causal": causal, "block_q": bq, "block_k": bk,
                                 "tiled_vs_ref_max": e_ref[0], "tiled_vs_ref_mean": e_ref[1],
                                 "tiled_vs_sdpa_max": e_fused[0], "ref_vs_sdpa_max": e_ref_fused[0],
                                 "pass": e_ref[0] <= TOL[dt] and e_fused[0] <= 2 * TOL[dt], **stats})
    n_pass = sum(r["pass"] for r in rows)
    print(f"[correctness] {n_pass}/{len(rows)} configurations within tolerance; "
          f"worst tiled-vs-ref fp32 = {max(r['tiled_vs_ref_max'] for r in rows if r['dtype']=='float32'):.2e}"
          + (f", fp16 = {max(r['tiled_vs_ref_max'] for r in rows if r['dtype']=='float16'):.2e}"
             if device.type == "cuda" else ""))
    return {"tolerance_abs": {"float32": TOL[torch.float32], "float16": TOL[torch.float16]},
            "n_pass": n_pass, "n": len(rows), "rows": rows}


def timed(fn, device, repeats):
    fn()                                                             # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter(); fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    mem = (torch.cuda.max_memory_allocated() - base) / 2**20 if device.type == "cuda" else None
    return 1e3 * st.median(ts), mem


def performance(device, lens, repeats):
    dt = torch.float16 if device.type == "cuda" else torch.float32
    rows = []
    for L in lens:
        q, k, v = (torch.randn(1, 12, L, 128, device=device, dtype=dt) for _ in range(3))
        row = {"seq_len": L, "B": 1, "H": 12, "D": 128, "dtype": str(dt).split(".")[-1], "causal": True,
               "score_matrix_mb_fp32": 12 * L * L * 4 / 2**20}
        for name, fn in (("reference", lambda: reference_attention(q, k, v, causal=True)),
                         ("tiled_128", lambda: tiled_attention(q, k, v, causal=True, block_q=128, block_k=128)),
                         ("torch_sdpa_fused", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))):
            try:
                ms, mem = timed(fn, device, repeats)
                row[name] = {"ms": ms, "peak_extra_mem_mb": mem}
            except torch.OutOfMemoryError:
                row[name] = {"ms": None, "peak_extra_mem_mb": None, "oom": True}
                torch.cuda.empty_cache()
        print(f"[perf] L={L:5d}  " + "  ".join(f"{n}={row[n]['ms']:.2f}ms/{row[n]['peak_extra_mem_mb'] or 0:.0f}MB"
                                               if row[n]["ms"] is not None else f"{n}=OOM"
                                               for n in ("reference", "tiled_128", "torch_sdpa_fused")))
        rows.append(row)
    return rows


def model_end_to_end(adapter, device, config):
    import yaml
    from trialmatch.data import load_split
    from trialmatch.kv_decode import eos_ids_of, greedy_kv
    from trialmatch.modeling import load_model, load_tokenizer
    from trialmatch.prompting import build_prompt
    cfg = yaml.safe_load(open(config))
    name = cfg["model"]["target"]
    tok = load_tokenizer(name)
    res = {}
    for dt in ([torch.float32, torch.float16] if device.type == "cuda" else [torch.float32]):
        model = load_model(name, adapter, dtype=dt)
        eos = eos_ids_of(model, tok)
        cases = []
        for r in load_split(ROOT / cfg["paths"]["processed_dir"], "val")[:5]:
            prompt, _ = build_prompt(r, tok, cfg["model"]["max_input_tokens"])
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            with torch.no_grad():
                base_logits = model(input_ids=ids, use_cache=False).logits.float()
                with use_attention_impl("tiled", 64, 64) as ctx:
                    tiled_logits = model(input_ids=ids, use_cache=False).logits.float()
                    tiled_tokens = greedy_kv(model, ids, 48, eos)["tokens"]
                base_tokens = greedy_kv(model, ids, 48, eos)["tokens"]
            cases.append({"annotation_id": r["annotation_id"], "prompt_tokens": ids.shape[1],
                          "max_abs_logit_diff": float((base_logits - tiled_logits).abs().max()),
                          "greedy_tokens_equal": base_tokens == tiled_tokens,
                          "tiled_attention_calls": ctx.calls})
        res[str(dt).split(".")[-1]] = {"cases": cases,
                                      "tokens_equal": sum(c["greedy_tokens_equal"] for c in cases),
                                      "n": len(cases)}
        print(f"[model {dt}] greedy tokens equal with tiled attention in every layer: "
              f"{res[str(dt).split('.')[-1]]['tokens_equal']}/{len(cases)}; max logit diff "
              f"{max(c['max_abs_logit_diff'] for c in cases):.2e}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--lens", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--skip-model", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "torch": torch.__version__, "correctness": correctness(device),
           "performance": performance(device, a.lens if device.type == "cuda" else [128, 256, 512], a.repeats)}
    if not a.skip_model:
        out["model_end_to_end"] = model_end_to_end(a.adapter, device, a.config)
    d = ROOT / "results" / "bench"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / "attention.json", "w"), indent=2)
    print("saved", d / "attention.json")


if __name__ == "__main__":
    main()
