"""Task 3: correctness and speed of KV-cached vs no-cache greedy decoding (same model, same inputs).

Part A (correctness on the real task): chat prompts from the validation split, natural EOS
  stop; tokens of no-cache, KV-cache and HF generate() must agree.
Part B (benchmark grid): input lengths x output lengths with EOS ignored (exactly N new
  tokens), 1 warm-up + R timed repeats, torch.cuda.synchronize around every timing.
  Reports prefill time, decode time, total, output tokens/s, decode tokens/s, peak extra
  memory, token agreement and, on any mismatch, the logit-level investigation.

Usage:  python scripts/bench_kv.py --adapter <adapter> --dtype float16
        python scripts/bench_kv.py --adapter <adapter> --dtype float32 --no-grid   # exactness check
Writes: results/bench/kv_<dtype>.json
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trialmatch.data import load_split  # noqa: E402
from trialmatch.kv_decode import (eos_ids_of, greedy_kv, greedy_no_cache,  # noqa: E402
                                  kv_cache_bytes_per_token, last_logits_both_ways)
from trialmatch.modeling import GREEDY_KWARGS, load_model, load_tokenizer, model_card  # noqa: E402
from trialmatch.prompting import build_prompt  # noqa: E402


def peak_extra_mb(fn, device):
    if device.type != "cuda":
        return fn(), None
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    r = fn()
    torch.cuda.synchronize()
    return r, (torch.cuda.max_memory_allocated() - base) / 2**20


def first_mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--in-lens", type=int, nargs="+", default=[128, 512, 1024])
    ap.add_argument("--out-lens", type=int, nargs="+", default=[32, 128, 256])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-task", type=int, default=20)
    ap.add_argument("--no-grid", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    name = a.model or cfg["model"]["target"]
    tok = load_tokenizer(name)
    model = load_model(name, a.adapter, dtype=getattr(torch, a.dtype))
    device = next(model.parameters()).device
    eos = eos_ids_of(model, tok)
    out = {"model": model_card(model, name), "adapter": a.adapter, "dtype": a.dtype,
           "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "kv_cache_bytes_per_token": kv_cache_bytes_per_token(model)}

    # ---------------- Part A: task prompts, natural stop
    recs = load_split(ROOT / cfg["paths"]["processed_dir"], "val")[: a.n_task]
    task = []
    for r in recs:
        prompt, _ = build_prompt(r, tok, cfg["model"]["max_input_tokens"])
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        nc = greedy_no_cache(model, ids, 96, eos)
        kv = greedy_kv(model, ids, 96, eos, trace_cache=True)
        hf = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=96,
                            pad_token_id=tok.pad_token_id, **GREEDY_KWARGS)
        hf_toks = hf[0, ids.shape[1]:].tolist()
        task.append({"annotation_id": r["annotation_id"], "prompt_tokens": ids.shape[1], "n_new": len(kv["tokens"]),
                     "nocache_eq_kv": nc["tokens"] == kv["tokens"], "kv_eq_hf_generate": kv["tokens"] == hf_toks,
                     "first_mismatch_nc_kv": first_mismatch(nc["tokens"], kv["tokens"]),
                     "cache_len_start_end": [kv["cache_trace"][0]["seq_len"], kv["cache_trace"][-1]["seq_len"]],
                     "layer0_key_shape_end": kv["cache_trace"][-1]["layer0_key_shape"],
                     "text": tok.decode(kv["tokens"], skip_special_tokens=True)})
    out["task_agreement"] = {"n": len(task), "nocache_eq_kv": sum(t["nocache_eq_kv"] for t in task),
                             "kv_eq_hf_generate": sum(t["kv_eq_hf_generate"] for t in task), "cases": task}
    print(f"[task] no-cache == KV: {out['task_agreement']['nocache_eq_kv']}/{len(task)}   "
          f"KV == HF generate: {out['task_agreement']['kv_eq_hf_generate']}/{len(task)}")

    # ---------------- Part B: length grid, EOS ignored
    if not a.no_grid:
        text = " ".join(s["text"] for r in load_split(ROOT / cfg["paths"]["processed_dir"], "val")
                        for s in r["sentences"])
        all_ids = tok(text, add_special_tokens=False)["input_ids"]
        grid = []
        for L in a.in_lens:
            ids = torch.tensor([all_ids[:L]], device=device)
            for N in a.out_lens:
                row = {"input_len": L, "output_len": N}
                runs = {}
                for mode, fn in (("no_cache", greedy_no_cache), ("kv_cache", greedy_kv)):
                    fn(model, ids, min(N, 8), eos, stop_at_eos=False)          # warm-up
                    rs, mems = [], []
                    for _ in range(a.repeats):
                        r, m = peak_extra_mb(lambda: fn(model, ids, N, eos, stop_at_eos=False), device)
                        rs.append(r); mems.append(m)
                    runs[mode] = rs[0]["tokens"]
                    pre = st.median(r["prefill_s"] for r in rs)
                    dec = st.median(r["decode_s"] for r in rs)
                    tot = st.median(r["total_s"] for r in rs)
                    row[mode] = {"prefill_ms": 1e3 * pre, "decode_ms": 1e3 * dec, "total_ms": 1e3 * tot,
                                 "tokens_per_s": N / tot, "decode_tokens_per_s": (N - 1) / dec if dec > 0 else None,
                                 "peak_extra_mem_mb": max(mems) if mems[0] is not None else None,
                                 "total_ms_all_repeats": [1e3 * r["total_s"] for r in rs]}
                mm = first_mismatch(runs["no_cache"], runs["kv_cache"])
                row["tokens_equal"] = mm is None
                row["first_mismatch_index"] = mm
                if mm is not None:   # investigate: compare logits for the shared prefix
                    prefix = torch.tensor([all_ids[:L] + runs["kv_cache"][:mm]], device=device)
                    row["mismatch_investigation"] = last_logits_both_ways(model, prefix)
                row["speedup_total"] = row["no_cache"]["total_ms"] / row["kv_cache"]["total_ms"]
                print(f"[grid] in={L:5d} out={N:4d}  no-cache {row['no_cache']['total_ms']:8.0f} ms  "
                      f"KV {row['kv_cache']['total_ms']:7.0f} ms  x{row['speedup_total']:.2f}  "
                      f"equal={row['tokens_equal']}" + ("" if mm is None else f" (first diff @ {mm})"))
                grid.append(row)
        out["grid"] = grid

    d = ROOT / "results" / "bench"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / f"kv_{a.dtype}.json", "w"), indent=2)
    print("saved", d / f"kv_{a.dtype}.json")


if __name__ == "__main__":
    main()
