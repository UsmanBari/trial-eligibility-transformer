"""Task 5: speculative decoding vs ordinary greedy decoding of the target (same prompts).

For every draft model and draft length k: token equality with target greedy (our KV
decoder), acceptance rate, iterations, end-to-end latency, tokens/s, speedup and peak
memory. Two prompt sets:
  * task:    the real eligibility prompts (short JSON answers, natural EOS stop)
  * summary: "summarise this patient note" prompts (longer free-text answers)

Also measures the cost model: one draft step / one target step (c) and the cost of
verifying k+1 tokens at once (v_k), then compares predicted speedup E(a,k)/(v_k + k*c)
with the measured one, so any slowdown is explained with numbers.

Usage:  python scripts/bench_spec.py --adapter <target adapter> \
            --drafts lookup=prompt_lookup base=Qwen/Qwen2.5-0.5B-Instruct \
                     tuned=Qwen/Qwen2.5-0.5B-Instruct@<draft adapter> --ks 2 4 6 8 --dtype float16
Writes: results/bench/spec_<dtype>.json
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
from trialmatch.kv_decode import eos_ids_of, greedy_kv  # noqa: E402
from trialmatch.modeling import load_model, load_tokenizer, model_card  # noqa: E402
from trialmatch.prompting import build_prompt, format_note  # noqa: E402
from trialmatch.speculative import (PROMPT_LOOKUP, check_draft_compatible, expected_tokens_per_iter,  # noqa: E402
                                    predicted_speedup, speculative_greedy, step_costs_ms)


def mem_reset(device):
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()


def mem_peak(device):
    return torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None


def build_prompts(tok, cfg, n_task, n_summary):
    recs = load_split(ROOT / cfg["paths"]["processed_dir"], "val")
    task = [build_prompt(r, tok, cfg["model"]["max_input_tokens"])[0] for r in recs[:n_task]]
    seen, notes = set(), []
    for r in recs:
        if r["patient_id"] not in seen:
            seen.add(r["patient_id"])
            notes.append(r["sentences"])
    summ = [tok.apply_chat_template(
        [{"role": "user", "content": "Summarise the key clinical facts of this patient note in 4-5 sentences:\n"
                                     + format_note(s)}], tokenize=False, add_generation_prompt=True)
        for s in notes[:n_summary]]
    return {"task": (task, 64), "summary": (summ, 160)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--adapter", default=None, help="target adapter")
    ap.add_argument("--drafts", nargs="+", default=["base=Qwen/Qwen2.5-0.5B-Instruct"],
                    help="name=hf_id, name=hf_id@adapter_path, or name=prompt_lookup")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--n-task", type=int, default=20)
    ap.add_argument("--n-summary", type=int, default=6)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    dt = getattr(torch, a.dtype)
    tname = cfg["model"]["target"]
    tok = load_tokenizer(tname)
    target = load_model(tname, a.adapter, dtype=dt)
    device = next(target.parameters()).device
    eos = eos_ids_of(target, tok)
    prompts = build_prompts(tok, cfg, a.n_task, a.n_summary)
    enc = {s: [tok(p, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device) for p in ps]
           for s, (ps, _) in prompts.items()}

    out = {"target": model_card(target, tname), "target_adapter": a.adapter, "dtype": a.dtype,
           "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "baseline": {}, "drafts": {}}

    # ---- cost model inputs: target single-step and (k+1)-token verify latency at a typical context
    ctx = int(sum(x.shape[1] for x in enc["task"]) / len(enc["task"]))
    t_costs = step_costs_ms(target, ctx, [1] + [k + 1 for k in a.ks])
    out["cost_model"] = {"context_len": ctx, "target_step_ms": t_costs[1],
                         "verify_ms": {k: t_costs[k + 1] for k in a.ks},
                         "v_k": {k: t_costs[k + 1] / t_costs[1] for k in a.ks}}
    print(f"[cost] target 1-token step {t_costs[1]:.1f} ms; verify k+1 tokens: "
          + ", ".join(f"k={k}: {t_costs[k + 1]:.1f} ms (x{t_costs[k + 1] / t_costs[1]:.2f})" for k in a.ks))

    # ---- baseline: ordinary greedy decoding of the target (KV cached), warm-up first
    greedy_kv(target, enc["task"][0], 8, eos)
    base_tokens = {}
    for s, ids_list in enc.items():
        max_new = prompts[s][1]
        mem_reset(device)
        runs = [greedy_kv(target, ids, max_new, eos) for ids in ids_list]
        base_tokens[s] = [r["tokens"] for r in runs]
        n_tok = sum(len(r["tokens"]) for r in runs)
        tot = sum(r["total_s"] for r in runs)
        out["baseline"][s] = {"n_prompts": len(runs), "new_tokens": n_tok, "total_s": tot,
                              "tokens_per_s": n_tok / tot, "mean_latency_ms": 1e3 * tot / len(runs),
                              "peak_mem_mb": mem_peak(device), "max_new_tokens": max_new}
        print(f"[baseline] {s:8s} {n_tok:5d} tokens  {n_tok / tot:6.1f} tok/s  "
              f"latency {1e3 * tot / len(runs):7.1f} ms/prompt")

    for spec in a.drafts:
        dname, rest = spec.split("=", 1)
        if rest == PROMPT_LOOKUP:
            draft, c_ratio = PROMPT_LOOKUP, 0.0
            res = {"draft_model": PROMPT_LOOKUP, "draft_adapter": None,
                   "compatibility": {"compatible": True, "note": "drafts are copied target-vocabulary tokens"},
                   "draft_step_ms": 0.0, "c": 0.0, "by_k": {}}
        else:
            did, dadapter = (rest.split("@", 1) + [None])[:2]
            draft = load_model(did, dadapter, dtype=dt)
            dtok = load_tokenizer(did)
            compat = check_draft_compatible(tok, dtok, target, draft)
            assert compat["compatible"], f"draft {dname} is not tokenizer-compatible: {compat}"
            d_ms = step_costs_ms(draft, ctx, [1])[1]
            c_ratio = d_ms / t_costs[1]
            res = {"draft_model": model_card(draft, did), "draft_adapter": dadapter, "compatibility": compat,
                   "draft_step_ms": d_ms, "c": c_ratio, "by_k": {}}
            print(f"[cost] {dname}: draft step {d_ms:.1f} ms = {c_ratio:.2f} x target step")
        speculative_greedy(target, draft, enc["task"][0], 8, k=2, eos_ids=eos)      # warm-up
        for k in a.ks:
            res["by_k"][k] = {}
            for s, ids_list in enc.items():
                max_new = prompts[s][1]
                mem_reset(device)
                runs = [speculative_greedy(target, draft, ids, max_new, k=k, eos_ids=eos) for ids in ids_list]
                eq = [r["tokens"] == b for r, b in zip(runs, base_tokens[s])]
                first_diff = []
                for r, b in zip(runs, base_tokens[s]):
                    fd = next((i for i, (x, y) in enumerate(zip(r["tokens"], b)) if x != y), None)
                    first_diff.append(fd)
                n_tok = sum(len(r["tokens"]) for r in runs)
                tot = sum(r["total_s"] for r in runs)
                drafted = sum(r["stats"]["drafted"] for r in runs)
                accepted = sum(r["stats"]["accepted"] for r in runs)
                tfw = sum(r["stats"]["target_forwards"] for r in runs)
                row = {"tokens_equal": sum(eq), "n_prompts": len(runs), "first_diff_index": first_diff,
                       "new_tokens": n_tok, "total_s": tot, "tokens_per_s": n_tok / tot,
                       "mean_latency_ms": 1e3 * tot / len(runs),
                       "speedup_vs_greedy": out["baseline"][s]["total_s"] / tot,
                       "acceptance_rate": accepted / drafted if drafted else None,
                       "mean_accepted_per_iter": st.mean(x for r in runs for x in r["stats"]["accepted_per_iter"])
                       if drafted else None,
                       "tokens_per_target_forward": n_tok / tfw, "target_forwards": tfw,
                       "draft_forwards": sum(r["stats"]["draft_forwards"] for r in runs),
                       "peak_mem_mb": mem_peak(device),
                       "empty_drafts": sum(r["stats"]["empty_drafts"] for r in runs)}
                alpha = row["acceptance_rate"]
                v_k = out["cost_model"]["v_k"][k]
                row["predicted"] = {"E_tokens_per_iter": expected_tokens_per_iter(alpha, k),
                                    "speedup_formula": predicted_speedup(alpha, k, c_ratio, v_k) if alpha is not None else None,
                                    "c": c_ratio, "v_k": v_k}
                res["by_k"][k][s] = row
                print(f"[{dname:6s} k={k}] {s:8s} equal {sum(eq)}/{len(runs)}  accept {(row['acceptance_rate'] or 0):.2f}  "
                      f"{row['tokens_per_s']:6.1f} tok/s  speedup x{row['speedup_vs_greedy']:.2f}  "
                      f"tok/target-pass {row['tokens_per_target_forward']:.2f}  "
                      f"predicted x{(row['predicted']['speedup_formula'] or 0):.2f}")
        out["drafts"][dname] = res
        del draft
        if device.type == "cuda":
            torch.cuda.empty_cache()

    d = ROOT / "results" / "bench"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / f"spec_{a.dtype}.json", "w"), indent=2)
    print("saved", d / f"spec_{a.dtype}.json")


if __name__ == "__main__":
    main()
