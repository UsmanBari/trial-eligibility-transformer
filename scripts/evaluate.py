"""Criterion-level evaluation (classification + evidence) with deterministic greedy decoding.

Examples
  # zero-shot baseline on validation
  python scripts/evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --split val --tag baseline_qwen15
  # fine-tuned adapter on validation
  python scripts/evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/lora --split val --tag ft_qwen15
  # FINAL held-out test (only once, at the very end)
  python scripts/evaluate.py ... --split test --final-test

Outputs in results/<tag>/: predictions.jsonl, metrics.json, run_info.json
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trialmatch.data import load_split  # noqa: E402
from trialmatch.metrics import full_report, summary_line  # noqa: E402
from trialmatch.modeling import GREEDY_KWARGS, load_model, load_tokenizer, model_card  # noqa: E402
from trialmatch.prompting import build_prompt, parse_output  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--model", default=None, help="HF model id (default: config target)")
    ap.add_argument("--adapter", default=None, help="path or HF id of a LoRA adapter")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--final-test", action="store_true", help="required to touch the test split")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N records only")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.split == "test":
        if not args.final_test:
            sys.exit("Refusing to evaluate on the TEST split without --final-test "
                     "(test cases must stay untouched until final evaluation).")
        log = ROOT / cfg["paths"]["results_dir"] / "TEST_ACCESS_LOG.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write(f"{dt.datetime.now().isoformat()}  tag={args.tag}  model={args.model}  adapter={args.adapter}\n")

    model_name = args.model or cfg["model"]["target"]
    gen = cfg["generation"]
    bs = args.batch_size or gen["batch_size"]
    records = load_split(ROOT / cfg["paths"]["processed_dir"], args.split)
    if args.limit:
        records = records[: args.limit]

    tok = load_tokenizer(model_name)
    model = load_model(model_name, args.adapter)
    device = next(model.parameters()).device

    prompts, infos = zip(*[build_prompt(r, tok, cfg["model"]["max_input_tokens"]) for r in records])
    order = sorted(range(len(records)), key=lambda i: infos[i]["prompt_tokens"])  # length-bucketed batches
    outputs = [None] * len(records)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for s in tqdm(range(0, len(order), bs), desc=f"{args.tag}:{args.split}"):
        idx = order[s: s + bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False, return_token_type_ids=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=gen["max_new_tokens"],
                                 pad_token_id=tok.pad_token_id, **GREEDY_KWARGS)
        new = out[:, enc["input_ids"].shape[1]:]
        for j, i in enumerate(idx):
            outputs[i] = tok.decode(new[j], skip_special_tokens=True)
    elapsed = time.perf_counter() - t0

    rows = []
    for r, info, text in zip(records, infos, outputs):
        valid = {s["id"] for s in r["sentences"]}
        p = parse_output(text, r["criterion_type"], valid)
        rows.append({
            "annotation_id": r["annotation_id"], "patient_id": r["patient_id"], "trial_id": r["trial_id"],
            "criterion_type": r["criterion_type"], "criterion_text": r["criterion_text"],
            "gold_label": r["gold_label"], "pred_label": p["label"],
            "gold_evidence": r["gold_evidence"], "pred_evidence": p["evidence"],
            "parse_ok": p["parse_ok"], "hallucinated_ids": p["hallucinated_ids"],
            "raw_output": text, **info,
        })

    report = full_report(rows)
    out_dir = ROOT / cfg["paths"]["results_dir"] / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    json.dump(report, open(out_dir / "metrics.json", "w"), indent=2)

    run_info = {
        "tag": args.tag, "split": args.split, "n_records": len(rows), "model": model_card(model, model_name),
        "adapter": args.adapter,
        "decoding": {"strategy": "greedy", "do_sample": False, "repetition_penalty": 1.0, "max_new_tokens": gen["max_new_tokens"],
                     "batch_size": bs, "padding_side": "left"},
        "max_input_tokens": cfg["model"]["max_input_tokens"],
        "truncation": {
            "n_truncated": sum(1 for r in rows if r["dropped_sentence_ids"]),
            "n_gold_evidence_lost": sum(1 for r in rows if r["gold_evidence_lost"]),
            "max_prompt_tokens": max(r["prompt_tokens"] for r in rows),
            "mean_prompt_tokens": sum(r["prompt_tokens"] for r in rows) / len(rows),
        },
        "parse_failures": sum(1 for r in rows if not r["parse_ok"]),
        "rows_with_hallucinated_ids": sum(1 for r in rows if r["hallucinated_ids"]),
        "wall_time_s": elapsed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "peak_gpu_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "timestamp": dt.datetime.now().isoformat(),
    }
    json.dump(run_info, open(out_dir / "run_info.json", "w"), indent=2)
    print("\n" + summary_line(report))
    print(f"inclusion acc={report['inclusion']['classification']['accuracy']:.3f}  "
          f"exclusion acc={report['exclusion']['classification']['accuracy']:.3f}")
    print(f"truncation: {run_info['truncation']}  parse_failures={run_info['parse_failures']}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
