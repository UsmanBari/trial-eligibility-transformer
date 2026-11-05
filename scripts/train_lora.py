"""LoRA fine-tuning of the target model on the TRAIN split (validation loss tracked each epoch).

Usage:  python scripts/train_lora.py [--config configs/default.yaml] [--epochs N] [--lora-r R]
Writes: <output_dir>/checkpoint-*/ (one adapter per epoch), train_log.json, train_config.json
Checkpoint choice is done afterwards by generation-based val macro-F1 (select_checkpoint.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trialmatch.data import load_split  # noqa: E402
from trialmatch.modeling import load_tokenizer, pick_device_dtype  # noqa: E402
from trialmatch.sft import ListDataset, encode_split, make_collator  # noqa: E402


def oversample(records, cap=4):
    """Duplicate minority (criterion_type, label) classes: copies = min(cap, round(sqrt(max/count))).
    Only the TRAIN split is changed; validation/test are never resampled."""
    from collections import Counter
    cnt = Counter((r["criterion_type"], r["gold_label"]) for r in records)
    biggest = max(cnt.values())
    copies = {c: max(1, min(cap, round((biggest / n) ** 0.5))) for c, n in cnt.items()}
    out = [r for r in records for _ in range(copies[(r["criterion_type"], r["gold_label"])])]
    return out, {f"{t}/{l}": [n, copies[(t, l)]] for (t, l), n in sorted(cnt.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--model", default=None, help="base model (default: config target); e.g. the draft model")
    ap.add_argument("--balance", choices=["none", "sqrt"], default="none",
                    help="sqrt: oversample each (criterion type, label) class by round(sqrt(max/count)), capped at 4")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ft = cfg["finetune"]
    if args.epochs is not None: ft["epochs"] = args.epochs
    if args.lora_r is not None: ft["lora_r"], ft["lora_alpha"] = args.lora_r, 2 * args.lora_r
    if args.lr is not None: ft["learning_rate"] = args.lr
    out_dir = ROOT / (args.output_dir or ft["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(ft["seed"])

    name = args.model or cfg["model"]["target"]
    tok = load_tokenizer(name)
    tok.padding_side = "right"                       # right padding for training
    proc = ROOT / cfg["paths"]["processed_dir"]
    train_recs = load_split(proc, "train")
    balance_report = None
    if args.balance == "sqrt":
        train_recs, balance_report = oversample(train_recs)
        print("class balancing (type, label): count -> copies:", balance_report)
    train_items, train_skip = encode_split(train_recs, tok, ft["max_seq_len"])
    val_items, val_skip = encode_split(load_split(proc, "val"), tok, ft["max_seq_len"])
    lens = [len(x["input_ids"]) for x in train_items]
    print(f"train examples: {len(train_items)} (skipped {len(train_skip)} over max_seq_len)  "
          f"val: {len(val_items)}  max len: {max(lens)}  mean len: {sum(lens)/len(lens):.0f}  "
          f"mean target tokens: {sum(x['n_target'] for x in train_items)/len(train_items):.1f}")

    device, dtype = pick_device_dtype()
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="sdpa")
    model.config.use_cache = False                   # no KV cache during training
    if ft.get("gradient_checkpointing"):
        # recompute activations in the backward pass: less memory, ~30% slower
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=ft["lora_r"], lora_alpha=ft["lora_alpha"], lora_dropout=ft["lora_dropout"],
                      target_modules=ft["target_modules"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    for p in model.parameters():                     # fp16 base + fp32 trainable LoRA weights
        if p.requires_grad:
            p.data = p.data.float()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train:,} / {n_all:,} ({100 * n_train / n_all:.2f}%)")

    targs = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=ft["epochs"], learning_rate=ft["learning_rate"],
        per_device_train_batch_size=ft["per_device_batch_size"], per_device_eval_batch_size=ft["per_device_batch_size"],
        gradient_accumulation_steps=ft["grad_accum"], warmup_ratio=ft["warmup_ratio"],
        lr_scheduler_type=ft["lr_scheduler"], weight_decay=ft["weight_decay"],
        logging_steps=ft["logging_steps"], eval_strategy="epoch", save_strategy="epoch",
        fp16=(dtype == torch.float16), bf16=(dtype == torch.bfloat16),
        seed=ft["seed"], data_seed=ft["seed"], report_to="none", remove_unused_columns=False,
        dataloader_num_workers=0, eval_accumulation_steps=1,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ListDataset(train_items),
                      eval_dataset=ListDataset(val_items), data_collator=make_collator(tok.pad_token_id))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - t0

    json.dump(trainer.state.log_history, open(out_dir / "train_log.json", "w"), indent=2)
    json.dump({"base_model": name, "finetune": ft, "balance": args.balance, "balance_report": balance_report, "n_train_examples": len(train_items),
               "n_val_examples": len(val_items), "skipped_train": train_skip, "skipped_val": val_skip,
               "trainable_params": n_train, "total_params": n_all, "train_time_s": train_time,
               "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
               "dtype": str(dtype)},
              open(out_dir / "train_config.json", "w"), indent=2)
    print(f"done in {train_time / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
