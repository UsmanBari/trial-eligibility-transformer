"""Pick the best epoch checkpoint by VALIDATION metrics and copy its adapter to outputs/<name>_best.

Selection rule (fixed before looking at test): highest val macro-F1; ties broken by
val evidence micro-F1, then by the earlier epoch.

Usage:  python scripts/select_checkpoint.py --run outputs/lora_qwen25_1p5b --tags ft_ep1 ft_ep2 ft_ep3
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="training output dir containing checkpoint-*")
ap.add_argument("--tags", nargs="+", required=True, help="result tags of the val runs, in epoch order")
ap.add_argument("--selection-file", default="checkpoint_selection.json")
args = ap.parse_args()

run = ROOT / args.run
ckpts = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
assert len(ckpts) == len(args.tags), f"{len(ckpts)} checkpoints but {len(args.tags)} tags"

rows = []
for ep, (ck, tag) in enumerate(zip(ckpts, args.tags), start=1):
    m = json.load(open(ROOT / "results" / tag / "metrics.json"))["overall"]
    rows.append({"epoch": ep, "checkpoint": ck.name, "tag": tag,
                 "val_accuracy": m["classification"]["accuracy"],
                 "val_macro_f1": m["classification"]["macro_f1"],
                 "val_evidence_micro_f1": m["evidence"]["micro"]["f1"]})
best = max(rows, key=lambda r: (r["val_macro_f1"], r["val_evidence_micro_f1"], -r["epoch"]))

dst = ROOT / "outputs" / (run.name + "_best")
if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True)
for f in ["adapter_config.json", "adapter_model.safetensors"]:
    shutil.copy(run / best["checkpoint"] / f, dst / f)
sel = {"rule": "max val macro-F1, tie -> val evidence micro-F1, tie -> earlier epoch",
       "candidates": rows, "selected": best, "adapter_dir": str(dst.relative_to(ROOT))}
(ROOT / "results").mkdir(exist_ok=True)
json.dump(sel, open(ROOT / "results" / args.selection_file, "w"), indent=2)
for r in rows:
    flag = "  <- selected" if r is best else ""
    print(f"epoch {r['epoch']}: acc={r['val_accuracy']:.3f} macroF1={r['val_macro_f1']:.3f} "
          f"evF1={r['val_evidence_micro_f1']:.3f}{flag}")
print(f"best adapter -> {dst}")
