"""Download the dataset, drop GPT-4 columns, build the fixed 37/8/8 patient split.

Usage:  python scripts/prepare_data.py [--config configs/default.yaml]
Writes: data/processed/{train,val,test}.jsonl and data/splits/split_manifest.json
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trialmatch.data import load_raw, prepare  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ds = cfg["dataset"]

    df = load_raw(ds["hf_name"], ds["hf_split"])
    print(f"raw rows: {len(df)}  columns: {list(df.columns)}")
    manifest = prepare(df, ROOT / cfg["paths"]["processed_dir"], ROOT / cfg["paths"]["manifest"],
                       seed=ds["seed"], n_train=ds["n_train_patients"], n_val=ds["n_val_patients"],
                       n_test=ds["n_test_patients"], source=ds["hf_name"])

    print("\nleakage checks:", json.dumps(manifest["leakage_checks"]))
    print("dropped rows:", json.dumps(manifest["rows_dropped"]))
    print(f"\n{'split':<6}{'patients':>9}{'annots':>8}{'incl':>6}{'excl':>6}{'empty_ev':>9}")
    for k, v in manifest["splits"].items():
        print(f"{k:<6}{v['n_patients']:>9}{v['n_annotations']:>8}{v['n_inclusion']:>6}"
              f"{v['n_exclusion']:>6}{v['n_empty_gold_evidence']:>9}")
    print("\nlabel counts per split:")
    for k, v in manifest["splits"].items():
        print(f"  {k}: {v['label_counts']}")
    print(f"\nmanifest -> {cfg['paths']['manifest']}")


if __name__ == "__main__":
    main()
