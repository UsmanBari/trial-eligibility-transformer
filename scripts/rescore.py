"""Recompute metrics.json from an existing predictions.jsonl (no model needed).

Usage:  python scripts/rescore.py results/<tag> [results/<tag2> ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trialmatch.metrics import full_report, summary_line  # noqa: E402

for d in sys.argv[1:]:
    d = Path(d)
    rows = [json.loads(l) for l in open(d / "predictions.jsonl")]
    rep = full_report(rows)
    json.dump(rep, open(d / "metrics.json", "w"), indent=2)
    print(f"{d.name:32s} {summary_line(rep)}")
