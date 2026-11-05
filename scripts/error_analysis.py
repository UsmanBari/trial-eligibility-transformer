"""Categorise errors and dump representative cases (with the actual note sentences) to markdown.

Usage:  python scripts/error_analysis.py results/<tag> [--per-category 2]
Writes: results/<tag>/error_analysis.md and error_categories.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trialmatch.data import load_split  # noqa: E402

NEI, NA = "not enough information", "not applicable"
CATEGORIES = {
    "inclusion_flip": "Inclusion decision reversed (included <-> not included)",
    "exclusion_flip": "Exclusion decision reversed (excluded <-> not excluded)",
    "overconfident": "Model decided although the expert said not enough information "
                     "(absent evidence treated as proof)",
    "missed_information": "Model said not enough information although the note does decide it",
    "applicability": "Confusion involving not applicable",
    "invalid_output": "Unparseable output or label not allowed for the criterion type",
    "evidence_missing": "Correct label, but no sentence cited although the expert cited some",
    "evidence_wrong": "Correct label, cited sentences do not overlap the expert's",
    "evidence_hallucinated": "Cited sentence IDs that do not exist in the note",
}


def categorise(r):
    g, p = r["gold_label"], r["pred_label"]
    cats = []
    if p == "invalid":
        cats.append("invalid_output")
    elif g != p:
        if {g, p} == {"included", "not included"}:
            cats.append("inclusion_flip")
        elif {g, p} == {"excluded", "not excluded"}:
            cats.append("exclusion_flip")
        elif g == NEI:
            cats.append("overconfident")
        elif p == NEI:
            cats.append("missed_information")
        if NA in (g, p):
            cats.append("applicability")
    else:
        G, P = set(r["gold_evidence"]), set(r["pred_evidence"])
        if G and not P:
            cats.append("evidence_missing")
        elif G and P and not (G & P):
            cats.append("evidence_wrong")
    if r.get("hallucinated_ids"):
        cats.append("evidence_hallucinated")
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir")
    ap.add_argument("--per-category", type=int, default=2)
    a = ap.parse_args()
    d = Path(a.result_dir)
    info = json.load(open(d / "run_info.json"))
    sents = {r["annotation_id"]: {s["id"]: s["text"] for s in r["sentences"]}
             for r in load_split(ROOT / "data/processed", info["split"])}
    rows = [json.loads(l) for l in open(d / "predictions.jsonl")]

    by_cat = defaultdict(list)
    for r in rows:
        for c in categorise(r):
            by_cat[c].append(r)
    counts = {c: len(by_cat[c]) for c in CATEGORIES}
    json.dump({"n_records": len(rows), "counts": counts,
               "annotation_ids": {c: [r["annotation_id"] for r in by_cat[c]] for c in CATEGORIES}},
              open(d / "error_categories.json", "w"), indent=2)

    def quote(aid, ids):
        return "\n".join(f"  - [{i}] {sents[aid].get(i, '(not in note)')}" for i in ids) or "  - (none)"

    lines = [f"# Error analysis: `{d.name}` ({info['split']} split, {len(rows)} criteria)\n",
             "| Category | Count |", "|---|---|"]
    lines += [f"| {CATEGORIES[c]} | {counts[c]} |" for c in CATEGORIES]
    n_shown = 0
    for c, desc in CATEGORIES.items():
        if not by_cat[c]:
            continue
        lines.append(f"\n## {desc}\n")
        for r in sorted(by_cat[c], key=lambda x: x["annotation_id"])[: a.per_category]:
            n_shown += 1
            aid = r["annotation_id"]
            lines += [f"### Case {n_shown}: annotation {aid} (patient {r['patient_id']}, {r['trial_id']}, "
                      f"{r['criterion_type']})",
                      f"**Criterion:** {r['criterion_text']}  ",
                      f"**Expert:** `{r['gold_label']}`, evidence {r['gold_evidence']}  ",
                      f"**Model:** `{r['pred_label']}`, evidence {r['pred_evidence']}  ",
                      f"**Raw output:** `{r['raw_output'].strip()}`\n",
                      "Expert evidence sentences:", quote(aid, r["gold_evidence"]),
                      "Model evidence sentences:", quote(aid, r["pred_evidence"]), ""]
    (d / "error_analysis.md").write_text("\n".join(lines))
    print(json.dumps(counts, indent=1))
    print(f"{n_shown} cases written -> {d / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
