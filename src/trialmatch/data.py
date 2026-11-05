"""Dataset loading, sentence parsing, the fixed patient-grouped split and leakage checks.

Rules from the assignment that this file enforces:
  * only ncbi/TrialGPT-Criterion-Annotations is used;
  * split = sort unique patient IDs -> shuffle with np.random.default_rng(42)
    -> first 37 train, next 8 val, last 8 test;
  * every annotation of a patient lives in exactly one split;
  * GPT-4 predictions/explanations are dropped before anything else sees the data.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import ALL_LABELS, allowed_labels

# Columns produced by an earlier model (GPT-4) or derived from its output.
# They must never become input features or evaluation targets.
LEAKY_COLUMNS = ["gpt4_explanation", "gpt4_sentences", "gpt4_eligibility",
                 "explanation_correctness"]
# The original repo's own train flag; we ignore it in favour of the prescribed split.
IGNORED_COLUMNS = ["training"]
REQUIRED_COLUMNS = ["annotation_id", "patient_id", "note", "trial_id", "criterion_type",
                    "criterion_text", "expert_eligibility", "expert_sentences"]


# --------------------------------------------------------------------------- loading
def load_raw(hf_name: str = "ncbi/TrialGPT-Criterion-Annotations", split: str = "train") -> pd.DataFrame:
    from datasets import load_dataset
    return load_dataset(hf_name, split=split).to_pandas()


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"dataset is missing expected columns {missing}; got {list(df.columns)}")
    return df.drop(columns=[c for c in LEAKY_COLUMNS + IGNORED_COLUMNS if c in df.columns])


# --------------------------------------------------------------------------- note parsing
def split_note_into_sentences(note: str) -> list[dict]:
    """Split a numbered note ("0. ... 1. ... 2. ...") into [{id, text}].

    We scan for the *next expected* number only, so numbers inside the text
    (doses, "2. episodes") are not mistaken for sentence boundaries. The
    sentence ids returned are exactly the ids the expert annotations refer to.
    """
    note = note.strip()
    first = re.match(r"\s*0\.\s+", note)
    if not first:
        return [{"id": 0, "text": note}]
    # Prefer line-start numbering when the note has newlines (much stricter);
    # fall back to "whitespace + number + dot" for single-line notes.
    boundary = r"(?m)^\s*" if "\n" in note else r"(?:\s)"
    starts = [(0, first.start(), first.end())]
    pos, nxt = first.end(), 1
    while True:
        m = re.compile(rf"{boundary}{nxt}\.\s+").search(note, pos)
        if not m:
            break
        starts.append((nxt, m.start(), m.end()))
        pos, nxt = m.end(), nxt + 1
    sents = []
    for i, (sid, _s, text_start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(note)
        sents.append({"id": sid, "text": note[text_start:end].strip()})
    return sents


def parse_id_list(value) -> list[int]:
    """'[0, 1, 2]' / '[]' / list / NaN -> sorted unique int list."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        items = list(value)
    else:
        s = str(value).strip()
        if not s:
            return []
        try:
            items = json.loads(s)
        except json.JSONDecodeError:
            items = re.findall(r"-?\d+", s)
    return sorted({int(x) for x in items})


# --------------------------------------------------------------------------- split
def make_patient_split(patient_ids, seed=42, n_train=37, n_val=8, n_test=8) -> dict:
    ids = np.array(sorted(set(patient_ids)))
    if len(ids) != n_train + n_val + n_test:
        raise ValueError(f"expected {n_train + n_val + n_test} patients, found {len(ids)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)                    # in-place shuffle of the sorted id array
    ids = ids.tolist()
    return {"train": ids[:n_train],
            "val": ids[n_train:n_train + n_val],
            "test": ids[n_train + n_val:]}


def check_no_leakage(split: dict, df: pd.DataFrame) -> dict:
    """Hard asserts; returns a small report that goes into the manifest."""
    sets = {k: set(v) for k, v in split.items()}
    for a in sets:
        for b in sets:
            if a < b:
                overlap = sets[a] & sets[b]
                assert not overlap, f"patient leakage between {a} and {b}: {overlap}"
    assigned = set().union(*sets.values())
    assert assigned == set(df["patient_id"]), "some patients not assigned to any split"
    # Same note text under two different patient ids would also be leakage.
    note_owner = {}
    for pid, note in df[["patient_id", "note"]].drop_duplicates().itertuples(index=False):
        h = hashlib.sha1(note.encode()).hexdigest()
        note_owner.setdefault(h, set()).add(pid)
    split_of = {pid: k for k, v in split.items() for pid in v}
    for pids in note_owner.values():
        assert len({split_of[p] for p in pids}) == 1, f"identical note in two splits: {pids}"
    # Every annotation in exactly one split.
    counts = df["patient_id"].map(split_of).value_counts().to_dict()
    assert sum(counts.values()) == len(df)
    return {"patient_overlap": 0, "duplicate_note_across_splits": 0,
            "annotations_assigned": int(sum(counts.values())), "annotations_total": int(len(df))}


# --------------------------------------------------------------------------- records
def build_records(df: pd.DataFrame) -> tuple[list[dict], dict]:
    """Turn rows into clean model-ready records. Returns (records, drop_report)."""
    records, dropped = [], Counter()
    sent_cache = {}
    for row in df.itertuples(index=False):
        ctext = row.criterion_text
        if ctext is None or (isinstance(ctext, float) and np.isnan(ctext)) or not str(ctext).strip():
            dropped["empty_criterion_text"] += 1
            continue
        label = str(row.expert_eligibility).strip().lower()
        ctype = str(row.criterion_type).strip().lower()
        if label not in allowed_labels(ctype):
            dropped[f"label_not_valid_for_type:{ctype}:{label}"] += 1
            continue
        if row.patient_id not in sent_cache:
            sent_cache[row.patient_id] = split_note_into_sentences(row.note)
        sentences = sent_cache[row.patient_id]
        valid_ids = {s["id"] for s in sentences}
        evidence = parse_id_list(row.expert_sentences)
        bad = [e for e in evidence if e not in valid_ids]
        if bad:
            dropped["evidence_id_out_of_range(kept,filtered)"] += 1
            evidence = [e for e in evidence if e in valid_ids]
        records.append({
            "annotation_id": int(row.annotation_id),
            "patient_id": row.patient_id,
            "trial_id": row.trial_id,
            "trial_title": getattr(row, "trial_title", ""),
            "criterion_type": ctype,
            "criterion_text": str(ctext).strip(),
            "sentences": sentences,
            "gold_label": label,
            "gold_evidence": evidence,
        })
    return records, dict(dropped)


def label_counts(records) -> dict:
    c = Counter(r["gold_label"] for r in records)
    return {lab: c.get(lab, 0) for lab in ALL_LABELS}


def prepare(df_raw: pd.DataFrame, out_dir: str | Path, manifest_path: str | Path,
            seed=42, n_train=37, n_val=8, n_test=8, source="ncbi/TrialGPT-Criterion-Annotations") -> dict:
    df = drop_leaky_columns(df_raw)
    split = make_patient_split(df["patient_id"].unique(), seed, n_train, n_val, n_test)
    leak = check_no_leakage(split, df)
    records, dropped = build_records(df)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_of = {pid: k for k, v in split.items() for pid in v}
    by_split = {k: [r for r in records if split_of[r["patient_id"]] == k] for k in split}
    for k, recs in by_split.items():
        with open(out_dir / f"{k}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    manifest = {
        "dataset": source,
        "split_rule": f"sorted unique patient_id -> np.random.default_rng({seed}).shuffle -> "
                      f"first {n_train} train / next {n_val} val / last {n_test} test",
        "seed": seed,
        "dropped_columns": [c for c in LEAKY_COLUMNS + IGNORED_COLUMNS if c in df_raw.columns],
        "rows_raw": int(len(df_raw)),
        "rows_kept": len(records),
        "rows_dropped": dropped,
        "leakage_checks": leak,
        "splits": {},
    }
    for k in ["train", "val", "test"]:
        recs = by_split[k]
        manifest["splits"][k] = {
            "patient_ids": split[k],
            "n_patients": len(split[k]),
            "n_annotations": len(recs),
            "n_inclusion": sum(r["criterion_type"] == "inclusion" for r in recs),
            "n_exclusion": sum(r["criterion_type"] == "exclusion" for r in recs),
            "n_trials": len({r["trial_id"] for r in recs}),
            "n_empty_gold_evidence": sum(not r["gold_evidence"] for r in recs),
            "label_counts": label_counts(recs),
            "annotation_ids": sorted(r["annotation_id"] for r in recs),
        }
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_split(processed_dir: str | Path, name: str) -> list[dict]:
    with open(Path(processed_dir) / f"{name}.jsonl") as f:
        return [json.loads(line) for line in f]
