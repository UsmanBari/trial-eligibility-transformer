"""Unit tests that need no GPU and no download (synthetic data with the real schema)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trialmatch import INVALID_LABEL  # noqa: E402
from trialmatch.data import (make_patient_split, parse_id_list, prepare,  # noqa: E402
                             split_note_into_sentences)
from trialmatch.metrics import evidence_report, full_report  # noqa: E402
from trialmatch.prompting import build_explanation, parse_output  # noqa: E402


def fake_df(n_patients=53, per_patient=19, seed=0):
    rng = np.random.default_rng(seed)
    rows, aid = [], 0
    for p in range(n_patients):
        note = "\n".join(f"{i}. Sentence {i} of patient {p}, dose 2. mg daily." for i in range(6))
        for _ in range(per_patient):
            ctype = rng.choice(["inclusion", "exclusion"])
            lab = rng.choice(["included", "not included", "not enough information", "not applicable"]
                             if ctype == "inclusion" else
                             ["excluded", "not excluded", "not enough information", "not applicable"])
            rows.append(dict(annotation_id=aid, patient_id=f"sigir-{1000 + p}", note=note,
                             trial_id=f"NCT{p:05d}", trial_title="t", criterion_type=ctype,
                             criterion_text="Age 18 to 65", gpt4_explanation="LEAK", explanation_correctness="Correct",
                             gpt4_sentences="[0]", expert_sentences=str(sorted(rng.choice(6, 2, replace=False).tolist())),
                             gpt4_eligibility="included", expert_eligibility=lab, training=True))
            aid += 1
    return pd.DataFrame(rows)


def test_split_is_exact_rule_and_disjoint():
    ids = [f"p{i:02d}" for i in range(53)]
    s = make_patient_split(ids[::-1], seed=42)          # input order must not matter
    expect = np.array(sorted(ids))
    np.random.default_rng(42).shuffle(expect)
    assert s["train"] == expect[:37].tolist() and s["val"] == expect[37:45].tolist()
    assert s["test"] == expect[45:].tolist()
    assert not (set(s["train"]) & set(s["val"])) and not (set(s["val"]) & set(s["test"]))


def test_prepare_drops_gpt4_and_writes_manifest(tmp_path):
    m = prepare(fake_df(), tmp_path / "proc", tmp_path / "manifest.json")
    assert [m["splits"][k]["n_patients"] for k in ("train", "val", "test")] == [37, 8, 8]
    assert sum(m["splits"][k]["n_annotations"] for k in m["splits"]) == m["rows_kept"]
    rec = json.loads(open(tmp_path / "proc" / "train.jsonl").readline())
    assert not any(k.startswith("gpt4") for k in rec) and "explanation_correctness" not in rec
    assert "LEAK" not in open(tmp_path / "proc" / "train.jsonl").read()


def test_sentence_parsing_ignores_numbers_in_text():
    note = "0. Patient got 2. mg of drug.\n1. She has 3. children.\n2. BP 140/90."
    sents = split_note_into_sentences(note)
    assert [s["id"] for s in sents] == [0, 1, 2]
    assert sents[0]["text"] == "Patient got 2. mg of drug."
    single = "0. A 58-year-old woman. 1. She smokes. 2. No diabetes."
    assert [s["text"] for s in split_note_into_sentences(single)] == \
        ["A 58-year-old woman.", "She smokes.", "No diabetes."]


def test_parse_id_list():
    assert parse_id_list("[2, 0, 2]") == [0, 2] and parse_id_list("[]") == [] and parse_id_list(None) == []


def test_parse_output_cases():
    assert parse_output('{"label": "not included", "evidence": [1, 9]}', "inclusion", {0, 1, 2}) == \
        {"label": "not included", "evidence": [1], "parse_ok": True, "hallucinated_ids": [9]}
    # wrong label set for the type -> invalid, never silently remapped
    assert parse_output('{"label": "excluded", "evidence": []}', "inclusion")["label"] == INVALID_LABEL
    # lenient fallback picks the longest matching label
    p = parse_output('Answer: not included because "evidence": [0, 2]', "inclusion")
    assert p["label"] == "not included" and p["evidence"] == [0, 2] and not p["parse_ok"]
    assert parse_output("garbage", "exclusion")["label"] == INVALID_LABEL


def test_evidence_empty_set_rules():
    r = evidence_report([[], [], [1, 2], [3]], [[], [0], [1], []])
    m = r["macro_all_records"]
    # per-record F1: both empty=1, gold empty+pred=0, P=1 R=.5 -> .667, gold non-empty+pred empty=0
    assert abs(m["f1"] - (1 + 0 + 2 / 3 + 0) / 4) < 1e-9
    assert r["macro_nonempty_gold"]["n"] == 2
    assert r["micro"]["tp"] == 1 and r["micro"]["n_pred_ids"] == 2 and r["micro"]["n_gold_ids"] == 3


def test_full_report_shapes():
    rows = [dict(criterion_type="inclusion", gold_label="included", pred_label="included", gold_evidence=[0], pred_evidence=[0]),
            dict(criterion_type="exclusion", gold_label="excluded", pred_label=INVALID_LABEL, gold_evidence=[], pred_evidence=[])]
    rep = full_report(rows)
    assert rep["overall"]["classification"]["accuracy"] == 0.5
    assert rep["exclusion"]["classification"]["n_invalid"] == 1
    assert rep["inclusion"]["classification"]["accuracy"] == 1.0


def test_explanation_only_quotes_note():
    sents = [{"id": 0, "text": "A 58-year-old woman."}, {"id": 1, "text": "Smoker."}]
    e = build_explanation("included", [0, 7], sents)
    assert e["evidence"] == [{"id": 0, "text": "A 58-year-old woman."}] and "[0]" in e["text"]


class _WordTok:
    """Tiny stand-in tokenizer: one token per whitespace word."""
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "\n".join(f"<{m['role']}> {m['content']}" for m in msgs) + "\n<assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_truncation_drops_from_end_and_reports_lost_evidence():
    from trialmatch.prompting import build_prompt
    rec = {"criterion_type": "inclusion", "criterion_text": "Age over 18",
           "sentences": [{"id": i, "text": "word " * 50} for i in range(6)], "gold_evidence": [0, 5]}
    full, info = build_prompt(rec, _WordTok(), 10_000)
    assert info["dropped_sentence_ids"] == []
    short, info = build_prompt(rec, _WordTok(), info["prompt_tokens"] - 60)
    assert info["dropped_sentence_ids"] == [4, 5] and info["gold_evidence_lost"] == [5]
    assert "[3]" in short and "[4]" not in short
