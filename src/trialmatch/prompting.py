"""Prompt construction, output schema, robust parsing and evidence-grounded explanations.

Output schema (what the model is trained to emit, one line of JSON):
    {"label": "<one allowed label>", "evidence": [<sentence ids>]}

The explanation shown to the user is NOT free text from the model. It is built
deterministically from the predicted label and the cited sentences, so every
word of evidence in it traces back to the patient record (no hallucinated
rationale, and no dependency on the GPT-4 explanations in the dataset).
"""
from __future__ import annotations

import json
import re

from . import INVALID_LABEL, allowed_labels

SYSTEM_PROMPT = (
    "You are a clinical trial screening assistant. You read a patient note whose "
    "sentences are numbered, and one eligibility criterion of a clinical trial. "
    "Decide how the criterion applies to this patient using ONLY the note. "
    "If the note does not contain enough information, say so instead of guessing. "
    "Answer with a single line of JSON and nothing else."
)

LABEL_GUIDE = {
    "inclusion": (
        '"included" = the patient meets the criterion; '
        '"not included" = the patient does not meet it; '
        '"not enough information" = the note cannot decide it; '
        '"not applicable" = the criterion does not apply to this patient.'
    ),
    "exclusion": (
        '"excluded" = the patient has the excluding condition; '
        '"not excluded" = the patient does not have it; '
        '"not enough information" = the note cannot decide it; '
        '"not applicable" = the criterion does not apply to this patient.'
    ),
}


def format_note(sentences: list[dict]) -> str:
    return "\n".join(f"[{s['id']}] {s['text']}" for s in sentences)


def user_message(sentences: list[dict], criterion_type: str, criterion_text: str) -> str:
    ctype = criterion_type.lower()
    labels = ", ".join(f'"{l}"' for l in allowed_labels(ctype))
    return (
        f"Patient note:\n{format_note(sentences)}\n\n"
        f"Criterion type: {ctype}\n"
        f"Criterion: {criterion_text}\n\n"
        f"Allowed labels: {labels}.\n{LABEL_GUIDE[ctype]}\n"
        'Return JSON: {"label": <label>, "evidence": [<ids of the sentences that support '
        "your decision; [] if none>]}"
    )


def target_text(label: str, evidence: list[int]) -> str:
    return json.dumps({"label": label, "evidence": sorted(evidence)})


def build_messages(record: dict, sentences: list[dict] | None = None, with_target: bool = False):
    sents = record["sentences"] if sentences is None else sentences
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(sents, record["criterion_type"], record["criterion_text"])}]
    if with_target:
        msgs.append({"role": "assistant", "content": target_text(record["gold_label"], record["gold_evidence"])})
    return msgs


def build_prompt(record: dict, tokenizer, max_input_tokens: int):
    """Chat-templated prompt string that fits the token budget.

    If too long, whole sentences are dropped from the END of the note (the
    opening sentences carry demographics and the chief complaint). We record
    exactly what was dropped and whether any gold evidence was lost so the
    report can document truncation honestly.
    """
    sents = list(record["sentences"])
    dropped = []
    while True:
        prompt = tokenizer.apply_chat_template(build_messages(record, sents), tokenize=False,
                                               add_generation_prompt=True)
        n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_tok <= max_input_tokens or len(sents) <= 1:
            break
        dropped.append(sents.pop()["id"])
    info = {
        "prompt_tokens": n_tok,
        "dropped_sentence_ids": sorted(dropped),
        "gold_evidence_lost": sorted(set(dropped) & set(record.get("gold_evidence", []))),
    }
    return prompt, info


# --------------------------------------------------------------------------- parsing
_JSON_OBJ = re.compile(r"\{.*?\}", re.S)
_EVID = re.compile(r'"?evidence"?\s*:\s*\[([^\]]*)\]', re.I)


def parse_output(text: str, criterion_type: str, valid_ids: set[int] | None = None) -> dict:
    """Parse model text into {label, evidence, parse_ok, hallucinated_ids}.

    Never raises. A label that is not allowed for this criterion type becomes
    INVALID_LABEL (scored as wrong), it is NOT silently mapped to a valid one.
    """
    allowed = allowed_labels(criterion_type)
    label, evidence, ok = None, [], False
    m = _JSON_OBJ.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            label = str(obj.get("label", "")).strip().lower()
            ev = obj.get("evidence", [])
            evidence = [int(x) for x in ev if str(x).lstrip("-").isdigit()] if isinstance(ev, list) else []
            ok = True
        except (json.JSONDecodeError, AttributeError, ValueError, TypeError):
            pass
    if not ok:  # lenient fallback: find a label string and an evidence list anywhere
        low = text.lower()
        for cand in sorted(allowed, key=len, reverse=True):   # "not included" before "included"
            if cand in low:
                label = cand
                break
        em = _EVID.search(text)
        if em:
            evidence = [int(x) for x in re.findall(r"\d+", em.group(1))]
    if label not in allowed:
        label = INVALID_LABEL
    evidence = sorted(set(evidence))
    hallucinated = []
    if valid_ids is not None:
        hallucinated = [e for e in evidence if e not in valid_ids]
        evidence = [e for e in evidence if e in valid_ids]
    return {"label": label, "evidence": evidence, "parse_ok": ok, "hallucinated_ids": hallucinated}


# --------------------------------------------------------------------------- explanation
_LABEL_SENTENCE = {
    "included": "The patient appears to meet this inclusion criterion.",
    "not included": "The patient appears not to meet this inclusion criterion.",
    "excluded": "The patient appears to meet this exclusion criterion, which would exclude them.",
    "not excluded": "The patient does not appear to have this excluding condition.",
    "not enough information": "The note does not contain enough information to decide this criterion.",
    "not applicable": "This criterion does not appear to apply to this patient.",
    INVALID_LABEL: "The model did not return a valid decision for this criterion.",
}


def build_explanation(label: str, evidence: list[int], sentences: list[dict]) -> dict:
    by_id = {s["id"]: s["text"] for s in sentences}
    cited = [{"id": i, "text": by_id[i]} for i in evidence if i in by_id]
    text = _LABEL_SENTENCE.get(label, "")
    if cited:
        text += " Supporting note sentences: " + " ".join(f"[{c['id']}] \"{c['text']}\"" for c in cited)
    elif label not in ("not enough information", INVALID_LABEL):
        text += " No supporting sentence was cited, so treat this decision with caution."
    return {"text": text, "evidence": cited}
