"""High-level screening used by the browser app: parse inputs, run every criterion, aggregate.

Decoders (all deterministic greedy, all produce the same tokens in exact arithmetic):
  kv            our KV-cached decoder; the system prompt + patient note are prefilled ONCE
                and that prefix cache is reused for every criterion (prefix caching)
  nocache       our no-cache reference (slow; for comparison only)
  spec_lookup   speculative decoding with the prompt-lookup drafter (no draft model)
  spec_draft    speculative decoding with the fine-tuned Qwen2.5-0.5B draft model
"""
from __future__ import annotations

import re
import threading
import time

import torch

from . import INVALID_LABEL
from .data import split_note_into_sentences
from .kv_decode import eos_ids_of, greedy_kv, greedy_no_cache, prefill_prefix
from .prompting import build_explanation, build_prompt, parse_output

MAX_CRITERIA = 60
MAX_NOTE_CHARS = 20_000
MAX_CRITERION_CHARS = 1_500
DECODERS = ("kv", "nocache", "spec_lookup", "spec_draft")


class InputError(ValueError):
    """Problem with what the user typed (shown to them as-is)."""


# --------------------------------------------------------------------------- input parsing
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def parse_patient_note(text: str) -> list[dict]:
    """Free text or a "0. ... 1. ..." numbered note -> [{id, text}] with ids 0..n-1."""
    text = (text or "").strip()
    if not text:
        raise InputError("The patient note is empty.")
    if len(text) > MAX_NOTE_CHARS:
        raise InputError(f"The patient note is too long ({len(text):,} characters; limit {MAX_NOTE_CHARS:,}).")
    if re.match(r"\s*0\.\s+", text):
        sents = split_note_into_sentences(text)
    else:
        pieces = []
        for para in re.split(r"\n\s*\n|\n", text):
            pieces += [p.strip() for p in _SENT_SPLIT.split(para.strip()) if p.strip()]
        sents = [{"id": i, "text": p} for i, p in enumerate(pieces)]
    sents = [s for s in sents if s["text"]]
    return [{"id": i, "text": s["text"]} for i, s in enumerate(sents)]


# a heading is the whole line ("Inclusion Criteria", "Exclusion criteria:") or "Inclusion: <criterion>";
# "Exclusion of patients with ..." is a criterion, not a heading
_HEAD_INC = re.compile(r"^(key\s+)?inclusion(\s+criteria)?\s*(?::\s*(.*))?$", re.I)
_HEAD_EXC = re.compile(r"^(key\s+)?exclusion(\s+criteria)?\s*(?::\s*(.*))?$", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•·▪]|\(?\d+[.)]|\(?[a-z][.)])\s+")


def parse_criteria(text: str) -> list[dict]:
    """Criteria text with 'Inclusion Criteria:' / 'Exclusion Criteria:' headings, one criterion
    per line (bullets/numbers optional) -> [{criterion_type, criterion_text}]."""
    out, current = [], None
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m_inc, m_exc = _HEAD_INC.match(line), _HEAD_EXC.match(line)
        if m_inc or m_exc:
            current = "inclusion" if m_inc else "exclusion"
            rest = ((m_inc or m_exc).group(3) or "").strip()
            if not rest:
                continue
            line = rest                                  # "Inclusion: age 18-65" on one line
        if current is None:
            raise InputError("Put each criterion under an 'Inclusion Criteria:' or 'Exclusion Criteria:' heading.")
        crit = _BULLET.sub("", line).strip()
        if len(crit) < 3:
            continue
        if len(crit) > MAX_CRITERION_CHARS:
            raise InputError(f"A criterion is longer than {MAX_CRITERION_CHARS} characters: '{crit[:60]}...'")
        out.append({"criterion_type": current, "criterion_text": crit})
    if not out:
        raise InputError("No criteria found. Add at least one line under 'Inclusion Criteria:' or 'Exclusion Criteria:'.")
    if len(out) > MAX_CRITERIA:
        raise InputError(f"Too many criteria ({len(out)}); the limit is {MAX_CRITERIA} per run.")
    return out


def normalise_criteria(items) -> list[dict]:
    """API form: list of {criterion_type|type, criterion_text|text}."""
    out = []
    for it in items or []:
        t = str(it.get("criterion_type", it.get("type", ""))).strip().lower()
        c = str(it.get("criterion_text", it.get("text", ""))).strip()
        if t not in ("inclusion", "exclusion"):
            raise InputError(f"criterion type must be 'inclusion' or 'exclusion', got {t!r}")
        if c:
            out.append({"criterion_type": t, "criterion_text": c[:MAX_CRITERION_CHARS]})
    if not out:
        raise InputError("No criteria given.")
    if len(out) > MAX_CRITERIA:
        raise InputError(f"Too many criteria ({len(out)}); the limit is {MAX_CRITERIA} per run.")
    return out


# --------------------------------------------------------------------------- aggregation
NEGATIVE = {"not included", "excluded"}


def overall_decision(results: list[dict]) -> dict:
    labels = [r["label"] for r in results]
    blocking = [r["index"] for r in results if r["label"] in NEGATIVE]
    unknown = [r["index"] for r in results if r["label"] in ("not enough information", INVALID_LABEL)]
    if blocking:
        status, text = "likely_ineligible", ("At least one criterion appears to rule the patient out "
                                             "(an inclusion criterion not met or an exclusion criterion met).")
    elif unknown:
        status, text = "needs_review", ("No criterion rules the patient out, but some criteria cannot be decided "
                                        "from the note. A clinician needs the missing information.")
    else:
        status, text = "likely_eligible", "Every evaluated criterion appears to be satisfied."
    return {"status": status, "summary": text, "blocking_criteria": blocking, "undecided_criteria": unknown,
            "counts": {l: labels.count(l) for l in sorted(set(labels))}}


# --------------------------------------------------------------------------- screener
class Screener:
    def __init__(self, base_model, adapter=None, draft_model=None, draft_adapter=None, dtype=None,
                 max_input_tokens=3072, max_new_tokens=96, load_draft=False):
        self.cfg = dict(base_model=base_model, adapter=adapter, draft_model=draft_model,
                        draft_adapter=draft_adapter, dtype=dtype, max_input_tokens=max_input_tokens,
                        max_new_tokens=max_new_tokens)
        self.model = self.tok = self.draft = None
        self.status, self.error = "not_loaded", None
        self.draft_status, self.draft_error = ("not_loaded" if draft_model else "not_configured"), None
        self.lock = threading.Lock()
        self._load_draft_on_start = load_draft
        self.load_seconds = None

    # ---- loading
    def load(self):
        from .modeling import load_model, load_tokenizer
        self.status = "loading"
        t0 = time.perf_counter()
        try:
            dt = getattr(torch, self.cfg["dtype"]) if self.cfg["dtype"] else None
            self.tok = load_tokenizer(self.cfg["base_model"])
            self.model = load_model(self.cfg["base_model"], self.cfg["adapter"], dtype=dt)
            self.eos = eos_ids_of(self.model, self.tok)
            self.status = "ready"
            self.load_seconds = time.perf_counter() - t0
        except Exception as e:                                   # shown on the status pill
            self.status, self.error = "error", f"{type(e).__name__}: {e}"
            return
        if self._load_draft_on_start:
            self.load_draft()

    def load_draft(self):
        from .modeling import load_model, load_tokenizer
        from .speculative import check_draft_compatible
        if not self.cfg["draft_model"]:
            raise InputError("No draft model is configured (start the app with --draft-model).")
        if self.draft is not None:
            return
        self.draft_status = "loading"
        try:
            dt = getattr(torch, self.cfg["dtype"]) if self.cfg["dtype"] else None
            draft = load_model(self.cfg["draft_model"], self.cfg["draft_adapter"], dtype=dt)
            compat = check_draft_compatible(self.tok, load_tokenizer(self.cfg["draft_model"]), self.model, draft)
            if not compat["compatible"]:
                raise RuntimeError(f"draft is not tokenizer-compatible with the target: {compat}")
            self.draft, self.draft_status = draft, "ready"
        except Exception as e:
            self.draft_status, self.draft_error = "error", f"{type(e).__name__}: {e}"
            raise

    def info(self) -> dict:
        from .modeling import model_card
        d = {"status": self.status, "error": self.error, "draft_status": self.draft_status,
             "draft_error": self.draft_error, "config": self.cfg, "load_seconds": self.load_seconds,
             "decoders": list(DECODERS)}
        if self.model is not None:
            d["model"] = model_card(self.model, self.cfg["base_model"])
            d["device"] = str(next(self.model.parameters()).device)
        return d

    # ---- screening
    def _require_ready(self):
        if self.status != "ready":
            raise RuntimeError(f"model is {self.status}" + (f": {self.error}" if self.error else ""))

    def screen(self, sentences: list[dict], criteria: list[dict], decoder="kv", k=4) -> dict:
        self._require_ready()
        if decoder not in DECODERS:
            raise InputError(f"unknown decoder {decoder!r}; choose one of {', '.join(DECODERS)}")
        if decoder == "spec_draft":
            self.load_draft()
        k = int(max(1, min(int(k), 12)))
        device = next(self.model.parameters()).device
        with self.lock:
            t_start = time.perf_counter()
            prompts = []
            for c in criteria:
                rec = {**c, "sentences": sentences, "gold_evidence": []}
                prompt, info = build_prompt(rec, self.tok, self.cfg["max_input_tokens"])
                ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                prompts.append((rec, ids, info))

            # prefix caching: longest common token prefix of all criterion prompts
            prefix_cache, P = None, 0
            if decoder == "kv" and len(prompts) > 1:
                first = prompts[0][1][0]
                P = first.shape[0]
                for _, ids, _ in prompts[1:]:
                    n = min(P, ids.shape[1])
                    neq = (first[:n] != ids[0, :n]).nonzero()
                    P = int(neq[0]) if len(neq) else n
                P = min(P, min(ids.shape[1] for _, ids, _ in prompts) - 1)
                if P > 0:
                    t0 = time.perf_counter()
                    prefix_cache = prefill_prefix(self.model, prompts[0][1][:, :P])
                    prefix_s = time.perf_counter() - t0

            results = []
            for i, (rec, ids, info) in enumerate(prompts):
                t0 = time.perf_counter()
                stats = None
                if decoder == "kv":
                    r = greedy_kv(self.model, ids, self.cfg["max_new_tokens"], self.eos,
                                  prefix_cache=prefix_cache, prefix_len=P) if prefix_cache is not None \
                        else greedy_kv(self.model, ids, self.cfg["max_new_tokens"], self.eos)
                elif decoder == "nocache":
                    r = greedy_no_cache(self.model, ids, self.cfg["max_new_tokens"], self.eos)
                else:
                    from .speculative import speculative_greedy
                    drafter = "prompt_lookup" if decoder == "spec_lookup" else self.draft
                    r = speculative_greedy(self.model, drafter, ids, self.cfg["max_new_tokens"], k=k,
                                           eos_ids=self.eos)
                    stats = {key: r["stats"][key] for key in ("iterations", "drafted", "accepted",
                                                              "acceptance_rate", "tokens_per_target_forward")}
                text = self.tok.decode(r["tokens"], skip_special_tokens=True)
                p = parse_output(text, rec["criterion_type"], {s["id"] for s in sentences})
                expl = build_explanation(p["label"], p["evidence"], sentences)
                results.append({
                    "index": i, "criterion_type": rec["criterion_type"], "criterion_text": rec["criterion_text"],
                    "label": p["label"], "evidence": p["evidence"], "explanation": expl["text"],
                    "evidence_sentences": expl["evidence"], "raw_output": text, "parse_ok": p["parse_ok"],
                    "hallucinated_ids": p["hallucinated_ids"], "prompt_tokens": info["prompt_tokens"],
                    "dropped_sentence_ids": info["dropped_sentence_ids"], "new_tokens": len(r["tokens"]),
                    "latency_ms": 1e3 * (time.perf_counter() - t0), "speculative": stats,
                })
            total = time.perf_counter() - t_start
        warnings = []
        if any(r["dropped_sentence_ids"] for r in results):
            warnings.append("The note was too long for the model's input budget; the last sentences were "
                            "left out for some criteria (see 'dropped sentences').")
        if any(not r["parse_ok"] for r in results):
            warnings.append("The model returned output that was not valid JSON for some criteria; those were "
                            "parsed leniently or marked invalid.")
        if any(r["hallucinated_ids"] for r in results):
            warnings.append("Some cited sentence numbers do not exist in the note and were removed.")
        n_tok = sum(r["new_tokens"] for r in results)
        return {"decoder": decoder, "k": k if decoder.startswith("spec") else None,
                "sentences": sentences, "results": results, "overall": overall_decision(results),
                "warnings": warnings,
                "timing": {"total_s": total, "per_criterion_ms": 1e3 * total / len(results),
                           "new_tokens": n_tok, "tokens_per_s": n_tok / total if total else None,
                           "shared_prefix_tokens": P,
                           "prefix_prefill_ms": 1e3 * prefix_s if prefix_cache is not None else None}}

    def compare_decoders(self, sentences, criterion, k=4, decoders=("nocache", "kv", "spec_lookup")) -> dict:
        """Live benchmark for one criterion: same prompt through each decoder."""
        self._require_ready()
        rows, ref = [], None
        for d in decoders:
            if d == "spec_draft" and self.draft is None:
                continue
            res = self.screen(sentences, [criterion], decoder=d, k=k)["results"][0]
            if ref is None:
                ref = res["raw_output"]
            rows.append({"decoder": d, "latency_ms": res["latency_ms"], "new_tokens": res["new_tokens"],
                         "tokens_per_s": res["new_tokens"] / (res["latency_ms"] / 1e3),
                         "label": res["label"], "same_output_as_first": res["raw_output"] == ref,
                         "speculative": res["speculative"]})
        return {"criterion": criterion, "rows": rows}
