"""FastAPI backend for the browser app.

Endpoints
  GET  /                  the single-page UI (app/static)
  GET  /api/health        model status (loading / ready / error), config, model card
  GET  /api/examples      demo patient + trial, and validation-split patients if prepared
  POST /api/parse         split a note into numbered sentences and parse criteria (no model)
  POST /api/screen        run every criterion -> labels, evidence, explanations, overall decision
  POST /api/compare       live benchmark of decoders on one criterion
  POST /api/export        Markdown or printable HTML report from a /api/screen result
  POST /api/load-draft    load the draft model for speculative decoding
  GET  /api/benchmarks    offline evaluation + Task 3/4/5 benchmark results from results/
  GET  /figures/<name>    figures from results/figures
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from trialmatch.inference import (DECODERS, InputError, Screener, normalise_criteria, parse_criteria,
                                  parse_patient_note)
from trialmatch.report import to_html, to_markdown

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"

DEMO = {
    "id": "demo",
    "title": "Demo (invented patient, invented trial)",
    "note": ("A 62-year-old man presents with exertional chest pain for 3 months. "
             "He has type 2 diabetes treated with metformin and a 20 pack-year smoking history. "
             "Blood pressure is 148/92 mmHg. "
             "An exercise stress test showed reversible ischemia in the inferior wall. "
             "He has no history of heart failure. "
             "Serum creatinine is 1.1 mg/dL."),
    "criteria": ("Inclusion Criteria:\n"
                 "- Age 40 to 75 years\n"
                 "- Stable angina with documented ischemia on stress testing\n"
                 "- Signed informed consent\n"
                 "Exclusion Criteria:\n"
                 "- Heart failure with reduced ejection fraction\n"
                 "- Severe renal impairment (creatinine > 2.5 mg/dL)\n"
                 "- Pregnancy"),
}


class ScreenRequest(BaseModel):
    note: str = Field(..., description="free text, or a note numbered '0. ... 1. ...'")
    criteria_text: str | None = Field(None, description="criteria under Inclusion/Exclusion headings")
    criteria: list[dict] | None = Field(None, description="alternative: [{type, text}]")
    decoder: str = "kv"
    k: int = 4


class CompareRequest(BaseModel):
    note: str
    criterion_type: str
    criterion_text: str
    k: int = 4
    include_draft: bool = False


class ExportRequest(BaseModel):
    result: dict
    format: str = "markdown"
    title: str = "Trial eligibility screening report"
    meta: dict | None = None


def _criteria_from(req) -> list[dict]:
    if req.criteria:
        return normalise_criteria(req.criteria)
    return parse_criteria(req.criteria_text or "")


def _load_json(p: Path):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def summarise_results(results_dir: Path) -> dict:
    """Small, UI-friendly summary of everything the experiments produced."""
    out = {"evaluation": [], "kv": None, "attention": None, "speculative": None, "figures": [],
           "final_model": _load_json(results_dir / "final_model.json")}
    tags = [("Zero-shot Qwen2.5-1.5B", "baseline_test_greedy", "test"),
            ("LoRA v1", "ft_v1_test_greedy", "test"),
            ("LoRA v2 (class-balanced)", "ft_v2_test_greedy", "test"),
            ("Zero-shot Qwen2.5-1.5B", "baseline_val_greedy", "val"),
            ("LoRA v1", "ft_v1_val_greedy", "val"),
            ("LoRA v2 (class-balanced)", "ft_v2_ep2_val", "val")]
    for name, tag, split in tags:
        m = _load_json(results_dir / tag / "metrics.json")
        if not m:
            continue
        o = m["overall"]
        out["evaluation"].append({
            "model": name, "split": split, "tag": tag, "n": o["classification"]["n"],
            "accuracy": o["classification"]["accuracy"], "macro_f1": o["classification"]["macro_f1"],
            "inclusion_accuracy": m["inclusion"]["classification"]["accuracy"],
            "exclusion_accuracy": m["exclusion"]["classification"]["accuracy"],
            "evidence_micro_f1": o["evidence"]["micro"]["f1"],
            "per_class_f1": {k: v["f1"] for k, v in o["classification"]["per_class"].items()}})
    kv = _load_json(results_dir / "bench" / "kv_float16.json")
    kv32 = _load_json(results_dir / "bench" / "kv_float32.json")
    if kv:
        out["kv"] = {"device": kv.get("device"),
                     "task_agreement_fp16": {k: kv["task_agreement"][k] for k in ("n", "nocache_eq_kv", "kv_eq_hf_generate")},
                     "task_agreement_fp32": {k: kv32["task_agreement"][k] for k in ("n", "nocache_eq_kv", "kv_eq_hf_generate")} if kv32 else None,
                     "grid": [{"input_len": r["input_len"], "output_len": r["output_len"],
                               "no_cache_ms": r["no_cache"]["total_ms"], "kv_ms": r["kv_cache"]["total_ms"],
                               "kv_prefill_ms": r["kv_cache"]["prefill_ms"],
                               "no_cache_tok_s": r["no_cache"]["tokens_per_s"], "kv_tok_s": r["kv_cache"]["tokens_per_s"],
                               "no_cache_mem_mb": r["no_cache"]["peak_extra_mem_mb"],
                               "kv_mem_mb": r["kv_cache"]["peak_extra_mem_mb"],
                               "speedup": r["speedup_total"], "tokens_equal": r["tokens_equal"],
                               "first_mismatch": r["first_mismatch_index"]} for r in kv.get("grid", [])]}
    at = _load_json(results_dir / "bench" / "attention.json")
    if at:
        c = at["correctness"]
        out["attention"] = {"device": at.get("device"), "n_pass": c["n_pass"], "n": c["n"],
                            "tolerance": c["tolerance_abs"],
                            "worst": {dt: max((r["tiled_vs_ref_max"] for r in c["rows"] if r["dtype"] == dt), default=None)
                                      for dt in ("float32", "float16")},
                            "performance": at["performance"],
                            "model_end_to_end": {k: {"tokens_equal": v["tokens_equal"], "n": v["n"],
                                                     "max_logit_diff": max(x["max_abs_logit_diff"] for x in v["cases"])}
                                                 for k, v in at.get("model_end_to_end", {}).items()}}
    sp = _load_json(results_dir / "bench" / "spec_float16.json")
    if sp:
        rows = []
        for dname, res in sp["drafts"].items():
            for k, by_set in res["by_k"].items():
                for s, r in by_set.items():
                    rows.append({"draft": dname, "k": int(k), "prompts": s, "equal": f"{r['tokens_equal']}/{r['n_prompts']}",
                                 "acceptance": r["acceptance_rate"], "tok_s": r["tokens_per_s"],
                                 "speedup": r["speedup_vs_greedy"], "tok_per_target_pass": r["tokens_per_target_forward"],
                                 "predicted_speedup": (r.get("predicted") or {}).get("speedup_formula")})
        out["speculative"] = {"device": sp.get("device"), "baseline": sp["baseline"], "rows": rows,
                              "cost_model": sp.get("cost_model"),
                              "draft_costs": {d: {"c": r.get("c"), "draft_step_ms": r.get("draft_step_ms")}
                                              for d, r in sp["drafts"].items()}}
    fig_dir = results_dir / "figures"
    if fig_dir.exists():
        out["figures"] = sorted(p.name for p in fig_dir.glob("*.png"))
    return out


def create_app(screener: Screener, results_dir: Path | str = ROOT / "results",
               processed_dir: Path | str = ROOT / "data" / "processed", load_in_background=True) -> FastAPI:
    results_dir, processed_dir = Path(results_dir), Path(processed_dir)
    app = FastAPI(title="Trial eligibility screener", version="1.0")
    app.state.screener = screener

    if load_in_background:
        threading.Thread(target=screener.load, daemon=True).start()

    @app.exception_handler(InputError)
    async def _input_error(_: Request, exc: InputError):
        return JSONResponse(status_code=400, content={"error": str(exc), "kind": "input"})

    def ready_or_503():
        if screener.status != "ready":
            msg = {"loading": "The model is still loading. Try again in a moment.",
                   "not_loaded": "The model has not been loaded yet.",
                   "error": f"The model failed to load: {screener.error}"}.get(screener.status, screener.status)
            raise HTTPException(status_code=503, detail=msg)

    @app.get("/api/health")
    def health():
        return screener.info()

    @app.get("/api/examples")
    def examples():
        ex = [DEMO]
        val = processed_dir / "val.jsonl"
        if val.exists():
            by_patient = {}
            for line in open(val):
                r = json.loads(line)
                by_patient.setdefault((r["patient_id"], r["trial_id"]), r | {"_crit": []})["_crit"].append(r)
            for (pid, tid), r in list(by_patient.items())[:6]:
                one_line = lambda t: " ".join(t.split())          # criteria are one per line in the UI
                inc = [one_line(c["criterion_text"]) for c in r["_crit"] if c["criterion_type"] == "inclusion"]
                exc = [one_line(c["criterion_text"]) for c in r["_crit"] if c["criterion_type"] == "exclusion"]
                crit = ""
                if inc:
                    crit += "Inclusion Criteria:\n" + "\n".join(f"- {c}" for c in inc) + "\n"
                if exc:
                    crit += "Exclusion Criteria:\n" + "\n".join(f"- {c}" for c in exc)
                ex.append({"id": f"{pid}/{tid}", "title": f"Validation patient {pid} / {tid}",
                           "note": "\n".join(f"{s['id']}. {s['text']}" for s in r["sentences"]),
                           "criteria": crit.strip(),
                           "gold": [{"criterion_text": one_line(c["criterion_text"]), "label": c["gold_label"],
                                     "evidence": c["gold_evidence"]} for c in r["_crit"]]})
        return {"examples": ex}

    @app.post("/api/parse")
    def parse(req: ScreenRequest):
        return {"sentences": parse_patient_note(req.note), "criteria": _criteria_from(req)}

    @app.post("/api/screen")
    def screen(req: ScreenRequest):
        sentences = parse_patient_note(req.note)
        criteria = _criteria_from(req)
        if req.decoder not in DECODERS:
            raise InputError(f"Unknown decoder '{req.decoder}'.")
        ready_or_503()
        try:
            return screener.screen(sentences, criteria, decoder=req.decoder, k=req.k)
        except InputError:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Screening failed: {type(e).__name__}: {e}")

    @app.post("/api/compare")
    def compare(req: CompareRequest):
        sentences = parse_patient_note(req.note)
        crit = normalise_criteria([{"type": req.criterion_type, "text": req.criterion_text}])[0]
        ready_or_503()
        decs = ["nocache", "kv", "spec_lookup"] + (["spec_draft"] if req.include_draft else [])
        if req.include_draft:
            try:
                screener.load_draft()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Could not load the draft model: {e}")
        return screener.compare_decoders(sentences, crit, k=req.k, decoders=decs)

    @app.post("/api/load-draft")
    def load_draft():
        ready_or_503()
        try:
            screener.load_draft()
        except InputError:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not load the draft model: {e}")
        return screener.info()

    @app.post("/api/export")
    def export(req: ExportRequest):
        need = {"results", "overall", "sentences", "timing", "decoder"}
        if not need <= set(req.result):
            raise InputError("This is not a screening result; run a screening first.")
        if req.format == "markdown":
            return PlainTextResponse(to_markdown(req.result, req.title, req.meta), media_type="text/markdown")
        if req.format == "html":
            return HTMLResponse(to_html(req.result, req.title, req.meta))
        raise InputError("format must be 'markdown' or 'html'")

    @app.get("/api/benchmarks")
    def benchmarks():
        return summarise_results(results_dir)

    if (results_dir / "figures").exists():
        app.mount("/figures", StaticFiles(directory=results_dir / "figures"), name="figures")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
