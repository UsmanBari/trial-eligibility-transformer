# Trial Eligibility Transformer

Criterion-level clinical-trial eligibility screening with a fine-tuned open-weight decoder
Transformer, plus hand-implemented KV-cached decoding, tiled online-softmax attention and
greedy speculative decoding. GenAI (AI-4009) Assignment 2, FAST-NUCES.

> **Educational research prototype.** Not a clinical enrollment decision tool.

## Status

| Milestone | Content | State |
|---|---|---|
| M1 | Data prep, fixed 37/8/8 split + manifest, metrics, zero-shot baseline | done |
| M2 | Fine-tuning (LoRA) + checkpoint selection + before/after + final test + error analysis | done |
| M3 | KV-cache decoder, tiled attention, speculative decoding, class-balanced v2 | done |
| M4 | Browser app, prompt-lookup speculative drafter, speculative cost model | done |
| M5 | IEEE report (`report/main.pdf`), demo video | done |

**Demo video (5-7 min):** https://youtu.be/yTGi0hUCeyk

## Task setup

* **Dataset:** `ncbi/TrialGPT-Criterion-Annotations` only (1,015 criterion annotations, 53 patients).
* **Dropped before anything else:** `gpt4_explanation`, `gpt4_sentences`, `gpt4_eligibility`,
  `explanation_correctness` (GPT-4 outputs) and the original `training` flag.
* **Split:** sorted unique `patient_id` → `np.random.default_rng(42).shuffle` → first 37 train,
  next 8 val, last 8 test. Manifest with patient IDs, counts and annotation IDs:
  `data/splits/split_manifest.json`. Leakage checks (patient overlap, identical notes across
  splits, every annotation assigned once) are hard asserts in `src/trialmatch/data.py`.
* **Test split guard:** `scripts/evaluate.py` refuses `--split test` without `--final-test`, and
  every test access is appended to `results/TEST_ACCESS_LOG.txt`.

### Model I/O

Input: system prompt + numbered patient sentences (`[0] ...`), criterion type, criterion text and
the 4 labels allowed for that type. Output (one line of JSON):

```json
{"label": "not enough information", "evidence": [2, 5]}
```

Labels kept as in the dataset: inclusion → `included / not included / not enough information /
not applicable`; exclusion → `excluded / not excluded / not enough information / not applicable`.
A label not allowed for the criterion type, or unparseable output, is scored as `invalid` (wrong),
never remapped. Explanations are built deterministically from the cited sentences
(`prompting.build_explanation`), so they only quote the patient record.

**Truncation:** prompt budget 3,072 tokens; if exceeded, whole sentences are dropped from the end
of the note, and the dropped IDs plus any lost gold-evidence IDs are logged per record.

### Metrics (`src/trialmatch/metrics.py`)

* Classification: accuracy, macro-F1, per-class P/R/F1, confusion matrix (with an `invalid`
  column); overall, inclusion-only and exclusion-only.
* Evidence: micro P/R/F1 (pooled IDs), macro P/R/F1 over all records (both-empty = perfect,
  one side empty = 0 for that side), and macro over records with non-empty gold evidence.

## Reproduce

```bash
pip install torch            # skip on Kaggle/Colab
pip install -r requirements.txt && pip install -e .
python -m pytest -q tests                     # no GPU needed
python scripts/prepare_data.py
python scripts/evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --split val --tag baseline_qwen25_1p5b
```

On Kaggle: open `notebooks/01_data_and_baseline.ipynb`, set **GPU T4 x1** and **Internet on**, Run All.

## Browser app

Runs on a normal laptop (CPU is fine). The first start downloads Qwen2.5-1.5B-Instruct (~3 GB)
and the LoRA adapters from HuggingFace.

**Windows, no commands:** double-click `setup_app.bat` once (creates `.venv` with Python 3.10-3.12
and installs everything), then double-click `start_app.bat`; the browser opens at
http://127.0.0.1:8000. Close the console window to stop the app.

**Any OS, manually** (use Python 3.10-3.12; Python 3.13+ has no pre-built `tokenizers` wheel for the
pinned version on Windows):

```bash
python -m venv .venv
.venv\Scripts\activate                      # Windows  (Linux/macOS: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && pip install -e .
python scripts/prepare_data.py             # optional: adds validation patients as examples
python -m app                              # then open http://127.0.0.1:8000
```

Options: `--dtype bfloat16` halves RAM (~3.5 GB instead of ~7 GB), `--adapter none` runs the
zero-shot base model, `--load-draft` loads the 0.5B draft model at start, `--port 8080`.

What it does:
* paste a patient note (free text or numbered sentences) and trial criteria under
  `Inclusion Criteria:` / `Exclusion Criteria:` headings, or load an example;
* every criterion gets a label, cited note sentences (highlighted in the note) and an explanation
  built only from those sentences, plus an overall rule-based decision
  (likely eligible / likely not eligible / needs review);
* decoders: our KV-cache loop with the shared system-prompt + note prefix encoded once
  (default), speculative decoding with the prompt-lookup drafter or the 0.5B draft, and the
  no-cache reference; "compare decoders" runs one criterion through all of them live;
* export as JSON, Markdown or a printable HTML report;
* **Benchmarks** tab: evaluation table and the Task 3/4/5 results read from `results/`;
* input errors, model loading state and truncation are reported in the UI.

Model weights: [LoRA v2 (final)](https://huggingface.co/usmanbari2005/trialmatch-qwen25-1p5b-lora-v2) ·
[LoRA v1](https://huggingface.co/usmanbari2005/trialmatch-qwen25-1p5b-lora) ·
[0.5B draft LoRA](https://huggingface.co/usmanbari2005/trialmatch-qwen25-0p5b-draft-lora)

## Report

IEEE-format paper: [`report/main.pdf`](report/main.pdf) (LaTeX source `report/main.tex`;
rebuild with `cd report && latexmk -pdf main.tex`).

## Repository layout

```
configs/default.yaml        paths, split, models, generation settings
src/trialmatch/             data, prompting, metrics, modeling, sft,
                            kv_decode (Task 3), attention (Task 4), speculative (Task 5),
                            inference + report (used by the app)
app/                        FastAPI server + static web UI (python -m app)
scripts/                    prepare_data, evaluate, train_lora, select_checkpoint, rescore,
                            error_analysis, make_figures, bench_kv, bench_attention,
                            bench_spec, make_bench_figures
notebooks/                  Kaggle notebooks per milestone
data/splits/                published split manifest
results/<tag>/              predictions.jsonl, metrics.json, run_info.json
tests/                      unit tests (synthetic data, CPU)
```

## AI use

Code was written with the help of Claude (Anthropic) and reviewed, run and tested by the author.
Details are in the report's AI-use appendix.
