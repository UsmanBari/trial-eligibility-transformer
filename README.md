# Trial Eligibility Transformer

Criterion-level clinical-trial eligibility screening with a fine-tuned open-weight decoder
Transformer, plus hand-implemented KV-cached decoding, tiled online-softmax attention and
greedy speculative decoding. GenAI (AI-4009) Assignment 2, FAST-NUCES.

> **Educational research prototype.** Not a clinical enrollment decision tool.

## Status

| Milestone | Content | State |
|---|---|---|
| M1 | Data prep, fixed 37/8/8 split + manifest, metrics, zero-shot baseline | done |
| M2 | Fine-tuning (LoRA) + before/after evaluation + error analysis | next |
| M3 | KV-cache decoder, tiled attention, speculative decoding | |
| M4 | Browser application | |
| M5 | IEEE report, demo video | |

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

## Repository layout

```
configs/default.yaml        paths, split, models, generation settings
src/trialmatch/             data, prompting, metrics, modeling
scripts/                    prepare_data.py, evaluate.py
notebooks/                  Kaggle notebooks per milestone
data/splits/                published split manifest
results/<tag>/              predictions.jsonl, metrics.json, run_info.json
tests/                      unit tests (synthetic data, CPU)
```

## AI use

Code was written with the help of Claude (Anthropic) and reviewed, run and tested by the author.
Details are in the report's AI-use appendix.
