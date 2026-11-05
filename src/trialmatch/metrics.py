"""Common evaluation metrics (classification + evidence sentence retrieval).

Classification
  * accuracy, macro-F1, per-class P/R/F1, confusion matrix;
  * overall (6 labels) and separately for inclusion (4 labels) and exclusion (4 labels);
  * an unparseable / not-allowed prediction is the extra column "invalid" in the
    confusion matrix and always counts as wrong. Macro-F1 averages over the
    real label classes of that subset (zero_division=0).

Evidence (sentence-ID sets per record)
  * micro P/R/F1: pool true positives / predicted / gold over all records;
    records with empty gold and empty prediction contribute nothing.
  * macro P/R/F1 (all records), per-record rule:
        P = |G∩P|/|P| if P non-empty else (1 if G empty else 0)
        R = |G∩P|/|G| if G non-empty else (1 if P empty else 0)
        F1 = harmonic mean (0 if P+R == 0)
    i.e. correctly predicting "no evidence" when the expert cited none is a
    perfect match, and citing sentences when none were annotated is penalised.
  * macro P/R/F1 restricted to records with non-empty gold evidence (reported
    separately so the empty-set rule cannot inflate the headline number).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import ALL_LABELS, EXCLUSION_LABELS, INCLUSION_LABELS, INVALID_LABEL


def classification_report(gold: list[str], pred: list[str], labels: list[str]) -> dict:
    gold, pred = list(gold), list(pred)
    n = len(gold)
    if n == 0:
        return {"n": 0}
    acc = float(np.mean([g == p for g, p in zip(gold, pred)]))
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=labels, average=None, zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(gold, pred, labels=labels, average="macro", zero_division=0)
    cm_labels = labels + [INVALID_LABEL]
    cm = confusion_matrix(gold, pred, labels=cm_labels)
    return {
        "n": n,
        "accuracy": acc,
        "macro_f1": float(macro_f1),
        "n_invalid": int(sum(x == INVALID_LABEL for x in pred)),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                            "support": int(s[i])} for i, lab in enumerate(labels)},
        "confusion_matrix": {"labels_rows_gold": labels, "labels_cols_pred": cm_labels,
                             "matrix": cm[: len(labels)].tolist()},
    }


def _record_prf(g: set, p: set):
    tp = len(g & p)
    prec = tp / len(p) if p else (1.0 if not g else 0.0)
    rec = tp / len(g) if g else (1.0 if not p else 0.0)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def evidence_report(gold_sets: list[list[int]], pred_sets: list[list[int]]) -> dict:
    G = [set(g) for g in gold_sets]
    P = [set(p) for p in pred_sets]
    tp = sum(len(g & p) for g, p in zip(G, P))
    npred = sum(len(p) for p in P)
    ngold = sum(len(g) for g in G)
    micro_p = tp / npred if npred else 0.0
    micro_r = tp / ngold if ngold else 0.0
    micro_f = 0.0 if micro_p + micro_r == 0 else 2 * micro_p * micro_r / (micro_p + micro_r)

    per = [_record_prf(g, p) for g, p in zip(G, P)]
    nonempty = [x for x, g in zip(per, G) if g]

    def avg(rows):
        if not rows:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
        a = np.array(rows)
        return {"precision": float(a[:, 0].mean()), "recall": float(a[:, 1].mean()),
                "f1": float(a[:, 2].mean()), "n": len(rows)}

    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f,
                  "tp": tp, "n_pred_ids": npred, "n_gold_ids": ngold},
        "macro_all_records": avg(per),
        "macro_nonempty_gold": avg(nonempty),
        "n_records": len(G),
        "n_empty_gold": sum(1 for g in G if not g),
        "n_empty_pred": sum(1 for p in P if not p),
        "n_both_empty": sum(1 for g, p in zip(G, P) if not g and not p),
    }


def full_report(rows: list[dict]) -> dict:
    """rows: dicts with criterion_type, gold_label, pred_label, gold_evidence, pred_evidence."""
    out = {}
    subsets = {"overall": (rows, ALL_LABELS),
               "inclusion": ([r for r in rows if r["criterion_type"] == "inclusion"], INCLUSION_LABELS),
               "exclusion": ([r for r in rows if r["criterion_type"] == "exclusion"], EXCLUSION_LABELS)}
    for name, (rs, labels) in subsets.items():
        out[name] = {
            "classification": classification_report([r["gold_label"] for r in rs],
                                                    [r["pred_label"] for r in rs], labels),
            "evidence": evidence_report([r["gold_evidence"] for r in rs],
                                        [r["pred_evidence"] for r in rs]),
        }
    return out


def summary_line(report: dict) -> str:
    o = report["overall"]
    c, e = o["classification"], o["evidence"]
    return (f"n={c['n']}  acc={c['accuracy']:.3f}  macroF1={c['macro_f1']:.3f}  invalid={c['n_invalid']}  "
            f"| evidence microF1={e['micro']['f1']:.3f}  macroF1(all)={e['macro_all_records']['f1']:.3f}  "
            f"macroF1(nonempty gold)={e['macro_nonempty_gold']['f1']:.3f}")
