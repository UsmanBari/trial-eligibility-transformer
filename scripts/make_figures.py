"""Figures for the report (PNG + PDF in results/figures/).

Usage:
  python scripts/make_figures.py --train-log outputs/lora_qwen25_1p5b/train_log.json \
      --cm ft_best_val --compare baseline_qwen25_1p5b ft_best_val --compare-names "Zero-shot" "Fine-tuned"
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
SHORT = {"included": "incl", "not included": "not incl", "excluded": "excl", "not excluded": "not excl",
         "not enough information": "NEI", "not applicable": "N/A", "invalid": "invalid"}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG / f"{name}.png")


def training_curve(log_path):
    log = json.load(open(log_path))
    tr = [(e["step"], e["loss"]) for e in log if "loss" in e and "eval_loss" not in e]
    ev = [(e["step"], e["eval_loss"], e.get("epoch")) for e in log if "eval_loss" in e]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(*zip(*tr), label="train loss", color="#3b6ea5", lw=1.6)
    if ev:
        s, l, _ = zip(*ev)
        ax.plot(s, l, "o-", label="validation loss", color="#d1495b", lw=1.6)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (target tokens)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    save(fig, "training_curve")


def confusion(tag, subset="overall"):
    m = json.load(open(ROOT / "results" / tag / "metrics.json"))[subset]["classification"]["confusion_matrix"]
    M = np.array(m["matrix"])
    rows, cols = m["labels_rows_gold"], m["labels_cols_pred"]
    if M[:, -1].sum() == 0:                       # hide empty "invalid" column
        M, cols = M[:, :-1], cols[:-1]
    fig, ax = plt.subplots(figsize=(0.75 * len(cols) + 1.6, 0.6 * len(rows) + 1.2))
    ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(cols)), [SHORT[c] for c in cols], rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), [SHORT[r] for r in rows])
    ax.set_xlabel("predicted")
    ax.set_ylabel("expert (gold)")
    vmax = M.max() if M.size else 1
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.6 * vmax else "black")
    save(fig, f"cm_{tag}_{subset}")


def compare(tags, names):
    keys = [("Accuracy", lambda m: m["overall"]["classification"]["accuracy"]),
            ("Macro-F1", lambda m: m["overall"]["classification"]["macro_f1"]),
            ("Incl. acc", lambda m: m["inclusion"]["classification"]["accuracy"]),
            ("Excl. acc", lambda m: m["exclusion"]["classification"]["accuracy"]),
            ("Evid. micro-F1", lambda m: m["overall"]["evidence"]["micro"]["f1"])]
    ms = [json.load(open(ROOT / "results" / t / "metrics.json")) for t in tags]
    x = np.arange(len(keys))
    w = 0.8 / len(tags)
    colors = ["#9aa5b1", "#3b6ea5", "#d1495b", "#66a182"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, (m, n) in enumerate(zip(ms, names)):
        vals = [f(m) for _, f in keys]
        bars = ax.bar(x + (i - (len(tags) - 1) / 2) * w, vals, w, label=n, color=colors[i % 4])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, [k for k, _ in keys])
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=len(tags), loc="lower left", bbox_to_anchor=(0, 1.0))
    save(fig, "compare_" + "_vs_".join(tags))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-log")
    ap.add_argument("--cm", nargs="*", default=[])
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--compare-names", nargs="*", default=None)
    a = ap.parse_args()
    if a.train_log:
        training_curve(a.train_log)
    for t in a.cm:
        for sub in ("overall", "inclusion", "exclusion"):
            confusion(t, sub)
    if a.compare:
        compare(a.compare, a.compare_names or a.compare)
