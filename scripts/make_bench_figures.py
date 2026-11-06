"""Figures for Tasks 3-5 from results/bench/*.json (PNG + PDF in results/figures/)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "results" / "bench"
FIG = ROOT / "results" / "figures"
C = {"no_cache": "#9aa5b1", "kv_cache": "#3b6ea5", "reference": "#d1495b", "tiled_128": "#edae49",
     "torch_sdpa_fused": "#3b6ea5"}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG / f"{name}.png")


def kv():
    p = B / "kv_float16.json"
    if not p.exists():
        return
    g = json.load(open(p)).get("grid", [])
    ins = sorted({r["input_len"] for r in g})
    outs = sorted({r["output_len"] for r in g})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for L, ls in zip(ins, ["-", "--", ":"]):
        rows = [r for r in g if r["input_len"] == L]
        for mode in ("no_cache", "kv_cache"):
            axes[0].plot([r["output_len"] for r in rows], [r[mode]["total_ms"] / 1e3 for r in rows], ls,
                         marker="o", color=C[mode], label=f"{mode.replace('_', '-')}, in={L}")
            axes[1].plot([r["output_len"] for r in rows], [r[mode]["tokens_per_s"] for r in rows], ls,
                         marker="o", color=C[mode])
    axes[0].set_xlabel("generated tokens"); axes[0].set_ylabel("end-to-end time (s)")
    axes[1].set_xlabel("generated tokens"); axes[1].set_ylabel("output tokens / s")
    for ax in axes:
        ax.grid(alpha=0.3); ax.set_xticks(outs)
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "kv_cache_benchmark")


def attention():
    p = B / "attention.json"
    if not p.exists():
        return
    rows = json.load(open(p))["performance"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for name in ("reference", "tiled_128", "torch_sdpa_fused"):
        xs = [r["seq_len"] for r in rows if r[name]["ms"] is not None]
        axes[0].plot(xs, [r[name]["ms"] for r in rows if r[name]["ms"] is not None], marker="o", color=C[name],
                     label=name.replace("_", " "))
        mem = [(r["seq_len"], r[name]["peak_extra_mem_mb"]) for r in rows if r[name].get("peak_extra_mem_mb")]
        if mem:
            axes[1].plot(*zip(*mem), marker="o", color=C[name])
    for ax, yl in zip(axes, ("time per call (ms)", "peak extra memory (MB)")):
        ax.set_xscale("log", base=2); ax.set_xlabel("sequence length"); ax.set_ylabel(yl)
        if ax.lines:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "attention_benchmark")


def spec():
    p = B / "spec_float16.json"
    if not p.exists():
        return
    d = json.load(open(p))
    nan = float("nan")
    val = lambda x: nan if x is None else x
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3))
    colors = ["#3b6ea5", "#d1495b", "#66a182", "#edae49"]
    for (dname, res), col in zip(d["drafts"].items(), colors):
        ks = sorted(res["by_k"], key=int)
        x = [int(k) for k in ks]
        for s, ls in (("task", "-"), ("summary", "--")):
            rows = [res["by_k"][k][s] for k in ks]
            axes[0].plot(x, [val(r["acceptance_rate"]) for r in rows], ls, marker="o", color=col,
                         label=f"{dname}, {s}")
            axes[1].plot(x, [r["speedup_vs_greedy"] for r in rows], ls, marker="o", color=col)
            if s == "task" and "predicted" in rows[0]:
                axes[1].plot(x, [val(r["predicted"]["speedup_formula"]) for r in rows], "x", color=col, ms=7)
            axes[2].plot(x, [r["tokens_per_target_forward"] for r in rows], ls, marker="o", color=col)
    axes[1].axhline(1.0, color="black", lw=0.8)
    axes[0].plot([], [], "x", color="gray", label="predicted speedup (task)")
    titles = ("draft acceptance rate", "speedup vs target greedy", "tokens per target forward")
    for ax, yl in zip(axes, titles):
        ax.set_xlabel("draft length k"); ax.set_ylabel(yl); ax.grid(alpha=0.3)
        ax.set_xticks(sorted({int(k) for r in d["drafts"].values() for k in r["by_k"]}))
    h, l = axes[0].get_legend_handles_labels()            # one shared legend under the panels
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "speculative_benchmark")


if __name__ == "__main__":
    kv(); attention(); spec()
