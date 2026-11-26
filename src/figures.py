"""Figures for the report, from results/summary/summary.json.

Usage: python -m src.figures results/summary/summary.json --out report/figures
"""
from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Match the paper's Times body text.
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})

COLORS = {"qwen3-0.6b": "#4C72B0", "qwen3-1.7b": "#DD8452", "qwen3-4b": "#55A868"}
LABEL = {"qwen3-0.6b": "Qwen3-0.6B", "qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B"}
RULE_NAME = {"fixed": "Fixed budget", "consistency": "Consistency", "confidence": "Confidence", "cons+conf": "Consistency + confidence"}
RULE_MARK = {"fixed": "s", "consistency": "o", "confidence": "^", "cons+conf": "D"}


def pareto(points):
    """Upper-left frontier of (tokens, acc) points."""
    pts = sorted(points)
    out, best = [], -1
    for t, a in pts:
        if a > best:
            out.append((t, a))
            best = a
    return out


def fig_trajectory(S, split, path):
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for m, res in S.items():
        tr = res[split]["trajectory"]
        # Bin 0 is before any probe exists, so it is left out rather than drawn as 0% accuracy.
        xs = [i / (len(tr) - 1) for i in range(1, len(tr))]
        ax.plot(xs, [100 * v for v in tr[1:]], color=COLORS.get(m), label=LABEL.get(m, m), lw=1.6, marker="o", ms=2)
        ax.axhline(100 * res[split]["acc_oracle"], color=COLORS.get(m), ls=":", lw=1)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Fraction of thinking completed")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_pareto(S, split, path):
    models = list(S)
    fig, axes = plt.subplots(1, len(models), figsize=(7.0, 2.35), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        r = S[m][split]
        base_t, base_a = r["tokens_end"], 100 * r["acc_end"]
        for kind, mk in RULE_MARK.items():
            pts = [(x["tokens"] / base_t, 100 * x["acc"]) for x in r["rules"] if x["rule"] == kind]
            fr = pareto(pts)
            ax.plot([p[0] for p in fr], [p[1] for p in fr], marker=mk, ms=3, lw=1, label=RULE_NAME[kind])
        ax.scatter([1.0], [base_a], color="k", zorder=5, s=18, label="Full thinking")
        ax.scatter([r["tokens_oracle"] / base_t], [100 * r["acc_oracle"]], color="r", marker="*", s=50, zorder=5,
                   label="Oracle stop")
        cv = r["cv"]
        ax.scatter([cv["tokens"] / base_t], [100 * cv["acc"]], color="purple", marker="X", s=40, zorder=6,
                   label="CV-selected rule")
        ax.set_title(LABEL.get(m, m), fontsize=8)
        ax.set_xlabel("Thinking tokens (relative to full)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, ncol=len(lab), loc="upper center", fontsize=7, handletextpad=0.3,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path)
    plt.close(fig)


def fig_flip_by_level(S, path):
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    w = 0.8 / max(len(S), 1)
    for i, (m, res) in enumerate(S.items()):
        ys = [100 * res.get(f"math500-L{lv}", {}).get("flip_rate", 0) for lv in range(1, 6)]
        ax.bar([lv + (i - (len(S) - 1) / 2) * w for lv in range(1, 6)], ys, width=w, color=COLORS.get(m),
               label=LABEL.get(m, m))
    ax.set_xlabel("MATH-500 difficulty level")
    ax.set_xticks(range(1, 6))
    ax.set_ylabel("Flip rate (%)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = json.load(open(a.summary))
    fig_trajectory(S, "all", os.path.join(a.out, "trajectory.pdf"))
    fig_pareto(S, "all", os.path.join(a.out, "pareto.pdf"))
    ex = os.path.join(os.path.dirname(os.path.dirname(a.summary)), "qwen3-1.7b")
    if os.path.isdir(ex):  # worked example for the overview figure
        from src.analysis import load_traces

        tr = [r for r in load_traces(ex) if r["id"] == "gsm8k-837" and r["sample"] == 1]
        if tr:
            fig_mechanism(tr[0], os.path.join(a.out, "mechanism.pdf"))
    if any("math500-L1" in r for r in S.values()):
        fig_flip_by_level(S, os.path.join(a.out, "flip_by_level.pdf"))
    print("figures written to", a.out)



def fig_mechanism(trace, path):
    """(a) How a trace is probed; (b) the answer trajectory of one real correct-then-wrong trace."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    green, red, grey, blue = "#2E8B57", "#C0392B", "#7F8C8D", "#2C5D8F"
    fig = plt.figure(figsize=(7.16, 2.35))
    axa = fig.add_axes([0.005, 0.02, 0.43, 0.9])
    axb = fig.add_axes([0.535, 0.17, 0.455, 0.72])

    # (a) schematic -------------------------------------------------------------------------
    axa.set_xlim(0, 10)
    axa.set_ylim(0, 6)
    axa.axis("off")
    fig.text(0.01, 0.95, "(a) Probing a thinking trace", fontsize=8.5, weight="bold")
    box = dict(boxstyle="round,pad=0.02,rounding_size=0.12", lw=0.8)
    axa.add_patch(FancyBboxPatch((0.2, 3.9), 1.5, 0.9, fc="#ECF0F1", ec=grey, **box))
    axa.text(0.95, 4.35, "problem $x$", ha="center", va="center", fontsize=7)
    axa.add_patch(FancyBboxPatch((1.85, 3.9), 6.9, 0.9, fc="#EAF2FB", ec=blue, **box))
    axa.text(5.3, 4.35, r"thinking $z_1 \dots z_L$ (sampled once)", ha="center", va="center", fontsize=7,
             bbox=dict(fc="#EAF2FB", ec="none", pad=0.5))
    axa.add_patch(FancyBboxPatch((8.9, 3.9), 0.95, 0.9, fc="#ECF0F1", ec=grey, **box))
    axa.text(9.37, 4.35, "answer", ha="center", va="center", fontsize=6.5)
    cuts = [3.0, 4.7, 6.4, 8.75]
    labels = ["$t_1$", "$t_2$", r"$t_3$", "$t_m{=}L$"]
    for c, lab in zip(cuts, labels):  # cut marks on the box edges only, so the box label stays readable
        axa.plot([c, c], [4.6, 4.95], color=blue, lw=1.2)
        axa.plot([c, c], [3.75, 4.1], color=blue, lw=1.2)
        axa.text(c, 5.05, lab, ha="center", va="bottom", fontsize=7)
        axa.add_patch(FancyArrowPatch((c, 3.8), (c, 2.75), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=grey))
    axa.add_patch(FancyBboxPatch((1.95, 1.65), 7.6, 1.0, fc="white", ec=grey, **box))
    axa.text(5.75, 2.32, r"prefix $z_{1:t_j}$ + forced suffix:", ha="center", va="center", fontsize=6.5)
    axa.text(5.75, 1.92, r"</think> The final answer is $\backslash$boxed{", ha="center", va="center", fontsize=6.2,
             family="monospace")
    axa.add_patch(FancyArrowPatch((5.75, 1.6), (5.75, 1.0), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=grey))
    axa.text(5.75, 0.6, r"greedy answer $a_j$ and its probability $c_j$ at every $t_j$", ha="center", va="center",
             fontsize=7)
    axa.text(5.75, 0.12, r"$\Rightarrow$ answer trajectory $(a_j, c_j)_{j=1}^{m}$", ha="center", va="center",
             fontsize=7)

    # (b) real trajectory ---------------------------------------------------------------------
    ps = trace["probes"]
    ts = [p["t"] for p in ps]
    cs = [math.exp(p["logp"]) if p["ans"] is not None else 0 for p in ps]
    oks = [p["ok"] for p in ps]
    axb.plot(ts, cs, color=grey, lw=0.8, zorder=1)
    axb.scatter(ts, cs, c=[green if o else red for o in oks], s=16, zorder=2, edgecolor="white", lw=0.4)
    prev = None
    for t, c, p in zip(ts, cs, ps):
        if p["ans"] != prev:
            axb.annotate(p["ans"], (t, c), textcoords="offset points", xytext=(0, -11 if p["ok"] else 5), ha="center", fontsize=6.5,
                         color=green if p["ok"] else red)
        prev = p["ans"]
    first = next(t for t, o in zip(ts, oks) if o)
    final = ps[-1]["ans"]
    j = len(ps) - 1
    while j > 0 and ps[j - 1]["ans"] == final:
        j -= 1
    stab = ts[j]
    axb.axvspan(stab, ts[-1], color="#F4D03F", alpha=0.18, lw=0)
    axb.axvline(first, color=green, lw=0.8, ls="--")
    axb.text(first - 30, 1.2, "oracle stop\n(first correct)", ha="right", va="top", fontsize=6.3, color=green)
    axb.text((stab + ts[-1]) / 2, 1.2, "answer fixed, thinking\ncontinues", ha="center", va="top",
             fontsize=6.3, color="#9A7D0A")
    axb.set_xlim(0, ts[-1] * 1.03)
    axb.set_ylim(0, 1.22)
    axb.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axb.set_xlabel("Thinking tokens")
    axb.set_ylabel("Probability $c_j$ of forced answer", fontsize=7.5)
    fig.text(0.535, 0.95, f"(b) A real flip trace (Qwen3-1.7B, GSM8K, gold answer {trace['gold']})",
             fontsize=8.5, weight="bold")
    axb.scatter([], [], c=green, s=16, label="correct")
    axb.scatter([], [], c=red, s=16, label="wrong")
    axb.legend(frameon=False, fontsize=6.5, loc="lower right", handletextpad=0.2)
    axb.grid(alpha=0.25)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
