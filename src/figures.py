"""Figures for the report, from results/summary/summary.json.

Usage: python -m src.figures results/summary/summary.json --out report/figures
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"qwen3-0.6b": "#4C72B0", "qwen3-1.7b": "#DD8452", "qwen3-4b": "#55A868"}
LABEL = {"qwen3-0.6b": "Qwen3-0.6B", "qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B"}
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
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for m, res in S.items():
        tr = res[split]["trajectory"]
        xs = [i / (len(tr) - 1) for i in range(len(tr))]
        ax.plot(xs, [100 * v for v in tr], color=COLORS.get(m), label=LABEL.get(m, m), lw=1.6)
        ax.axhline(100 * res[split]["acc_oracle"], color=COLORS.get(m), ls=":", lw=1)
    ax.set_xlabel("fraction of thinking completed")
    ax.set_ylabel("accuracy (%)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_pareto(S, split, path):
    models = list(S)
    fig, axes = plt.subplots(1, len(models), figsize=(3.0 * len(models), 2.6), sharey=False)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        r = S[m][split]
        base_t, base_a = r["tokens_end"], 100 * r["acc_end"]
        for kind, mk in RULE_MARK.items():
            pts = [(x["tokens"] / base_t, 100 * x["acc"]) for x in r["rules"] if x["rule"] == kind]
            fr = pareto(pts)
            ax.plot([p[0] for p in fr], [p[1] for p in fr], marker=mk, ms=3, lw=1, label=kind)
        ax.scatter([1.0], [base_a], color="k", zorder=5, s=18, label="full thinking")
        ax.scatter([r["tokens_oracle"] / base_t], [100 * r["acc_oracle"]], color="r", marker="*", s=50, zorder=5,
                   label="oracle stop")
        cv = r["cv"]
        ax.scatter([cv["tokens"] / base_t], [100 * cv["acc"]], color="purple", marker="X", s=40, zorder=6,
                   label="CV-selected rule")
        ax.set_title(LABEL.get(m, m), fontsize=9)
        ax.set_xlabel("thinking tokens (× full)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy (%)")
    axes[-1].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
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
    ax.set_ylabel("flip traces (%)")
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
    if any("math500-L1" in r for r in S.values()):
        fig_flip_by_level(S, os.path.join(a.out, "flip_by_level.pdf"))
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
