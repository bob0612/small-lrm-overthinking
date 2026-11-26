"""LaTeX tables for the report, from results/summary/summary.json (and intervene summaries).

Usage: python -m src.tables results/summary/summary.json --out report/tables
"""
from __future__ import annotations

import argparse
import json
import os

LABEL = {"qwen3-0.6b": "0.6B", "qwen3-1.7b": "1.7B", "qwen3-4b": "4B"}
DS = [("gsm8k", "GSM8K"), ("math500", "MATH-500"), ("aime", "AIME"), ("all", "All")]


def pct(x):
    return f"{100 * x:.1f}"


def table_outcomes(S):
    rows = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
            r"Model & Data & Acc. & Oracle & Flip & Never & Tokens & Oracle tok. \\", r"\midrule"]
    for m, res in S.items():
        for i, (k, name) in enumerate(DS):
            if k not in res:
                continue
            r = res[k]
            n = r["n_traces"]
            never = r["categories"].get("never", 0) / n
            rows.append(f"{LABEL.get(m, m) if i == 0 else ''} & {name} & {pct(r['acc_end'])} & {pct(r['acc_oracle'])} & "
                        f"{pct(r['flip_rate'])} & {pct(never)} & {r['tokens_end']:.0f} & {r['tokens_oracle']:.0f} \\\\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    rows.append(r"\end{tabular}")
    return "\n".join(rows)


def table_cv(S):
    rows = [r"\begin{tabular}{lrrrrr}", r"\toprule",
            r"Model & Full acc. & Full tok. & Rule acc. & Rule tok. & Saved \\", r"\midrule"]
    for m, res in S.items():
        r = res["all"]
        cv = r["cv"]
        rows.append(f"{LABEL.get(m, m)} & {pct(r['acc_end'])} & {r['tokens_end']:.0f} & {pct(cv['acc'])} & "
                    f"{cv['tokens']:.0f} & {pct(1 - cv['tokens'] / r['tokens_end'])}\\% \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def table_predictors(S):
    feats = ["conf_at_first_correct", "distinct_answers_so_far", "relative_position",
             "hesitations_per_1k_tokens_so_far", "difficulty_level"]
    names = {"conf_at_first_correct": "Answer confidence", "distinct_answers_so_far": "Distinct answers so far",
             "relative_position": "Relative position", "hesitations_per_1k_tokens_so_far": "Hesitations / 1k tok.",
             "difficulty_level": "Difficulty level"}
    ms = list(S)
    rows = [r"\begin{tabular}{l" + "r" * len(ms) + "}", r"\toprule",
            "Signal at first correct probe & " + " & ".join(LABEL.get(m, m) for m in ms) + r" \\", r"\midrule"]
    for f in feats:
        vals = []
        for m in ms:
            p = S[m]["all"]["flip_predictors"].get(f)
            vals.append("--" if p is None else f"{p['auc_flip_vs_kept']:.2f}")
        rows.append(f"{names[f]} & " + " & ".join(vals) + r" \\")
    nflip = " & ".join(str(S[m]["all"]["flip_predictors"]["difficulty_level"]["n_flip"]) for m in ms)
    rows += [r"\midrule", f"\\# flip traces & {nflip} \\\\", r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def table_commitment(S):
    """Answer changes after continuing from a cut, clean vs with the last number corrupted."""
    rows = [r"\begin{tabular}{llrrrr}", r"\toprule",
            r"Model & Cut & $n$ & Clean & Corrupt & Effect \\", r"\midrule"]
    for m, res in S.items():
        iv = res.get("all", {}).get("intervene") or res.get("intervene")
        if not iv:
            continue
        blocks = [(f"{float(p):.0%}".replace("%", r"\%"), d) for p, d in iv["by_position"].items()]
        if "before_stabilization" in iv:
            blocks += [("before stabil.", iv["before_stabilization"]), ("after stabil.", iv["after_stabilization"])]
        for i, (name, d) in enumerate(blocks):
            rows.append(f"{LABEL.get(m, m) if i == 0 else ''} & {name} & {d['corrupt']['n']} & "
                        f"{pct(d['clean']['changed'])} & {pct(d['corrupt']['changed'])} & {100 * d['effect']:+.1f} \\\\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    rows.append(r"\end{tabular}")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = json.load(open(a.summary))
    for name, fn in [("outcomes", table_outcomes), ("cv", table_cv), ("predictors", table_predictors),
                     ("commitment", table_commitment)]:
        with open(os.path.join(a.out, f"{name}.tex"), "w") as f:
            f.write(fn(S) + "\n")
    print("tables written to", a.out)


if __name__ == "__main__":
    main()
