"""Analysis of answer trajectories: harmful overthinking, oracle frontier, label-free
stopping rules (with cross-validated thresholds), and predictors of flips.

Usage: python -m src.analysis results/qwen3-0.6b results/qwen3-1.7b ... --out results/summary
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grade import Grader  # noqa: E402

GRADE = Grader()


# ---------------------------------------------------------------- loading
def load_traces(run_dir: str) -> list[dict]:
    out = []
    with open(os.path.join(run_dir, "traces.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            for p in r["probes"]:
                p["ok"] = GRADE(p["ans"], r["gold"])
                p["conf"] = math.exp(p["logp"]) if p["ans"] is not None else 0.0
            r["natural_ok"] = GRADE(r["natural_answer"], r["gold"]) if r["natural_answer"] else r["probes"][-1]["ok"]
            out.append(r)
    return out


# ---------------------------------------------------------------- per-trace facts
def classify(r: dict) -> str:
    oks = [p["ok"] for p in r["probes"]]
    if not any(oks):
        return "never"
    if oks[-1]:
        return "kept" if oks.index(True) < len(oks) - 1 else "late"
    return "flip"  # was correct at some point, wrong when thinking ended


def first_correct_t(r: dict) -> int | None:
    for p in r["probes"]:
        if p["ok"]:
            return p["t"]
    return None


# ---------------------------------------------------------------- stopping rules
# A rule looks at the probes seen so far and says stop or not. The answer is the answer of
# the probe where it stops; the cost is the number of thinking tokens used up to there.
def rule_fixed(B):
    return ("fixed", {"B": B}, lambda ps, j: ps[j]["t"] >= B)


def rule_cons(k, tmin=0):
    def f(ps, j):
        if j + 1 < k or ps[j]["t"] < tmin:
            return False
        a = [ps[i]["ans"] for i in range(j - k + 1, j + 1)]
        return a[0] is not None and all(x == a[0] for x in a)

    return ("consistency", {"k": k, "tmin": tmin}, f)


def rule_conf(tau, tmin=0):
    return ("confidence", {"tau": tau, "tmin": tmin}, lambda ps, j: ps[j]["t"] >= tmin and ps[j]["conf"] >= tau)


def rule_cons_conf(k, tau):
    c = rule_cons(k)[2]
    return ("cons+conf", {"k": k, "tau": tau}, lambda ps, j: c(ps, j) and ps[j]["conf"] >= tau)


def all_rules():
    rules = [rule_fixed(B) for B in (256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144)]
    rules += [rule_cons(k, tmin) for k in (2, 3, 4, 5, 6, 8, 10) for tmin in (0, 256, 512, 1024)]
    rules += [rule_conf(t) for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)]
    rules += [rule_cons_conf(k, t) for k in (2, 3, 4) for t in (0.7, 0.9, 0.95)]
    return rules


def apply_rule(r: dict, rule) -> tuple[bool, int]:
    ps = r["probes"]
    for j in range(len(ps)):
        if rule[2](ps, j):
            return ps[j]["ok"], ps[j]["t"]
    return ps[-1]["ok"], ps[-1]["t"]


def evaluate(traces, rule):
    res = [apply_rule(r, rule) for r in traces]
    return sum(a for a, _ in res) / len(res), sum(t for _, t in res) / len(res)


def oracle(traces):
    """Stop at the first probe whose answer is correct (needs the gold answer)."""
    acc = sum(any(p["ok"] for p in r["probes"]) for r in traces) / len(traces)
    tok = sum(first_correct_t(r) or r["probes"][-1]["t"] for r in traces) / len(traces)
    return acc, tok


def baseline(traces):
    return (sum(r["probes"][-1]["ok"] for r in traces) / len(traces),
            sum(r["probes"][-1]["t"] for r in traces) / len(traces))


# ---------------------------------------------------------------- cross-validated selection
def cv_select(traces, rules, folds=5, tol=0.005, seed=0):
    """For each fold, pick on the other folds the rule with the fewest tokens whose accuracy
    is within `tol` of full thinking; report its accuracy and tokens on the held-out fold.
    Folds are grouped by problem id so samples of one problem never straddle folds."""
    ids = sorted({r["id"] for r in traces})
    random.Random(seed).shuffle(ids)
    fold_of = {pid: i % folds for i, pid in enumerate(ids)}
    picks, acc_sum, tok_sum, n = [], 0.0, 0.0, 0
    for k in range(folds):
        tr = [r for r in traces if fold_of[r["id"]] != k]
        te = [r for r in traces if fold_of[r["id"]] == k]
        if not te:
            continue
        base_acc, _ = baseline(tr)
        ev = [evaluate(tr, ru) for ru in rules]
        ok = [(t, i) for i, (a, t) in enumerate(ev) if a >= base_acc - tol]
        if ok:
            ru = rules[min(ok)[1]]
            a, t = evaluate(te, ru)
            picks.append(f"{ru[0]}{ru[1]}")
        else:
            a, t = baseline(te)
            picks.append("full")
        acc_sum += a * len(te)
        tok_sum += t * len(te)
        n += len(te)
    return dict(acc=acc_sum / n, tokens=tok_sum / n, picks=picks)


# ---------------------------------------------------------------- statistics
def bootstrap_ci(traces, fn, B=1000, seed=0):
    by = defaultdict(list)
    for r in traces:
        by[r["id"]].append(r)
    ids = list(by)
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        sample = [r for pid in (rng.choice(ids) for _ in ids) for r in by[pid]]
        vals.append(fn(sample))
    vals.sort()
    return vals[int(0.025 * B)], vals[int(0.975 * B) - 1]


def auc(pos, neg):
    """Mann-Whitney AUC: probability a random positive scores above a random negative."""
    if not pos or not neg:
        return float("nan")
    s = 0.0
    for p in pos:
        for q in neg:
            s += 1.0 if p > q else 0.5 if p == q else 0.0
    return s / (len(pos) * len(neg))


def flip_predictors(traces):
    """Among traces that were correct at some point: which signals, available without the
    gold answer at the first-correct probe, separate later flips from kept answers?"""
    feats = defaultdict(lambda: ([], []))
    for r in traces:
        c = classify(r)
        if c == "never":
            continue
        ps = r["probes"]
        j = next(i for i, p in enumerate(ps) if p["ok"])
        f = {
            "conf_at_first_correct": ps[j]["conf"],
            "relative_position": ps[j]["t"] / max(ps[-1]["t"], 1),
            "same_as_previous_probe": float(j > 0 and ps[j - 1]["ans"] == ps[j]["ans"]),
            "hesitations_per_1k_tokens_so_far": 1000 * ps[j]["n_hes"] / max(ps[j]["t"], 1),
            "difficulty_level": r.get("level", 0),
        }
        for k, v in f.items():
            feats[k][0 if c == "flip" else 1].append(v)
    return {k: dict(auc_flip_vs_kept=round(auc(a, b), 3), n_flip=len(a), n_kept=len(b),
                    mean_flip=round(sum(a) / max(len(a), 1), 3), mean_kept=round(sum(b) / max(len(b), 1), 3))
            for k, (a, b) in feats.items()}


def trajectory_curve(traces, bins=20):
    """Accuracy of the probed answer as a function of the fraction of thinking done."""
    acc = [[0, 0] for _ in range(bins + 1)]
    for r in traces:
        L = max(r["probes"][-1]["t"], 1)
        for b in range(bins + 1):
            x = b / bins * L
            # answer of the latest probe at or before x (no answer yet -> wrong)
            ok = False
            for p in r["probes"]:
                if p["t"] <= x + 1e-9:
                    ok = p["ok"]
                else:
                    break
            acc[b][0] += ok
            acc[b][1] += 1
    return [round(a / max(n, 1), 4) for a, n in acc]


# ---------------------------------------------------------------- report
def summarize(traces):
    cats = defaultdict(int)
    for r in traces:
        cats[classify(r)] += 1
    n = len(traces)
    ever = n - cats["never"]
    base = baseline(traces)
    orc = oracle(traces)
    rules = all_rules()
    curve = [dict(rule=ru[0], params=ru[1], acc=round(a, 4), tokens=round(t, 1))
             for ru in rules for a, t in [evaluate(traces, ru)]]
    wasted = [1 - (first_correct_t(r) / max(r["probes"][-1]["t"], 1)) for r in traces if classify(r) in ("kept", "late")]
    return dict(
        n_traces=n, n_problems=len({r["id"] for r in traces}),
        acc_end=round(base[0], 4), acc_end_ci=[round(x, 4) for x in bootstrap_ci(traces, lambda s: baseline(s)[0])],
        acc_natural=round(sum(r["natural_ok"] for r in traces) / n, 4),
        tokens_end=round(base[1], 1), truncated=round(sum(r["truncated"] for r in traces) / n, 4),
        acc_oracle=round(orc[0], 4), tokens_oracle=round(orc[1], 1),
        categories=dict(cats),
        flip_rate=round(cats["flip"] / n, 4),
        flip_rate_ci=[round(x, 4) for x in bootstrap_ci(traces, lambda s: sum(classify(r) == "flip" for r in s) / len(s))],
        flip_share_of_ever_correct=round(cats["flip"] / max(ever, 1), 4),
        mean_wasted_fraction_after_first_correct=round(sum(wasted) / max(len(wasted), 1), 4),
        rules=curve,
        cv=cv_select(traces, rules),
        flip_predictors=flip_predictors(traces),
        trajectory=trajectory_curve(traces),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allres = {}
    for rd in a.runs:
        tr = load_traces(rd)
        name = os.path.basename(rd.rstrip("/"))
        res = {"all": summarize(tr)}
        for ds in sorted({r["dataset"] for r in tr}):
            res[ds] = summarize([r for r in tr if r["dataset"] == ds])
        if any(r["dataset"] == "math500" for r in tr):
            for lv in range(1, 6):
                sub = [r for r in tr if r["dataset"] == "math500" and r["level"] == lv]
                if sub:
                    s = summarize(sub)
                    res[f"math500-L{lv}"] = {k: s[k] for k in ("n_traces", "acc_end", "acc_oracle", "flip_rate", "tokens_end")}
        allres[name] = res
        print(name, json.dumps({k: res["all"][k] for k in ("acc_end", "acc_oracle", "flip_rate", "tokens_end", "tokens_oracle", "cv")}))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(allres, f, indent=1)


if __name__ == "__main__":
    main()
