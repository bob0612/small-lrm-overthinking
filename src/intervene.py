"""Commitment test. For a sample of traces, cut the thinking at several relative positions,
then continue sampling from (a) the clean prefix and (b) the same prefix with its last
number changed. If the final answer moves with the corruption more often than it moves
under clean resampling, the model is still using its reasoning at that point; if not, the
answer is already committed and the rest of the thinking is mostly epiphenomenal.

Output: <run>/intervene.jsonl, one line per (trace, position, condition).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data  # noqa: E402
from src.grade import extract_boxed  # noqa: E402

NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")


def corrupt_last_number(text: str, rng: random.Random, window: int = 400) -> str | None:
    """Change the last number in the final `window` characters (x -> x + d, d in +-1..3 or x*2)."""
    start = max(0, len(text) - window)
    ms = list(NUM.finditer(text, start))
    if not ms:
        return None
    m = ms[-1]
    x = m.group(1)
    if "." in x:
        new = str(round(float(x) + rng.choice([-1, 1]) * rng.choice([0.5, 1, 2]), 3))
    else:
        v = int(x)
        new = str(rng.choice([v + rng.choice([-3, -2, -1, 1, 2, 3]), v * 2 if v else 2]))
        if new == x:
            new = str(v + 1)
    return text[: m.start(1)] + new + text[m.end(1) :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with traces.jsonl and config.json")
    ap.add_argument("--per-category", type=int, default=60, help="traces sampled per outcome category")
    ap.add_argument("--positions", nargs="+", type=float, default=[0.25, 0.5, 0.75, 0.9])
    ap.add_argument("--max-new", type=int, default=6144, help="cap on new tokens; each job gets 1.5x the tokens the original trace still had, plus 512")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    from src.analysis import classify, load_traces, stabilization_fraction
    from src.backend import VLLMBackend

    traces = [r for r in load_traces(a.run) if r["sample"] == 0 and not r["truncated"]]
    rng = random.Random(a.seed)
    chosen = []
    for cat in ("kept", "flip", "late"):
        pool = [r for r in traces if classify(r) == cat]
        rng.shuffle(pool)
        chosen += pool[: a.per_category]
    items = {it["id"]: it for it in data.load(sorted({r["dataset"] for r in chosen}))}
    be = VLLMBackend(cfg["model"], cfg.get("revision"), max_model_len=cfg["max_think"] + 2048, seed=a.seed)

    # Intervene on the exact thinking text of each chosen trace.
    jobs, meta, budgets = [], [], []
    for r in chosen:
        p = be.chat_prompt_ids(data.user_message(items[r["id"]]))
        think_text = r["think_text"]
        stab = stabilization_fraction(r)
        for pos in a.positions:
            cut = think_text[: int(len(think_text) * pos)]
            cut = cut[: cut.rfind("\n") + 1] or cut
            bad = corrupt_last_number(cut, rng)
            if bad is None:
                continue
            budget = min(a.max_new, int(1.5 * (1 - pos) * r["think_len"]) + 512)
            for cond, txt in (("clean", cut), ("corrupt", bad)):
                jobs.append(p + be.encode(txt))
                budgets.append(budget)
                meta.append(dict(id=r["id"], dataset=r["dataset"], category=classify(r), pos=pos, cond=cond,
                                 stab=round(stab, 4), after_stab=pos >= stab,
                                 orig_answer=r["probes"][-1]["ans"], gold=r["gold"]))
    print(f"{len(chosen)} traces, {len(jobs)} continuations", flush=True)
    outs = be.continue_from(jobs, max_tokens=budgets, seed=a.seed + 7)
    with open(os.path.join(a.run, "intervene.jsonl"), "w") as f:
        for m, o in zip(meta, outs):
            text = be.decode(o)
            m["answer"] = extract_boxed(text.split("</think>")[-1]) if "</think>" in text else None
            m["new_tokens"] = len(o)
            f.write(json.dumps(m) + "\n")
    print(f"wrote {len(meta)} rows")


if __name__ == "__main__":
    main()
