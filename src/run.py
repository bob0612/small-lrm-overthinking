"""Stage 1: sample thinking traces. Stage 2: probe the answer the model would give if it
stopped thinking at regular points of each trace ("answer trajectory").

Output: one JSON line per trace in <out>/traces.jsonl with
  id, dataset, level, gold, sample, prompt_len, think_len, truncated, natural_answer,
  probes: [{t, ans, logp, n_hes, chars}]
where t is the number of thinking tokens kept, logp the summed log-probability of the
answer tokens, n_hes the number of hesitation markers ("wait", "hmm", ...) seen so far.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data  # noqa: E402
from src.backend import PROBE_SUFFIX  # noqa: E402
from src.grade import complete_probe, extract_boxed  # noqa: E402

HES = re.compile(r"\b(wait|hmm|alternatively|but let me|let me double[- ]check|let me check|actually)\b", re.I)


def cut_points(think_ids: list[int], decode, every: int) -> list[int]:
    """Probe positions: roughly every `every` tokens, moved back to the end of a line when
    one is within the last half-window, so a probe never splits a sentence mid-way."""
    L = len(think_ids)
    pts = []
    t = every
    while t < L:
        best = t
        for k in range(t, max(t - every // 2, 1), -1):
            if "\n" in decode([think_ids[k - 1]]):
                best = k
                break
        if not pts or best > pts[-1]:
            pts.append(best)
        t += every
    if not pts or pts[-1] != L:
        pts.append(L)  # end of thinking (forced answer at the natural stopping point)
    return pts


def split_trace(gen: list[int], think_end_id: int) -> tuple[list[int], list[int], bool]:
    if think_end_id in gen:
        k = gen.index(think_end_id)
        return gen[:k], gen[k + 1 :], False
    return gen, [], True


def answer_from_probe(backend, ids: list[int], lps: list[float]) -> tuple[str | None, float]:
    """Decode the forced answer and sum log-probs over the tokens up to the closing brace."""
    text, total = "", 0.0
    for tid, lp in zip(ids, lps):
        text += backend.decode([tid])
        total += lp
        a = complete_probe(text)
        if a is not None:
            return a, total
    return None, total


def run(backend, items, out_dir, n, max_think, every, seed, batch_traces=256, log=print):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "traces.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["id"], r["sample"]))
    todo = [it for it in items if any((it["id"], s) not in done for s in range(n))]
    log(f"{len(items)} items, {len(todo)} to run")
    suffix = backend.encode(PROBE_SUFFIX)
    t0 = time.time()
    for b in range(0, len(todo), batch_traces // n or 1):
        chunk = todo[b : b + (batch_traces // n or 1)]
        prompts = [backend.chat_prompt_ids(data.user_message(it)) for it in chunk]
        gens = backend.generate_traces(prompts, n=n, max_tokens=max_think, seed=seed + b)
        recs, probe_prompts, owners = [], [], []
        for it, p, samples in zip(chunk, prompts, gens):
            for s, g in enumerate(samples):
                if (it["id"], s) in done:
                    continue
                think, after, trunc = split_trace(g, backend.think_end_id)
                pts = cut_points(think, backend.decode, every)
                rec = dict(id=it["id"], dataset=it["dataset"], level=it.get("level", 0), gold=it["answer"],
                           sample=s, prompt_len=len(p), think_len=len(think), truncated=trunc,
                           answer_len=len(after), natural_answer=extract_boxed(backend.decode(after)), probes=[],
                           think_text=backend.decode(think))
                think_text_cache = {}
                for t in pts:
                    probe_prompts.append(p + think[:t] + suffix)
                    owners.append((len(recs), t))
                    pre = backend.decode(think[:t])
                    think_text_cache[t] = (len(HES.findall(pre)), len(pre))
                rec["_hes"] = think_text_cache
                recs.append(rec)
        outs = backend.probe(probe_prompts)
        for (ri, t), (ids, lps) in zip(owners, outs):
            ans, lp = answer_from_probe(backend, ids, lps)
            nh, nc = recs[ri]["_hes"][t]
            recs[ri]["probes"].append(dict(t=t, ans=ans, logp=round(lp, 4), n_hes=nh, chars=nc))
        with open(path, "a") as f:
            for r in recs:
                r.pop("_hes")
                f.write(json.dumps(r) + "\n")
        log(f"  {min(b + len(chunk), len(todo))}/{len(todo)} items, {len(probe_prompts)} probes, {time.time() - t0:.0f}s")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "aime"])
    ap.add_argument("--limit", type=int, default=None, help="max problems per dataset (seeded subsample)")
    ap.add_argument("--n", type=int, default=2, help="samples per problem")
    ap.add_argument("--max-think", type=int, default=8192)
    ap.add_argument("--every", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from src.backend import VLLMBackend

    items = data.load(a.datasets, a.limit, seed=a.seed)
    be = VLLMBackend(a.model, a.revision, max_model_len=a.max_think + 2048, seed=a.seed)
    os.makedirs(a.out, exist_ok=True)
    cfg = vars(a)
    try:  # record the exact model commit that was used
        from huggingface_hub import model_info

        cfg["resolved_revision"] = model_info(a.model, revision=a.revision).sha
    except Exception as e:  # noqa: BLE001
        cfg["resolved_revision"] = f"unknown ({e.__class__.__name__})"
    import vllm

    cfg["vllm_version"] = vllm.__version__
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    run(be, items, a.out, a.n, a.max_think, a.every, a.seed)


if __name__ == "__main__":
    main()
