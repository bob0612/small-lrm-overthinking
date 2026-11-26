"""Benchmark loading. Every item is a dict with keys: id, dataset, problem, answer, level.

All datasets are public on the Hugging Face Hub. Each loader pins the dataset id
and split; the README lists them. Subsampling uses a fixed seed so runs are reproducible.
"""
from __future__ import annotations

import random
import re

PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def _gsm8k():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i, r in enumerate(ds):
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append(dict(id=f"gsm8k-{i}", dataset="gsm8k", problem=r["question"], answer=ans, level=0))
    return out


def _math500():
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [
        dict(id=f"math500-{i}", dataset="math500", problem=r["problem"], answer=r["answer"],
             level=int(r["level"]), subject=r["subject"])
        for i, r in enumerate(ds)
    ]


def _aime():
    from datasets import load_dataset

    out = []
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    for i, r in enumerate(ds):
        out.append(dict(id=f"aime24-{i}", dataset="aime", problem=r["problem"], answer=str(r["answer"]), level=6))
    # AIME 2025: try a few public mirrors (same 30 problems).
    for name, split, qk in [("math-ai/aime25", "test", "problem"), ("yentinglin/aime_2025", "train", "problem")]:
        try:
            ds = load_dataset(name, split=split)
        except Exception:  # noqa: BLE001
            continue
        for i, r in enumerate(ds):
            out.append(dict(id=f"aime25-{i}", dataset="aime", problem=r[qk], answer=str(r["answer"]), level=6))
        break
    return out


LOADERS = {"gsm8k": _gsm8k, "math500": _math500, "aime": _aime}


def load(names: list[str], limit_per_dataset: int | None = None, seed: int = 0) -> list[dict]:
    items = []
    for n in names:
        rows = LOADERS[n]()
        if limit_per_dataset is not None and len(rows) > limit_per_dataset:
            rng = random.Random(seed)
            rows = sorted(rng.sample(rows, limit_per_dataset), key=lambda r: int(re.sub(r"\D", "", r["id"].split("-")[-1]) or 0))
        items.extend(rows)
    return items


def user_message(item: dict) -> str:
    return item["problem"] + PROMPT_SUFFIX
