"""Answer extraction and grading."""
from __future__ import annotations

import re


def extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...} in text (balanced braces), or None."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
        j += 1
    return None  # unbalanced (e.g. probe cut off)


def complete_probe(gen: str) -> str | None:
    """A probe is forced to start with '\\boxed{'; gen is what the model wrote after it."""
    return extract_boxed("\\boxed{" + gen)


_STRIP = [r"\\left", r"\\right", r"\\!", r"\\,", r"\\;", r"\\ ", r"\$", r"\\text\{\s*\}", r"^\s*\\displaystyle"]


def normalize(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    for p in _STRIP:
        s = re.sub(p, "", s)
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("^\\circ", "").replace("^{\\circ}", "")
    s = s.replace(" ", "").rstrip(".").lower()
    s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)  # 1,000 -> 1000
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def _mv_equal(gold: str, pred: str) -> bool | None:
    try:
        from math_verify import parse, verify
    except Exception:  # noqa: BLE001
        return None
    try:
        g = parse("$" + gold + "$")
        p = parse("$" + pred + "$")
        if not g or not p:
            return None
        return bool(verify(g, p))
    except Exception:  # noqa: BLE001
        return None


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    a, b = normalize(pred), normalize(gold)
    if a == b:
        return True
    mv = _mv_equal(gold, pred)
    return bool(mv)


class Grader:
    """Caches (pred, gold) -> bool, since trajectories repeat the same answers many times."""

    def __init__(self):
        self.cache: dict[tuple[str | None, str], bool] = {}

    def __call__(self, pred: str | None, gold: str) -> bool:
        k = (pred, gold)
        if k not in self.cache:
            self.cache[k] = is_correct(pred, gold)
        return self.cache[k]
