import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import run as R, analysis as A, data
from src.backend import FakeBackend
from src.grade import extract_boxed, is_correct, complete_probe

def test_grade():
    assert extract_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert complete_probe(r"\frac{1}{2}}$") == r"\frac{1}{2}"
    assert complete_probe(r"\frac{1}{2") is None
    assert is_correct("1,000", "1000") and is_correct(r"\dfrac{1}{2}", r"\frac12") and is_correct("0.5", r"\frac{1}{2}")
    assert not is_correct("3", "4")

def make_items():
    return [dict(id="p-0", dataset="toy", problem="A", answer="5", level=1),
            dict(id="p-1", dataset="toy", problem="B", answer="7", level=2),
            dict(id="p-2", dataset="toy", problem="C", answer="9", level=3)]

SCRIPTS = {  # A: correct then flips; B: wrong then correct; C: never correct
    "A" + data.PROMPT_SUFFIX: "x\n" * 10 + "ANS=5\n" + "y\n" * 10 + "ANS=5\n" + "wait\n" * 10 + "ANS=6\n" + "z\n" * 5 + "</think>\\boxed{6}",
    "B" + data.PROMPT_SUFFIX: "ANS=1\n" + "x\n" * 20 + "ANS=7\n" + "y\n" * 20 + "ANS=7\n</think>\\boxed{7}",
    "C" + data.PROMPT_SUFFIX: "ANS=2\n" * 30 + "</think>\\boxed{2}",
}

def test_end_to_end(tmp_path):
    be = FakeBackend(SCRIPTS)
    path = R.run(be, make_items(), str(tmp_path), n=2, max_think=10_000, every=8, seed=0, log=lambda *_: None)
    tr = A.load_traces(str(tmp_path))
    assert len(tr) == 6
    cats = {r["id"]: A.classify(r) for r in tr}
    assert cats == {"p-0": "flip", "p-1": "kept", "p-2": "never"}
    a = [r for r in tr if r["id"] == "p-0"][0]
    assert a["natural_answer"] == "6" and not a["natural_ok"]
    # probes are at line ends and the last probe is the end of thinking
    assert a["probes"][-1]["t"] == a["think_len"]
    # resume: running again adds nothing
    R.run(be, make_items(), str(tmp_path), n=2, max_think=10_000, every=8, seed=0, log=lambda *_: None)
    assert sum(1 for _ in open(path)) == 6
    s = A.summarize(tr)
    assert s["flip_rate"] == round(2 / 6, 4)
    assert s["acc_oracle"] == round(4 / 6, 4) and s["acc_end"] == round(2 / 6, 4)
    # the fake model is confident once the same answer is written twice, so a confidence
    # rule stops A at its second ANS=5 (correct) and beats full thinking
    acc, tok = A.evaluate(tr, A.rule_conf(0.5))
    assert acc == round(4 / 6, 4) or abs(acc - 4 / 6) < 1e-9
    assert tok < s["tokens_end"]
    # early consistent guesses are a real failure mode of the consistency rule
    acc2, _ = A.evaluate(tr, A.rule_cons(2))
    assert acc2 <= acc
    assert "acc" in s["cv"]


def test_reports(tmp_path):
    from src import tables, figures
    be = FakeBackend(SCRIPTS)
    rd = tmp_path / "qwen3-0.6b"
    R.run(be, make_items(), str(rd), n=2, max_think=10_000, every=8, seed=0, log=lambda *_: None)
    with open(rd / "intervene.jsonl", "w") as f:
        for pos in (0.5, 0.9):
            for cond, ans in (("clean", "5"), ("corrupt", "6")):
                f.write(json.dumps(dict(id="p-0", pos=pos, cond=cond, answer=ans, orig_answer="5")) + "\n")
    import subprocess, sys as _s
    out = tmp_path / "summary"
    subprocess.run([_s.executable, "-m", "src.analysis", str(rd), "--out", str(out)], check=True,
                   cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    S = json.load(open(out / "summary.json"))
    assert S["qwen3-0.6b"]["intervene"]["0.5"]["effect"] == 1.0
    _s.argv = ["x", str(out / "summary.json"), "--out", str(tmp_path / "tab")]
    tables.main()
    assert "Flip" in open(tmp_path / "tab" / "outcomes.tex").read()
    _s.argv = ["x", str(out / "summary.json"), "--out", str(tmp_path / "fig")]
    figures.main()
    assert (tmp_path / "fig" / "pareto.pdf").exists()
