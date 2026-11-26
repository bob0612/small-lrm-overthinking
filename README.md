# small-lrm-overthinking

**When do small reasoning models think themselves out of a correct answer, and can a stopping rule that needs no
gold answer catch it?**

Code, results and report for the DASE7506 (HKU) final research project. We sample thinking traces from
Qwen3-0.6B, 1.7B and 4B, force an answer every ~128 thinking tokens ("answer trajectory"), label each trace as
*never / kept / late / flip* (correct-then-wrong), and compare label-free stopping rules against an oracle that
stops at the first correct answer. A corruption test checks whether thinking after the answer stabilises is still
causally used.

## Layout
| Path | What it does |
|---|---|
| `src/data.py` | Loads GSM8K (500-problem seeded sample), MATH-500, AIME 2024 + 2025 from Hugging Face |
| `src/backend.py` | vLLM wrapper: thinking-mode sampling, greedy answer probes with log-probs, continuations |
| `src/run.py` | Stage 1 + 2: sample traces, probe the answer every `--every` tokens, write `traces.jsonl` |
| `src/grade.py` | `\boxed{}` extraction, normalisation, `math-verify` equivalence |
| `src/analysis.py` | Outcome categories, stopping rules, grouped 5-fold CV, bootstrap CIs, flip predictors |
| `src/intervene.py` | Commitment test: corrupt the last number at 25/50/75/90% of the trace, continue, compare to clean resampling |
| `src/figures.py`, `src/tables.py` | Figures and LaTeX tables for the report |
| `tests/` | CPU tests with a fake character-level model (no GPU or download needed) |
| `report/` | IEEE-format LaTeX source of the paper |

## Setup
One GPU with ≥ 40 GB (we used an NVIDIA A40 48 GB on RunPod, image with CUDA 12 + PyTorch 2.8), Python 3.11.
```bash
bash scripts/pod_setup.sh     # pip install -r requirements.txt (vllm 0.11.0, transformers 4.57.1, datasets 4.1.1, math-verify 0.8.0)
```
Note: vLLM 0.11.0 breaks with transformers ≥ 5, and `datasets` 5.x needs a newer `huggingface-hub`, hence the pins.
Models and data download automatically from the Hugging Face Hub (`Qwen/Qwen3-{0.6B,1.7B,4B}`, `openai/gsm8k`,
`HuggingFaceH4/MATH-500`, `HuggingFaceH4/aime_2024`, `math-ai/aime25`). No API keys are needed.

## Reproduce
```bash
python -m pytest -q tests                      # 3 CPU tests, ~2 s
bash scripts/run_all.sh                           # traces + probes for 3 models (~5 GPU-hours on an A40)
python -m src.analysis results/qwen3-0.6b results/qwen3-1.7b results/qwen3-4b --out results/summary
python -m src.intervene --run results/qwen3-1.7b   # commitment test (repeat per model)
python -m src.figures results/summary/summary.json --out report/figures
python -m src.tables  results/summary/summary.json --out report/tables
cd report && latexmk -pdf main.tex
```
`--limit 500` caps each dataset at 500 problems, which only affects GSM8K (seeded sample, seed 0). Each run writes
`config.json` with the exact model commit and vLLM version. Sampling: temperature 0.6, top-p 0.95, top-k 20,
2 samples per problem, 8,192 thinking tokens max; probes are greedy.

## Outputs
`results/summary/summary.json` holds every number in the report (per model, per dataset, per MATH level).
Raw `traces.jsonl` files (~100 MB per model, full thinking text) are not in git; see the release assets if
present, or regenerate them with the commands above.

## AI use
Code and drafting were done with help from an AI assistant (Claude, Anthropic). All numbers come from the scripts
in this repository.
