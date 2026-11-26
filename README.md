# small-lrm-overthinking

When do small reasoning models think themselves out of a correct answer, and can a stopping rule that
needs no gold answer catch it?

DASE7506 (HKU) final research project. Work in progress; this README will list the full setup, data and
commands to reproduce every table and figure in the report.

## Quick start (GPU, ~48 GB)
```bash
bash scripts/pod_setup.sh          # installs pinned dependencies
bash scripts/run_all.sh --limit 20 # smoke test; drop --limit for the full run
```

## Tests (CPU, no model needed)
```bash
python -m pytest -q tests
```
