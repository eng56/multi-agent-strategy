# Evidence-quality eval harness

This directory contains offline evaluation inputs for source-aware investment research runs.

The runner is intentionally local and explicit: it does not run during tests, does not require
production credentials for `--help`, and writes outputs under `evals/output/` so generated
artifacts are not committed.

## Run

```bash
ORCHESTRATOR_API_URL=http://34.27.225.48 \
ORCHESTRATOR_API_TOKEN=... \
bash scripts/run-evidence-eval.sh
```

Useful options:

```bash
bash scripts/run-evidence-eval.sh --help
bash scripts/run-evidence-eval.sh --limit 2 --budget 1.25 --timeout-seconds 900
bash scripts/run-evidence-eval.sh --questions evals/research_questions.json --output-dir evals/output/manual
```

## Outputs

Each run gets a directory containing `run-detail.json`. The runner also writes:

- `summary.json`
- `summary.csv`

The summary includes run status, final-answer presence, judge score, budget spent, artifact and
claim counts, dead-letter count, source count, and stop reasons.
