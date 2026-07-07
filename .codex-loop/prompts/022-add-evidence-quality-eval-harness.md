---
title: Add evidence-quality eval harness
mode: code
allow_secrets: false
checks:
  - ["bash", "scripts/run-evidence-eval.sh", "--help"]
  - ["ruff", "check", "."]
  - ["pytest"]
---

# Task

Add an offline evidence-quality evaluation harness.

# Context

The next bottleneck is research quality, not more architecture. We need to run a set of canonical prompts and inspect whether the system produces useful, grounded, source-aware reports.

# Required files

Create:

```text
evals/
  research_questions.json
  README.md
scripts/run-evidence-eval.sh
```

# Evaluation questions

Add 5-10 canonical investment/macro research questions, including:

```text
Fed surprise rate cut: gold/USD/equities
CPI upside surprise: rates/gold/equities
Oil supply shock: oil/inflation/equities/USD
China stimulus: commodities/EM/FX
AI capex slowdown: semiconductors/cloud/software
```

# Script behavior

`run-evidence-eval.sh` should:

1. Require `ORCHESTRATOR_API_URL` and `ORCHESTRATOR_API_TOKEN`.
2. Submit each question with a configurable budget.
3. Poll until terminal status or timeout.
4. Save each `run-detail.json`.
5. Write a summary CSV/JSON with:
   - run_id
   - status
   - final_present
   - judge_score
   - budget_spent
   - artifact_count
   - verified_claim_count
   - rejected_claim_count
   - disputed_claim_count
   - dead_letter_count
   - source_count
   - stop_reasons
6. Exit non-zero if too many runs fail before final answer.

# Acceptance criteria

```bash
bash scripts/run-evidence-eval.sh --help
ruff check .
pytest
```

passes.

# Constraints

- Do not call production automatically in tests.
- Do not commit eval output files.
- Add output directory to `.gitignore` if needed.
