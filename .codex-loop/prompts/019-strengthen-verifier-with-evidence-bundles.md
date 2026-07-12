---
title: Strengthen verifier with evidence bundles
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_evidence_engine.py", "tests/test_workflow.py", "tests/test_run_state.py"]
---

# Task

Strengthen claim verification using EvidenceEngine outputs and source quality.

# Context

The verifier currently risks treating search results as generic support. We need verification to reason over linked observations/evidence items, source quality, contradictions, and limitations.

# Required changes

Extend Verification or add a companion artifact so verification captures:

```text
supported_parts
unsupported_parts
contradictions
required_caveats
source_quality_summary
evidence_item_refs
```

If changing the `Verification` model is too broad, store these fields in a VERIFICATION artifact text/metadata-like summary first.

# Verifier prompt requirements

The verifier prompt should receive:

```text
claim text
linked observation summaries
evidence items with URLs/snippets/source quality
contradictory/contextual evidence if available
```

It should output strict JSON:

```json
{
  "verdict": "verified|rejected|uncertain",
  "rationale": "...",
  "confidence": 0.0,
  "supported_parts": ["..."],
  "unsupported_parts": ["..."],
  "contradictions": ["..."],
  "required_caveats": ["..."],
  "source_quality_summary": "..."
}
```

# Rules

- A claim cannot be `verified` based only on weak/unknown sources unless it is low-stakes/contextual.
- If evidence supports only part of a claim, verdict should be `uncertain` or include required caveats.
- If sources conflict, verdict should be `uncertain` or `rejected` depending on strength.
- Do not promote claim artifact to PUBLIC_VERIFIED if verifier verdict is uncertain.

# Tests

- strong primary source support -> verified
- weak source only -> uncertain
- contradiction -> rejected or uncertain
- partial support creates required caveats
- verified claim artifact promotion still works
- uncertain claim artifact becomes disputed

# Acceptance criteria

```bash
ruff check .
pytest tests/test_evidence_engine.py tests/test_workflow.py tests/test_run_state.py
```

passes.
