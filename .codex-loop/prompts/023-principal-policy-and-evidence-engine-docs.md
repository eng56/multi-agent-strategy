---
title: Document PrincipalPolicy and EvidenceEngine architecture
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest"]
---

# Task

Document the new PrincipalPolicy + EvidenceEngine architecture.

# Context

EvidenceEngine is not a named paper component. It is our systems abstraction for implementing the paper-style observation function in open-world research.

# Required documentation

Create or update:

```text
docs/PRINCIPAL_POLICY.md
docs/EVIDENCE_ENGINE.md
```

# PRINCIPAL_POLICY.md should explain

- goal
- relation to paper loop
- RunState input
- PrincipalAction output
- deterministic policy v0
- shadow vs active mode
- action lifecycle
- idempotency
- budget constraints
- current limitations

# EVIDENCE_ENGINE.md should explain

- why the old web_search is insufficient
- EvidenceRequest / EvidenceBundle / EvidenceItem
- provider routing
- source quality scoring
- contradiction search
- verifier integration
- how artifacts are written
- current limitations
- next steps

# Root README

Add concise links to both docs.

# Required wording

Include this point clearly:

```text
EvidenceEngine is not a named component from the papers. It operationalizes the observation layer when moving from closed simulations to open-world research.
```

# Acceptance criteria

```bash
ruff check .
pytest
```

passes.
