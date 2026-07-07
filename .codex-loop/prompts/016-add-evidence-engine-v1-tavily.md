---
title: Add EvidenceEngine v1 using Tavily
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_evidence_engine.py", "tests/test_workflow.py"]
---

# Task

Add `EvidenceEngine` v1 using the existing Tavily integration.

# Context

The current web search tool is too basic. It only takes a query, calls Tavily, records result count, and returns raw provider JSON. We need a structured observation layer for decision-grade research.

This is an engineering abstraction. It is not a named component in the papers. It operationalizes the paper-style “observation” function for open-world web research.

# Required files

Create:

```text
src/agents/evidence_engine.py
tests/test_evidence_engine.py
```

# Required models

Add internal Pydantic or dataclass models:

```text
EvidenceRequest
EvidenceQuery
EvidenceItem
EvidenceSource
EvidenceBundle
SearchMode
SourceTier
```

Keep them internal unless API exposure is needed.

Suggested fields:

EvidenceRequest:
```text
run_id
task_id optional
branch
objective
claim_id optional
claim_text optional
search_mode: exploratory | verification | contradiction | primary_source | historical
max_sources
max_cost_usd optional
preferred_domains
excluded_domains
freshness_window optional
```

EvidenceItem:
```text
source_url
source_title
publisher optional
published_at optional
retrieved_at
source_tier
snippet
support_type: supports | contradicts | contextual | unknown
confidence
limitations
```

EvidenceBundle:
```text
request
queries
items
raw_result_artifact_uri optional
source_quality_summary
```

# Tavily implementation

Implement an engine wrapper around the current Tavily search path:

```python
EvidenceEngine.search(request: EvidenceRequest) -> EvidenceBundle
```

Use Tavily only for v1. No Brave/Exa/Firecrawl yet.

# Behavior

- Generate 1-3 search queries from EvidenceRequest.
- Use branch/search_mode to shape queries.
- For verification mode, include claim text in query.
- For contradiction mode, include terms like "risk", "counterargument", "opposite", "failed", etc.
- Convert Tavily results into EvidenceItems.
- Do not invent source dates if not present.
- Save raw provider output to GCS if the artifact store is available, or keep this integrated through existing tool flow later.
- Keep budget consumption through existing ResearchTools / budget path where possible.

# Integration

Do not replace `web_search` globally yet.

Add a method to `ResearchTools` or an adjacent class:

```python
evidence_search(run_id, request) -> EvidenceBundle
```

If full integration is too much, implement the module and unit tests first.

# Tests

- exploratory request creates search queries
- verification request includes claim text
- contradiction request creates opposition/counterargument query
- Tavily-like raw response converts into EvidenceItems
- source URLs and snippets are preserved
- no dates are invented
- empty results produce empty bundle without crash

# Acceptance criteria

```bash
ruff check .
pytest tests/test_evidence_engine.py tests/test_workflow.py
```

passes.

# Constraints

- No new external providers yet.
- No new credentials.
- No vector DB.
- No broad runtime refactor.
