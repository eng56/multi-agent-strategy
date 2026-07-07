---
title: Add SearchProvider interface and optional Brave provider
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_evidence_engine.py", "tests/test_workflow.py"]
---

# Task

Add a provider interface for EvidenceEngine and implement an optional Brave Search provider.

# Context

EvidenceEngine v1 uses Tavily. To improve recall and avoid one-provider bias, add a provider interface. Brave should be optional and disabled unless credentials exist.

# Required design

Create provider abstraction:

```python
class SearchProvider:
    name: str
    async def search(self, query: EvidenceQuery) -> ProviderSearchResult:
        ...
```

Providers:

```text
TavilyProvider
BraveProvider
```

Brave provider should be optional:

```text
BRAVE_SEARCH_API_KEY optional
BRAVE_SEARCH_BASE_URL optional
```

If no Brave key is configured, EvidenceEngine should continue with Tavily only.

# Provider routing v1

Use:
- Tavily for normal/exploratory semantic search.
- Brave for broad recall, site queries, and contradiction/primary-source modes when configured.
- If Brave fails, fall back to Tavily and record provider failure in bundle limitations.

# Security

- Do not print API keys.
- Add env vars to `.env.example` as optional placeholders if needed.
- Readiness should report configured true/false, not secret values.

# Tests

- no Brave key -> provider router uses Tavily only
- Brave key present -> BraveProvider selected for primary_source/contradiction where appropriate
- Brave failure falls back to Tavily
- provider name is preserved in EvidenceItem/EvidenceSource

# Acceptance criteria

```bash
ruff check .
pytest tests/test_evidence_engine.py tests/test_workflow.py
```

passes.
