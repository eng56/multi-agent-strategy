---
title: Add optional Exa/Firecrawl retrieval hooks
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_evidence_engine.py", "tests/test_workflow.py"]
---

# Task

Add optional Exa and Firecrawl hooks to EvidenceEngine.

# Context

Tavily/Brave search return search results. For selected high-value URLs, the system may need cleaner fetched content. Add optional hooks without making them required.

# Required behavior

Add optional configuration:

```text
EXA_API_KEY optional
EXA_BASE_URL optional
FIRECRAWL_API_KEY optional
FIRECRAWL_BASE_URL optional
```

Add provider/fetcher stubs:

```text
ExaProvider
FirecrawlFetcher
```

# Scope

Do not overbuild.

V1 should:
- expose clean provider classes
- support disabled/no-key mode
- allow EvidenceEngine to call FirecrawlFetcher for top N URLs if configured
- allow ExaProvider to be used for semantic discovery if configured
- store fetched content snippets/markdown in raw artifact output when available
- never fail the whole evidence request if optional providers fail

# Tests

- no keys -> no optional provider usage
- Firecrawl configured -> fetcher called for top URL
- Exa configured -> provider can return normalized result
- optional provider failure is recorded as limitation, not fatal
- secrets are not logged

# Acceptance criteria

```bash
ruff check .
pytest tests/test_evidence_engine.py tests/test_workflow.py
```

passes.

# Constraints

- Do not require these providers for deployment.
- Do not add paid calls to normal tests.
- Use fake clients/mocks in tests.
