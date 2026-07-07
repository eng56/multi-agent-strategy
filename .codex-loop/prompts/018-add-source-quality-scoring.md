---
title: Add source quality scoring
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_evidence_engine.py", "tests/test_workflow.py"]
---

# Task

Add source quality scoring to EvidenceEngine.

# Context

The current system can collect sources, but it does not yet distinguish high-quality primary evidence from weak web content. Verification should not treat every source equally.

# Required behavior

Add a deterministic source-quality scorer.

Suggested fields:

```text
source_tier
quality_score
domain
publisher
reason
```

Suggested tiers:

```text
primary
high_quality_secondary
news
blog_or_opinion
unknown
weak
```

Suggested scoring factors:

```text
primary domain bonus
government/central bank/statistical source bonus
major financial/news outlet bonus
recency if date available
data specificity
penalty for missing title/snippet/url
penalty for weak domains/content farms
```

# Initial domain rules

Add a small configurable or hard-coded map for finance/macro:

Primary:
```text
federalreserve.gov
stlouisfed.org
fred.stlouisfed.org
bls.gov
bea.gov
treasury.gov
sec.gov
cmegroup.com
nasdaq.com
nyse.com
```

High-quality secondary / news examples:
```text
reuters.com
bloomberg.com
ft.com
wsj.com
economist.com
imf.org
worldbank.org
bis.org
oecd.org
```

Keep this list small and easy to extend.

# Integration

EvidenceItem should include:

```text
source_tier
quality_score
source_quality_reason
```

Observation artifacts should include enough text for the UI/verifier to understand source quality.

# Tests

- federalreserve.gov classified primary
- FRED/St Louis Fed classified primary
- Reuters/Bloomberg/FT classified high-quality secondary or news
- unknown domain classified unknown
- missing URL/title/snippet penalized
- source quality is deterministic

# Acceptance criteria

```bash
ruff check .
pytest tests/test_evidence_engine.py tests/test_workflow.py
```

passes.
