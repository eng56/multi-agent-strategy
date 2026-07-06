---
title: Improve artifact graph UI
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest"]
  - ["bash", "-lc", "cd frontend && npm run build"]
---

# Task

Improve the frontend run detail view so the artifact graph is easier to inspect.

# Context

The backend now emits artifacts, legacy links, supports/contradicts/dependencies, visibility, branch, status, confidence, and principal actions. The UI should help understand the research process, not just show raw JSON.

# Required UI improvements

In the run detail page, add or improve sections for:

1. Principal actions timeline
2. Artifact list grouped by:
   - artifact type
   - status
   - branch
3. Evidence graph basics:
   - supports links
   - contradicts links
   - depends_on links
4. Final answer panel
5. Failure reason / stop reasons panel
6. Budget summary panel if available

# Constraints

- Keep UI simple.
- No new heavy graph visualization dependency unless already present.
- Text/table representation is acceptable.
- Do not break password gate or API proxy behavior.
- Do not expose API tokens to browser.

# Required tests/checks

Run frontend build and existing tests.

# Acceptance criteria

```bash
ruff check .
pytest
cd frontend && npm run build
```

passes.

The UI should make it possible to answer:
- What did the principal decide?
- Which artifacts are verified/rejected/disputed?
- Which evidence supports the final report?
- Why did the run stop?
