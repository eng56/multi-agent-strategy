---
title: Add KnowledgeRouter skeleton
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_workflow.py", "tests/test_run_state.py"]
  - ["pytest", "tests/test_knowledge_router.py"]
---

# Task

Add a minimal KnowledgeRouter skeleton.

# Context

The project is moving toward a multi-agent research system where agents should not all see everything. They should retrieve relevant public verified evidence, branch-local evidence, and selected cross-branch artifacts.

The next architectural piece is a KnowledgeRouter that decides what context each agent receives.

# Required changes

Create a small module, for example:

```text
src/agents/knowledge_router.py
```

It should expose a simple class/function such as:

```python
select_context_for_agent(
    *,
    agent_spec: AgentSpec,
    run_state: RunState,
    artifacts: list[Artifact],
    tasks: list[ResearchTask],
    max_items: int = 12,
) -> list[Artifact]
```

# Selection rules v1

Return artifacts ordered by relevance:

1. PUBLIC_VERIFIED artifacts relevant to the agent branch/tags.
2. Same-branch TEAM artifacts.
3. Artifacts explicitly linked by dependency/support/contradiction.
4. High-confidence verified claim/forecast artifacts.
5. Exclude rejected artifacts unless the agent is verifier, judge, or skeptic/counterargument role.

# Required integration

Do not fully refactor all prompts yet. Add the router and use it in at least one place where context is prepared for:
- claim creation, or
- aggregation, or
- verification

Keep this integration minimal.

# Required tests

- same-branch artifacts are selected
- unrelated branch artifacts are not selected by default
- public verified artifacts can cross branch
- rejected artifacts are excluded for normal research agents
- verifier/judge can see rejected/disputed artifacts
- max_items is respected

# Constraints

- No vector DB.
- No embeddings.
- No new external services.
- Deterministic heuristic routing only for v1.

# Acceptance criteria

```bash
ruff check .
pytest tests/test_workflow.py tests/test_run_state.py
pytest tests/test_knowledge_router.py
```

passes.
