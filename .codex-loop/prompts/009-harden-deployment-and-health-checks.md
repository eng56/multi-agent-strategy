---
title: Harden deployment and health checks
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest"]
  - ["bash", "scripts/check-deployment.sh", "--help"]
---

# Task

Harden deployment and health checks for the managed-services demo.

# Context

The project deploys to managed services, including GKE/Vercel and external managed integrations. We need fast checks proving the running system has the expected commit and can reach critical dependencies without exposing secrets.

# Required changes

Add or improve health/debug endpoints or scripts so that we can verify:

- deployed git SHA / image tag
- backend version
- API reachable
- worker reachable or recently alive
- Redis/blackboard connectivity
- event bus publish ability or configured status
- artifact store configured status
- LLM provider configured status without making expensive calls
- Tavily/market data configured status without exposing keys

# Suggested implementation

Add an endpoint such as:

```text
GET /health
GET /version
GET /ready
```

or improve existing ones.

`/health` can be shallow.
`/ready` can check managed-service connectivity.
`/version` should expose commit SHA/build time if available.

# Security

- Never return secret values.
- Redact provider URLs if they contain credentials.
- Do not run expensive LLM calls in health checks.
- Make readiness checks timeout quickly.

# Scripts

Add:

```text
scripts/check-deployment.sh
```

It should call the health/version/readiness endpoints and print clear pass/fail output.

# Required tests

- health endpoint returns expected structure
- version endpoint works even if build SHA missing
- readiness redacts secrets
- deployment script supports `--help`

# Acceptance criteria

```bash
ruff check .
pytest
bash scripts/check-deployment.sh --help
```

passes.
