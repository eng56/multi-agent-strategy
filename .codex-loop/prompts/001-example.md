Task: Add a harmless repo health check script.

Acceptance:
- create `scripts/repo-health.sh`
- it runs `ruff check .`
- it runs `pytest`
- it prints clear success/failure
- do not change app runtime code
