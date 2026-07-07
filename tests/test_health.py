import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.common.health import worker_heartbeat_key, worker_heartbeat_payload


class FakeBlackboard:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    async def command(self, *parts: object) -> object:
        if parts[0] == "PING":
            return "PONG"
        if parts[0] == "GET":
            return self.values.get(str(parts[1]))
        raise AssertionError(f"unexpected command {parts}")


class FakeEvents:
    def __init__(self) -> None:
        self.published: list[object] = []

    def publish(self, topic: str, event: object, timeout: float = 10) -> None:
        self.published.append((topic, event, timeout))


@pytest.fixture(autouse=True)
def restore_app_runtime() -> object:
    had_runtime = "runtime" in app.state._state
    old_runtime = app.state._state.get("runtime")
    yield
    if had_runtime:
        app.state.runtime = old_runtime
    else:
        app.state._state.pop("runtime", None)


def test_health_endpoint_returns_expected_structure() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "managed-multi-agent-strategy-api",
        "checks": {"api": {"status": "ok"}},
    }


def test_version_endpoint_works_without_build_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GIT_SHA", "COMMIT_SHA", "GITHUB_SHA", "VERCEL_GIT_COMMIT_SHA", "SOURCE_VERSION"):
        monkeypatch.delenv(name, raising=False)

    response = TestClient(app).get("/version")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "managed-multi-agent-strategy-api"
    assert payload["version"]
    assert payload["git_sha"] is None


def test_readiness_redacts_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEALTH_WORKER_ROLES", "worker-agents")
    heartbeat = json.dumps(worker_heartbeat_payload("worker-agents", datetime.now(UTC)))
    events = FakeEvents()
    app.state.runtime = SimpleNamespace(
        settings=SimpleNamespace(
            confluent_bootstrap_servers="broker:9092",
            runtime_topic="agent-runtime",
            gcs_bucket_name="artifact-bucket",
            openrouter_base_url="https://user:open-secret@openrouter.example.com/api?api_key=open-secret&keep=1",
            openrouter_api_key="open-secret",
            tavily_base_url="https://tavily.example.com/search?token=tavily-secret",
            tavily_api_key="tavily-secret",
            brave_search_base_url="https://brave.example.com/search?token=brave-secret",
            brave_search_api_key="brave-secret",
            exa_base_url="https://exa.example.com/search?token=exa-secret",
            exa_api_key="exa-secret",
            firecrawl_base_url="https://firecrawl.example.com/scrape?token=firecrawl-secret",
            firecrawl_api_key="firecrawl-secret",
            market_data_provider="massive",
            market_data_base_url="https://market.example.com?password=market-secret",
            market_data_api_key="market-secret",
        ),
        blackboard=FakeBlackboard({worker_heartbeat_key("worker-agents"): heartbeat}),
        events=events,
    )

    response = TestClient(app).get("/ready")

    assert response.status_code == 200
    payload = response.json()
    serialized = json.dumps(payload)
    assert payload["status"] == "ok"
    assert events.published
    assert "open-secret" not in serialized
    assert "tavily-secret" not in serialized
    assert "brave-secret" not in serialized
    assert "exa-secret" not in serialized
    assert "firecrawl-secret" not in serialized
    assert "market-secret" not in serialized
    assert "user:" not in serialized
    assert payload["checks"]["llm_provider"]["base_url"] == (
        "https://openrouter.example.com/api?api_key=<redacted>&keep=1"
    )
    assert payload["checks"]["tavily"]["base_url"] == (
        "https://tavily.example.com/search?token=<redacted>"
    )
    assert payload["checks"]["brave_search"]["configured"] is True
    assert payload["checks"]["brave_search"]["base_url"] == (
        "https://brave.example.com/search?token=<redacted>"
    )
    assert payload["checks"]["exa"]["configured"] is True
    assert payload["checks"]["exa"]["base_url"] == (
        "https://exa.example.com/search?token=<redacted>"
    )
    assert payload["checks"]["firecrawl"]["configured"] is True
    assert payload["checks"]["firecrawl"]["base_url"] == (
        "https://firecrawl.example.com/scrape?token=<redacted>"
    )


def test_check_deployment_script_supports_help() -> None:
    script = Path("scripts/check-deployment.sh")

    result = subprocess.run(
        ["bash", str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "/health, /version, and /ready" in result.stdout
