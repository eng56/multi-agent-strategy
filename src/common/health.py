from __future__ import annotations

import os
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SERVICE_NAME = "managed-multi-agent-strategy-api"
PACKAGE_NAME = "managed-multi-agent-strategy"
DEFAULT_WORKER_ROLES = (
    "planner-agent",
    "worker-agents",
    "verifier-agent",
    "skeptic-agent",
    "aggregator-agent",
    "judge-agent",
    "tool-runner",
)
WORKER_HEARTBEAT_INTERVAL_SECONDS = 30
WORKER_HEARTBEAT_TTL_SECONDS = 300
WORKER_HEARTBEAT_MAX_AGE_SECONDS = 180

_SENSITIVE_QUERY_PARTS = ("key", "token", "secret", "password", "auth", "credential")


def utc_timestamp(value: datetime | None = None) -> str:
    current = value or datetime.now(UTC)
    return current.astimezone(UTC).isoformat().replace("+00:00", "Z")


def worker_heartbeat_key(role: str) -> str:
    return f"worker:heartbeat:{role}"


def worker_heartbeat_payload(role: str, now: datetime | None = None) -> dict[str, object]:
    return {
        "service": SERVICE_NAME,
        "role": role,
        "pid": os.getpid(),
        "last_seen_at": utc_timestamp(now),
    }


def expected_worker_roles() -> list[str]:
    raw_roles = os.getenv("HEALTH_WORKER_ROLES")
    if raw_roles is None:
        return list(DEFAULT_WORKER_ROLES)
    return [role.strip() for role in raw_roles.split(",") if role.strip()]


def worker_heartbeat_max_age_seconds() -> int:
    raw_value = os.getenv("WORKER_HEARTBEAT_MAX_AGE_SECONDS")
    if not raw_value:
        return WORKER_HEARTBEAT_MAX_AGE_SECONDS
    try:
        value = int(raw_value)
    except ValueError:
        return WORKER_HEARTBEAT_MAX_AGE_SECONDS
    return value if value > 0 else WORKER_HEARTBEAT_MAX_AGE_SECONDS


def build_version_info() -> dict[str, str | None]:
    return {
        "service": SERVICE_NAME,
        "version": _backend_version(),
        "git_sha": _first_env(
            "GIT_SHA",
            "COMMIT_SHA",
            "GITHUB_SHA",
            "VERCEL_GIT_COMMIT_SHA",
            "SOURCE_VERSION",
        ),
        "image_tag": _first_env("IMAGE_TAG", "DOCKER_IMAGE_TAG", "CONTAINER_IMAGE"),
        "build_time": _first_env("BUILD_TIME", "BUILD_DATE", "SOURCE_DATE_EPOCH"),
        "environment": os.getenv("ENVIRONMENT", "unknown"),
    }


def redact_url(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return text
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError:
        return "<redacted-url>"
    if not parsed.scheme or not parsed.netloc:
        return text

    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if port is not None:
        netloc = f"{netloc}:{port}"

    query = urlencode(
        [
            (key, "<redacted>" if _is_sensitive_query_key(key) else item_value)
            for key, item_value in parse_qsl(parsed.query, keep_blank_values=True)
        ],
        safe="<>",
    )
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, ""))


def _backend_version() -> str:
    env_version = os.getenv("BACKEND_VERSION")
    if env_version:
        return env_version
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "unknown"


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _is_sensitive_query_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _SENSITIVE_QUERY_PARTS)
