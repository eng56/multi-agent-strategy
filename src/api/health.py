from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from src.common.health import (
    SERVICE_NAME,
    build_version_info,
    expected_worker_roles,
    redact_url,
    utc_timestamp,
    worker_heartbeat_key,
    worker_heartbeat_max_age_seconds,
)
from src.common.models import EventEnvelope, EventType

DEFAULT_READINESS_TIMEOUT_SECONDS = 2.0


def health_payload() -> dict[str, object]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "checks": {"api": {"status": "ok"}},
    }


async def readiness_payload(runtime: Any) -> dict[str, object]:
    checked_at = datetime.now(UTC)
    timeout = _readiness_timeout_seconds()
    settings = runtime.settings

    checks = {
        "api": {"status": "ok"},
        "blackboard": await _check_blackboard(runtime, timeout),
        "workers": await _check_workers(runtime, checked_at, timeout),
        "event_bus": await _check_event_bus(runtime, checked_at, timeout),
        "artifact_store": _artifact_store_check(settings),
        "llm_provider": _provider_check(
            "openrouter",
            settings.openrouter_base_url,
            settings.openrouter_api_key,
        ),
        "tavily": _provider_check("tavily", settings.tavily_base_url, settings.tavily_api_key),
        "brave_search": _optional_provider_check(
            "brave",
            getattr(settings, "brave_search_base_url", ""),
            getattr(settings, "brave_search_api_key", None),
        ),
        "market_data": _provider_check(
            settings.market_data_provider,
            settings.market_data_base_url,
            settings.market_data_api_key,
        ),
    }
    status = "ok" if all(check["status"] == "ok" for check in checks.values()) else "fail"
    return {
        "status": status,
        "service": SERVICE_NAME,
        "checked_at": utc_timestamp(checked_at),
        "version": build_version_info(),
        "checks": checks,
    }


async def missing_runtime_payload() -> dict[str, object]:
    return {
        "status": "fail",
        "service": SERVICE_NAME,
        "checked_at": utc_timestamp(),
        "version": build_version_info(),
        "checks": {
            "api": {"status": "ok"},
            "runtime": {"status": "fail", "reason": "runtime is not initialized"},
        },
    }


async def _check_blackboard(runtime: Any, timeout: float) -> dict[str, object]:
    try:
        result = await asyncio.wait_for(runtime.blackboard.command("PING"), timeout)
    except Exception as exc:
        return _failed_check(exc)
    status = "ok" if result in ("PONG", "OK", True) else "fail"
    return {"status": status, "operation": "PING"}


async def _check_workers(runtime: Any, now: datetime, timeout: float) -> dict[str, object]:
    roles = expected_worker_roles()
    max_age_seconds = worker_heartbeat_max_age_seconds()
    if not roles:
        return {"status": "ok", "max_age_seconds": max_age_seconds, "roles": {}}
    try:
        values = await asyncio.wait_for(
            asyncio.gather(
                *(runtime.blackboard.command("GET", worker_heartbeat_key(role)) for role in roles)
            ),
            timeout,
        )
    except Exception as exc:
        return _failed_check(exc, max_age_seconds=max_age_seconds, roles={})

    role_checks = {
        role: _worker_heartbeat_status(role, value, now, max_age_seconds)
        for role, value in zip(roles, values, strict=True)
    }
    status = "ok" if all(check["status"] == "ok" for check in role_checks.values()) else "fail"
    return {"status": status, "max_age_seconds": max_age_seconds, "roles": role_checks}


async def _check_event_bus(runtime: Any, checked_at: datetime, timeout: float) -> dict[str, object]:
    settings = runtime.settings
    if not settings.confluent_bootstrap_servers or not settings.runtime_topic:
        return {"status": "fail", "configured": False, "publish": "not_attempted"}
    event = EventEnvelope(
        type=EventType.HEALTH_CHECK,
        run_id=uuid4(),
        producer="orchestrator-api-readiness",
        payload={"checked_at": utc_timestamp(checked_at)},
    )
    try:
        await asyncio.wait_for(
            asyncio.to_thread(runtime.events.publish, settings.runtime_topic, event, timeout),
            timeout + 0.5,
        )
    except Exception as exc:
        return _failed_check(exc, configured=True, publish="failed")
    return {"status": "ok", "configured": True, "publish": "ok"}


def _artifact_store_check(settings: Any) -> dict[str, object]:
    configured = bool(getattr(settings, "gcs_bucket_name", ""))
    return {
        "status": "ok" if configured else "fail",
        "configured": configured,
        "bucket": "configured" if configured else "missing",
    }


def _provider_check(provider: str, base_url: object, api_key: object) -> dict[str, object]:
    configured = bool(str(base_url or "").strip()) and bool(str(api_key or "").strip())
    return {
        "status": "ok" if configured else "fail",
        "provider": provider,
        "configured": configured,
        "base_url": redact_url(base_url),
        "api_key_configured": bool(str(api_key or "").strip()),
    }


def _optional_provider_check(provider: str, base_url: object, api_key: object) -> dict[str, object]:
    return {
        "status": "ok",
        "provider": provider,
        "configured": bool(str(base_url or "").strip()) and bool(str(api_key or "").strip()),
        "base_url": redact_url(base_url),
        "api_key_configured": bool(str(api_key or "").strip()),
    }


def _worker_heartbeat_status(
    role: str, value: object, now: datetime, max_age_seconds: int
) -> dict[str, object]:
    if not value:
        return {"status": "fail", "reason": "missing"}
    try:
        payload = _heartbeat_payload(value)
        last_seen_at = _parse_timestamp(str(payload["last_seen_at"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {"status": "fail", "reason": "invalid"}

    age_seconds = max(0, int((now - last_seen_at).total_seconds()))
    status = "ok" if age_seconds <= max_age_seconds else "fail"
    result: dict[str, object] = {
        "status": status,
        "last_seen_at": utc_timestamp(last_seen_at),
        "age_seconds": age_seconds,
    }
    payload_role = payload.get("role")
    if payload_role and payload_role != role:
        result["status"] = "fail"
        result["reason"] = "role_mismatch"
    elif status != "ok":
        result["reason"] = "stale"
    return result


def _heartbeat_payload(value: object) -> dict[str, object]:
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        payload = json.loads(value)
    elif isinstance(value, dict):
        payload = value
    else:
        raise TypeError("unsupported heartbeat payload")
    if not isinstance(payload, dict):
        raise TypeError("heartbeat payload must be an object")
    return payload


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _failed_check(exc: Exception, **extra: object) -> dict[str, object]:
    return {
        "status": "fail",
        "error_type": type(exc).__name__,
        **extra,
    }


def _readiness_timeout_seconds() -> float:
    raw_value = os.getenv("READINESS_CHECK_TIMEOUT_SECONDS")
    if not raw_value:
        return DEFAULT_READINESS_TIMEOUT_SECONDS
    try:
        value = float(raw_value)
    except ValueError:
        return DEFAULT_READINESS_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_READINESS_TIMEOUT_SECONDS
