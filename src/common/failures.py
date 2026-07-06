import re
from json import JSONDecodeError

import httpx
from pydantic import ValidationError


class PermanentEventError(RuntimeError):
    """Raised for malformed or impossible internal coordination events."""


class TransientEventError(RuntimeError):
    """Raised when a runtime event failed for a retryable provider/tool reason."""


HTTP_STATUS_RE = re.compile(r"\bHTTP\s+(\d{3})\b", re.IGNORECASE)
SENSITIVE_TEXT_RE = re.compile(
    r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+|((?:api[_-]?key|token|secret|password)=)[^&\s]+"
)
TRANSIENT_CLASS_MARKERS = (
    "timeout",
    "temporar",
    "toomanyrequests",
    "serviceunavailable",
    "deadlineexceeded",
)
TRANSIENT_MESSAGE_MARKERS = (
    "rate limit",
    "too many requests",
    "timeout",
    "timed out",
    "temporar",
    "network",
    "connection reset",
    "connection refused",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
)
SENSITIVE_KEY_MARKERS = ("key", "secret", "token", "password", "authorization")


def sanitize_text(value: object, limit: int = 500) -> str:
    text = str(value)[:limit]
    return SENSITIVE_TEXT_RE.sub(lambda match: f"{match.group(1) or match.group(2)}<redacted>", text)


def sanitize_payload(payload: dict[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in payload.items():
        lowered = key.lower()
        if any(marker in lowered for marker in SENSITIVE_KEY_MARKERS):
            sanitized[key] = "<redacted>"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_payload(value)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_payload(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            sanitized[key] = value
    return sanitized


def concise_exception(exc: Exception, limit: int = 240) -> str:
    return f"{type(exc).__name__}: {sanitize_text(exc, limit)}"


def _status_from_message(message: str) -> int | None:
    match = HTTP_STATUS_RE.search(message)
    return int(match.group(1)) if match else None


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code <= 599


def is_transient_failure(exc: Exception) -> bool:
    if isinstance(exc, TransientEventError):
        return True
    if isinstance(exc, PermanentEventError):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return _is_retryable_status(exc.response.status_code)

    class_name = type(exc).__name__.lower()
    if any(marker in class_name for marker in TRANSIENT_CLASS_MARKERS):
        return True

    message = str(exc).lower()
    status = _status_from_message(message)
    if status is not None:
        return _is_retryable_status(status)
    return any(marker in message for marker in TRANSIENT_MESSAGE_MARKERS)


def is_permanent_failure(exc: Exception) -> bool:
    if isinstance(exc, PermanentEventError):
        return True
    if isinstance(exc, TransientEventError):
        return False
    if isinstance(exc, (ValidationError, ValueError, KeyError, TypeError, JSONDecodeError)):
        return True
    return type(exc).__name__ == "LLMOutputError"
