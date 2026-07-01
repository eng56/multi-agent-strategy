import logging
from collections.abc import Awaitable, Callable

import httpx

from src.common.models import RunRequest

logger = logging.getLogger(__name__)


class ProviderValidationError(RuntimeError):
    """Raised when deployment-owned provider validation fails."""


async def _checked_response(
    provider: str,
    action: str,
    call: Callable[[], Awaitable[httpx.Response]],
) -> httpx.Response:
    logger.info("validating provider=%s action=%s", provider, action)
    try:
        response = await call()
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        body = exc.response.text[:500]
        logger.warning(
            "provider validation failed provider=%s action=%s status=%s response=%s",
            provider,
            action,
            status_code,
            body,
        )
        raise ProviderValidationError(
            f"{provider} {action} returned HTTP {status_code}: {body}"
        ) from None
    except httpx.RequestError as exc:
        logger.warning(
            "provider validation request failed provider=%s action=%s error_type=%s",
            provider,
            action,
            type(exc).__name__,
        )
        raise ProviderValidationError(
            f"{provider} {action} request failed: {type(exc).__name__}"
        ) from None
    logger.info("provider validation passed provider=%s action=%s", provider, action)
    return response


async def validate_demo_configuration(
    request: RunRequest,
    openrouter_api_key: str,
    tavily_api_key: str,
    market_data_api_key: str,
    openrouter_base_url: str,
    tavily_base_url: str,
    market_data_base_url: str,
) -> None:
    """Validate deployment-owned provider keys and selected models before starting a demo run."""
    timeout = httpx.Timeout(10.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        key_response = await _checked_response(
            "openrouter",
            "key",
            lambda: client.get(
                f"{openrouter_base_url.rstrip('/')}/key",
                headers={"Authorization": f"Bearer {openrouter_api_key}"},
            ),
        )
        key_data = key_response.json().get("data", {})
        remaining = key_data.get("limit_remaining")
        if remaining is not None and float(remaining) < request.llm_budget_usd:
            logger.warning(
                "provider validation failed provider=openrouter action=budget remaining=%s requested=%s",
                remaining,
                request.llm_budget_usd,
            )
            raise ProviderValidationError(
                "OpenRouter key remaining limit is below the requested LLM budget"
            )

        model_response = await _checked_response(
            "openrouter",
            "models",
            lambda: client.get(
                f"{openrouter_base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {openrouter_api_key}"},
            ),
        )
        available = {item["id"] for item in model_response.json().get("data", [])}
        selected = {
            policy.model for role in request.models.model_fields if (policy := getattr(request.models, role))
        }
        missing = sorted(selected - available)
        if missing:
            logger.warning(
                "provider validation failed provider=openrouter action=models missing=%s",
                missing,
            )
            raise ProviderValidationError(f"OpenRouter model IDs not found: {', '.join(missing)}")

        await _checked_response(
            "tavily",
            "search",
            lambda: client.post(
                f"{tavily_base_url.rstrip('/')}/search",
                json={
                    "api_key": tavily_api_key,
                    "query": "credential validation",
                    "search_depth": "basic",
                    "max_results": 1,
                },
            ),
        )

        await _checked_response(
            "market-data",
            "ticker-reference",
            lambda: client.get(
                f"{market_data_base_url.rstrip('/')}/v3/reference/tickers/AAPL",
                headers={"Authorization": f"Bearer {market_data_api_key}"},
            ),
        )
