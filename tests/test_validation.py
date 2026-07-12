import asyncio

import httpx
import pytest

from src.integrations.validation import ProviderValidationError, _checked_response


def test_provider_validation_error_does_not_chain_secret_bearing_http_exception() -> None:
    async def call() -> httpx.Response:
        request = httpx.Request("GET", "https://api.polygon.io/v3/reference/tickers/AAPL?apiKey=secret-key")
        response = httpx.Response(401, request=request, text='{"error":"Unknown API Key"}')
        response.raise_for_status()
        return response

    with pytest.raises(ProviderValidationError) as exc_info:
        asyncio.run(_checked_response("market-data", "ticker-reference", call))

    assert exc_info.value.__cause__ is None
    assert "secret-key" not in str(exc_info.value)
