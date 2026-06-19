import httpx

from src.common.models import RunRequest


async def validate_demo_configuration(
    request: RunRequest,
    openrouter_api_key: str,
    tavily_api_key: str,
    market_data_api_key: str,
    openrouter_base_url: str,
    tavily_base_url: str,
    polygon_base_url: str,
) -> None:
    """Validate deployment-owned provider keys and selected models before starting a demo run."""
    async with httpx.AsyncClient(timeout=30) as client:
        key_response = await client.get(
            f"{openrouter_base_url.rstrip('/')}/key",
            headers={"Authorization": f"Bearer {openrouter_api_key}"},
        )
        key_response.raise_for_status()
        key_data = key_response.json().get("data", {})
        remaining = key_data.get("limit_remaining")
        if remaining is not None and float(remaining) < request.llm_budget_usd:
            raise ValueError("OpenRouter key remaining limit is below the requested LLM budget")

        model_response = await client.get(
            f"{openrouter_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {openrouter_api_key}"},
        )
        model_response.raise_for_status()
        available = {item["id"] for item in model_response.json().get("data", [])}
        selected = {
            policy.model for role in request.models.model_fields if (policy := getattr(request.models, role))
        }
        missing = sorted(selected - available)
        if missing:
            raise ValueError(f"OpenRouter model IDs not found: {', '.join(missing)}")

        tavily = await client.post(
            f"{tavily_base_url.rstrip('/')}/search",
            json={
                "api_key": tavily_api_key,
                "query": "credential validation",
                "search_depth": "basic",
                "max_results": 1,
            },
        )
        tavily.raise_for_status()

        market = await client.get(
            f"{polygon_base_url.rstrip('/')}/v3/reference/tickers/AAPL",
            params={"apiKey": market_data_api_key},
        )
        market.raise_for_status()
