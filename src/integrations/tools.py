from datetime import date, timedelta
import asyncio
import re
import time
from typing import Any
from uuid import UUID, uuid4

import httpx

from src.agents.evidence_engine import (
    BraveProvider,
    EvidenceBundle,
    EvidenceEngine,
    EvidenceRequest,
    ExaProvider,
    FirecrawlFetcher,
    TavilyProvider,
)
from src.common.budget import PersistentBudget
from src.integrations.llm import LangfuseRecorder


class MarketDataRateLimitExceeded(RuntimeError):
    pass


class ProductRateLimiter:
    def __init__(self, limit_per_minute: int, *, clock=time.monotonic) -> None:
        self.limit_per_minute = max(1, int(limit_per_minute))
        self.clock = clock
        self._timestamps_by_product: dict[str, list[float]] = {}

    def allow(self, product: str) -> bool:
        now = self.clock()
        timestamps = self._active_timestamps(product, now)
        if len(timestamps) >= self.limit_per_minute:
            self._timestamps_by_product[product] = timestamps
            return False
        timestamps.append(now)
        self._timestamps_by_product[product] = timestamps
        return True

    def delay_seconds(self, product: str) -> float:
        now = self.clock()
        timestamps = self._active_timestamps(product, now)
        if len(timestamps) < self.limit_per_minute:
            self._timestamps_by_product[product] = timestamps
            return 0.0
        oldest = min(timestamps)
        self._timestamps_by_product[product] = timestamps
        return max(0.0, 60 - (now - oldest))

    def _active_timestamps(self, product: str, now: float) -> list[float]:
        window_start = now - 60
        return [
            value
            for value in self._timestamps_by_product.get(product, [])
            if value > window_start
        ]


def clean_search_query(query: str, max_chars: int = 360) -> str:
    cleaned = re.sub(r"\s+", " ", query).strip()
    cleaned = cleaned.replace("```", "") or "investment research"
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(" ", 1)[0] or cleaned[:max_chars]


class ResearchTools:
    def __init__(
        self,
        budget: PersistentBudget,
        recorder: LangfuseRecorder,
        tavily_api_key: str,
        market_data_api_key: str,
        tavily_base_url: str,
        market_data_base_url: str,
        market_data_product_rate_limit_per_minute: int = 5,
        market_data_enabled_products: str = "stocks",
        brave_search_api_key: str | None = None,
        brave_search_base_url: str | None = None,
        exa_api_key: str | None = None,
        exa_base_url: str | None = None,
        firecrawl_api_key: str | None = None,
        firecrawl_base_url: str | None = None,
    ) -> None:
        self.budget = budget
        self.recorder = recorder
        self.tavily_api_key = tavily_api_key
        self.market_data_api_key = market_data_api_key
        self.tavily_base_url = tavily_base_url.rstrip("/")
        self.market_data_base_url = market_data_base_url.rstrip("/")
        self.market_data_enabled_products = {
            product.strip().lower()
            for product in market_data_enabled_products.split(",")
            if product.strip()
        } or {"stocks"}
        self.market_data_rate_limiter = ProductRateLimiter(
            market_data_product_rate_limit_per_minute
        )
        self._market_data_cache: dict[tuple[UUID, str, str], dict[str, Any]] = {}
        self.brave_search_api_key = (brave_search_api_key or "").strip()
        self.brave_search_base_url = (
            brave_search_base_url or "https://api.search.brave.com/res/v1/web/search"
        ).rstrip("/")
        self.exa_api_key = (exa_api_key or "").strip()
        self.exa_base_url = (exa_base_url or "https://api.exa.ai/search").rstrip("/")
        self.firecrawl_api_key = (firecrawl_api_key or "").strip()
        self.firecrawl_base_url = (
            firecrawl_base_url or "https://api.firecrawl.dev/v1/scrape"
        ).rstrip("/")

    async def web_search(self, run_id: UUID, query: str) -> dict[str, Any]:
        query = clean_search_query(query)
        await self.budget.consume_tool(run_id, "tavily", 1)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.tavily_base_url}/search",
                json={"api_key": self.tavily_api_key, "query": query, "search_depth": "basic"},
            )
            response.raise_for_status()
            result = response.json()
        await self.recorder.span(
            str(uuid4()),
            {
                "traceId": str(run_id),
                "name": "tavily-search",
                "input": query,
                "output": {"result_count": len(result.get("results", []))},
            },
        )
        return result

    async def brave_search(self, run_id: UUID, query: str) -> dict[str, Any]:
        if not self.brave_search_api_key:
            raise RuntimeError("Brave Search is not configured.")
        query = clean_search_query(query)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                self.brave_search_base_url,
                params={"q": query, "count": 10},
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self.brave_search_api_key,
                },
            )
            response.raise_for_status()
            result = response.json()
        web = result.get("web") if isinstance(result, dict) else None
        results = web.get("results", []) if isinstance(web, dict) else []
        await self.recorder.span(
            str(uuid4()),
            {
                "traceId": str(run_id),
                "name": "brave-search",
                "input": query,
                "output": {"result_count": len(results) if isinstance(results, list) else 0},
            },
        )
        return result

    async def exa_search(self, run_id: UUID, query: str) -> dict[str, Any]:
        if not self.exa_api_key:
            raise RuntimeError("Exa is not configured.")
        query = clean_search_query(query)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                self.exa_base_url,
                json={
                    "query": query,
                    "numResults": 10,
                    "contents": {"text": True, "highlights": True},
                },
                headers={
                    "Accept": "application/json",
                    "x-api-key": self.exa_api_key,
                },
            )
            response.raise_for_status()
            result = response.json()
        results = result.get("results", []) if isinstance(result, dict) else []
        await self.recorder.span(
            str(uuid4()),
            {
                "traceId": str(run_id),
                "name": "exa-search",
                "input": query,
                "output": {"result_count": len(results) if isinstance(results, list) else 0},
            },
        )
        return result

    async def firecrawl_fetch(self, run_id: UUID, source_url: str) -> dict[str, Any]:
        if not self.firecrawl_api_key:
            raise RuntimeError("Firecrawl is not configured.")
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                self.firecrawl_base_url,
                json={"url": source_url, "formats": ["markdown"]},
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.firecrawl_api_key}",
                },
            )
            response.raise_for_status()
            result = response.json()
        await self.recorder.span(
            str(uuid4()),
            {
                "traceId": str(run_id),
                "name": "firecrawl-fetch",
                "input": source_url,
                "output": {"fetched": True},
            },
        )
        return result

    async def evidence_search(
        self, run_id: UUID, request: EvidenceRequest
    ) -> EvidenceBundle:
        if request.run_id != run_id:
            request = request.model_copy(update={"run_id": run_id})
        providers = [TavilyProvider(self.web_search, run_id)]
        if self.brave_search_api_key:
            providers.append(BraveProvider(self.brave_search, run_id))
        if self.exa_api_key:
            providers.append(ExaProvider(self.exa_search, run_id))
        fetcher = (
            FirecrawlFetcher(self.firecrawl_fetch, run_id, max_urls=1)
            if self.firecrawl_api_key
            else None
        )
        engine = EvidenceEngine(providers=providers, fetcher=fetcher)
        return await engine.search(request)

    async def market_data(
        self, run_id: UUID, ticker: str, *, product: str = "stocks"
    ) -> dict[str, Any]:
        product = _normalize_market_product(product)
        if product not in self.market_data_enabled_products:
            raise RuntimeError(f"market data product is not enabled: {product}")
        cache_key = (run_id, product, ticker.upper())
        if cache_key in self._market_data_cache:
            return self._market_data_cache[cache_key]
        if not self.market_data_rate_limiter.allow(product):
            delay = self.market_data_rate_limiter.delay_seconds(product)
            if delay > 0:
                await asyncio.sleep(delay)
            if not self.market_data_rate_limiter.allow(product):
                raise MarketDataRateLimitExceeded(
                    f"market data product rate limit reached for {product}: "
                    f"{self.market_data_rate_limiter.limit_per_minute}/minute"
                )
        await self.budget.consume_tool(run_id, "market_data", 1)
        end = date.today()
        start = end - timedelta(days=365)
        provider_ticker = _provider_ticker(ticker, product)
        url = f"{self.market_data_base_url}/v2/aggs/ticker/{provider_ticker}/range/1/day/{start}/{end}"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                url,
                params={"adjusted": "true"},
                headers={"Authorization": f"Bearer {self.market_data_api_key}"},
            )
            response.raise_for_status()
            result = response.json()
        await self.recorder.span(
            str(uuid4()),
            {
                "traceId": str(run_id),
                "name": "massive-market-data",
                "input": {"ticker": ticker, "product": product},
                "output": {"result_count": len(result.get("results", []))},
            },
        )
        self._market_data_cache[cache_key] = result
        return result


def _normalize_market_product(product: str) -> str:
    normalized = product.strip().lower()
    aliases = {
        "stock": "stocks",
        "equities": "stocks",
        "equity": "stocks",
        "forex": "currencies",
        "fx": "currencies",
        "currency": "currencies",
        "index": "indices",
        "future": "futures",
        "option": "options",
    }
    return aliases.get(normalized, normalized)


def _provider_ticker(ticker: str, product: str) -> str:
    symbol = ticker.upper()
    if product == "currencies" and not symbol.startswith("C:"):
        return f"C:{symbol}"
    if product == "indices" and not symbol.startswith("I:"):
        return f"I:{symbol}"
    if product == "futures" and not symbol.startswith("F:"):
        return f"F:{symbol}"
    if product == "options" and not symbol.startswith("O:"):
        return f"O:{symbol}"
    return symbol
