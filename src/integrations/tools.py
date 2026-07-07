from datetime import date, timedelta
import re
from typing import Any
from uuid import UUID, uuid4

import httpx

from src.agents.evidence_engine import (
    BraveProvider,
    EvidenceBundle,
    EvidenceEngine,
    EvidenceRequest,
    TavilyProvider,
)
from src.common.budget import PersistentBudget
from src.integrations.llm import LangfuseRecorder


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
        brave_search_api_key: str | None = None,
        brave_search_base_url: str | None = None,
    ) -> None:
        self.budget = budget
        self.recorder = recorder
        self.tavily_api_key = tavily_api_key
        self.market_data_api_key = market_data_api_key
        self.tavily_base_url = tavily_base_url.rstrip("/")
        self.market_data_base_url = market_data_base_url.rstrip("/")
        self.brave_search_api_key = (brave_search_api_key or "").strip()
        self.brave_search_base_url = (
            brave_search_base_url or "https://api.search.brave.com/res/v1/web/search"
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

    async def evidence_search(
        self, run_id: UUID, request: EvidenceRequest
    ) -> EvidenceBundle:
        if request.run_id != run_id:
            request = request.model_copy(update={"run_id": run_id})
        providers = [TavilyProvider(self.web_search, run_id)]
        if self.brave_search_api_key:
            providers.append(BraveProvider(self.brave_search, run_id))
        engine = EvidenceEngine(providers=providers)
        return await engine.search(request)

    async def market_data(self, run_id: UUID, ticker: str) -> dict[str, Any]:
        await self.budget.consume_tool(run_id, "market_data", 1)
        end = date.today()
        start = end - timedelta(days=365)
        url = f"{self.market_data_base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
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
                "input": ticker,
                "output": {"result_count": len(result.get("results", []))},
            },
        )
        return result
