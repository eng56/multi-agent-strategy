from datetime import date, timedelta
from typing import Any
from uuid import UUID, uuid4

import httpx

from src.common.budget import PersistentBudget
from src.integrations.credentials import EphemeralCredentialStore
from src.integrations.llm import LangfuseRecorder


class ResearchTools:
    def __init__(self, budget: PersistentBudget, credentials: EphemeralCredentialStore, recorder: LangfuseRecorder, tavily_base_url: str, polygon_base_url: str) -> None:
        self.budget = budget
        self.credentials = credentials
        self.recorder = recorder
        self.tavily_base_url = tavily_base_url.rstrip("/")
        self.polygon_base_url = polygon_base_url.rstrip("/")

    async def web_search(self, run_id: UUID, query: str) -> dict[str, Any]:
        await self.budget.consume_tool(run_id, "tavily", 1)
        key = await self.credentials.get_key(run_id, "tavily")
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(f"{self.tavily_base_url}/search", json={"api_key": key, "query": query, "search_depth": "basic"})
            response.raise_for_status()
            result = response.json()
        await self.recorder.span(str(uuid4()), {"traceId": str(run_id), "name": "tavily-search", "input": query, "output": {"result_count": len(result.get("results", []))}})
        return result

    async def market_data(self, run_id: UUID, ticker: str) -> dict[str, Any]:
        await self.budget.consume_tool(run_id, "market_data", 1)
        key = await self.credentials.get_key(run_id, "market_data")
        end = date.today()
        start = end - timedelta(days=365)
        url = f"{self.polygon_base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, params={"adjusted": "true", "apiKey": key})
            response.raise_for_status()
            result = response.json()
        await self.recorder.span(str(uuid4()), {"traceId": str(run_id), "name": "polygon-market-data", "input": ticker, "output": {"result_count": len(result.get("results", []))}})
        return result
