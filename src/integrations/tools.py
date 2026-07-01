from datetime import date, timedelta
from typing import Any
from uuid import UUID, uuid4

import httpx

from src.common.budget import PersistentBudget
from src.integrations.llm import LangfuseRecorder


class ResearchTools:
    def __init__(
        self,
        budget: PersistentBudget,
        recorder: LangfuseRecorder,
        tavily_api_key: str,
        market_data_api_key: str,
        tavily_base_url: str,
        market_data_base_url: str,
    ) -> None:
        self.budget = budget
        self.recorder = recorder
        self.tavily_api_key = tavily_api_key
        self.market_data_api_key = market_data_api_key
        self.tavily_base_url = tavily_base_url.rstrip("/")
        self.market_data_base_url = market_data_base_url.rstrip("/")

    async def web_search(self, run_id: UUID, query: str) -> dict[str, Any]:
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

    async def market_data(self, run_id: UUID, ticker: str) -> dict[str, Any]:
        await self.budget.consume_tool(run_id, "market_data", 1)
        end = date.today()
        start = end - timedelta(days=365)
        url = f"{self.market_data_base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                url, params={"adjusted": "true"}, headers={"Authorization": f"Bearer {self.market_data_api_key}"}
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
