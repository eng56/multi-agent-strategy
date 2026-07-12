import asyncio
from uuid import uuid4

from src.integrations.tools import ProductRateLimiter, ResearchTools
from src.markets.snapshot import MarketDataGap, MarketSnapshot, get_market_snapshot


def test_gc_f_is_not_sent_to_stocks_endpoint_and_uses_gld() -> None:
    calls = []

    async def fetch(candidate):
        calls.append(candidate.symbol)
        return {"results": [{"c": 187.4, "v": 1000}]}

    result = asyncio.run(
        get_market_snapshot(
            "market/gold",
            "Get GC=F market data for gold",
            fetch_market_data=fetch,
        )
    )

    assert isinstance(result, MarketSnapshot)
    assert result.candidate.symbol == "GLD"
    assert calls == ["GLD"]


def test_dxy_is_not_sent_to_stocks_endpoint_and_uses_uup_proxy() -> None:
    calls = []

    def fetch(candidate):
        calls.append(candidate.symbol)
        return {"results": [{"c": 28.5}]}

    result = asyncio.run(
        get_market_snapshot("market/fx", "Fetch DXY price data", fetch_market_data=fetch)
    )

    assert isinstance(result, MarketSnapshot)
    assert result.candidate.symbol == "UUP"
    assert calls == ["UUP"]


def test_spx_maps_to_spy_proxy_snapshot() -> None:
    calls = []

    async def fetch(candidate):
        calls.append(candidate.symbol)
        return {"results": [{"c": 640.0}]}

    result = asyncio.run(
        get_market_snapshot("market/equities", "Fetch SPX data", fetch_market_data=fetch)
    )

    assert isinstance(result, MarketSnapshot)
    assert result.candidate.symbol == "SPY"
    assert calls == ["SPY"]


def test_market_bonds_maps_to_tlt_snapshot() -> None:
    calls = []

    async def fetch(candidate):
        calls.append(candidate.symbol)
        return {"results": [{"c": 85.45}]}

    result = asyncio.run(
        get_market_snapshot("market/bonds", "Long-duration Treasury TLT data", fetch_market_data=fetch)
    )

    assert isinstance(result, MarketSnapshot)
    assert result.candidate.symbol == "TLT"
    assert calls == ["TLT"]


def test_zero_results_trigger_fallback_candidate() -> None:
    calls = []

    async def fetch(candidate):
        calls.append(candidate.symbol)
        if candidate.symbol == "SPY":
            return {"results": [], "resultsCount": 0}
        return {"results": [{"c": 501.0}], "resultsCount": 1}

    result = asyncio.run(
        get_market_snapshot(
            "market/equities",
            "Fetch tech growth equity market data",
            fetch_market_data=fetch,
        )
    )

    assert isinstance(result, MarketSnapshot)
    assert result.candidate.symbol == "QQQ"
    assert calls == ["SPY", "QQQ"]
    assert [attempt.status for attempt in result.attempts] == ["zero_results", "ok"]


def test_all_zero_results_return_data_gap() -> None:
    calls = []

    async def fetch(candidate):
        calls.append(candidate.symbol)
        return {"ticker": candidate.symbol, "results": [], "resultsCount": 0}

    result = asyncio.run(
        get_market_snapshot("market/gold", "Gold market data", fetch_market_data=fetch)
    )

    assert isinstance(result, MarketDataGap)
    assert result.reason == "all_candidates_failed"
    assert calls == ["GLD"]
    assert "Do not convert this provider/data gap into an investment thesis" in result.summary()


def test_rates_return_data_gap_without_stock_endpoint_attempt_even_when_tlt_is_mentioned() -> None:
    calls = []

    async def fetch(candidate):
        calls.append(candidate.symbol)
        return {"results": [{"c": 1}]}

    result = asyncio.run(
        get_market_snapshot("macro/rates", "Fed funds, TLT, and yield curve data", fetch_market_data=fetch)
    )

    assert isinstance(result, MarketDataGap)
    assert result.reason == "no_resolved_candidates"
    assert calls == []
    assert "FRED, Treasury, CME" in result.summary()


def test_global_long_duration_question_keeps_snapshot_grounded_to_task_branch() -> None:
    question = (
        "If the Fed signals faster rate cuts, compare U.S. equities, the U.S. dollar, "
        "gold, and long-duration bonds."
    )

    async def fetch(candidate):
        return {"results": [{"c": 1}], "resultsCount": 1}

    equities = asyncio.run(
        get_market_snapshot("market/equities Equity impact", question, fetch_market_data=fetch)
    )
    fx = asyncio.run(get_market_snapshot("market/fx USD impact", question, fetch_market_data=fetch))
    gold = asyncio.run(
        get_market_snapshot("market/gold Gold impact", question, fetch_market_data=fetch)
    )
    bonds = asyncio.run(
        get_market_snapshot("market/bonds Long-duration bond impact", question, fetch_market_data=fetch)
    )

    assert isinstance(equities, MarketSnapshot)
    assert isinstance(fx, MarketSnapshot)
    assert isinstance(gold, MarketSnapshot)
    assert isinstance(bonds, MarketSnapshot)
    assert equities.candidate.symbol == "SPY"
    assert fx.candidate.symbol == "UUP"
    assert gold.candidate.symbol == "GLD"
    assert bonds.candidate.symbol == "TLT"


def test_product_rate_limiter_blocks_sixth_same_product_call_in_one_minute() -> None:
    now = 1000.0
    limiter = ProductRateLimiter(5, clock=lambda: now)

    assert [limiter.allow("stocks") for _ in range(5)] == [True] * 5
    assert limiter.allow("stocks") is False


def test_product_rate_limiter_is_independent_by_product() -> None:
    now = 1000.0
    limiter = ProductRateLimiter(1, clock=lambda: now)

    assert limiter.allow("stocks") is True
    assert limiter.allow("stocks") is False
    assert limiter.allow("currencies") is True


def test_market_data_duplicate_product_symbol_request_reuses_run_cache(monkeypatch) -> None:
    calls = []

    class FakeBudget:
        async def consume_tool(self, run_id, provider, units):
            calls.append(("budget", run_id, provider, units))

    class FakeRecorder:
        async def span(self, *_args, **_kwargs):
            calls.append(("span",))

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"c": 1}], "resultsCount": 1}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, **_kwargs):
            calls.append(("http", url))
            return FakeResponse()

    monkeypatch.setattr("src.integrations.tools.httpx.AsyncClient", FakeClient)
    tools = ResearchTools(
        FakeBudget(),
        FakeRecorder(),
        "tavily",
        "market",
        "https://tavily.example.com",
        "https://market.example.com",
        market_data_product_rate_limit_per_minute=5,
        market_data_enabled_products="stocks,currencies",
    )
    run_id = uuid4()

    first = asyncio.run(tools.market_data(run_id, "SPY", product="stocks"))
    second = asyncio.run(tools.market_data(run_id, "SPY", product="stocks"))

    assert first == second
    assert len([call for call in calls if call[0] == "budget"]) == 1
    assert len([call for call in calls if call[0] == "http"]) == 1
