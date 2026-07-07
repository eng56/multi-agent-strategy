import asyncio

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


def test_tlt_maps_to_tlt_snapshot() -> None:
    calls = []

    async def fetch(candidate):
        calls.append(candidate.symbol)
        return {"results": [{"c": 85.45}]}

    result = asyncio.run(
        get_market_snapshot("macro/rates", "Long-duration Treasury TLT data", fetch_market_data=fetch)
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


def test_rates_return_data_gap_without_stock_endpoint_attempt() -> None:
    calls = []

    async def fetch(candidate):
        calls.append(candidate.symbol)
        return {"results": [{"c": 1}]}

    result = asyncio.run(
        get_market_snapshot("macro/rates", "Fed funds and yield curve data", fetch_market_data=fetch)
    )

    assert isinstance(result, MarketDataGap)
    assert result.reason == "no_resolved_candidates"
    assert calls == []
    assert "FRED, Treasury, CME" in result.summary()

