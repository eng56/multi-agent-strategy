from src.markets.resolver import (
    AssetClass,
    MarketDataCapabilities,
    is_valid_for_stocks_endpoint,
    resolve_instruments,
)


def symbols(candidates):
    return [candidate.symbol for candidate in candidates]


def test_gold_resolves_to_gld_proxy_without_spot_or_futures_provider() -> None:
    candidates = resolve_instruments("market/gold", "Get GC=F gold market data")

    assert symbols(candidates) == ["GLD"]
    assert candidates[0].asset_class == AssetClass.ETF
    assert candidates[0].endpoint_family == "stocks"
    assert candidates[0].is_proxy is True
    assert is_valid_for_stocks_endpoint(candidates[0])


def test_gold_uses_real_instruments_before_etf_when_providers_are_configured() -> None:
    candidates = resolve_instruments(
        "market/gold",
        "Gold price snapshot",
        capabilities=MarketDataCapabilities(forex=True, futures=True),
    )

    assert symbols(candidates) == ["XAUUSD", "GC", "GLD"]
    assert candidates[0].asset_class == AssetClass.COMMODITY_SPOT
    assert candidates[1].asset_class == AssetClass.COMMODITY_FUTURE
    assert candidates[2].asset_class == AssetClass.ETF


def test_dxy_defaults_to_uup_proxy_not_stocks_index_symbol() -> None:
    candidates = resolve_instruments("market/fx", "Get DXY market data")

    assert symbols(candidates) == ["UUP"]
    assert candidates[0].asset_class == AssetClass.ETF
    assert candidates[0].endpoint_family == "stocks"
    assert is_valid_for_stocks_endpoint(candidates[0])


def test_dxy_index_candidate_requires_index_provider() -> None:
    candidates = resolve_instruments(
        "market/fx",
        "Get DXY market data",
        capabilities=MarketDataCapabilities(indices=True),
    )

    assert symbols(candidates) == ["DXY", "UUP"]
    assert candidates[0].asset_class == AssetClass.INDEX
    assert candidates[0].endpoint_family == "indices"
    assert not is_valid_for_stocks_endpoint(candidates[0])


def test_spx_maps_to_spy_proxy() -> None:
    candidates = resolve_instruments("market/equities", "Get SPX market data")

    assert symbols(candidates)[0] == "SPY"
    assert candidates[0].asset_class == AssetClass.ETF
    assert is_valid_for_stocks_endpoint(candidates[0])


def test_us_equities_adds_growth_and_small_cap_proxies_when_requested() -> None:
    candidates = resolve_instruments(
        "market/equities",
        "Compare tech growth and small caps after Fed cuts",
    )

    assert symbols(candidates) == ["SPY", "QQQ", "IWM"]


def test_tlt_maps_to_tlt() -> None:
    candidates = resolve_instruments("macro/rates", "Long-duration bond TLT data")

    assert symbols(candidates) == ["TLT"]
    assert candidates[0].asset_class == AssetClass.ETF
    assert candidates[0].is_proxy is False


def test_rates_do_not_use_stock_endpoint() -> None:
    candidates = resolve_instruments("macro/rates", "Fed funds and yield curve evidence")

    assert candidates == []


def test_plain_stock_ticker_can_use_stocks_endpoint() -> None:
    candidates = resolve_instruments("market/equities", "Get SAP ticker market data")

    assert symbols(candidates) == ["SAP"]
    assert candidates[0].asset_class == AssetClass.STOCK
    assert is_valid_for_stocks_endpoint(candidates[0])


def test_global_long_duration_bond_question_does_not_force_tlt_for_other_branches() -> None:
    question = (
        "If the Fed signals faster rate cuts, compare U.S. equities, the U.S. dollar, "
        "gold, and long-duration bonds."
    )

    equities = resolve_instruments("market/equities Equity market impact", question)
    fx = resolve_instruments("market/fx U.S. dollar impact", question)
    gold = resolve_instruments("market/gold Gold impact", question)
    bonds = resolve_instruments("market/bonds Long-duration bond impact", question)

    assert symbols(equities)[0] == "SPY"
    assert symbols(fx) == ["UUP"]
    assert symbols(gold) == ["GLD"]
    assert symbols(bonds) == ["TLT"]
    assert "TLT" not in symbols(equities)
    assert "TLT" not in symbols(fx)
    assert "TLT" not in symbols(gold)
