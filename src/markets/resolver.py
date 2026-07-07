import re
from dataclasses import dataclass
from enum import StrEnum


class AssetClass(StrEnum):
    STOCK = "stock"
    ETF = "etf"
    INDEX = "index"
    FX = "fx"
    COMMODITY_SPOT = "commodity_spot"
    COMMODITY_FUTURE = "commodity_future"
    RATES = "rates"
    MACRO_SERIES = "macro_series"


@dataclass(frozen=True)
class MarketDataCapabilities:
    """Provider endpoint families available to the market snapshot layer."""

    provider: str = "massive"
    stocks: bool = True
    indices: bool = False
    forex: bool = False
    futures: bool = False
    macro_series: bool = False


@dataclass(frozen=True)
class InstrumentCandidate:
    asset_name: str
    symbol: str
    provider: str
    asset_class: AssetClass
    endpoint_family: str
    currency: str
    is_proxy: bool
    priority: int
    rationale: str


STOCK_ENDPOINT_FAMILY = "stocks"
INDEX_ENDPOINT_FAMILY = "indices"
FOREX_ENDPOINT_FAMILY = "forex"
FUTURES_ENDPOINT_FAMILY = "futures"
MACRO_SERIES_ENDPOINT_FAMILY = "macro_series"

_STOCK_TOKEN_RE = re.compile(r"\b[A-Z]{1,5}\b")
_INVALID_STOCK_ENDPOINT_SYMBOLS = {
    "DXY",
    "SPX",
    "SPXW",
    "NDX",
    "RUT",
    "DJI",
    "VIX",
    "XAU",
    "XAUUSD",
    "GC",
    "GC=F",
    "SI=F",
    "CL=F",
}
_NON_TICKER_WORDS = {
    "A",
    "AN",
    "API",
    "CME",
    "CPI",
    "DXY",
    "ETF",
    "F",
    "FED",
    "FOMC",
    "FX",
    "GDP",
    "GET",
    "I",
    "PCE",
    "PMI",
    "QQQ",
    "SPX",
    "S",
    "THE",
    "TLT",
    "U",
    "USD",
    "US",
    "UST",
}


def resolve_instruments(
    asset_or_branch: str,
    question: str,
    capabilities: MarketDataCapabilities | None = None,
) -> list[InstrumentCandidate]:
    """Return deterministic, endpoint-valid candidates for an asset research task."""
    capabilities = capabilities or MarketDataCapabilities()
    text = f"{asset_or_branch} {question}"
    explicit_stock_symbol = _extract_stock_symbol(question)
    if (
        explicit_stock_symbol
        and capabilities.stocks
        and not _contains_non_stock_asset_terms(question)
    ):
        return [_stock_candidate(explicit_stock_symbol, capabilities, priority=5)]

    asset_key = _classify_branch(asset_or_branch) or _classify_asset(text)

    if asset_key == "gold":
        return _gold_candidates(capabilities)
    if asset_key == "us_equities":
        return _us_equity_candidates(text, capabilities)
    if asset_key == "usd":
        return _usd_candidates(capabilities)
    if asset_key == "long_duration_bonds":
        return _long_duration_bond_candidates(capabilities)
    if asset_key == "rates":
        return _rates_candidates(capabilities)

    stock_symbol = _extract_stock_symbol(question) or _extract_stock_symbol(asset_or_branch)
    if stock_symbol and capabilities.stocks:
        return [_stock_candidate(stock_symbol, capabilities, priority=100)]
    return []


def is_valid_for_stocks_endpoint(candidate: InstrumentCandidate) -> bool:
    return (
        candidate.endpoint_family == STOCK_ENDPOINT_FAMILY
        and candidate.asset_class in {AssetClass.STOCK, AssetClass.ETF}
        and candidate.symbol.upper() not in _INVALID_STOCK_ENDPOINT_SYMBOLS
        and "=" not in candidate.symbol
        and "/" not in candidate.symbol
    )


def _gold_candidates(capabilities: MarketDataCapabilities) -> list[InstrumentCandidate]:
    candidates: list[InstrumentCandidate] = []
    if capabilities.forex:
        candidates.append(
            InstrumentCandidate(
                asset_name="gold",
                symbol="XAUUSD",
                provider=capabilities.provider,
                asset_class=AssetClass.COMMODITY_SPOT,
                endpoint_family=FOREX_ENDPOINT_FAMILY,
                currency="USD",
                is_proxy=False,
                priority=10,
                rationale="Gold spot requires a configured spot/FX endpoint; do not use stocks.",
            )
        )
    if capabilities.futures:
        candidates.append(
            InstrumentCandidate(
                asset_name="gold",
                symbol="GC",
                provider=capabilities.provider,
                asset_class=AssetClass.COMMODITY_FUTURE,
                endpoint_family=FUTURES_ENDPOINT_FAMILY,
                currency="USD",
                is_proxy=False,
                priority=20,
                rationale="COMEX gold futures require a configured futures endpoint.",
            )
        )
    if capabilities.stocks:
        candidates.append(
            InstrumentCandidate(
                asset_name="gold",
                symbol="GLD",
                provider=capabilities.provider,
                asset_class=AssetClass.ETF,
                endpoint_family=STOCK_ENDPOINT_FAMILY,
                currency="USD",
                is_proxy=True,
                priority=30,
                rationale=(
                    "GLD is a liquid ETF proxy that is valid for the configured stocks "
                    "aggregate endpoint when spot/futures endpoints are unavailable."
                ),
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.priority)


def _us_equity_candidates(
    text: str, capabilities: MarketDataCapabilities
) -> list[InstrumentCandidate]:
    if not capabilities.stocks:
        return []
    candidates = [
        InstrumentCandidate(
            asset_name="us_equities",
            symbol="SPY",
            provider=capabilities.provider,
            asset_class=AssetClass.ETF,
            endpoint_family=STOCK_ENDPOINT_FAMILY,
            currency="USD",
            is_proxy=True,
            priority=10,
            rationale="SPY is the stocks-endpoint ETF proxy for broad U.S. equities/SPX.",
        )
    ]
    normalized = text.lower()
    if any(term in normalized for term in ("tech", "technology", "growth", "nasdaq", "qqq")):
        candidates.append(
            InstrumentCandidate(
                asset_name="us_equities_growth",
                symbol="QQQ",
                provider=capabilities.provider,
                asset_class=AssetClass.ETF,
                endpoint_family=STOCK_ENDPOINT_FAMILY,
                currency="USD",
                is_proxy=True,
                priority=20,
                rationale="QQQ is an ETF proxy for technology/growth-heavy U.S. equities.",
            )
        )
    if any(term in normalized for term in ("small cap", "small-cap", "small caps", "iwm")):
        candidates.append(
            InstrumentCandidate(
                asset_name="us_equities_small_caps",
                symbol="IWM",
                provider=capabilities.provider,
                asset_class=AssetClass.ETF,
                endpoint_family=STOCK_ENDPOINT_FAMILY,
                currency="USD",
                is_proxy=True,
                priority=30,
                rationale="IWM is an ETF proxy for U.S. small-cap equity exposure.",
            )
        )
    return candidates


def _usd_candidates(capabilities: MarketDataCapabilities) -> list[InstrumentCandidate]:
    candidates: list[InstrumentCandidate] = []
    if capabilities.indices:
        candidates.append(
            InstrumentCandidate(
                asset_name="usd",
                symbol="DXY",
                provider=capabilities.provider,
                asset_class=AssetClass.INDEX,
                endpoint_family=INDEX_ENDPOINT_FAMILY,
                currency="USD",
                is_proxy=False,
                priority=10,
                rationale="DXY requires a configured index endpoint; do not use stocks.",
            )
        )
    if capabilities.forex:
        candidates.extend(
            [
                InstrumentCandidate(
                    asset_name="usd",
                    symbol="EURUSD",
                    provider=capabilities.provider,
                    asset_class=AssetClass.FX,
                    endpoint_family=FOREX_ENDPOINT_FAMILY,
                    currency="USD",
                    is_proxy=True,
                    priority=20,
                    rationale="EURUSD is an FX endpoint proxy for broad USD pressure.",
                ),
                InstrumentCandidate(
                    asset_name="usd",
                    symbol="USDJPY",
                    provider=capabilities.provider,
                    asset_class=AssetClass.FX,
                    endpoint_family=FOREX_ENDPOINT_FAMILY,
                    currency="JPY",
                    is_proxy=True,
                    priority=30,
                    rationale="USDJPY is an FX endpoint proxy for USD/rate-differential exposure.",
                ),
            ]
        )
    if capabilities.stocks:
        candidates.append(
            InstrumentCandidate(
                asset_name="usd",
                symbol="UUP",
                provider=capabilities.provider,
                asset_class=AssetClass.ETF,
                endpoint_family=STOCK_ENDPOINT_FAMILY,
                currency="USD",
                is_proxy=True,
                priority=40,
                rationale="UUP is the stocks-endpoint ETF proxy for broad U.S. dollar exposure.",
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.priority)


def _long_duration_bond_candidates(
    capabilities: MarketDataCapabilities,
) -> list[InstrumentCandidate]:
    if not capabilities.stocks:
        return []
    return [
        InstrumentCandidate(
            asset_name="long_duration_bonds",
            symbol="TLT",
            provider=capabilities.provider,
            asset_class=AssetClass.ETF,
            endpoint_family=STOCK_ENDPOINT_FAMILY,
            currency="USD",
            is_proxy=False,
            priority=10,
            rationale="TLT is an ETF traded through the stocks aggregate endpoint.",
        )
    ]


def _stock_candidate(
    symbol: str,
    capabilities: MarketDataCapabilities,
    *,
    priority: int,
) -> InstrumentCandidate:
    return InstrumentCandidate(
        asset_name=symbol,
        symbol=symbol,
        provider=capabilities.provider,
        asset_class=AssetClass.STOCK,
        endpoint_family=STOCK_ENDPOINT_FAMILY,
        currency="USD",
        is_proxy=False,
        priority=priority,
        rationale=(
            "Detected a plain equity ticker that is valid for the configured "
            "stocks aggregate endpoint."
        ),
    )


def _rates_candidates(capabilities: MarketDataCapabilities) -> list[InstrumentCandidate]:
    if not capabilities.macro_series:
        return []
    return [
        InstrumentCandidate(
            asset_name="rates",
            symbol="FEDFUNDS",
            provider=capabilities.provider,
            asset_class=AssetClass.MACRO_SERIES,
            endpoint_family=MACRO_SERIES_ENDPOINT_FAMILY,
            currency="USD",
            is_proxy=False,
            priority=10,
            rationale=(
                "Rates are macro series and must use a configured macro data endpoint, "
                "not the stocks aggregate endpoint."
            ),
        )
    ]


def _classify_asset(text: str) -> str:
    normalized = _normalize(text)
    if any(term in normalized for term in ("long duration", "long-duration", "tlt")):
        return "long_duration_bonds"
    if any(term in normalized for term in ("gold", "xauusd", "xau/usd", "gc=f", "comex gold")):
        return "gold"
    if any(term in normalized for term in ("dxy", "u.s. dollar", "us dollar", "usd", "fx")):
        return "usd"
    if any(
        term in normalized
        for term in (
            "spx",
            "s&p 500",
            "s&p500",
            "us equities",
            "u.s. equities",
            "equity",
            "equities",
            "stocks",
            "qqq",
            "iwm",
            "nasdaq",
        )
    ):
        return "us_equities"
    if any(
        term in normalized
        for term in ("macro/rates", "fed funds", "policy rate", "rates", "yield curve")
    ):
        return "rates"
    return "unknown"


def _classify_branch(asset_or_branch: str) -> str | None:
    """Classify explicit semantic branches before scanning the global question.

    Market-data tasks receive both a task/branch string and the original user question.
    The global question can mention several assets at once. A long-duration-bonds mention
    in that global text must not override a task already grounded to market/equities,
    market/fx, or market/gold.
    """
    normalized = _normalize(asset_or_branch)
    if "market/equities" in normalized:
        return "us_equities"
    if "market/fx" in normalized:
        return "usd"
    if "market/gold" in normalized:
        return "gold"
    if "market/bonds" in normalized:
        return "long_duration_bonds"
    if "macro/rates" in normalized:
        return "rates"
    return None


def _extract_stock_symbol(text: str) -> str | None:
    for match in _STOCK_TOKEN_RE.finditer(text):
        symbol = match.group(0).upper()
        if symbol in _NON_TICKER_WORDS or symbol in _INVALID_STOCK_ENDPOINT_SYMBOLS:
            continue
        return symbol
    return None


def _contains_non_stock_asset_terms(text: str) -> bool:
    normalized = _normalize(text)
    return any(
        term in normalized
        for term in (
            "gold",
            "gc=f",
            "xau",
            "dxy",
            "spx",
            "s&p",
            "usd",
            "u.s. dollar",
            "us dollar",
            "fx",
            "rates",
            "fed funds",
            "yield",
            "treasury",
        )
    )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()
