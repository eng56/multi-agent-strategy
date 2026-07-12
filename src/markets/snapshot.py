import inspect
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any

from src.common.budget import BudgetExceeded
from src.markets.resolver import (
    InstrumentCandidate,
    MarketDataCapabilities,
    is_valid_for_stocks_endpoint,
    resolve_instruments,
)


MarketDataFetcher = Callable[[InstrumentCandidate], dict[str, Any] | Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class MarketDataAttempt:
    candidate: InstrumentCandidate
    status: str
    reason: str
    result_count: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "candidate": _candidate_payload(self.candidate),
            "status": self.status,
            "reason": self.reason,
            "result_count": self.result_count,
        }


@dataclass(frozen=True)
class MarketSnapshot:
    asset: str
    question: str
    candidate: InstrumentCandidate
    raw: dict[str, Any]
    result_count: int
    attempts: list[MarketDataAttempt]

    @property
    def sources(self) -> list[str]:
        return [_source_url(self.candidate)]

    def summary(self) -> str:
        latest = _latest_result(self.raw)
        proxy_note = " proxy" if self.candidate.is_proxy else ""
        lines = [
            (
                f"MarketSnapshot for {self.asset}: selected {self.candidate.symbol}"
                f" ({self.candidate.asset_class.value}{proxy_note}) via "
                f"{self.candidate.provider}/{self.candidate.endpoint_family}; "
                f"{self.result_count} result(s) returned."
            ),
            f"Resolver rationale: {self.candidate.rationale}",
        ]
        if latest:
            close = latest.get("c") or latest.get("close") or latest.get("price")
            volume = latest.get("v") or latest.get("volume")
            if close is not None:
                lines.append(f"Latest available close/price: {close}.")
            if volume is not None:
                lines.append(f"Latest available volume: {volume}.")
        elif "price" in self.raw:
            lines.append(f"Latest available price: {self.raw['price']}.")
        return " ".join(lines)

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "market_snapshot",
            "asset": self.asset,
            "question": self.question,
            "candidate": _candidate_payload(self.candidate),
            "result_count": self.result_count,
            "sources": self.sources,
            "attempts": [attempt.to_payload() for attempt in self.attempts],
            "raw": self.raw,
        }


@dataclass(frozen=True)
class MarketDataGap:
    asset: str
    question: str
    attempts: list[MarketDataAttempt]
    reason: str

    @property
    def sources(self) -> list[str]:
        return []

    def summary(self) -> str:
        if not self.attempts:
            return (
                f"Market data gap for {self.asset}: no endpoint-valid candidate was "
                "available for the configured market-data provider. Use EvidenceEngine, "
                "FRED, Treasury, CME, or another primary source instead of converting "
                "this gap into an investment claim."
            )
        attempted = ", ".join(
            f"{attempt.candidate.symbol} ({attempt.reason})" for attempt in self.attempts
        )
        return (
            f"Market data gap for {self.asset}: no configured market-data candidate "
            f"returned usable data. Attempted: {attempted}. Do not convert this "
            "provider/data gap into an investment thesis."
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "market_data_gap",
            "asset": self.asset,
            "question": self.question,
            "reason": self.reason,
            "sources": [],
            "attempts": [attempt.to_payload() for attempt in self.attempts],
        }


async def get_market_snapshot(
    asset: str,
    question: str,
    *,
    fetch_market_data: MarketDataFetcher | None = None,
    capabilities: MarketDataCapabilities | None = None,
    stocks_endpoint_only: bool = True,
) -> MarketSnapshot | MarketDataGap:
    """Resolve an asset to endpoint-valid candidates and return the first usable snapshot."""
    candidates = resolve_instruments(asset, question, capabilities=capabilities)
    if not candidates:
        return MarketDataGap(
            asset=asset,
            question=question,
            attempts=[],
            reason="no_resolved_candidates",
        )
    if fetch_market_data is None:
        return MarketDataGap(
            asset=asset,
            question=question,
            attempts=[
                MarketDataAttempt(
                    candidate=candidate,
                    status="skipped",
                    reason="market_data_fetcher_not_configured",
                )
                for candidate in candidates
            ],
            reason="market_data_fetcher_not_configured",
        )

    attempts: list[MarketDataAttempt] = []
    for candidate in candidates:
        if stocks_endpoint_only and not is_valid_for_stocks_endpoint(candidate):
            attempts.append(
                MarketDataAttempt(
                    candidate=candidate,
                    status="skipped",
                    reason=(
                        f"{candidate.symbol} requires {candidate.endpoint_family}; "
                        "the current market_data fetcher is stocks-endpoint only."
                    ),
                )
            )
            continue
        try:
            raw = await _maybe_await(fetch_market_data(candidate))
        except BudgetExceeded:
            raise
        except Exception as exc:
            attempts.append(
                MarketDataAttempt(
                    candidate=candidate,
                    status="error",
                    reason=f"{type(exc).__name__}: {str(exc)[:240]}",
                )
            )
            continue
        result_count = market_data_result_count(raw)
        if result_count > 0:
            attempts.append(
                MarketDataAttempt(
                    candidate=candidate,
                    status="ok",
                    reason="returned_data",
                    result_count=result_count,
                )
            )
            return MarketSnapshot(
                asset=asset,
                question=question,
                candidate=candidate,
                raw=raw,
                result_count=result_count,
                attempts=attempts,
            )
        attempts.append(
            MarketDataAttempt(
                candidate=candidate,
                status="zero_results",
                reason="provider_returned_zero_results",
                result_count=0,
            )
        )

    return MarketDataGap(
        asset=asset,
        question=question,
        attempts=attempts,
        reason="all_candidates_failed",
    )


def market_data_result_count(raw: dict[str, Any]) -> int:
    results = raw.get("results")
    if isinstance(results, list):
        return len(results)
    for key in ("resultsCount", "result_count", "count", "queryCount"):
        value = raw.get(key)
        if isinstance(value, int):
            return max(0, value)
    if raw:
        return 1
    return 0


async def _maybe_await(value: dict[str, Any] | Awaitable[dict[str, Any]]) -> dict[str, Any]:
    if inspect.isawaitable(value):
        return await value
    return value


def _latest_result(raw: dict[str, Any]) -> dict[str, Any] | None:
    results = raw.get("results")
    if isinstance(results, list) and results:
        latest = results[-1]
        if isinstance(latest, dict):
            return latest
    return None


def _candidate_payload(candidate: InstrumentCandidate) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["asset_class"] = candidate.asset_class.value
    return payload


def _source_url(candidate: InstrumentCandidate) -> str:
    if candidate.endpoint_family == "stocks":
        return f"https://massive.com/stocks/{candidate.symbol}"
    return f"https://massive.com/{candidate.endpoint_family}/{candidate.symbol}"
