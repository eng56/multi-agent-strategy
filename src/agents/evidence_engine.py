from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, date, datetime
from enum import StrEnum
import html
import re
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import UUID

from pydantic import BaseModel, Field


class SearchMode(StrEnum):
    EXPLORATORY = "exploratory"
    VERIFICATION = "verification"
    CONTRADICTION = "contradiction"
    PRIMARY_SOURCE = "primary_source"
    HISTORICAL = "historical"


class SourceTier(StrEnum):
    PRIMARY = "primary"
    HIGH_QUALITY_SECONDARY = "high_quality_secondary"
    NEWS = "news"
    BLOG_OR_OPINION = "blog_or_opinion"
    UNKNOWN = "unknown"
    WEAK = "weak"


PRIMARY_SOURCE_DOMAINS = {
    "federalreserve.gov",
    "stlouisfed.org",
    "fred.stlouisfed.org",
    "bls.gov",
    "bea.gov",
    "treasury.gov",
    "sec.gov",
    "cmegroup.com",
    "nasdaq.com",
    "nyse.com",
}

HIGH_QUALITY_SECONDARY_DOMAINS = {
    "imf.org",
    "worldbank.org",
    "bis.org",
    "oecd.org",
}

MAJOR_NEWS_DOMAINS = {
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "economist.com",
}

BLOG_OR_OPINION_DOMAINS = {
    "blogspot.com",
    "medium.com",
    "seekingalpha.com",
    "substack.com",
    "wordpress.com",
}

WEAK_SOURCE_DOMAINS = {
    "answers.com",
    "contentfarm.com",
    "quora.com",
    "reddit.com",
    "wikipedia.org",
}

DATA_SPECIFICITY_TERMS = {
    "balance sheet",
    "basis point",
    "bps",
    "cpi",
    "dataset",
    "earnings",
    "filing",
    "gdp",
    "inflation",
    "index",
    "minutes",
    "pce",
    "release",
    "report",
    "series",
    "statement",
    "statistic",
    "survey",
    "table",
    "yield",
}

WEAK_CONTENT_PATTERNS = (
    "affiliate",
    "clickbait",
    "content farm",
    "rumor",
    "sponsored content",
    "top 10",
)


SupportType = Literal["supports", "contradicts", "contextual", "unknown"]
SearchClient = Callable[[UUID, str], Awaitable[dict[str, Any]]]


class EvidenceRequest(BaseModel):
    run_id: UUID
    task_id: UUID | None = None
    branch: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    claim_id: UUID | None = None
    claim_text: str | None = None
    search_mode: SearchMode = SearchMode.EXPLORATORY
    max_sources: int = Field(default=5, ge=0, le=25)
    max_cost_usd: float | None = Field(default=None, ge=0)
    preferred_domains: list[str] = Field(default_factory=list)
    excluded_domains: list[str] = Field(default_factory=list)
    freshness_window: str | None = None


class EvidenceQuery(BaseModel):
    query: str = Field(min_length=1)
    search_mode: SearchMode
    branch: str
    rationale: str


class ProviderSearchResult(BaseModel):
    provider: str = Field(min_length=1)
    raw: dict[str, Any] = Field(default_factory=dict)


class SearchProvider(ABC):
    name: str

    @abstractmethod
    async def search(self, query: EvidenceQuery) -> ProviderSearchResult:
        ...


class TavilyProvider(SearchProvider):
    name = "tavily"

    def __init__(self, search_client: SearchClient, run_id: UUID) -> None:
        self.search_client = search_client
        self.run_id = run_id

    async def search(self, query: EvidenceQuery) -> ProviderSearchResult:
        return ProviderSearchResult(
            provider=self.name,
            raw=await self.search_client(self.run_id, query.query),
        )


class BraveProvider(SearchProvider):
    name = "brave"

    def __init__(self, search_client: SearchClient, run_id: UUID) -> None:
        self.search_client = search_client
        self.run_id = run_id

    async def search(self, query: EvidenceQuery) -> ProviderSearchResult:
        return ProviderSearchResult(
            provider=self.name,
            raw=await self.search_client(self.run_id, query.query),
        )


class EvidenceSource(BaseModel):
    source_url: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    provider: str = Field(default="unknown", min_length=1)
    domain: str | None = None
    publisher: str | None = None
    published_at: str | None = None
    source_tier: SourceTier = SourceTier.UNKNOWN


class EvidenceItem(EvidenceSource):
    retrieved_at: datetime
    snippet: str
    support_type: SupportType = "unknown"
    confidence: float = Field(ge=0, le=1)
    quality_score: int = Field(default=0, ge=0, le=100)
    source_quality_reason: str = ""
    limitations: list[str] = Field(default_factory=list)


class EvidenceBundle(BaseModel):
    request: EvidenceRequest
    queries: list[EvidenceQuery] = Field(default_factory=list)
    items: list[EvidenceItem] = Field(default_factory=list)
    raw_result_artifact_uri: str | None = None
    limitations: list[str] = Field(default_factory=list)
    source_quality_summary: dict[str, Any] = Field(default_factory=dict)


class SourceQuality(BaseModel):
    source_tier: SourceTier
    quality_score: int = Field(ge=0, le=100)
    domain: str | None = None
    publisher: str | None = None
    reason: str


class EvidenceEngine:
    """Structured observation wrapper for provider-backed research search."""

    def __init__(
        self,
        search_client: SearchClient | None = None,
        *,
        providers: Iterable[SearchProvider] | None = None,
        artifact_store: Any | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.search_client = search_client
        self.providers = list(providers or [])
        self.artifact_store = artifact_store
        self.clock = clock or (lambda: datetime.now(UTC))

    async def search(self, request: EvidenceRequest) -> EvidenceBundle:
        queries = generate_evidence_queries(request)
        retrieved_at = self.clock()
        raw_outputs: list[dict[str, Any]] = []
        bundle_limitations: list[str] = []
        items: list[EvidenceItem] = []
        seen_urls: set[str] = set()
        providers = self._providers_for_request(request)

        if request.max_sources > 0:
            for evidence_query in queries:
                result = await self._search_with_fallback(
                    evidence_query,
                    providers,
                    bundle_limitations,
                    raw_outputs,
                )
                raw_outputs.append(
                    {
                        "query": evidence_query.model_dump(mode="json"),
                        "provider": result.provider,
                        "response": result.raw,
                    }
                )
                for item in provider_results_to_items(result, request, retrieved_at):
                    normalized_url = item.source_url.rstrip("/")
                    if normalized_url in seen_urls:
                        continue
                    seen_urls.add(normalized_url)
                    items.append(item)
                    if len(items) >= request.max_sources:
                        break
                if len(items) >= request.max_sources:
                    break

        artifact_uri = self._persist_raw_results(request, queries, raw_outputs)
        return EvidenceBundle(
            request=request,
            queries=queries,
            items=items,
            raw_result_artifact_uri=artifact_uri,
            limitations=_unique_preserving_order(bundle_limitations),
            source_quality_summary=summarize_source_quality(items, queries),
        )

    def _providers_for_request(self, request: EvidenceRequest) -> list[SearchProvider]:
        providers = list(self.providers)
        if self.search_client is not None and _provider_by_name(providers, "tavily") is None:
            providers.insert(0, TavilyProvider(self.search_client, request.run_id))
        if not providers:
            raise ValueError("EvidenceEngine requires at least one search provider.")
        return providers

    async def _search_with_fallback(
        self,
        query: EvidenceQuery,
        providers: list[SearchProvider],
        bundle_limitations: list[str],
        raw_outputs: list[dict[str, Any]],
    ) -> ProviderSearchResult:
        primary = select_search_provider(query, providers)
        try:
            return await primary.search(query)
        except Exception as exc:
            fallback = _provider_by_name(providers, "tavily")
            if primary.name == "tavily" or fallback is None or fallback is primary:
                raise
            raw_outputs.append(
                {
                    "query": query.model_dump(mode="json"),
                    "provider": primary.name,
                    "error": {"type": type(exc).__name__},
                    "fallback_provider": fallback.name,
                }
            )
            bundle_limitations.append(
                f"{primary.name} provider failed with {type(exc).__name__}; "
                f"fell back to {fallback.name}."
            )
            return await fallback.search(query)

    def _persist_raw_results(
        self,
        request: EvidenceRequest,
        queries: list[EvidenceQuery],
        raw_outputs: list[dict[str, Any]],
    ) -> str | None:
        if self.artifact_store is None or not raw_outputs:
            return None
        pointer = self.artifact_store.put_json(
            request.run_id,
            "evidence-search",
            {
                "request": request.model_dump(mode="json"),
                "queries": [query.model_dump(mode="json") for query in queries],
                "raw_outputs": raw_outputs,
            },
        )
        return getattr(pointer, "uri", None)


def generate_evidence_queries(request: EvidenceRequest) -> list[EvidenceQuery]:
    branch_terms = _branch_terms(request.branch)
    subject = request.claim_text or request.objective
    domain_suffix = _domain_query_suffix(request)
    freshness_suffix = f" {request.freshness_window}" if request.freshness_window else ""

    candidates: list[tuple[str, str]] = []
    if request.search_mode == SearchMode.VERIFICATION:
        verification_subject = request.claim_text or request.objective
        candidates.append(
            (
                f"{verification_subject} {branch_terms} evidence verification{freshness_suffix}"
                f"{domain_suffix}",
                "Verify the claim with directly relevant corroborating evidence.",
            )
        )
    elif request.search_mode == SearchMode.CONTRADICTION:
        candidates.extend(
            [
                (
                    f"{subject} {branch_terms} risk counterargument opposite failed"
                    f"{freshness_suffix}{domain_suffix}",
                    "Search for evidence that would weaken or contradict the claim.",
                ),
                (
                    f"{subject} {branch_terms} criticism downside challenge rebuttal"
                    f"{freshness_suffix}{domain_suffix}",
                    "Search for critical discussion and contrary evidence.",
                ),
            ]
        )
    elif request.search_mode == SearchMode.PRIMARY_SOURCE:
        candidates.append(
            (
                f"{request.objective} {branch_terms} official report primary source data"
                f"{freshness_suffix}{domain_suffix}",
                "Prioritize official or primary-source material.",
            )
        )
    elif request.search_mode == SearchMode.HISTORICAL:
        candidates.append(
            (
                f"{request.objective} {branch_terms} historical data previous cases"
                f"{freshness_suffix}{domain_suffix}",
                "Find historical context and comparable prior evidence.",
            )
        )
    else:
        candidates.append(
            (
                f"{request.objective} {branch_terms}{freshness_suffix}{domain_suffix}",
                "Explore decision-relevant sources for the research objective.",
            )
        )

    queries: list[EvidenceQuery] = []
    seen: set[str] = set()
    for query, rationale in candidates:
        cleaned = _clean_query(query)
        if cleaned in seen:
            continue
        seen.add(cleaned)
        queries.append(
            EvidenceQuery(
                query=cleaned,
                search_mode=request.search_mode,
                branch=request.branch,
                rationale=rationale,
            )
        )
        if len(queries) == 3:
            break
    return queries


def select_search_provider(
    query: EvidenceQuery,
    providers: list[SearchProvider],
) -> SearchProvider:
    brave = _provider_by_name(providers, "brave")
    if brave is not None and _should_use_brave(query):
        return brave
    tavily = _provider_by_name(providers, "tavily")
    return tavily or providers[0]


def provider_results_to_items(
    result: ProviderSearchResult,
    request: EvidenceRequest,
    retrieved_at: datetime,
) -> list[EvidenceItem]:
    if result.provider == "brave":
        return brave_results_to_items(result.raw, request, retrieved_at)
    return tavily_results_to_items(
        result.raw,
        request,
        retrieved_at,
        provider=result.provider,
    )


def tavily_results_to_items(
    raw: dict[str, Any],
    request: EvidenceRequest,
    retrieved_at: datetime,
    *,
    provider: str = "tavily",
) -> list[EvidenceItem]:
    results = raw.get("results", [])
    if not isinstance(results, list):
        return []

    items: list[EvidenceItem] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        source_url = _string_value(result, "url", "source_url")
        if not source_url:
            continue
        if _domain_matches_any(source_url, request.excluded_domains):
            continue
        raw_title = _string_value(result, "title", "source_title")
        source_title = raw_title or source_url
        raw_snippet = _string_value(result, "content", "snippet", "description", "raw_content")
        snippet = raw_snippet or ""
        publisher = _string_value(result, "publisher", "source")
        published_at = _string_value(result, "published_at", "published_date", "date")
        confidence = _confidence_from_score(result.get("score"))
        limitations = _limitations(result, request.search_mode)
        source_quality = score_source_quality(
            source_url=source_url,
            source_title=raw_title,
            snippet=raw_snippet,
            publisher=publisher,
            published_at=published_at,
            preferred_domains=request.preferred_domains,
            retrieved_at=retrieved_at,
        )
        items.append(
            EvidenceItem(
                source_url=source_url,
                source_title=source_title,
                provider=provider,
                domain=source_quality.domain,
                publisher=source_quality.publisher,
                published_at=published_at,
                retrieved_at=retrieved_at,
                source_tier=source_quality.source_tier,
                snippet=snippet,
                support_type=_support_type(request.search_mode),
                confidence=confidence,
                quality_score=source_quality.quality_score,
                source_quality_reason=source_quality.reason,
                limitations=limitations,
            )
        )
    return items


def brave_results_to_items(
    raw: dict[str, Any],
    request: EvidenceRequest,
    retrieved_at: datetime,
) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for result in _brave_result_list(raw):
        source_url = _string_value(result, "url", "source_url")
        if not source_url:
            continue
        if _domain_matches_any(source_url, request.excluded_domains):
            continue
        raw_title = _clean_provider_text(_string_value(result, "title", "source_title"))
        source_title = raw_title or source_url
        raw_snippet = _clean_provider_text(
            _string_value(result, "description", "snippet", "content", "raw_content")
        )
        snippet = raw_snippet or ""
        publisher = _brave_publisher(result)
        published_at = _string_value(result, "published_at", "published_date", "date", "age")
        confidence = _confidence_from_score(result.get("score"))
        limitations = _limitations(result, request.search_mode)
        if "score" not in result:
            limitations.append("Brave result did not include a normalized relevance score.")
        source_quality = score_source_quality(
            source_url=source_url,
            source_title=raw_title,
            snippet=raw_snippet,
            publisher=publisher,
            published_at=published_at,
            preferred_domains=request.preferred_domains,
            retrieved_at=retrieved_at,
        )
        items.append(
            EvidenceItem(
                source_url=source_url,
                source_title=source_title,
                provider="brave",
                domain=source_quality.domain,
                publisher=source_quality.publisher,
                published_at=published_at,
                retrieved_at=retrieved_at,
                source_tier=source_quality.source_tier,
                snippet=snippet,
                support_type=_support_type(request.search_mode),
                confidence=confidence,
                quality_score=source_quality.quality_score,
                source_quality_reason=source_quality.reason,
                limitations=limitations,
            )
        )
        if len(items) >= request.max_sources:
            break
    return items


def summarize_source_quality(
    items: list[EvidenceItem], queries: list[EvidenceQuery]
) -> dict[str, Any]:
    tiers = {tier.value: 0 for tier in SourceTier}
    domains: set[str] = set()
    missing_published_at_count = 0
    total_quality_score = 0
    top_quality_score = 0
    for item in items:
        tiers[item.source_tier.value] += 1
        total_quality_score += item.quality_score
        top_quality_score = max(top_quality_score, item.quality_score)
        domain = item.domain or _domain(item.source_url)
        if domain:
            domains.add(domain)
        if item.published_at is None:
            missing_published_at_count += 1
    average_quality_score = round(total_quality_score / len(items), 1) if items else 0.0
    return {
        "query_count": len(queries),
        "item_count": len(items),
        "unique_domain_count": len(domains),
        "source_tiers": tiers,
        "average_quality_score": average_quality_score,
        "top_quality_score": top_quality_score,
        "missing_published_at_count": missing_published_at_count,
    }


def score_source_quality(
    *,
    source_url: str | None,
    source_title: str | None = None,
    snippet: str | None = None,
    publisher: str | None = None,
    published_at: str | None = None,
    preferred_domains: list[str] | None = None,
    retrieved_at: datetime | None = None,
) -> SourceQuality:
    """Score source quality with deterministic finance/macro domain rules."""

    preferred_domains = preferred_domains or []
    domain = _domain(source_url or "")
    inferred_publisher = publisher or _publisher_from_domain(domain)
    tier = _classify_source_tier(domain, preferred_domains)
    score = 40
    reasons: list[str] = []

    if tier == SourceTier.PRIMARY:
        score += 35
        reasons.append("primary finance/macro or preferred source domain")
    elif tier == SourceTier.HIGH_QUALITY_SECONDARY:
        score += 25
        reasons.append("high-quality institutional secondary source domain")
    elif tier == SourceTier.NEWS:
        score += 22
        reasons.append("major financial/news outlet domain")
    elif tier == SourceTier.BLOG_OR_OPINION:
        score -= 5
        reasons.append("blog or opinion-oriented domain")
    elif tier == SourceTier.WEAK:
        score -= 30
        reasons.append("weak or crowd-sourced domain")
    else:
        reasons.append("domain not in source-quality rules")

    if domain and (domain.endswith(".gov") or _domain_matches(domain, PRIMARY_SOURCE_DOMAINS)):
        score += 8
        reasons.append("government, central bank, statistical, exchange, or filing source bonus")

    recency_score, recency_reason = _recency_score(published_at, retrieved_at)
    score += recency_score
    reasons.append(recency_reason)

    specificity_score, specificity_reasons = _data_specificity_score(source_title, snippet)
    score += specificity_score
    reasons.extend(specificity_reasons)

    missing_penalties: list[str] = []
    if not source_url or not source_url.strip():
        score -= 25
        missing_penalties.append("missing url")
    if not source_title or not source_title.strip():
        score -= 10
        missing_penalties.append("missing title")
    if not snippet or not snippet.strip():
        score -= 12
        missing_penalties.append("missing snippet")
    if missing_penalties:
        reasons.append("penalized for " + ", ".join(missing_penalties))

    if _has_weak_content_pattern(source_title, snippet):
        score -= 10
        reasons.append("weak-content language penalty")

    quality_score = max(0, min(100, score))
    return SourceQuality(
        source_tier=tier,
        quality_score=quality_score,
        domain=domain,
        publisher=inferred_publisher,
        reason="; ".join(reasons),
    )


def _support_type(search_mode: SearchMode) -> SupportType:
    if search_mode == SearchMode.VERIFICATION:
        return "supports"
    if search_mode == SearchMode.CONTRADICTION:
        return "contradicts"
    return "contextual"


def _limitations(result: dict[str, Any], search_mode: SearchMode) -> list[str]:
    limitations = ["Support type is inferred from search mode, not semantic adjudication."]
    if "score" not in result:
        limitations.append("Provider did not include a relevance score.")
    if not _string_value(result, "published_at", "published_date", "date"):
        limitations.append("Provider did not include a publication date.")
    if search_mode in {SearchMode.VERIFICATION, SearchMode.CONTRADICTION}:
        limitations.append("Result requires downstream claim-level verification.")
    return limitations


def _provider_by_name(
    providers: list[SearchProvider],
    name: str,
) -> SearchProvider | None:
    for provider in providers:
        if provider.name == name:
            return provider
    return None


def _should_use_brave(query: EvidenceQuery) -> bool:
    if query.search_mode in {SearchMode.CONTRADICTION, SearchMode.PRIMARY_SOURCE}:
        return True
    return "site:" in query.query.lower()


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _brave_result_list(raw: dict[str, Any]) -> list[dict[str, Any]]:
    web = raw.get("web")
    if isinstance(web, dict) and isinstance(web.get("results"), list):
        return [item for item in web["results"] if isinstance(item, dict)]
    results = raw.get("results")
    if isinstance(results, list):
        return [item for item in results if isinstance(item, dict)]
    return []


def _brave_publisher(result: dict[str, Any]) -> str | None:
    profile = result.get("profile")
    if isinstance(profile, dict):
        name = profile.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return _string_value(result, "publisher", "source")


def _clean_provider_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = html.unescape(re.sub(r"<[^>]+>", " ", value))
    cleaned = " ".join(text.split()).strip()
    return cleaned or None


def _classify_source_tier(
    domain: str | None,
    preferred_domains: list[str],
) -> SourceTier:
    if not domain:
        return SourceTier.UNKNOWN
    if _domain_matches(domain, WEAK_SOURCE_DOMAINS):
        return SourceTier.WEAK
    if _domain_matches(domain, preferred_domains) or _domain_matches(
        domain, PRIMARY_SOURCE_DOMAINS
    ):
        return SourceTier.PRIMARY
    if domain.endswith(".gov"):
        return SourceTier.PRIMARY
    if _domain_matches(domain, HIGH_QUALITY_SECONDARY_DOMAINS):
        return SourceTier.HIGH_QUALITY_SECONDARY
    if _domain_matches(domain, MAJOR_NEWS_DOMAINS):
        return SourceTier.NEWS
    if _domain_matches(domain, BLOG_OR_OPINION_DOMAINS):
        return SourceTier.BLOG_OR_OPINION
    return SourceTier.UNKNOWN


def _recency_score(
    published_at: str | None,
    retrieved_at: datetime | None,
) -> tuple[int, str]:
    published_date = _publication_date(published_at)
    if published_date is None:
        return -6, "publication date missing or unparseable"
    if retrieved_at is None:
        return 4, "publication date available"
    age_days = max(0, (retrieved_at.date() - published_date).days)
    if age_days <= 370:
        return 8, "recent publication date"
    if age_days <= 365 * 3:
        return 5, "moderately recent publication date"
    return 2, "older publication date available"


def _data_specificity_score(
    source_title: str | None,
    snippet: str | None,
) -> tuple[int, list[str]]:
    text = f"{source_title or ''} {snippet or ''}".lower()
    score = 0
    reasons: list[str] = []
    if re.search(r"\d", text):
        score += 5
        reasons.append("contains numeric or dated detail")
    if any(term in text for term in DATA_SPECIFICITY_TERMS):
        score += 5
        reasons.append("contains data-specific terminology")
    return score, reasons


def _has_weak_content_pattern(source_title: str | None, snippet: str | None) -> bool:
    text = f"{source_title or ''} {snippet or ''}".lower()
    return any(pattern in text for pattern in WEAK_CONTENT_PATTERNS)


def _publication_date(value: str | None) -> date | None:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    iso_candidate = cleaned.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_candidate).date()
    except ValueError:
        pass
    if len(cleaned) > 10:
        try:
            return datetime.fromisoformat(iso_candidate[:10]).date()
        except ValueError:
            pass
    for date_format in ("%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y", "%Y"):
        try:
            return datetime.strptime(cleaned, date_format).date()
        except ValueError:
            continue
    return None


def _publisher_from_domain(domain: str | None) -> str | None:
    if not domain:
        return None
    known_publishers = {
        "federalreserve.gov": "Federal Reserve",
        "fred.stlouisfed.org": "FRED",
        "stlouisfed.org": "Federal Reserve Bank of St. Louis",
        "bls.gov": "Bureau of Labor Statistics",
        "bea.gov": "Bureau of Economic Analysis",
        "treasury.gov": "U.S. Treasury",
        "sec.gov": "U.S. Securities and Exchange Commission",
        "cmegroup.com": "CME Group",
        "nasdaq.com": "Nasdaq",
        "nyse.com": "NYSE",
        "reuters.com": "Reuters",
        "bloomberg.com": "Bloomberg",
        "ft.com": "Financial Times",
        "wsj.com": "Wall Street Journal",
        "economist.com": "The Economist",
        "imf.org": "International Monetary Fund",
        "worldbank.org": "World Bank",
        "bis.org": "Bank for International Settlements",
        "oecd.org": "OECD",
    }
    for source_domain, publisher in known_publishers.items():
        if _domain_matches(domain, [source_domain]):
            return publisher
    return None


def _confidence_from_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _string_value(result: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _branch_terms(branch: str) -> str:
    return " ".join(part for part in branch.replace("/", " ").split() if part)


def _domain_query_suffix(request: EvidenceRequest) -> str:
    parts: list[str] = []
    preferred = [_normalize_domain(value) for value in request.preferred_domains[:3]]
    preferred = [value for value in preferred if value]
    excluded = [_normalize_domain(value) for value in request.excluded_domains[:5]]
    excluded = [value for value in excluded if value]
    if len(preferred) == 1:
        parts.append(f"site:{preferred[0]}")
    elif len(preferred) > 1:
        parts.append("(" + " OR ".join(f"site:{domain}" for domain in preferred) + ")")
    parts.extend(f"-site:{domain}" for domain in excluded)
    return f" {' '.join(parts)}" if parts else ""


def _clean_query(query: str, max_chars: int = 360) -> str:
    cleaned = " ".join(query.split()).strip() or "research evidence"
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(" ", 1)[0] or cleaned[:max_chars]


def _domain_matches_any(source_url: str, domains: list[str]) -> bool:
    source_domain = _domain(source_url)
    return _domain_matches(source_domain, domains) if source_domain else False


def _domain_matches(source_domain: str, domains: Iterable[str]) -> bool:
    normalized_source = _normalize_domain(source_domain)
    if not normalized_source:
        return False
    for domain in domains:
        normalized_domain = _normalize_domain(domain)
        if normalized_source == normalized_domain or normalized_source.endswith(
            f".{normalized_domain}"
        ):
            return True
    return False


def _domain(source_url: str) -> str | None:
    parsed = urlparse(source_url if "://" in source_url else f"https://{source_url}")
    return _normalize_domain(parsed.hostname or "") or None


def _normalize_domain(value: str) -> str:
    parsed = urlparse(value if "://" in value else f"https://{value}")
    domain = (parsed.hostname or value).lower().strip()
    return domain[4:] if domain.startswith("www.") else domain
