from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
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
    AUTHORITATIVE = "authoritative"
    SECONDARY = "secondary"
    UNKNOWN = "unknown"


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


class EvidenceSource(BaseModel):
    source_url: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    publisher: str | None = None
    published_at: str | None = None
    source_tier: SourceTier = SourceTier.UNKNOWN


class EvidenceItem(EvidenceSource):
    retrieved_at: datetime
    snippet: str
    support_type: SupportType = "unknown"
    confidence: float = Field(ge=0, le=1)
    limitations: list[str] = Field(default_factory=list)


class EvidenceBundle(BaseModel):
    request: EvidenceRequest
    queries: list[EvidenceQuery] = Field(default_factory=list)
    items: list[EvidenceItem] = Field(default_factory=list)
    raw_result_artifact_uri: str | None = None
    source_quality_summary: dict[str, Any] = Field(default_factory=dict)


class EvidenceEngine:
    """Structured observation wrapper for Tavily-backed research search."""

    def __init__(
        self,
        search_client: SearchClient,
        *,
        artifact_store: Any | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.search_client = search_client
        self.artifact_store = artifact_store
        self.clock = clock or (lambda: datetime.now(UTC))

    async def search(self, request: EvidenceRequest) -> EvidenceBundle:
        queries = generate_evidence_queries(request)
        retrieved_at = self.clock()
        raw_outputs: list[dict[str, Any]] = []
        items: list[EvidenceItem] = []
        seen_urls: set[str] = set()

        if request.max_sources > 0:
            for evidence_query in queries:
                raw = await self.search_client(request.run_id, evidence_query.query)
                raw_outputs.append(
                    {
                        "query": evidence_query.model_dump(mode="json"),
                        "response": raw,
                    }
                )
                for item in tavily_results_to_items(raw, request, retrieved_at):
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
            source_quality_summary=summarize_source_quality(items, queries),
        )

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


def tavily_results_to_items(
    raw: dict[str, Any],
    request: EvidenceRequest,
    retrieved_at: datetime,
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
        source_title = _string_value(result, "title", "source_title") or source_url
        snippet = _string_value(result, "content", "snippet", "description", "raw_content") or ""
        publisher = _string_value(result, "publisher", "source")
        published_at = _string_value(result, "published_at", "published_date", "date")
        confidence = _confidence_from_score(result.get("score"))
        limitations = _limitations(result, request.search_mode)
        items.append(
            EvidenceItem(
                source_url=source_url,
                source_title=source_title,
                publisher=publisher,
                published_at=published_at,
                retrieved_at=retrieved_at,
                source_tier=_source_tier(source_url, request.preferred_domains),
                snippet=snippet,
                support_type=_support_type(request.search_mode),
                confidence=confidence,
                limitations=limitations,
            )
        )
    return items


def summarize_source_quality(
    items: list[EvidenceItem], queries: list[EvidenceQuery]
) -> dict[str, Any]:
    tiers = {tier.value: 0 for tier in SourceTier}
    domains: set[str] = set()
    missing_published_at_count = 0
    for item in items:
        tiers[item.source_tier.value] += 1
        domain = _domain(item.source_url)
        if domain:
            domains.add(domain)
        if item.published_at is None:
            missing_published_at_count += 1
    return {
        "query_count": len(queries),
        "item_count": len(items),
        "unique_domain_count": len(domains),
        "source_tiers": tiers,
        "missing_published_at_count": missing_published_at_count,
    }


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


def _source_tier(source_url: str, preferred_domains: list[str]) -> SourceTier:
    domain = _domain(source_url)
    if not domain:
        return SourceTier.UNKNOWN
    if _domain_matches(domain, preferred_domains):
        return SourceTier.PRIMARY
    if domain.endswith((".gov", ".edu", ".int")):
        return SourceTier.AUTHORITATIVE
    return SourceTier.SECONDARY


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


def _domain_matches(source_domain: str, domains: list[str]) -> bool:
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
    return _normalize_domain(parsed.hostname or "")


def _normalize_domain(value: str) -> str:
    parsed = urlparse(value if "://" in value else f"https://{value}")
    domain = (parsed.hostname or value).lower().strip()
    return domain[4:] if domain.startswith("www.") else domain
