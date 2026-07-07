import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from src.agents.evidence_engine import (
    EvidenceEngine,
    EvidenceRequest,
    SearchMode,
    SourceTier,
    generate_evidence_queries,
    score_source_quality,
)
from src.integrations.tools import ResearchTools


class FakeSearchClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[tuple[object, str]] = []

    async def search(self, run_id, query):
        self.calls.append((run_id, query))
        if self.responses:
            return self.responses.pop(0)
        return {"results": []}


def request(**overrides) -> EvidenceRequest:
    values = {
        "run_id": uuid4(),
        "branch": "market/equities",
        "objective": "Assess whether SAP cloud growth supports buying the stock",
        "max_sources": 3,
    }
    values.update(overrides)
    return EvidenceRequest(**values)


RETRIEVED_AT = datetime(2026, 7, 7, 9, 0, tzinfo=UTC)


def test_exploratory_request_creates_search_queries() -> None:
    queries = generate_evidence_queries(request())

    assert 1 <= len(queries) <= 3
    assert queries[0].search_mode == SearchMode.EXPLORATORY
    assert "SAP cloud growth" in queries[0].query
    assert "market equities" in queries[0].query


def test_verification_request_includes_claim_text() -> None:
    claim = "SAP cloud revenue grew faster than license revenue in 2025"

    queries = generate_evidence_queries(
        request(search_mode=SearchMode.VERIFICATION, claim_text=claim)
    )

    assert claim in queries[0].query
    assert "verification" in queries[0].query


def test_contradiction_request_creates_opposition_query() -> None:
    queries = generate_evidence_queries(request(search_mode=SearchMode.CONTRADICTION))

    assert any(
        term in queries[0].query
        for term in ["risk", "counterargument", "opposite", "failed"]
    )


def test_tavily_response_converts_to_evidence_items() -> None:
    run_id = uuid4()
    fake = FakeSearchClient(
        {
            "results": [
                {
                    "url": "https://example.com/sap-cloud",
                    "title": "SAP cloud growth",
                    "content": "SAP reported cloud backlog growth.",
                    "score": 0.84,
                }
            ]
        }
    )
    engine = EvidenceEngine(
        fake.search,
        clock=lambda: datetime(2026, 7, 7, 9, 0, tzinfo=UTC),
    )

    bundle = asyncio.run(
        engine.search(
            request(
                run_id=run_id,
                objective="Find SAP cloud backlog evidence",
                max_sources=1,
            )
        )
    )

    assert len(bundle.items) == 1
    item = bundle.items[0]
    assert item.source_url == "https://example.com/sap-cloud"
    assert item.snippet == "SAP reported cloud backlog growth."
    assert item.confidence == 0.84
    assert item.published_at is None
    assert item.source_tier == SourceTier.UNKNOWN
    assert item.domain == "example.com"
    assert item.quality_score < 50
    assert "domain not in source-quality rules" in item.source_quality_reason
    assert bundle.source_quality_summary["missing_published_at_count"] == 1


def test_source_quality_classifies_primary_finance_macro_domains() -> None:
    fed = score_source_quality(
        source_url="https://www.federalreserve.gov/monetarypolicy/fomcminutes.htm",
        source_title="FOMC minutes",
        snippet="The statement includes inflation and labor market data.",
        published_at="2026-07-01",
        retrieved_at=RETRIEVED_AT,
    )
    fred = score_source_quality(
        source_url="https://fred.stlouisfed.org/series/FEDFUNDS",
        source_title="Federal funds effective rate",
        snippet="FRED monthly series with historical yield data.",
        published_at="2026-07-01",
        retrieved_at=RETRIEVED_AT,
    )
    st_louis_fed = score_source_quality(
        source_url="https://www.stlouisfed.org/research",
        source_title="St. Louis Fed research report",
        snippet="The report summarizes inflation and GDP data.",
        published_at="2026-07-01",
        retrieved_at=RETRIEVED_AT,
    )

    assert fed.source_tier == SourceTier.PRIMARY
    assert fred.source_tier == SourceTier.PRIMARY
    assert fred.publisher == "FRED"
    assert st_louis_fed.source_tier == SourceTier.PRIMARY


def test_source_quality_classifies_major_news_domains() -> None:
    for url in (
        "https://www.reuters.com/markets/rates/fed-policy-2026-07-01/",
        "https://www.bloomberg.com/news/articles/2026-07-01/fed-cuts-rates",
        "https://www.ft.com/content/fed-policy",
    ):
        quality = score_source_quality(
            source_url=url,
            source_title="Fed policy report",
            snippet="Markets priced a 25 bps rate cut after the release.",
            published_at="2026-07-01",
            retrieved_at=RETRIEVED_AT,
        )

        assert quality.source_tier in {
            SourceTier.HIGH_QUALITY_SECONDARY,
            SourceTier.NEWS,
        }


def test_source_quality_classifies_unknown_domain_as_unknown() -> None:
    quality = score_source_quality(
        source_url="https://research.example.com/macro-note",
        source_title="Macro note",
        snippet="GDP rose 2.1% in the latest data release.",
        published_at="2026-07-01",
        retrieved_at=RETRIEVED_AT,
    )

    assert quality.source_tier == SourceTier.UNKNOWN
    assert quality.domain == "research.example.com"


def test_source_quality_penalizes_missing_url_title_and_snippet() -> None:
    complete = score_source_quality(
        source_url="https://research.example.com/macro-note",
        source_title="Macro note 2026",
        snippet="GDP rose 2.1% in the latest data release.",
        published_at="2026-07-01",
        retrieved_at=RETRIEVED_AT,
    )
    missing = score_source_quality(
        source_url="",
        source_title="",
        snippet="",
        published_at=None,
        retrieved_at=RETRIEVED_AT,
    )

    assert missing.quality_score < complete.quality_score
    assert missing.quality_score == 0
    assert "missing url" in missing.reason
    assert "missing title" in missing.reason
    assert "missing snippet" in missing.reason


def test_source_quality_is_deterministic() -> None:
    kwargs = {
        "source_url": "https://www.reuters.com/markets/rates/fed-policy-2026-07-01/",
        "source_title": "Fed policy report",
        "snippet": "Markets priced a 25 bps rate cut after the release.",
        "published_at": "2026-07-01",
        "retrieved_at": RETRIEVED_AT,
    }

    assert score_source_quality(**kwargs) == score_source_quality(**kwargs)


def test_source_urls_and_snippets_are_preserved() -> None:
    fake = FakeSearchClient(
        {
            "results": [
                {
                    "url": "https://news.example.com/article",
                    "title": "Result title",
                    "snippet": "A provider snippet survives conversion.",
                    "published_date": "2026-07-01",
                }
            ]
        }
    )

    bundle = asyncio.run(
        EvidenceEngine(fake.search).search(
            request(objective="Find provider snippet evidence", max_sources=1)
        )
    )

    assert bundle.items[0].source_url == "https://news.example.com/article"
    assert bundle.items[0].snippet == "A provider snippet survives conversion."
    assert bundle.items[0].published_at == "2026-07-01"


def test_empty_results_produce_empty_bundle_without_crash() -> None:
    fake = FakeSearchClient({"results": []})

    bundle = asyncio.run(EvidenceEngine(fake.search).search(request()))

    assert bundle.items == []
    assert bundle.source_quality_summary["item_count"] == 0


def test_research_tools_evidence_search_uses_existing_web_search_path() -> None:
    class StubResearchTools(ResearchTools):
        def __init__(self) -> None:
            self.calls: list[tuple[object, str]] = []

        async def web_search(self, run_id, query):
            self.calls.append((run_id, query))
            return {
                "results": [
                    {
                        "url": "https://example.com/research-tools",
                        "content": "Evidence from existing web_search.",
                    }
                ]
            }

    run_id = uuid4()
    tools = StubResearchTools()

    bundle = asyncio.run(
        tools.evidence_search(
            run_id,
            request(run_id=uuid4(), objective="Use existing Tavily path", max_sources=1),
        )
    )

    assert bundle.request.run_id == run_id
    assert tools.calls[0][0] == run_id
    assert bundle.items[0].source_url == "https://example.com/research-tools"
