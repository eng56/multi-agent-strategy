import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from src.agents.evidence_engine import (
    EvidenceEngine,
    EvidenceRequest,
    SearchMode,
    generate_evidence_queries,
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
    assert bundle.source_quality_summary["missing_published_at_count"] == 1


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
