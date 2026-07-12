import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.agents.workflow import (
    HANDLERS,
    aggregate,
    create_claim,
    deterministic_partial,
    execute_tool,
    judge,
    plan,
    semantic_branches,
    skeptic_review,
    verify_claim,
)
from src.agents.principal_runtime import (
    ACTIVE_POLICY_PRODUCER,
    SHADOW_POLICY_PRODUCER,
    evaluate_principal_policy,
)
from src.common.budget import BudgetExceeded
from src.integrations.llm import LLMOutputError
from src.worker import MAX_HANDLER_RETRIES, dispatch
from src.common.models import (
    ActionStatus,
    AgentSpec,
    Artifact,
    ArtifactPointer,
    ArtifactStatus,
    ArtifactType,
    Budget,
    Claim,
    EventEnvelope,
    EventType,
    FinalReport,
    ModelPolicy,
    Observation,
    OrganizationPlan,
    PrincipalAction,
    PrincipalActionType,
    ResearchTask,
    RoleModelPolicy,
    Run,
    RunStatus,
    ToolBudget,
    Verification,
    VisibilityScope,
)


def role_policy(cap: float = 0.1, protected: float = 0) -> RoleModelPolicy:
    return RoleModelPolicy(
        model="provider/model",
        cap_usd=cap,
        max_call_cost_usd=min(cap, 0.05),
        protected_usd=protected,
    )


def model_policy(judge_enabled: bool = True) -> ModelPolicy:
    return ModelPolicy(
        planner=role_policy(),
        utility=role_policy(),
        research=role_policy(0.3),
        verifier=role_policy(0.2),
        aggregator=role_policy(0.25, 0.05),
        judge=role_policy(0.1, 0.05) if judge_enabled else None,
    )


class FakeBlackboard:
    def __init__(self, run: Run) -> None:
        self.run = run
        self.tasks: list[ResearchTask] = []
        self.observations = []
        self.claims: list[Claim] = []
        self.verifications: list[Verification] = []
        self.actions: list[PrincipalAction] = []
        self.agent_specs: list[AgentSpec] = []
        self.artifacts: list[Artifact] = []
        self.dead_letters = []
        self.final: FinalReport | None = None
        self.organization_plan: OrganizationPlan | None = None

    async def get_run(self, _run_id):
        return self.run

    async def put_run(self, value):
        self.run = value

    async def put_task(self, value):
        self.tasks = [item for item in self.tasks if item.id != value.id] + [value]

    async def put_observation(self, value):
        self.observations.append(value)

    async def put_claim(self, value):
        self.claims.append(value)

    async def put_verification(self, value):
        self.verifications.append(value)

    async def put_final(self, value):
        self.final = value

    async def get_final(self, _run_id):
        return self.final

    async def put_principal_action(self, value):
        self.actions.append(value)

    async def put_agent_spec(self, value):
        self.agent_specs.append(value)

    async def put_artifact(self, value):
        self.artifacts = [item for item in self.artifacts if item.id != value.id] + [value]

    async def put_dead_letter(self, value):
        self.dead_letters = [item for item in self.dead_letters if item.id != value.id] + [value]

    async def put_organization_plan(self, value):
        self.organization_plan = value

    async def get_organization_plan(self, _run_id):
        return self.organization_plan

    async def list_models(self, _run_id, kind, _model):
        return {
            "tasks": self.tasks,
            "observations": self.observations,
            "claims": self.claims,
            "verifications": self.verifications,
            "principal_actions": self.actions,
            "agent_specs": self.agent_specs,
            "artifacts": self.artifacts,
            "dead_letters": self.dead_letters,
        }.get(kind, [])


class FakeArtifacts:
    def __init__(self) -> None:
        self.payloads = []

    def put_json(self, run_id, kind, _raw):
        self.payloads.append((run_id, kind, _raw))
        return ArtifactPointer(
            uri=f"gs://bucket/runs/{run_id}/{kind}/raw.json",
            size_bytes=2,
            sha256="0" * 64,
        )


class FakeTools:
    async def web_search(self, _run_id, query):
        return {
            "results": [
                {
                    "url": "https://www.sec.gov/Archives/edgar/data/sap",
                    "title": "SAP filing",
                    "content": f"Primary-source evidence for {query[:80]}.",
                    "published_date": "2026-07-01",
                    "score": 0.91,
                }
            ]
        }

    async def market_data(self, _run_id, ticker):
        return {"ticker": ticker, "price": 123}


class QueueLLM:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    async def json(self, *_args, **_kwargs):
        call = {"args": _args, "kwargs": _kwargs}
        if len(_args) >= 5:
            call.update(
                {
                    "run_id": _args[0],
                    "role": _args[1],
                    "name": _args[2],
                    "system": _args[3],
                    "prompt": _args[4],
                }
            )
        self.calls.append(call)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def runtime(
    run: Run, llm: QueueLLM | None = None, *, principal_policy_mode: str = "shadow"
):
    published = []
    blackboard = FakeBlackboard(run)
    return SimpleNamespace(
        blackboard=blackboard,
        llm=llm or QueueLLM(),
        tools=FakeTools(),
        artifacts=FakeArtifacts(),
        settings=SimpleNamespace(
            runtime_topic="agent-runtime", principal_policy_mode=principal_policy_mode
        ),
        events=SimpleNamespace(publish=lambda topic, event: published.append((topic, event))),
        published=published,
    )


def pending_policy_runtime(
    *, principal_policy_mode: str = "shadow", run_status: RunStatus = RunStatus.RUNNING
):
    run = Run(
        question="Should I buy SAP stock?",
        status=run_status,
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="SAP stock catalysts",
        question="SAP stock catalysts",
        tool="web_search",
    )
    rt = runtime(run, QueueLLM(), principal_policy_mode=principal_policy_mode)
    rt.blackboard.tasks.append(task)
    return run, task, rt


def test_shadow_policy_safe_active_dispatches_pending_tool_task() -> None:
    run, task, rt = pending_policy_runtime()

    selected = asyncio.run(evaluate_principal_policy(rt, run.id, trigger="test"))

    assert selected is not None
    assert len(rt.blackboard.actions) == 1
    action = rt.blackboard.actions[0]
    assert action.status == ActionStatus.EXECUTED
    assert action.producer == ACTIVE_POLICY_PRODUCER
    assert action.reason.startswith("active policy selected:")
    assert action.action_type == PrincipalActionType.REQUEST_TOOL_CALL
    assert action.idempotency_key
    task_events = [event for _topic, event in rt.published if event.type == EventType.TASK_CREATED]
    assert len(task_events) == 1
    assert task_events[0].payload["task_id"] == str(task.id)


def test_principal_policy_off_mode_persists_nothing() -> None:
    run, _task, rt = pending_policy_runtime(principal_policy_mode="off")

    selected = asyncio.run(evaluate_principal_policy(rt, run.id, trigger="test"))

    assert selected is None
    assert rt.blackboard.actions == []
    assert rt.published == []


def test_active_assign_task_creates_deterministic_tasks_and_events() -> None:
    run = Run(
        question="Will gold and USD move if Fed cuts rates?",
        status=RunStatus.RUNNING,
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    rt = runtime(run, QueueLLM(), principal_policy_mode="active")

    selected = asyncio.run(evaluate_principal_policy(rt, run.id, trigger="test"))

    assert selected is not None
    assert selected.action_type == PrincipalActionType.ASSIGN_TASK
    assert selected.status == ActionStatus.EXECUTED
    assert selected.producer == ACTIVE_POLICY_PRODUCER
    assert {task.title for task in rt.blackboard.tasks} >= {
        "Gold reaction to surprise Fed cut",
        "US dollar reaction to surprise Fed cut",
    }
    published_types = [event.type for _topic, event in rt.published]
    assert published_types.count(EventType.TASK_CREATED) == len(rt.blackboard.tasks)
    assert EventType.PRINCIPAL_ACTION_CREATED in published_types


def test_active_request_verification_emits_claim_created() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        status=RunStatus.RUNNING,
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="SAP stock catalysts",
        question="SAP stock catalysts",
        tool="web_search",
        status="completed",
    )
    claim = Claim(
        run_id=run.id,
        task_id=task.id,
        statement="SAP backlog is rising.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    rt = runtime(run, QueueLLM(), principal_policy_mode="active")
    rt.blackboard.tasks.append(task)
    rt.blackboard.claims.append(claim)

    selected = asyncio.run(evaluate_principal_policy(rt, run.id, trigger="test"))

    assert selected is not None
    assert selected.action_type == PrincipalActionType.REQUEST_VERIFICATION
    assert selected.status == ActionStatus.EXECUTED
    claim_events = [event for _topic, event in rt.published if event.type == EventType.CLAIM_CREATED]
    assert len(claim_events) == 1
    assert claim_events[0].payload["claim_id"] == str(claim.id)


def test_active_request_aggregation_does_not_double_dispatch() -> None:
    run = Run(
        question="Should I buy SAP?",
        status=RunStatus.RUNNING,
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="SAP",
        question="SAP stock",
        tool="web_search",
        status="completed",
    )
    claim = Claim(
        run_id=run.id,
        task_id=task.id,
        statement="SAP has positive momentum.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    verification = Verification(
        run_id=run.id,
        claim_id=claim.id,
        verdict="verified",
        rationale="Supported.",
        confidence=0.7,
    )
    rt = runtime(run, QueueLLM(), principal_policy_mode="active")
    rt.blackboard.tasks.append(task)
    rt.blackboard.claims.append(claim)
    rt.blackboard.verifications.append(verification)
    rt.blackboard.organization_plan = OrganizationPlan(
        run_id=run.id,
        root_agent_id=uuid4(),
        branches=["market/equities", "synthesis/aggregator"],
    )

    first = asyncio.run(evaluate_principal_policy(rt, run.id, trigger="first"))
    second = asyncio.run(evaluate_principal_policy(rt, run.id, trigger="second"))

    assert first is not None
    assert first.action_type == PrincipalActionType.REQUEST_AGGREGATION
    assert second is None
    aggregation_events = [
        event for _topic, event in rt.published if event.type == EventType.CLAIM_VERIFIED
    ]
    assert len(aggregation_events) == 1
    assert aggregation_events[0].payload["force"] is True


def test_active_request_followup_creates_at_most_one_wave() -> None:
    run = Run(
        question="Should I buy SAP?",
        status=RunStatus.RUNNING,
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    final = FinalReport(
        run_id=run.id,
        answer="Buy with caution.",
        verified_claim_ids=[],
        sources=[],
        judge_score=0.62,
        judge_feedback="Needs stronger valuation evidence. Address downside risks.",
    )
    rt = runtime(run, QueueLLM(), principal_policy_mode="active")
    rt.blackboard.final = final

    first = asyncio.run(evaluate_principal_policy(rt, run.id, trigger="first"))
    second = asyncio.run(evaluate_principal_policy(rt, run.id, trigger="second"))

    assert first is not None
    assert first.action_type == PrincipalActionType.REQUEST_FOLLOWUP
    assert second is None or second.action_type != PrincipalActionType.REQUEST_FOLLOWUP
    followups = [task for task in rt.blackboard.tasks if task.wave_number == 1]
    assert 1 <= len(followups) <= 3
    assert max(task.wave_number for task in rt.blackboard.tasks) == 1
    followup_events = [
        event for _topic, event in rt.published if event.type == EventType.FOLLOWUP_REQUESTED
    ]
    assert len(followup_events) == 1


def test_active_idempotency_prevents_duplicate_tool_dispatch() -> None:
    run, task, rt = pending_policy_runtime(principal_policy_mode="active")
    rt.blackboard.organization_plan = OrganizationPlan(
        run_id=run.id,
        root_agent_id=uuid4(),
        branches=["market/equities"],
    )

    first = asyncio.run(evaluate_principal_policy(rt, run.id, trigger="first"))
    second = asyncio.run(evaluate_principal_policy(rt, run.id, trigger="second"))

    assert first is not None
    assert first.action_type == PrincipalActionType.REQUEST_TOOL_CALL
    assert second is None
    task_events = [event for _topic, event in rt.published if event.type == EventType.TASK_CREATED]
    assert len(task_events) == 1
    assert task_events[0].payload["task_id"] == str(task.id)


def test_shadow_policy_does_not_write_duplicate_proposals() -> None:
    run, _task, rt = pending_policy_runtime()

    first = asyncio.run(evaluate_principal_policy(rt, run.id, trigger="first"))
    second = asyncio.run(evaluate_principal_policy(rt, run.id, trigger="second"))

    assert first is not None
    assert second is None
    assert len(rt.blackboard.actions) == 1


def test_terminal_run_writes_no_shadow_policy_proposal() -> None:
    run, _task, rt = pending_policy_runtime(run_status=RunStatus.COMPLETED)

    selected = asyncio.run(evaluate_principal_policy(rt, run.id, trigger="test"))

    assert selected is None
    assert rt.blackboard.actions == []
    assert rt.published == []


def test_plan_hook_records_tool_dispatch_marker_without_shadow_duplicate() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    rt = runtime(
        run,
        QueueLLM(
            {
                "tasks": [
                    {
                        "title": "SAP stock catalysts",
                        "question": "SAP stock catalysts",
                        "tool": "web_search",
                    }
                ]
            }
        ),
    )

    asyncio.run(plan(rt, EventEnvelope(type=EventType.RUN_CREATED, run_id=run.id, producer="test")))

    assert not any(action.producer == SHADOW_POLICY_PRODUCER for action in rt.blackboard.actions)
    dispatch_actions = [
        action
        for action in rt.blackboard.actions
        if action.action_type == PrincipalActionType.REQUEST_TOOL_CALL
        and action.required_role == "tool_runner"
    ]
    assert len(dispatch_actions) == 1
    assert dispatch_actions[0].status == ActionStatus.EXECUTED
    assert "task_id=" in dispatch_actions[0].reason


def test_every_workflow_stage_has_an_event_handler() -> None:
    assert HANDLERS["planner-agent"] == {
        EventType.RUN_CREATED: HANDLERS["planner-agent"][EventType.RUN_CREATED]
    }
    assert EventType.TASK_CREATED in HANDLERS["tool-runner"]
    assert EventType.OBSERVATION_CREATED in HANDLERS["worker-agents"]
    assert EventType.CLAIM_CREATED in HANDLERS["verifier-agent"]
    assert EventType.SKEPTIC_REVIEW_REQUESTED in HANDLERS["skeptic-agent"]
    assert EventType.CLAIM_VERIFIED in HANDLERS["aggregator-agent"]
    assert EventType.FINAL_CREATED in HANDLERS["judge-agent"]


def test_future_event_types_serialize_without_handler_rewire() -> None:
    event = EventEnvelope(type=EventType.ARTIFACT_CREATED, run_id=uuid4(), producer="test")
    assert event.model_dump(mode="json")["type"] == "artifact.created"
    handled_events = {event_type for handlers in HANDLERS.values() for event_type in handlers}
    assert EventType.ARTIFACT_CREATED not in handled_events


def test_planner_creates_organization_and_actions_then_falls_back_when_no_tasks() -> None:
    run = Run(
        question="Will gold rise if Fed cuts rates?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    rt = runtime(run, QueueLLM({"tasks": []}))

    asyncio.run(plan(rt, EventEnvelope(type=EventType.RUN_CREATED, run_id=run.id, producer="test")))

    assert rt.blackboard.organization_plan is not None
    assert {agent.branch for agent in rt.blackboard.agent_specs} >= {
        "root",
        "market/gold",
        "macro/rates",
        "trust/source_verifier",
        "synthesis/aggregator",
        "synthesis/judge",
    }
    assert any(
        action.action_type == PrincipalActionType.SPAWN_AGENT for action in rt.blackboard.actions
    )
    assert rt.blackboard.run.status == RunStatus.RUNNING
    assert len(rt.blackboard.tasks) >= 2
    assert any(
        action.action_type == PrincipalActionType.ASSIGN_TASK for action in rt.blackboard.actions
    )


def test_planner_malformed_json_retries_and_creates_tasks() -> None:
    run = Run(
        question="Will gold rise if Fed cuts rates?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    rt = runtime(
        run,
        QueueLLM(
            LLMOutputError("model returned invalid JSON for planner: ```json truncated"),
            {
                "tasks": [
                    {
                        "title": "Gold and Fed cuts",
                        "question": "Find evidence on gold after surprise Fed cuts.",
                        "tool": "web_search",
                    }
                ]
            },
        ),
    )

    asyncio.run(plan(rt, EventEnvelope(type=EventType.RUN_CREATED, run_id=run.id, producer="test")))

    assert rt.blackboard.run.status == RunStatus.RUNNING
    assert [task.title for task in rt.blackboard.tasks] == ["Gold and Fed cuts"]
    assert any(
        action.action_type == PrincipalActionType.ASSIGN_TASK for action in rt.blackboard.actions
    )


def test_planner_malformed_json_twice_uses_deterministic_fallback() -> None:
    run = Run(
        question="Will gold and USD move if Fed cuts rates?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    rt = runtime(
        run,
        QueueLLM(
            LLMOutputError("model returned invalid JSON for planner: first bad output"),
            LLMOutputError(
                "model returned invalid JSON for planner-retry-compact: second bad output"
            ),
        ),
    )

    asyncio.run(plan(rt, EventEnvelope(type=EventType.RUN_CREATED, run_id=run.id, producer="test")))

    titles = {task.title for task in rt.blackboard.tasks}
    assert "Gold reaction to surprise Fed cut" in titles
    assert "US dollar reaction to surprise Fed cut" in titles
    assert rt.blackboard.run.status == RunStatus.RUNNING
    assert rt.blackboard.run.failure_reason is None


def test_planner_uses_deterministic_fallback_when_all_tasks_are_invalid() -> None:
    run = Run(
        question="Will gold and USD move if Fed cuts rates?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    rt = runtime(
        run,
        QueueLLM(
            {
                "tasks": [
                    {"title": "Missing question and tool"},
                    {
                        "title": "Unsupported tool",
                        "question": "Research this with an unsupported tool.",
                        "tool": "calculator",
                    },
                ]
            }
        ),
    )

    asyncio.run(plan(rt, EventEnvelope(type=EventType.RUN_CREATED, run_id=run.id, producer="test")))

    titles = {task.title for task in rt.blackboard.tasks}
    assert "Gold reaction to surprise Fed cut" in titles
    assert "US dollar reaction to surprise Fed cut" in titles
    assert rt.blackboard.run.status == RunStatus.RUNNING
    assert rt.blackboard.run.failure_reason is None


def test_deterministic_fallback_uses_semantic_branches_without_irrelevant_crypto() -> None:
    run = Run(
        question="Whether gold is better than equities after a Fed cut",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    rt = runtime(
        run,
        QueueLLM(
            LLMOutputError("model returned invalid JSON for planner: first bad output"),
            LLMOutputError(
                "model returned invalid JSON for planner-retry-compact: second bad output"
            ),
        ),
    )

    asyncio.run(plan(rt, EventEnvelope(type=EventType.RUN_CREATED, run_id=run.id, producer="test")))

    branches = {agent.branch for agent in rt.blackboard.agent_specs}
    assert "market/gold" in branches
    assert "market/equities" in branches
    assert "macro/rates" in branches
    assert "market/crypto" not in branches
    assert all("Crypto" not in task.title for task in rt.blackboard.tasks)


def test_semantic_branches_use_token_matching_for_crypto_false_positives() -> None:
    assert "market/crypto" not in semantic_branches("whether gold is better than equities")
    assert "market/crypto" not in semantic_branches("method for equities after Fed cuts")
    assert "market/crypto" in semantic_branches("ETH vs BTC after Fed cuts")
    assert "market/equities" in semantic_branches("SAP stock over the next two months")


def test_planner_persists_assign_task_actions_and_legacy_tasks() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    rt = runtime(
        run,
        QueueLLM(
            {
                "tasks": [
                    {
                        "title": "SAP stock catalysts",
                        "question": "SAP stock catalysts",
                        "tool": "web_search",
                    }
                ]
            }
        ),
    )

    asyncio.run(plan(rt, EventEnvelope(type=EventType.RUN_CREATED, run_id=run.id, producer="test")))

    assert len(rt.blackboard.tasks) == 1
    assert any(
        action.action_type == PrincipalActionType.ASSIGN_TASK for action in rt.blackboard.actions
    )
    assert rt.blackboard.organization_plan is not None


def test_execute_tool_dual_writes_observation_artifact() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="SAP stock catalysts",
        question="SAP stock catalysts",
        tool="web_search",
    )
    rt = runtime(run, QueueLLM({"summary": "SAP has cloud catalysts."}))
    rt.blackboard.tasks.append(task)

    asyncio.run(
        execute_tool(
            rt,
            EventEnvelope(
                type=EventType.TASK_CREATED,
                run_id=run.id,
                producer="test",
                payload={"task_id": str(task.id)},
            ),
        )
    )

    assert rt.blackboard.observations
    artifact = rt.blackboard.artifacts[-1]
    assert artifact.artifact_type == ArtifactType.OBSERVATION
    assert artifact.branch == "market/equities"
    assert any(
        action.action_type == PrincipalActionType.REQUEST_TOOL_CALL
        for action in rt.blackboard.actions
    )


def test_execute_tool_web_search_creates_evidence_bundle_backed_observation() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="SAP stock catalysts",
        question="SAP cloud backlog catalysts",
        tool="web_search",
    )
    rt = runtime(run, QueueLLM())
    rt.blackboard.tasks.append(task)

    class EvidenceTools(FakeTools):
        async def web_search(self, _run_id, _query):
            return {
                "results": [
                    {
                        "url": "https://example.com/sap-cloud",
                        "title": "SAP cloud backlog",
                        "content": "SAP cloud backlog expanded with resilient demand.",
                        "published_date": "2026-07-01",
                        "score": 0.9,
                    },
                    {
                        "url": "https://example.com/sap-margin",
                        "title": "SAP margin context",
                        "snippet": "Margins remain a constraint on the stock thesis.",
                        "score": 0.6,
                    },
                ]
            }

    rt.tools = EvidenceTools()

    asyncio.run(
        execute_tool(
            rt,
            EventEnvelope(
                type=EventType.TASK_CREATED,
                run_id=run.id,
                producer="test",
                payload={"task_id": str(task.id)},
            ),
        )
    )

    observation = rt.blackboard.observations[0]
    assert "2 source(s)" in observation.summary
    assert "SAP cloud backlog expanded" in observation.summary
    assert "Source quality:" in observation.summary
    assert observation.sources == [
        "https://example.com/sap-cloud",
        "https://example.com/sap-margin",
    ]
    artifact = rt.blackboard.artifacts[-1]
    assert artifact.source_refs == observation.sources
    assert "evidence_engine" in artifact.tags

    payloads = rt.artifacts.payloads
    assert any(kind == "evidence-search" for _run_id, kind, _raw in payloads)
    bundle_payload = next(raw for _run_id, kind, raw in payloads if kind == "web_search")
    assert bundle_payload["branch"] == "market/equities"
    assert bundle_payload["source_refs"] == observation.sources
    first_item = bundle_payload["evidence_bundle"]["items"][0]
    assert first_item["source_url"] == "https://example.com/sap-cloud"
    assert first_item["source_tier"] == "unknown"
    assert first_item["domain"] == "example.com"
    assert first_item["quality_score"] > 0
    assert "source_quality_reason" in first_item


def test_execute_tool_web_search_empty_evidence_bundle_still_observes() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Sparse SAP query",
        question="obscure SAP catalyst with no results",
        tool="web_search",
    )
    rt = runtime(run, QueueLLM())
    rt.blackboard.tasks.append(task)

    class EmptyTools(FakeTools):
        async def web_search(self, _run_id, _query):
            return {"results": []}

    rt.tools = EmptyTools()

    asyncio.run(
        execute_tool(
            rt,
            EventEnvelope(
                type=EventType.TASK_CREATED,
                run_id=run.id,
                producer="test",
                payload={"task_id": str(task.id)},
            ),
        )
    )

    observation = rt.blackboard.observations[0]
    assert rt.blackboard.tasks[0].status == "completed"
    assert observation.sources == []
    assert "0 source(s)" in observation.summary
    assert "No provider results were returned" in observation.summary
    assert rt.blackboard.artifacts[-1].source_refs == []


def test_execute_tool_market_data_path_uses_deterministic_snapshot() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="SAP market data",
        question="Get SAP ticker market data",
        tool="market_data",
    )
    llm = QueueLLM()
    rt = runtime(run, llm)
    rt.blackboard.tasks.append(task)

    asyncio.run(
        execute_tool(
            rt,
            EventEnvelope(
                type=EventType.TASK_CREATED,
                run_id=run.id,
                producer="test",
                payload={"task_id": str(task.id)},
            ),
        )
    )

    assert llm.calls == []
    observation = rt.blackboard.observations[0]
    assert "MarketSnapshot for market/equities SAP market data" in observation.summary
    assert "selected SAP" in observation.summary
    assert observation.sources == ["https://massive.com/stocks/SAP"]
    assert rt.blackboard.artifacts[-1].source_refs == observation.sources
    assert {"market_snapshot", "tool:market_data", "market:equities"}.issubset(
        set(rt.blackboard.artifacts[-1].tags)
    )
    assert rt.blackboard.claims == []
    assert not any(event.type == EventType.OBSERVATION_CREATED for _topic, event in rt.published)
    assert run.budget.tools.tavily_credits_used == 0


def test_execute_tool_market_data_gap_creates_artifact_without_observation_event() -> None:
    run = Run(
        question="Will gold rise if the Fed cuts rates?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Gold market data",
        question="Get GC=F gold market data",
        tool="market_data",
    )
    rt = runtime(run, QueueLLM())
    rt.blackboard.tasks.append(task)

    class EmptyMarketDataTools(FakeTools):
        def __init__(self) -> None:
            self.calls = []

        async def market_data(self, _run_id, ticker):
            self.calls.append(ticker)
            return {"ticker": ticker, "results": [], "resultsCount": 0}

    tools = EmptyMarketDataTools()
    rt.tools = tools

    asyncio.run(
        execute_tool(
            rt,
            EventEnvelope(
                type=EventType.TASK_CREATED,
                run_id=run.id,
                producer="test",
                payload={"task_id": str(task.id)},
            ),
        )
    )

    assert tools.calls == ["GLD"]
    assert rt.blackboard.observations == []
    assert rt.blackboard.claims == []
    assert rt.blackboard.tasks[0].status == "failed"
    assert not any(event.type == EventType.OBSERVATION_CREATED for _topic, event in rt.published)
    artifact = rt.blackboard.artifacts[-1]
    assert artifact.artifact_type == ArtifactType.DATA_GAP
    assert "market data gap" in artifact.text_or_summary.lower()
    assert "investment thesis" in artifact.text_or_summary
    assert {"data_gap", "market_data_gap", "tool:market_data", "market:gold"}.issubset(
        set(artifact.tags)
    )


def test_market_data_tasks_use_branch_grounded_symbols_despite_global_bond_question() -> None:
    run = Run(
        question=(
            "If the Fed signals faster rate cuts, compare U.S. equities, the U.S. dollar, "
            "gold, and long-duration bonds."
        ),
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget(market_data_max_requests=4)),
    )
    tasks = [
        ResearchTask(
            run_id=run.id,
            title="U.S. equities market data",
            question="Fetch market data for U.S. equities.",
            tool="market_data",
        ),
        ResearchTask(
            run_id=run.id,
            title="U.S. dollar market data",
            question="Fetch market data for the U.S. dollar.",
            tool="market_data",
        ),
        ResearchTask(
            run_id=run.id,
            title="Gold market data",
            question="Fetch market data for gold.",
            tool="market_data",
        ),
        ResearchTask(
            run_id=run.id,
            title="Long-duration bond market data",
            question="Fetch market data for long-duration bonds.",
            tool="market_data",
        ),
    ]
    rt = runtime(run, QueueLLM())
    rt.blackboard.tasks.extend(tasks)

    class RecordingMarketDataTools(FakeTools):
        def __init__(self) -> None:
            self.calls = []

        async def market_data(self, _run_id, ticker):
            self.calls.append(ticker)
            return {"ticker": ticker, "results": [{"c": 1}], "resultsCount": 1}

    tools = RecordingMarketDataTools()
    rt.tools = tools

    for task in tasks:
        asyncio.run(
            execute_tool(
                rt,
                EventEnvelope(
                    type=EventType.TASK_CREATED,
                    run_id=run.id,
                    producer="test",
                    payload={"task_id": str(task.id)},
                ),
            )
        )

    assert tools.calls == ["SPY", "UUP", "GLD", "TLT"]
    summaries = [observation.summary for observation in rt.blackboard.observations]
    assert any("selected SPY" in summary for summary in summaries)
    assert any("selected UUP" in summary for summary in summaries)
    assert any("selected GLD" in summary for summary in summaries)
    assert any("selected TLT" in summary for summary in summaries)


def test_explicit_research_task_branch_overrides_text_inference_for_market_data() -> None:
    run = Run(
        question="Compare U.S. equities and long-duration bonds.",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget(market_data_max_requests=1)),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Long-duration bond market data",
        question="Fetch TLT data, but this task is explicitly the equities branch.",
        tool="market_data",
        branch="market/equities",
    )
    rt = runtime(run, QueueLLM())
    rt.blackboard.tasks.append(task)

    class RecordingMarketDataTools(FakeTools):
        def __init__(self) -> None:
            self.calls = []

        async def market_data(self, _run_id, ticker):
            self.calls.append(ticker)
            return {"ticker": ticker, "results": [{"c": 1}], "resultsCount": 1}

    tools = RecordingMarketDataTools()
    rt.tools = tools

    asyncio.run(
        execute_tool(
            rt,
            EventEnvelope(
                type=EventType.TASK_CREATED,
                run_id=run.id,
                producer="test",
                payload={"task_id": str(task.id)},
            ),
        )
    )

    assert tools.calls == ["SPY"]
    assert rt.blackboard.artifacts[-1].branch == "market/equities"


def test_execute_tool_web_search_creates_evidence_observation_with_agent_branch() -> None:
    run = Run(
        question="Will gold rise if Fed cuts rates?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Gold after Fed cuts",
        question="Find gold evidence after Fed cuts.",
        tool="web_search",
    )
    llm = QueueLLM({"summary": "Gold context kept."})
    rt = runtime(run, llm)
    rt.blackboard.tasks.append(task)
    rt.blackboard.agent_specs.append(
        AgentSpec(
            run_id=run.id,
            name="gold agent",
            role_template="asset_research_agent",
            branch="market/gold",
            domain="market",
            objective="Research gold evidence relevant to Fed cuts.",
            allowed_tools=["web_search", "market_data"],
            retrieval_tags=["market:gold", "macro:rates"],
            local_budget_usd=0.07,
            visibility_scope=VisibilityScope.TEAM,
        )
    )

    asyncio.run(
        execute_tool(
            rt,
            EventEnvelope(
                type=EventType.TASK_CREATED,
                run_id=run.id,
                producer="test",
                payload={"task_id": str(task.id)},
            ),
        )
    )

    assert llm.calls == []
    observation = rt.blackboard.observations[0]
    assert "EvidenceEngine observation for branch market/gold" in observation.summary
    assert "1 source(s)" in observation.summary
    assert "Strongest evidence snippets" in observation.summary
    assert "Source limitations" in observation.summary
    assert "Branch: market/gold" in observation.summary
    artifact = rt.blackboard.artifacts[-1]
    assert artifact.branch == "market/gold"
    assert artifact.legacy_object_type == "observation"
    assert artifact.legacy_object_id == observation.id
    assert artifact.source_refs == observation.sources
    assert {"evidence_engine", "tool:web_search", "market:gold"}.issubset(
        set(artifact.tags)
    )


def test_create_claim_dual_writes_claim_artifact() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="SAP stock catalysts",
        question="SAP stock catalysts",
        tool="web_search",
    )
    observation_id = uuid4()
    observation = SimpleNamespace(
        id=observation_id,
        run_id=run.id,
        task_id=task.id,
        summary="SAP cloud backlog is rising.",
        sources=["https://example.com/sap"],
    )
    rt = runtime(run, QueueLLM({"statement": "SAP cloud backlog is rising.", "confidence": 0.8}))
    rt.blackboard.tasks.append(task)
    rt.blackboard.observations.append(observation)
    rt.blackboard.artifacts.append(
        Artifact(
            run_id=run.id,
            artifact_type=ArtifactType.OBSERVATION,
            branch="market/equities",
            text_or_summary=observation.summary,
            legacy_object_type="observation",
            legacy_object_id=observation_id,
        )
    )

    asyncio.run(
        create_claim(
            rt,
            EventEnvelope(
                type=EventType.OBSERVATION_CREATED,
                run_id=run.id,
                producer="test",
                payload={"observation_id": str(observation_id)},
            ),
        )
    )

    assert rt.blackboard.claims
    artifact = rt.blackboard.artifacts[-1]
    assert artifact.artifact_type == ArtifactType.CLAIM
    assert artifact.depends_on_artifact_ids


def test_create_claim_generates_multiple_atomic_asset_specific_claims() -> None:
    run = Run(
        question="What happens to rates, gold, and equities if the Fed cuts faster?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Cross-asset Fed cuts",
        question="Fed cuts and market reactions",
        tool="web_search",
        branch="macro/rates",
    )
    observation = Observation(
        run_id=run.id,
        task_id=task.id,
        tool="web_search",
        summary=(
            "Lower expected policy rates can reduce Treasury yields, support gold "
            "through lower real yields, and help equities through lower discount rates "
            "unless cuts signal a growth scare."
        ),
        artifact=ArtifactPointer(uri="gs://bucket/raw.json", size_bytes=2, sha256="0" * 64),
        sources=["https://www.federalreserve.gov/monetarypolicy"],
    )
    llm = QueueLLM(
        {
            "claims": [
                {
                    "statement": "Lower expected Fed policy rates tend to reduce front-end Treasury yields, all else equal.",
                    "claim_type": "mechanism",
                    "asset": "rates",
                    "direction": "down",
                    "time_horizon": "unspecified",
                    "confidence": 0.72,
                },
                {
                    "statement": "Lower real yields tend to support gold by reducing the opportunity cost of holding a non-yielding asset.",
                    "claim_type": "mechanism",
                    "asset": "gold",
                    "direction": "up",
                    "time_horizon": "unspecified",
                    "confidence": 0.7,
                },
                {
                    "statement": "U.S. equities can benefit from lower discount rates, but growth-scare cuts can offset that effect through weaker earnings expectations.",
                    "claim_type": "market_reaction",
                    "asset": "equities",
                    "direction": "mixed",
                    "time_horizon": "unspecified",
                    "confidence": 0.66,
                },
            ]
        }
    )
    rt = runtime(run, llm)
    rt.blackboard.tasks.append(task)
    rt.blackboard.observations.append(observation)
    rt.blackboard.artifacts.append(
        Artifact(
            run_id=run.id,
            artifact_type=ArtifactType.OBSERVATION,
            branch="macro/rates",
            text_or_summary=observation.summary,
            legacy_object_type="observation",
            legacy_object_id=observation.id,
        )
    )

    asyncio.run(
        create_claim(
            rt,
            EventEnvelope(
                type=EventType.OBSERVATION_CREATED,
                run_id=run.id,
                producer="test",
                payload={"observation_id": str(observation.id)},
            ),
        )
    )

    assert len(rt.blackboard.claims) == 3
    assert {claim.asset for claim in rt.blackboard.claims} == {"rates", "gold", "equities"}
    assert all(len(claim.statement) < 180 for claim in rt.blackboard.claims)
    assert all(claim.claim_type in {"mechanism", "market_reaction"} for claim in rt.blackboard.claims)
    assert sum(event.type == EventType.CLAIM_CREATED for _topic, event in rt.published) == 3


def test_create_claim_skips_data_gap_observation() -> None:
    run = Run(
        question="Will gold rise if the Fed cuts rates?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Gold market data",
        question="Get GC=F gold market data",
        tool="market_data",
    )
    observation_id = uuid4()
    observation = Observation(
        id=observation_id,
        run_id=run.id,
        task_id=task.id,
        tool="market_data",
        summary="Market data gap for gold: GLD returned zero results.",
        artifact=ArtifactPointer(uri="gs://bucket/gap.json", size_bytes=2, sha256="0" * 64),
        sources=[],
    )
    llm = QueueLLM({"statement": "Gold data is unavailable.", "confidence": 0.8})
    rt = runtime(run, llm)
    rt.blackboard.tasks.append(task)
    rt.blackboard.observations.append(observation)
    rt.blackboard.artifacts.append(
        Artifact(
            run_id=run.id,
            artifact_type=ArtifactType.DATA_GAP,
            branch="market/gold",
            text_or_summary="Market data gap for gold.",
            tags=["data_gap", "market_data_gap"],
            legacy_object_type="observation",
            legacy_object_id=observation_id,
        )
    )

    asyncio.run(
        create_claim(
            rt,
            EventEnvelope(
                type=EventType.OBSERVATION_CREATED,
                run_id=run.id,
                producer="test",
                payload={"observation_id": str(observation_id)},
            ),
        )
    )

    assert llm.calls == []
    assert rt.blackboard.claims == []
    assert any(
        action.action_type == PrincipalActionType.REQUEST_VERIFICATION
        and "Skipped claim extraction for data-gap" in action.reason
        for action in rt.blackboard.actions
    )


def test_create_claim_falls_back_when_claim_extractor_returns_invalid_json() -> None:
    run = Run(
        question="What happens to equities if the Fed cuts faster?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Equity evidence",
        question="Fed cuts and U.S. equities",
        tool="web_search",
    )
    observation_id = uuid4()
    observation = SimpleNamespace(
        id=observation_id,
        run_id=run.id,
        task_id=task.id,
        summary=(
            "Equities can benefit from lower discount rates after Fed cuts, while "
            "growth-scare cuts can pressure earnings expectations."
        ),
        sources=["https://example.com/equities"],
    )
    llm = QueueLLM(
        LLMOutputError('model returned non-object JSON for claim-extractor: ["statement"]')
    )
    rt = runtime(run, llm)
    rt.blackboard.tasks.append(task)
    rt.blackboard.observations.append(observation)
    rt.blackboard.artifacts.append(
        Artifact(
            run_id=run.id,
            artifact_type=ArtifactType.OBSERVATION,
            branch="market/equities",
            text_or_summary=observation.summary,
            legacy_object_type="observation",
            legacy_object_id=observation_id,
        )
    )

    asyncio.run(
        create_claim(
            rt,
            EventEnvelope(
                type=EventType.OBSERVATION_CREATED,
                run_id=run.id,
                producer="test",
                payload={"observation_id": str(observation_id)},
            ),
        )
    )

    assert len(rt.blackboard.claims) == 1
    assert rt.blackboard.claims[0].statement == (
        "Equities can benefit from easier expected policy, but the reaction depends on "
        "whether cuts signal easing support or a growth scare."
    )
    assert rt.blackboard.tasks[0].status == "created"
    assert any(event.type == EventType.CLAIM_CREATED for _topic, event in rt.published)
    assert any(
        action.action_type == PrincipalActionType.REQUEST_VERIFICATION
        and "Claim extraction model returned invalid output" in action.reason
        for action in rt.blackboard.actions
    )


def test_create_claim_skips_source_meta_observation_without_claim() -> None:
    run = Run(
        question="What happens to equities if the Fed cuts faster?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Equity evidence",
        question="Fed cuts and U.S. equities",
        tool="web_search",
        branch="market/equities",
    )
    observation = Observation(
        run_id=run.id,
        task_id=task.id,
        tool="web_search",
        summary="EvidenceEngine observation for branch market/equities: 5 source(s).",
        artifact=ArtifactPointer(uri="gs://bucket/raw.json", size_bytes=2, sha256="0" * 64),
        sources=["https://example.com/equities"],
    )
    llm = QueueLLM(
        {
            "statement": "EvidenceEngine retrieval includes 5 sources with missing publication dates.",
            "confidence": 0.8,
        }
    )
    rt = runtime(run, llm)
    rt.blackboard.tasks.append(task)
    rt.blackboard.observations.append(observation)

    asyncio.run(
        create_claim(
            rt,
            EventEnvelope(
                type=EventType.OBSERVATION_CREATED,
                run_id=run.id,
                producer="test",
                payload={"observation_id": str(observation.id)},
            ),
        )
    )

    assert rt.blackboard.claims == []
    assert llm.calls == []
    assert any(
        action.action_type == PrincipalActionType.ASSIGN_TASK
        and "Skipped claim extraction for source/provider metadata" in action.reason
        for action in rt.blackboard.actions
    )
    assert any("source_meta_claim_rejected" in artifact.tags for artifact in rt.blackboard.artifacts)


def test_create_claim_rejects_source_meta_model_output_before_verifier() -> None:
    run = Run(
        question="What happens to equities if the Fed cuts faster?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Equity mechanism evidence",
        question="Fed cuts and U.S. equities",
        tool="web_search",
        branch="market/equities",
    )
    observation = Observation(
        run_id=run.id,
        task_id=task.id,
        tool="web_search",
        summary="Lower discount rates can support equities, while weaker earnings expectations can offset that effect.",
        artifact=ArtifactPointer(uri="gs://bucket/raw.json", size_bytes=2, sha256="0" * 64),
        sources=["https://example.com/equities"],
    )
    llm = QueueLLM(
        {
            "claims": [
                {
                    "statement": "The source title says financial publications discuss Fed cuts.",
                    "claim_type": "data_point",
                    "asset": "equities",
                    "direction": "unknown",
                    "time_horizon": "current",
                    "confidence": 0.8,
                }
            ]
        }
    )
    rt = runtime(run, llm)
    rt.blackboard.tasks.append(task)
    rt.blackboard.observations.append(observation)
    rt.blackboard.artifacts.append(
        Artifact(
            run_id=run.id,
            artifact_type=ArtifactType.OBSERVATION,
            branch="market/equities",
            text_or_summary=observation.summary,
            legacy_object_type="observation",
            legacy_object_id=observation.id,
        )
    )

    asyncio.run(
        create_claim(
            rt,
            EventEnvelope(
                type=EventType.OBSERVATION_CREATED,
                run_id=run.id,
                producer="test",
                payload={"observation_id": str(observation.id)},
            ),
        )
    )

    assert rt.blackboard.claims == []
    assert not any(event.type == EventType.CLAIM_CREATED for _topic, event in rt.published)
    assert any("source_meta_claim_rejected" in artifact.tags for artifact in rt.blackboard.artifacts)


def test_create_claim_skips_trust_source_verifier_observation() -> None:
    run = Run(
        question="What happens if the Fed cuts faster?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Verifier observation",
        question="Verifier rationale",
        tool="web_search",
        branch="trust/source_verifier",
    )
    observation = Observation(
        run_id=run.id,
        task_id=task.id,
        tool="verification_web_search",
        summary="Verifier rationale with supported parts.",
        artifact=ArtifactPointer(uri="gs://bucket/verification.json", size_bytes=2, sha256="0" * 64),
        sources=["https://example.com/verifier"],
    )
    rt = runtime(run, QueueLLM({"statement": "This should not be called.", "confidence": 0.8}))
    rt.blackboard.tasks.append(task)
    rt.blackboard.observations.append(observation)
    rt.blackboard.artifacts.append(
        Artifact(
            run_id=run.id,
            artifact_type=ArtifactType.OBSERVATION,
            branch="trust/source_verifier",
            text_or_summary=observation.summary,
            legacy_object_type="observation",
            legacy_object_id=observation.id,
        )
    )

    asyncio.run(
        create_claim(
            rt,
            EventEnvelope(
                type=EventType.OBSERVATION_CREATED,
                run_id=run.id,
                producer="test",
                payload={"observation_id": str(observation.id)},
            ),
        )
    )

    assert rt.llm.calls == []
    assert rt.blackboard.claims == []
    assert any("trust-layer observation" in action.reason for action in rt.blackboard.actions)


def test_claim_cap_triggers_synthesis_path_without_more_claim_generation() -> None:
    run = Run(
        question="What happens if the Fed cuts faster?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Rates evidence",
        question="Fed cuts and rates",
        tool="web_search",
        branch="macro/rates",
    )
    observation = Observation(
        run_id=run.id,
        task_id=task.id,
        tool="web_search",
        summary="New observation should not produce claim after cap.",
        artifact=ArtifactPointer(uri="gs://bucket/raw.json", size_bytes=2, sha256="0" * 64),
        sources=["https://example.com/rates"],
    )
    rt = runtime(run, QueueLLM({"statement": "This should not be called.", "confidence": 0.8}))
    rt.blackboard.tasks.append(task)
    rt.blackboard.observations.append(observation)
    for index in range(20):
        claim = Claim(
            run_id=run.id,
            task_id=task.id,
            statement=f"Existing normal claim {index}.",
            evidence_observation_ids=[uuid4()],
            confidence=0.5,
        )
        rt.blackboard.claims.append(claim)
        rt.blackboard.artifacts.append(
            Artifact(
                run_id=run.id,
                artifact_type=ArtifactType.CLAIM,
                branch="macro/rates" if index < 5 else "market/equities",
                text_or_summary=claim.statement,
                legacy_object_type="claim",
                legacy_object_id=claim.id,
            )
        )
    rt.blackboard.artifacts.append(
        Artifact(
            run_id=run.id,
            artifact_type=ArtifactType.OBSERVATION,
            branch="macro/rates",
            text_or_summary=observation.summary,
            legacy_object_type="observation",
            legacy_object_id=observation.id,
        )
    )

    asyncio.run(
        create_claim(
            rt,
            EventEnvelope(
                type=EventType.OBSERVATION_CREATED,
                run_id=run.id,
                producer="test",
                payload={"observation_id": str(observation.id)},
            ),
        )
    )

    assert rt.llm.calls == []
    assert len(rt.blackboard.claims) == 20
    assert any(
        action.action_type == PrincipalActionType.REQUEST_AGGREGATION
        and "Claim cap reached" in action.reason
        for action in rt.blackboard.actions
    )
    assert any(event.type == EventType.CLAIM_VERIFIED for _topic, event in rt.published)


def test_create_claim_skips_market_snapshot_observation_without_claim_or_verification() -> None:
    run = Run(
        question="What happens to long-duration bonds if the Fed cuts faster?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Long-duration Treasury Yield & Price Response to Fed Cuts",
        question="TLT market data",
        tool="market_data",
    )
    observation_id = uuid4()
    observation = Observation(
        id=observation_id,
        run_id=run.id,
        task_id=task.id,
        tool="market_data",
        summary=(
            "MarketSnapshot for macro/rates Long-Duration Treasury Yield & Price Response "
            "to Fed Cuts: selected TLT (etf) via massive/stocks; 251 result(s) returned. "
            "Resolver rationale: TLT is an ETF traded through the stocks aggregate endpoint. "
            "Latest available close/price: 85.45. Latest available volume: 17940936.644408."
        ),
        artifact=ArtifactPointer(uri="gs://bucket/tlt.json", size_bytes=2, sha256="0" * 64),
        sources=["https://massive.com/stocks/TLT"],
    )
    llm = QueueLLM({"statement": "This should not be called.", "confidence": 0.9})
    rt = runtime(run, llm)
    rt.blackboard.tasks.append(task)
    rt.blackboard.observations.append(observation)
    rt.blackboard.artifacts.append(
        Artifact(
            run_id=run.id,
            artifact_type=ArtifactType.OBSERVATION,
            branch="macro/rates",
            text_or_summary=observation.summary,
            legacy_object_type="observation",
            legacy_object_id=observation_id,
        )
    )

    asyncio.run(
        create_claim(
            rt,
            EventEnvelope(
                type=EventType.OBSERVATION_CREATED,
                run_id=run.id,
                producer="test",
                payload={"observation_id": str(observation_id)},
            ),
        )
    )

    assert llm.calls == []
    assert rt.blackboard.claims == []
    assert not any(event.type == EventType.CLAIM_CREATED for _topic, event in rt.published)
    assert any(
        action.action_type == PrincipalActionType.ASSIGN_TASK
        and "Skipped claim extraction for market snapshot" in action.reason
        for action in rt.blackboard.actions
    )


def test_verify_claim_skips_legacy_market_snapshot_claim_without_search() -> None:
    run = Run(
        question="What happens to long-duration bonds if the Fed cuts faster?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    claim = Claim(
        run_id=run.id,
        task_id=uuid4(),
        statement="Market data snapshot selected TLT (etf) via massive/stocks.",
        evidence_observation_ids=[uuid4()],
        sources=["https://massive.com/stocks/TLT"],
        confidence=0.7,
    )
    claim_artifact = Artifact(
        run_id=run.id,
        artifact_type=ArtifactType.CLAIM,
        branch="market/bonds",
        text_or_summary=claim.statement,
        tags=["market_snapshot", "context_only"],
        legacy_object_type="claim",
        legacy_object_id=claim.id,
    )
    rt = runtime(
        run,
        QueueLLM({"verdict": "verified", "rationale": "Should not run", "confidence": 0.9}),
    )
    rt.blackboard.claims.append(claim)
    rt.blackboard.artifacts.append(claim_artifact)

    asyncio.run(
        verify_claim(
            rt,
            EventEnvelope(
                type=EventType.CLAIM_CREATED,
                run_id=run.id,
                producer="test",
                payload={"claim_id": str(claim.id)},
            ),
        )
    )

    assert rt.llm.calls == []
    assert rt.blackboard.verifications == []
    assert any(
        "market_snapshot_claim_rejected" in artifact.tags
        for artifact in rt.blackboard.artifacts
    )
    assert any(
        action.action_type == PrincipalActionType.REQUEST_AGGREGATION
        and "Skipped context-only/source-meta claim" in action.reason
        for action in rt.blackboard.actions
    )


def test_verify_claim_dual_writes_verification_artifact() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    claim = Claim(
        run_id=run.id,
        task_id=uuid4(),
        statement="SAP backlog is rising.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    rt = runtime(
        run, QueueLLM({"verdict": "verified", "rationale": "Supported.", "confidence": 0.7})
    )
    rt.blackboard.claims.append(claim)
    rt.blackboard.artifacts.append(
        Artifact(
            run_id=run.id,
            artifact_type=ArtifactType.CLAIM,
            branch="market/equities",
            text_or_summary=claim.statement,
            legacy_object_type="claim",
            legacy_object_id=claim.id,
        )
    )

    asyncio.run(
        verify_claim(
            rt,
            EventEnvelope(
                type=EventType.CLAIM_CREATED,
                run_id=run.id,
                producer="test",
                payload={"claim_id": str(claim.id)},
            ),
        )
    )

    assert rt.blackboard.verifications
    artifact = rt.blackboard.artifacts[-1]
    assert artifact.artifact_type == ArtifactType.VERIFICATION
    assert artifact.status == ArtifactStatus.VERIFIED
    assert artifact.supports_artifact_ids
    verification = rt.blackboard.verifications[-1]
    assert verification.evidence_item_refs == ["https://www.sec.gov/Archives/edgar/data/sap"]
    assert "primary" in verification.source_quality_summary


def test_verify_claim_prompt_includes_linked_observations_and_evidence_items() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    observation_id = uuid4()
    claim = Claim(
        run_id=run.id,
        task_id=uuid4(),
        statement="SAP cloud backlog is rising.",
        evidence_observation_ids=[observation_id],
        confidence=0.8,
    )
    rt = runtime(
        run,
        QueueLLM(
            {
                "verdict": "verified",
                "rationale": "The filing supports backlog growth.",
                "confidence": 0.83,
                "supported_parts": ["SAP cloud backlog is rising."],
                "unsupported_parts": [],
                "contradictions": [],
                "required_caveats": [],
                "source_quality_summary": "Primary filing support; Reuters risk context.",
            }
        ),
    )
    rt.blackboard.claims.append(claim)
    rt.blackboard.observations.append(
        Observation(
            id=observation_id,
            run_id=run.id,
            task_id=claim.task_id,
            tool="web_search",
            summary="Linked observation says SAP cloud backlog expanded year over year.",
            artifact=ArtifactPointer(
                uri="gs://bucket/observation.json",
                size_bytes=2,
                sha256="0" * 64,
            ),
            sources=["https://www.sap.com/investors"],
        )
    )

    class EvidenceTools(FakeTools):
        async def web_search(self, _run_id, query):
            if "risk counterargument" in query:
                return {
                    "results": [
                        {
                            "url": "https://www.reuters.com/markets/sap-risk",
                            "title": "SAP risk context",
                            "content": "Some customers delayed cloud migrations.",
                            "published_date": "2026-07-02",
                            "score": 0.72,
                        }
                    ]
                }
            return {
                "results": [
                    {
                        "url": "https://www.sec.gov/Archives/edgar/data/sap",
                        "title": "SAP annual filing",
                        "content": "SAP reported cloud backlog rose 27%.",
                        "published_date": "2026-07-01",
                        "score": 0.95,
                    }
                ]
            }

    rt.tools = EvidenceTools()

    asyncio.run(
        verify_claim(
            rt,
            EventEnvelope(
                type=EventType.CLAIM_CREATED,
                run_id=run.id,
                producer="test",
                payload={"claim_id": str(claim.id)},
            ),
        )
    )

    prompt = rt.llm.calls[0]["prompt"]
    assert '"claim_text": "SAP cloud backlog is rising."' in prompt
    assert "Linked observation says SAP cloud backlog expanded" in prompt
    assert "SAP reported cloud backlog rose 27%" in prompt
    assert "https://www.sec.gov/Archives/edgar/data/sap" in prompt
    assert "source_quality_reason" in prompt
    assert "contradictory_or_contextual_evidence" in prompt
    assert "https://www.reuters.com/markets/sap-risk" in prompt


def test_verify_claim_weak_source_only_becomes_uncertain_and_disputed() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    claim = Claim(
        run_id=run.id,
        task_id=uuid4(),
        statement="SAP backlog is rising.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    claim_artifact = Artifact(
        run_id=run.id,
        artifact_type=ArtifactType.CLAIM,
        branch="market/equities",
        text_or_summary=claim.statement,
        legacy_object_type="claim",
        legacy_object_id=claim.id,
    )
    rt = runtime(
        run,
        QueueLLM({"verdict": "verified", "rationale": "A forum says so.", "confidence": 0.8}),
    )
    rt.blackboard.claims.append(claim)
    rt.blackboard.artifacts.append(claim_artifact)

    class WeakTools(FakeTools):
        async def web_search(self, _run_id, _query):
            return {
                "results": [
                    {
                        "url": "https://www.quora.com/sap-backlog",
                        "title": "SAP backlog rumor",
                        "content": "A user claims SAP backlog is rising.",
                        "score": 0.8,
                    }
                ]
            }

    rt.tools = WeakTools()

    asyncio.run(
        verify_claim(
            rt,
            EventEnvelope(
                type=EventType.CLAIM_CREATED,
                run_id=run.id,
                producer="test",
                payload={"claim_id": str(claim.id)},
            ),
        )
    )

    verification = rt.blackboard.verifications[-1]
    assert verification.verdict == "uncertain"
    assert "weak, unknown, or missing" in verification.required_caveats[-1]
    promoted = next(item for item in rt.blackboard.artifacts if item.id == claim_artifact.id)
    assert promoted.status == ArtifactStatus.DISPUTED
    assert promoted.visibility == VisibilityScope.PUBLIC_UNVERIFIED


def test_verify_claim_contradiction_downgrades_verified_result_to_uncertain() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    claim = Claim(
        run_id=run.id,
        task_id=uuid4(),
        statement="SAP backlog is rising.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    claim_artifact = Artifact(
        run_id=run.id,
        artifact_type=ArtifactType.CLAIM,
        branch="market/equities",
        text_or_summary=claim.statement,
        legacy_object_type="claim",
        legacy_object_id=claim.id,
    )
    rt = runtime(
        run,
        QueueLLM(
            {
                "verdict": "verified",
                "rationale": "Support exists but conflicts remain.",
                "confidence": 0.78,
                "supported_parts": ["The filing suggests backlog growth."],
                "unsupported_parts": [],
                "contradictions": ["A high-quality source reports backlog contraction."],
                "required_caveats": [],
                "source_quality_summary": "Primary and news sources conflict.",
            }
        ),
    )
    rt.blackboard.claims.append(claim)
    rt.blackboard.artifacts.append(claim_artifact)

    asyncio.run(
        verify_claim(
            rt,
            EventEnvelope(
                type=EventType.CLAIM_CREATED,
                run_id=run.id,
                producer="test",
                payload={"claim_id": str(claim.id)},
            ),
        )
    )

    verification = rt.blackboard.verifications[-1]
    assert verification.verdict == "uncertain"
    assert verification.confidence == 0.55
    assert verification.contradictions == ["A high-quality source reports backlog contraction."]
    promoted = next(item for item in rt.blackboard.artifacts if item.id == claim_artifact.id)
    assert promoted.status == ArtifactStatus.DISPUTED


def test_verify_claim_ignores_non_material_contradiction_text() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    claim = Claim(
        run_id=run.id,
        task_id=uuid4(),
        statement="SAP backlog is rising.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    claim_artifact = Artifact(
        run_id=run.id,
        artifact_type=ArtifactType.CLAIM,
        branch="market/equities",
        text_or_summary=claim.statement,
        legacy_object_type="claim",
        legacy_object_id=claim.id,
    )
    rt = runtime(
        run,
        QueueLLM(
            {
                "verdict": "verified",
                "rationale": "Primary filing support remains intact.",
                "confidence": 0.78,
                "supported_parts": ["SAP backlog is rising."],
                "unsupported_parts": [],
                "contradictions": ["None material", "No direct contradiction"],
                "required_caveats": [],
                "source_quality_summary": "Primary filing support.",
            }
        ),
    )
    rt.blackboard.claims.append(claim)
    rt.blackboard.artifacts.append(claim_artifact)

    asyncio.run(
        verify_claim(
            rt,
            EventEnvelope(
                type=EventType.CLAIM_CREATED,
                run_id=run.id,
                producer="test",
                payload={"claim_id": str(claim.id)},
            ),
        )
    )

    verification = rt.blackboard.verifications[-1]
    assert verification.verdict == "verified"
    assert verification.contradictions == []
    assert not any(
        "Verifier identified contradictions" in caveat
        for caveat in verification.required_caveats
    )
    promoted = next(item for item in rt.blackboard.artifacts if item.id == claim_artifact.id)
    assert promoted.status == ArtifactStatus.VERIFIED


def test_verify_claim_partial_support_adds_required_caveat() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    claim = Claim(
        run_id=run.id,
        task_id=uuid4(),
        statement="SAP backlog is rising and valuation is attractive.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    rt = runtime(
        run,
        QueueLLM(
            {
                "verdict": "verified",
                "rationale": "Backlog growth is supported; valuation is not covered.",
                "confidence": 0.74,
                "supported_parts": ["SAP backlog is rising."],
                "unsupported_parts": ["The evidence does not establish attractive valuation."],
                "contradictions": [],
                "required_caveats": [],
                "source_quality_summary": "Primary filing supports backlog only.",
            }
        ),
    )
    rt.blackboard.claims.append(claim)

    asyncio.run(
        verify_claim(
            rt,
            EventEnvelope(
                type=EventType.CLAIM_CREATED,
                run_id=run.id,
                producer="test",
                payload={"claim_id": str(claim.id)},
            ),
        )
    )

    verification = rt.blackboard.verifications[-1]
    assert verification.supported_parts == ["SAP backlog is rising."]
    assert verification.unsupported_parts == [
        "The evidence does not establish attractive valuation."
    ]
    assert verification.required_caveats
    artifact = rt.blackboard.artifacts[-1]
    assert "Required caveats:" in artifact.text_or_summary


def test_verify_claim_stores_supported_parts_without_recursive_claims() -> None:
    run = Run(
        question="What happens if the Fed cuts faster?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget(tavily_max_credits=10)),
    )
    observation_id = uuid4()
    claim = Claim(
        run_id=run.id,
        task_id=uuid4(),
        statement="Fed cuts lower yields and guarantee a bullish cross-asset outcome.",
        evidence_observation_ids=[observation_id],
        sources=["https://www.federalreserve.gov/monetarypolicy"],
        confidence=0.8,
        asset="cross_asset",
    )
    rt = runtime(
        run,
        QueueLLM(
            {
                "verdict": "uncertain",
                "rationale": "Yield direction is supported but the guarantee is not.",
                "confidence": 0.62,
                "supported_parts": [
                    "Lower expected Fed policy rates tend to reduce front-end Treasury yields, all else equal."
                ],
                "unsupported_parts": ["The evidence does not support a guaranteed bullish cross-asset outcome."],
                "contradictions": [],
                "required_caveats": [],
                "source_quality_summary": "Primary and institutional sources support the rate mechanism.",
            },
        ),
    )
    rt.blackboard.claims.append(claim)
    rt.blackboard.observations.append(
        Observation(
            id=observation_id,
            run_id=run.id,
            task_id=claim.task_id,
            tool="web_search",
            summary="Fed cut evidence.",
            artifact=ArtifactPointer(uri="gs://bucket/obs.json", size_bytes=2, sha256="0" * 64),
            sources=claim.sources,
        )
    )

    asyncio.run(
        verify_claim(
            rt,
            EventEnvelope(
                type=EventType.CLAIM_CREATED,
                run_id=run.id,
                producer="test",
                payload={"claim_id": str(claim.id)},
            ),
        )
    )

    assert rt.blackboard.claims == [claim]
    assert any(
        "verifier_supported_part" in artifact.tags
        and "context_only" in artifact.tags
        and artifact.status == ArtifactStatus.DISPUTED
        and artifact.text_or_summary.startswith("Lower expected Fed policy rates")
        for artifact in rt.blackboard.artifacts
    )
    assert not any(
        event.type == EventType.CLAIM_CREATED
        and event.payload.get("derived_from_verification_id")
        for _topic, event in rt.published
    )


def test_verify_claim_requests_skeptic_when_second_verified_claim_exists() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    first_claim = Claim(
        run_id=run.id,
        task_id=uuid4(),
        statement="SAP backlog is rising.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    second_claim = Claim(
        run_id=run.id,
        task_id=uuid4(),
        statement="SAP cloud margins are improving.",
        evidence_observation_ids=[uuid4()],
        confidence=0.75,
    )
    rt = runtime(
        run, QueueLLM({"verdict": "verified", "rationale": "Supported.", "confidence": 0.7})
    )
    rt.blackboard.claims.extend([first_claim, second_claim])
    rt.blackboard.verifications.append(
        Verification(
            run_id=run.id,
            claim_id=first_claim.id,
            verdict="verified",
            rationale="Supported.",
            confidence=0.7,
        )
    )
    for claim in [first_claim, second_claim]:
        rt.blackboard.artifacts.append(
            Artifact(
                run_id=run.id,
                artifact_type=ArtifactType.CLAIM,
                branch="market/equities",
                text_or_summary=claim.statement,
                status=ArtifactStatus.VERIFIED if claim == first_claim else ArtifactStatus.UNVERIFIED,
                legacy_object_type="claim",
                legacy_object_id=claim.id,
            )
        )

    asyncio.run(
        verify_claim(
            rt,
            EventEnvelope(
                type=EventType.CLAIM_CREATED,
                run_id=run.id,
                producer="test",
                payload={"claim_id": str(second_claim.id)},
            ),
        )
    )

    assert any(
        action.action_type == PrincipalActionType.REQUEST_SKEPTIC_REVIEW
        and action.status == ActionStatus.EXECUTED
        for action in rt.blackboard.actions
    )
    published_types = [event.type for _topic, event in rt.published]
    assert EventType.SKEPTIC_REVIEW_REQUESTED in published_types
    assert EventType.CLAIM_VERIFIED not in published_types


def test_skeptic_review_creates_counterargument_artifact_and_triggers_aggregation() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    claims = [
        Claim(
            run_id=run.id,
            task_id=uuid4(),
            statement="SAP backlog is rising.",
            evidence_observation_ids=[uuid4()],
            sources=[f"https://example.com/{index}"],
            confidence=0.8,
        )
        for index in range(2)
    ]
    rt = runtime(
        run,
        QueueLLM(
            {
                "strongest_counterargument": "Valuation already prices the backlog.",
                "contradicting_evidence": "A margin miss would weaken the thesis.",
                "risks_regime_changes": "Rates could rise again.",
                "what_would_change_conclusion": "Lower guidance.",
                "confidence_calibration_comments": "Moderate confidence.",
            }
        ),
    )
    rt.blackboard.claims.extend(claims)
    for claim in claims:
        rt.blackboard.verifications.append(
            Verification(
                run_id=run.id,
                claim_id=claim.id,
                verdict="verified",
                rationale="Supported.",
                confidence=0.7,
            )
        )

    asyncio.run(
        skeptic_review(
            rt,
            EventEnvelope(
                type=EventType.SKEPTIC_REVIEW_REQUESTED,
                run_id=run.id,
                producer="test",
            ),
        )
    )

    artifact = rt.blackboard.artifacts[-1]
    assert artifact.artifact_type == ArtifactType.COUNTERARGUMENT
    assert artifact.status == ArtifactStatus.VERIFIED
    assert "Strongest counterargument: Valuation already prices the backlog." in artifact.text_or_summary
    assert "What would change conclusion: Lower guidance." in artifact.text_or_summary
    assert any(
        action.action_type == PrincipalActionType.REQUEST_SKEPTIC_REVIEW
        and action.status == ActionStatus.EXECUTED
        for action in rt.blackboard.actions
    )
    assert any(event.type == EventType.CLAIM_VERIFIED for _topic, event in rt.published)


def test_aggregator_stringifies_nested_answer_object_and_dual_writes_final_artifact() -> None:
    run = Run(
        question="Should I buy SAP?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id, title="SAP", question="SAP stock", tool="web_search", status="completed"
    )
    claim = Claim(
        run_id=run.id,
        task_id=task.id,
        statement="SAP has positive momentum.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    verification = Verification(
        run_id=run.id, claim_id=claim.id, verdict="verified", rationale="Supported.", confidence=0.7
    )
    rt = runtime(
        run, QueueLLM({"answer": {"summary": "Buy with caution.", "risks": ["Volatility"]}})
    )
    rt.blackboard.tasks.append(task)
    rt.blackboard.claims.append(claim)
    rt.blackboard.verifications.append(verification)

    asyncio.run(
        aggregate(rt, EventEnvelope(type=EventType.CLAIM_VERIFIED, run_id=run.id, producer="test"))
    )

    assert '"summary": "Buy with caution."' in rt.blackboard.final.answer
    assert any(
        action.action_type == PrincipalActionType.REQUEST_AGGREGATION
        for action in rt.blackboard.actions
    )
    assert rt.blackboard.artifacts[-1].artifact_type == ArtifactType.FINAL_REPORT


def test_aggregator_prompt_includes_synthesis_role_and_verified_evidence_requirement() -> None:
    run = Run(
        question="Should I buy gold or equities after a Fed cut?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Gold and equities",
        question="Compare verified gold and equities evidence.",
        tool="web_search",
        status="completed",
    )
    claim = Claim(
        run_id=run.id,
        task_id=task.id,
        statement="Verified gold evidence is stronger than equities evidence after the cut.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    verification = Verification(
        run_id=run.id, claim_id=claim.id, verdict="verified", rationale="Supported.", confidence=0.7
    )
    llm = QueueLLM({"answer": "Gold has the cleaner verified setup."})
    rt = runtime(run, llm)
    rt.blackboard.tasks.append(task)
    rt.blackboard.claims.append(claim)
    rt.blackboard.verifications.append(verification)
    rt.blackboard.agent_specs.append(
        AgentSpec(
            run_id=run.id,
            name="aggregator",
            role_template="aggregator_agent",
            branch="synthesis/aggregator",
            domain="synthesis",
            objective="Synthesize trusted claims into a decision-grade final report.",
            allowed_tools=[],
            retrieval_tags=["synthesis", "final"],
            local_budget_usd=0.05,
            visibility_scope=VisibilityScope.PUBLIC_VERIFIED,
        )
    )

    asyncio.run(
        aggregate(rt, EventEnvelope(type=EventType.CLAIM_VERIFIED, run_id=run.id, producer="test"))
    )

    prompt = llm.calls[-1]["prompt"]
    assert llm.calls[-1]["name"] == "aggregator"
    assert "branch: synthesis/aggregator" in prompt
    assert "domain: synthesis" in prompt
    assert "allowed_tools: none" in prompt
    assert "visibility_scope: public_verified" in prompt
    assert "Synthesis role: synthesize across verified claims only" in prompt
    assert "Evidence rule: Use verified claims and public_verified artifacts as final support" in prompt
    assert "trade-offs" in prompt
    assert 'Return {"answer":"..."}' in prompt
    assert rt.blackboard.final.answer.startswith("Gold has the cleaner verified setup.")
    assert "U.S. equities" in rt.blackboard.final.answer
    assert "long-duration bonds" in rt.blackboard.final.answer
    assert "Evidence tier used" in rt.blackboard.final.answer


def test_aggregator_requests_zero_verified_followup_before_deterministic_final() -> None:
    run = Run(
        question="What happens if the Fed cuts faster than expected?",
        models=model_policy(),
        budget=Budget(
            limit_usd=5,
            tools=ToolBudget(tavily_max_credits=20, tavily_credits_used=9),
        ),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Rates and cross-asset impact",
        question="Find evidence for faster Fed cuts.",
        tool="web_search",
        status="completed",
    )
    failed_task = ResearchTask(
        run_id=run.id,
        title="Equity impact",
        question="Find evidence for equity impact.",
        tool="web_search",
        status="failed",
        reason="Task failed during claim_generation: Equity impact (LLMOutputError)",
    )
    claim = Claim(
        run_id=run.id,
        task_id=task.id,
        statement="A faster Fed cutting path guarantees a bullish cross-asset outcome.",
        evidence_observation_ids=[uuid4()],
        sources=["https://example.com/fed"],
        confidence=0.8,
    )
    verification = Verification(
        run_id=run.id,
        claim_id=claim.id,
        verdict="uncertain",
        rationale="The evidence supports directionality but not the absolute guarantee.",
        confidence=0.52,
        supported_parts=["Fed rate cuts can support some rate-sensitive assets."],
        unsupported_parts=["The guaranteed bullish outcome is not supported."],
        required_caveats=["The result depends on whether cuts reflect disinflation or recession."],
        sources=["https://example.com/fed"],
    )
    llm = QueueLLM(
        {
            "diagnostic": "Need another branch-distributed repair wave.",
            "targeted_gaps": ["Split claims by asset class."],
            "repair_branches": ["macro/rates", "market/equities"],
        }
    )
    rt = runtime(run, llm)
    rt.blackboard.tasks.append(task)
    rt.blackboard.tasks.append(failed_task)
    rt.blackboard.claims.append(claim)
    rt.blackboard.verifications.append(verification)

    asyncio.run(
        aggregate(rt, EventEnvelope(type=EventType.CLAIM_VERIFIED, run_id=run.id, producer="test"))
    )

    assert [call["name"] for call in llm.calls] == ["evidence-gap-planner"]
    assert rt.blackboard.final is None
    followups = [task for task in rt.blackboard.tasks if task.wave_number == 1]
    assert len(followups) == 2
    followup_branches = {task.branch for task in followups}
    assert followup_branches == {"macro/rates", "market/equities"}
    assert all(task.title.startswith("Evidence repair:") for task in followups)
    assert all("split broad claims into atomic claims" in task.question for task in followups)
    assert all("search primary/date-bearing sources" in task.question for task in followups)
    assert all("run contradiction search" in task.question for task in followups)
    assert any("evidence_gap_planner" in artifact.tags for artifact in rt.blackboard.artifacts)
    assert any(
        action.action_type == PrincipalActionType.REQUEST_FOLLOWUP
        and action.status == ActionStatus.EXECUTED
        and "Zero claims passed verification" in action.reason
        for action in rt.blackboard.actions
    )
    assert not any(
        action.action_type == PrincipalActionType.STOP_RUN for action in rt.blackboard.actions
    )
    published_types = [event.type for _topic, event in rt.published]
    assert EventType.FOLLOWUP_REQUESTED in published_types
    assert published_types.count(EventType.TASK_CREATED) >= len(followups)


def test_zero_verified_repair_tasks_are_distributed_across_failed_branches() -> None:
    run = Run(
        question=(
            "If the Fed signals faster rate cuts, compare macro rates, U.S. equities, "
            "the U.S. dollar, gold, and long-duration bonds."
        ),
        models=model_policy(),
        budget=Budget(limit_usd=5, tools=ToolBudget(tavily_max_credits=20, tavily_credits_used=4)),
    )
    branches = ["macro/rates", "market/equities", "market/fx", "market/gold"]
    tasks = [
        ResearchTask(
            run_id=run.id,
            title=f"{branch} evidence",
            question=f"Find evidence for {branch}",
            tool="web_search",
            branch=branch,
            status="completed",
        )
        for branch in branches
    ]
    claims = [
        Claim(
            run_id=run.id,
            task_id=task.id,
            statement=f"Broad unsupported claim for {task.branch}.",
            evidence_observation_ids=[uuid4()],
            confidence=0.7,
        )
        for task in tasks
    ]
    verifications = [
        Verification(
            run_id=run.id,
            claim_id=claim.id,
            verdict="uncertain",
            rationale="Too broad.",
            confidence=0.4,
            unsupported_parts=[f"Unsupported part for {task.branch}"],
            required_caveats=[f"Caveat for {task.branch}"],
            source_quality_summary="No dated primary source.",
        )
        for task, claim in zip(tasks, claims)
    ]
    artifacts = [
        Artifact(
            run_id=run.id,
            artifact_type=ArtifactType.CLAIM,
            branch=task.branch,
            text_or_summary=claim.statement,
            status=ArtifactStatus.DISPUTED,
            legacy_object_type="claim",
            legacy_object_id=claim.id,
        )
        for task, claim in zip(tasks, claims)
    ]
    rt = runtime(run, QueueLLM({"answer": "This should not be called."}))
    rt.blackboard.tasks.extend(tasks)
    rt.blackboard.claims.extend(claims)
    rt.blackboard.verifications.extend(verifications)
    rt.blackboard.artifacts.extend(artifacts)

    asyncio.run(
        aggregate(rt, EventEnvelope(type=EventType.CLAIM_VERIFIED, run_id=run.id, producer="test"))
    )

    repair_tasks = [task for task in rt.blackboard.tasks if task.wave_number == 1]
    repair_branches = {task.branch for task in repair_tasks}
    assert len(repair_tasks) == 5
    assert len(repair_branches) >= 5
    assert "market/gold" in repair_branches
    assert "market/bonds" in repair_branches
    assert repair_branches != {"market/gold"}
    assert all(task.branch for task in repair_tasks)
    assert rt.blackboard.final is None


def test_latest_run_like_aggregate_blocks_final_and_scales_repair_to_budget() -> None:
    run = Run(
        question=(
            "If the Fed signals faster rate cuts, compare macro rates, U.S. equities, "
            "the U.S. dollar, gold, and long-duration bonds."
        ),
        models=model_policy(),
        budget=Budget(
            limit_usd=5,
            spent_usd=0.06,
            tools=ToolBudget(
                tavily_max_credits=100,
                tavily_credits_used=12,
                market_data_max_requests=50,
                market_data_requests_used=1,
            ),
        ),
    )
    branches = ["macro/rates", "market/equities", "market/fx", "market/gold", "market/bonds"]
    tasks = [
        ResearchTask(
            run_id=run.id,
            title=f"{branch} first repair",
            question=f"Find primary evidence for {branch}.",
            tool="web_search",
            branch=branch,
            status="completed",
            wave_number=1,
        )
        for branch in branches
    ]
    claims = [
        Claim(
            run_id=run.id,
            task_id=task.id,
            statement=f"Broad unsupported claim for {task.branch}.",
            evidence_observation_ids=[uuid4()],
            confidence=0.7,
        )
        for task in tasks
    ]
    verifications = [
        Verification(
            run_id=run.id,
            claim_id=claim.id,
            verdict="uncertain",
            rationale="Too broad for source verification.",
            confidence=0.4,
            unsupported_parts=[f"Unsupported part for {task.branch}"],
            source_quality_summary="No dated primary source.",
        )
        for task, claim in zip(tasks, claims)
    ]
    rt = runtime(
        run,
        QueueLLM(
            {
                "diagnostic": "Need another branch-distributed repair wave.",
                "targeted_gaps": ["Split claims by asset class."],
                "repair_branches": branches,
            }
        ),
    )
    rt.blackboard.tasks.extend(tasks)
    rt.blackboard.claims.extend(claims)
    rt.blackboard.verifications.extend(verifications)

    asyncio.run(
        aggregate(rt, EventEnvelope(type=EventType.CLAIM_VERIFIED, run_id=run.id, producer="test"))
    )

    repair_tasks = [task for task in rt.blackboard.tasks if task.wave_number == 2]
    repair_branches = {task.branch for task in repair_tasks}
    assert rt.blackboard.final is None
    assert not any(action.action_type == PrincipalActionType.STOP_RUN for action in rt.blackboard.actions)
    assert [call["name"] for call in rt.llm.calls] == ["evidence-gap-planner"]
    assert len(repair_tasks) >= 15
    assert len(repair_branches) == 5
    assert all(branch in repair_branches for branch in branches)
    assert max(
        sum(1 for task in repair_tasks if task.branch == branch) for branch in repair_branches
    ) < len(repair_tasks)
    assert any(
        action.action_type == PrincipalActionType.REQUEST_FOLLOWUP
        and action.status == ActionStatus.EXECUTED
        for action in rt.blackboard.actions
    )


def test_aggregator_writes_deterministic_final_after_zero_verified_repair_waves_exhausted() -> None:
    run = Run(
        question="What happens if the Fed cuts faster than expected?",
        models=model_policy(),
        budget=Budget(
            limit_usd=5,
            tools=ToolBudget(tavily_max_credits=20, tavily_credits_used=9),
        ),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Follow-up: primary dated sources",
        question="Find dated primary evidence for faster Fed cuts.",
        tool="web_search",
        status="completed",
        wave_number=2,
    )
    claim = Claim(
        run_id=run.id,
        task_id=task.id,
        statement="A faster Fed cutting path guarantees a bullish cross-asset outcome.",
        evidence_observation_ids=[uuid4()],
        sources=["https://example.com/claim-source"],
        confidence=0.8,
    )
    verification = Verification(
        run_id=run.id,
        claim_id=claim.id,
        verdict="uncertain",
        rationale="The evidence supports directionality but not the absolute guarantee.",
        confidence=0.52,
        supported_parts=["Fed rate cuts can support some rate-sensitive assets."],
        unsupported_parts=["The guaranteed bullish outcome is not supported."],
        required_caveats=["The result depends on whether cuts reflect disinflation or recession."],
        sources=["https://example.com/verifier-source"],
        evidence_item_refs=["https://example.com/evidence-item"],
    )
    claim_artifact = Artifact(
        run_id=run.id,
        artifact_type=ArtifactType.CLAIM,
        branch="macro/rates",
        text_or_summary=claim.statement,
        status=ArtifactStatus.DISPUTED,
        source_refs=claim.sources,
        legacy_object_type="claim",
        legacy_object_id=claim.id,
    )
    verification_artifact = Artifact(
        run_id=run.id,
        artifact_type=ArtifactType.VERIFICATION,
        branch="trust/source_verifier",
        text_or_summary=verification.rationale,
        status=ArtifactStatus.DISPUTED,
        source_refs=verification.sources,
        legacy_object_type="verification",
        legacy_object_id=verification.id,
    )
    llm = QueueLLM(
        {"diagnostic": "Search repair exhausted.", "targeted_gaps": []},
        {
            "answer": (
                "Caveated decision memo: evidence is disputed, but the main supported "
                "fragment is that Fed cuts can support some rate-sensitive assets."
            )
        },
    )
    rt = runtime(run, llm)
    rt.blackboard.tasks.append(task)
    rt.blackboard.claims.append(claim)
    rt.blackboard.verifications.append(verification)
    rt.blackboard.artifacts.extend([claim_artifact, verification_artifact])

    asyncio.run(
        aggregate(rt, EventEnvelope(type=EventType.CLAIM_VERIFIED, run_id=run.id, producer="test"))
    )

    assert [call["name"] for call in llm.calls] == [
        "evidence-gap-planner",
        "caveated-partial-aggregator",
    ]
    assert rt.blackboard.final is not None
    assert rt.blackboard.final.partial is True
    assert rt.blackboard.final.verified_claim_ids == []
    assert rt.blackboard.final.sources == [
        "https://example.com/verifier-source",
        "https://example.com/evidence-item",
        "https://example.com/claim-source",
    ]
    assert "Caveated decision memo" in rt.blackboard.final.answer
    assert "Fed cuts can support some rate-sensitive assets" in rt.blackboard.final.answer
    assert "U.S. equities" in rt.blackboard.final.answer
    assert "Evidence tier used" in rt.blackboard.final.answer
    assert "Conversion quality" in rt.blackboard.final.answer
    assert rt.blackboard.run.status == RunStatus.RUNNING
    assert rt.blackboard.run.final_answer == rt.blackboard.final.answer
    assert not any(action.action_type == PrincipalActionType.STOP_RUN for action in rt.blackboard.actions)
    assert any(event.type == EventType.FINAL_CREATED for _topic, event in rt.published)
    final_artifact = next(
        artifact
        for artifact in rt.blackboard.artifacts
        if artifact.artifact_type == ArtifactType.FINAL_REPORT
    )
    assert final_artifact.source_refs == rt.blackboard.final.sources
    assert set(final_artifact.depends_on_artifact_ids) == {
        claim_artifact.id,
        verification_artifact.id,
    }


def test_evidence_limited_final_runs_judge_before_terminal_stop() -> None:
    run = Run(
        question="What happens if the Fed cuts faster than expected?",
        models=model_policy(),
        budget=Budget(limit_usd=5, tools=ToolBudget(tavily_max_credits=2, tavily_credits_used=2)),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Exhausted search repair",
        question="Find dated primary evidence for faster Fed cuts.",
        tool="web_search",
        status="completed",
    )
    claim = Claim(
        run_id=run.id,
        task_id=task.id,
        statement="A faster Fed cutting path guarantees a bullish cross-asset outcome.",
        evidence_observation_ids=[uuid4()],
        sources=["https://example.com/claim-source"],
        confidence=0.8,
    )
    verification = Verification(
        run_id=run.id,
        claim_id=claim.id,
        verdict="uncertain",
        rationale="The evidence supports directionality but not the absolute guarantee.",
        confidence=0.52,
        unsupported_parts=["The guaranteed bullish outcome is not supported."],
        sources=["https://example.com/verifier-source"],
    )
    rt = runtime(
        run,
        QueueLLM(
            {"diagnostic": "Search exhausted; judge the partial.", "targeted_gaps": []},
            {"answer": "Caveated diagnostic synthesis: evidence remains partial."},
            {"score": 0.82, "feedback": "Evidence-limited status is clear."},
        ),
    )
    rt.blackboard.tasks.append(task)
    rt.blackboard.claims.append(claim)
    rt.blackboard.verifications.append(verification)

    asyncio.run(
        aggregate(rt, EventEnvelope(type=EventType.CLAIM_VERIFIED, run_id=run.id, producer="test"))
    )
    final_event = next(event for _topic, event in rt.published if event.type == EventType.FINAL_CREATED)
    asyncio.run(judge(rt, final_event))

    assert rt.blackboard.final is not None
    assert rt.blackboard.final.partial is True
    assert rt.blackboard.final.judge_score == 0.82
    assert [call["name"] for call in rt.llm.calls] == [
        "evidence-gap-planner",
        "caveated-partial-aggregator",
        "judge",
    ]
    assert "Conversion quality" in rt.blackboard.final.answer
    assert "2 Tavily credits produced 0 verified claim(s)" in rt.blackboard.final.answer
    assert "Judge payoff" in rt.blackboard.final.answer
    assert "Score: 0.82" in rt.blackboard.final.answer
    assert rt.blackboard.run.status == RunStatus.COMPLETED
    assert any(action.action_type == PrincipalActionType.STOP_RUN for action in rt.blackboard.actions)


def test_terminal_partial_final_is_idempotent_and_ignores_late_claim_events() -> None:
    run = Run(
        question="What happens if the Fed cuts faster than expected?",
        models=model_policy(judge_enabled=False),
        budget=Budget(limit_usd=5, tools=ToolBudget(tavily_max_credits=2, tavily_credits_used=2)),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Exhausted search repair",
        question="Find dated primary evidence for faster Fed cuts.",
        tool="web_search",
        status="completed",
    )
    claim = Claim(
        run_id=run.id,
        task_id=task.id,
        statement="A faster Fed cutting path guarantees a bullish cross-asset outcome.",
        evidence_observation_ids=[uuid4()],
        sources=["https://example.com/claim-source"],
        confidence=0.8,
    )
    verification = Verification(
        run_id=run.id,
        claim_id=claim.id,
        verdict="uncertain",
        rationale="The evidence supports directionality but not the absolute guarantee.",
        confidence=0.52,
        supported_parts=["Fed cuts can lower expected short rates."],
        unsupported_parts=["The guaranteed bullish outcome is not supported."],
        sources=["https://example.com/verifier-source"],
    )
    observation = Observation(
        run_id=run.id,
        task_id=task.id,
        tool="web_search",
        summary="Lower expected short rates can move yields.",
        artifact=ArtifactPointer(uri="gs://bucket/raw.json", size_bytes=2, sha256="0" * 64),
        sources=["https://example.com/source"],
    )
    rt = runtime(
        run,
        QueueLLM(
            {"diagnostic": "Search exhausted.", "targeted_gaps": []},
            {"answer": "Caveated diagnostic synthesis."},
        ),
    )
    rt.blackboard.tasks.append(task)
    rt.blackboard.claims.append(claim)
    rt.blackboard.verifications.append(verification)
    rt.blackboard.observations.append(observation)

    event = EventEnvelope(type=EventType.CLAIM_VERIFIED, run_id=run.id, producer="test")
    asyncio.run(aggregate(rt, event))
    asyncio.run(aggregate(rt, event))
    asyncio.run(
        create_claim(
            rt,
            EventEnvelope(
                type=EventType.OBSERVATION_CREATED,
                run_id=run.id,
                producer="late-test",
                payload={"observation_id": str(observation.id)},
            ),
        )
    )

    assert rt.blackboard.run.status == RunStatus.COMPLETED
    assert sum(
        artifact.artifact_type == ArtifactType.FINAL_REPORT
        for artifact in rt.blackboard.artifacts
    ) == 1
    assert sum(
        action.action_type == PrincipalActionType.STOP_RUN
        for action in rt.blackboard.actions
    ) == 1
    assert rt.blackboard.claims == [claim]


def test_forced_evidence_limited_final_links_unverified_claim_sources_when_capacity_exhausted() -> None:
    run = Run(
        question="What happens if the Fed cuts faster than expected?",
        models=model_policy(),
        budget=Budget(
            limit_usd=5,
            tools=ToolBudget(tavily_max_credits=2, tavily_credits_used=2),
        ),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Gold reaction",
        question="Find evidence for gold.",
        tool="web_search",
        status="completed",
    )
    claim = Claim(
        run_id=run.id,
        task_id=task.id,
        statement="Gold moved on rate-cut expectations.",
        evidence_observation_ids=[uuid4()],
        sources=["https://example.com/gold-source"],
        confidence=0.6,
    )
    claim_artifact = Artifact(
        run_id=run.id,
        artifact_type=ArtifactType.CLAIM,
        branch="market/gold",
        text_or_summary=claim.statement,
        status=ArtifactStatus.UNVERIFIED,
        source_refs=claim.sources,
        legacy_object_type="claim",
        legacy_object_id=claim.id,
    )
    rt = runtime(run, QueueLLM({"answer": "This should not be called."}))
    rt.blackboard.tasks.append(task)
    rt.blackboard.claims.append(claim)
    rt.blackboard.artifacts.append(claim_artifact)

    asyncio.run(
        aggregate(
            rt,
            EventEnvelope(
                type=EventType.CLAIM_VERIFIED,
                run_id=run.id,
                producer="test",
                payload={"force": True},
            ),
        )
    )

    assert rt.blackboard.final is not None
    assert rt.blackboard.final.partial is True
    assert rt.blackboard.final.sources == ["https://example.com/gold-source"]
    assert "Unassessed claims due to capacity exhaustion" in rt.blackboard.final.answer
    assert "Gold moved on rate-cut expectations" in rt.blackboard.final.answer
    final_artifact = next(
        artifact
        for artifact in rt.blackboard.artifacts
        if artifact.artifact_type == ArtifactType.FINAL_REPORT
    )
    assert final_artifact.source_refs == ["https://example.com/gold-source"]
    assert final_artifact.depends_on_artifact_ids == [claim_artifact.id]


def test_forced_aggregation_does_not_finalize_with_unverified_claims_when_capacity_remains() -> None:
    run = Run(
        question="What happens if the Fed cuts faster than expected?",
        models=model_policy(),
        budget=Budget(
            limit_usd=5,
            tools=ToolBudget(tavily_max_credits=20, tavily_credits_used=1),
        ),
    )
    task = ResearchTask(
        run_id=run.id,
        title="FX evidence",
        question="Find evidence for USD.",
        tool="web_search",
        status="completed",
        branch="market/fx",
    )
    claim = Claim(
        run_id=run.id,
        task_id=task.id,
        statement="USD weakens when Fed cuts faster.",
        evidence_observation_ids=[uuid4()],
        sources=["https://example.com/usd"],
        confidence=0.6,
    )
    rt = runtime(run, QueueLLM({"answer": "This should not be called."}))
    rt.blackboard.tasks.append(task)
    rt.blackboard.claims.append(claim)

    asyncio.run(
        aggregate(
            rt,
            EventEnvelope(
                type=EventType.CLAIM_VERIFIED,
                run_id=run.id,
                producer="test",
                payload={"force": True},
            ),
        )
    )

    assert rt.blackboard.final is None
    assert any(
        action.action_type == PrincipalActionType.REQUEST_VERIFICATION
        and action.status == ActionStatus.EXECUTED
        for action in rt.blackboard.actions
    )
    assert any(event.type == EventType.CLAIM_CREATED for _topic, event in rt.published)


def test_aggregation_prompt_consumes_skeptic_counterargument_artifact() -> None:
    run = Run(
        question="Should I buy SAP?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id, title="SAP", question="SAP stock", tool="web_search", status="completed"
    )
    claim = Claim(
        run_id=run.id,
        task_id=task.id,
        statement="SAP has positive momentum.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    verification = Verification(
        run_id=run.id, claim_id=claim.id, verdict="verified", rationale="Supported.", confidence=0.7
    )
    counterargument = Artifact(
        run_id=run.id,
        artifact_type=ArtifactType.COUNTERARGUMENT,
        branch="trust/skeptic",
        text_or_summary="Strongest counterargument: the thesis is already priced.",
        status=ArtifactStatus.VERIFIED,
        visibility=VisibilityScope.PUBLIC_VERIFIED,
    )
    llm = QueueLLM({"answer": "Buy only if valuation remains reasonable."})
    rt = runtime(run, llm)
    rt.blackboard.tasks.append(task)
    rt.blackboard.claims.append(claim)
    rt.blackboard.verifications.append(verification)
    rt.blackboard.artifacts.append(counterargument)

    asyncio.run(
        aggregate(rt, EventEnvelope(type=EventType.CLAIM_VERIFIED, run_id=run.id, producer="test"))
    )

    prompt = llm.calls[-1]["prompt"]
    assert "Skeptic/counterargument context" in prompt
    assert "the thesis is already priced" in prompt
    assert rt.blackboard.final.answer == "Buy only if valuation remains reasonable."


def test_skeptic_failure_records_artifact_and_allows_aggregation() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    tasks = [
        ResearchTask(
            run_id=run.id,
            title=f"SAP evidence {index}",
            question="SAP stock",
            tool="web_search",
            status="completed",
        )
        for index in range(2)
    ]
    claims = [
        Claim(
            run_id=run.id,
            task_id=tasks[index].id,
            statement=f"Verified SAP claim {index}.",
            evidence_observation_ids=[uuid4()],
            confidence=0.8,
        )
        for index in range(2)
    ]
    rt = runtime(run, QueueLLM(RuntimeError("skeptic model failed"), {"answer": "Proceed."}))
    rt.blackboard.tasks.extend(tasks)
    rt.blackboard.claims.extend(claims)
    for claim in claims:
        rt.blackboard.verifications.append(
            Verification(
                run_id=run.id,
                claim_id=claim.id,
                verdict="verified",
                rationale="Supported.",
                confidence=0.7,
            )
        )

    asyncio.run(
        skeptic_review(
            rt,
            EventEnvelope(
                type=EventType.SKEPTIC_REVIEW_REQUESTED,
                run_id=run.id,
                producer="test",
            ),
        )
    )
    aggregation_event = next(
        event for _topic, event in rt.published if event.type == EventType.CLAIM_VERIFIED
    )
    asyncio.run(aggregate(rt, aggregation_event))

    counterargument = next(
        artifact
        for artifact in rt.blackboard.artifacts
        if artifact.artifact_type == ArtifactType.COUNTERARGUMENT
    )
    assert counterargument.status == ArtifactStatus.DISPUTED
    assert "Skeptic review failed" in counterargument.text_or_summary
    assert any(
        action.action_type == PrincipalActionType.REQUEST_SKEPTIC_REVIEW
        and action.status == ActionStatus.FAILED
        for action in rt.blackboard.actions
    )
    assert rt.blackboard.final.answer == "Proceed."


def test_judge_normalizes_score_dual_writes_feedback_artifact_and_stop_action() -> None:
    run = Run(
        question="Should I buy SAP?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    final = FinalReport(
        run_id=run.id, answer="Buy with caution.", verified_claim_ids=[], sources=[]
    )
    rt = runtime(run, QueueLLM({"score": 9.5, "feedback": {"summary": "Strong answer."}}))
    rt.blackboard.final = final

    asyncio.run(
        judge(rt, EventEnvelope(type=EventType.FINAL_CREATED, run_id=run.id, producer="test"))
    )

    assert rt.blackboard.final.judge_score == 0.95
    assert '"summary": "Strong answer."' in rt.blackboard.final.judge_feedback
    assert rt.blackboard.run.status == RunStatus.COMPLETED
    assert rt.blackboard.artifacts[-1].artifact_type == ArtifactType.JUDGE_FEEDBACK
    assert any(
        action.action_type == PrincipalActionType.STOP_RUN for action in rt.blackboard.actions
    )
    assert not any(
        action.action_type == PrincipalActionType.REQUEST_FOLLOWUP
        for action in rt.blackboard.actions
    )


def test_low_judge_score_creates_auditable_followup_tasks_and_continues() -> None:
    run = Run(
        question="Should I buy SAP?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    final = FinalReport(
        run_id=run.id, answer="Buy with caution.", verified_claim_ids=[], sources=[]
    )
    rt = runtime(
        run,
        QueueLLM(
            {
                "score": 0.62,
                "feedback": {
                    "gaps": [
                        "Needs stronger valuation evidence.",
                        "Address downside risks.",
                    ]
                },
            }
        ),
    )
    rt.blackboard.final = final

    asyncio.run(
        judge(rt, EventEnvelope(type=EventType.FINAL_CREATED, run_id=run.id, producer="test"))
    )

    followups = [task for task in rt.blackboard.tasks if task.wave_number == 1]
    assert rt.blackboard.final.judge_score == 0.62
    assert rt.blackboard.run.status == RunStatus.RUNNING
    assert 1 <= len(followups) <= 3
    assert all(task.reason and "Judge score 0.62 below 0.75" in task.reason for task in followups)
    assert rt.blackboard.artifacts[-1].artifact_type == ArtifactType.JUDGE_FEEDBACK
    assert any(
        action.action_type == PrincipalActionType.REQUEST_FOLLOWUP
        and action.status == ActionStatus.EXECUTED
        for action in rt.blackboard.actions
    )
    assign_actions = [
        action
        for action in rt.blackboard.actions
        if action.action_type == PrincipalActionType.ASSIGN_TASK
        and action.reason.startswith("Follow-up wave 1:")
    ]
    assert len(assign_actions) == len(followups)
    assert not any(
        action.action_type == PrincipalActionType.STOP_RUN for action in rt.blackboard.actions
    )
    published_types = [event.type for _topic, event in rt.published]
    assert EventType.FOLLOWUP_REQUESTED in published_types
    assert published_types.count(EventType.TASK_CREATED) == len(followups)


def test_followup_wave_completion_refreshes_final_and_requests_judge_again() -> None:
    run = Run(
        question="Should I buy SAP?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="Follow-up: valuation",
        question="Find valuation evidence.",
        tool="web_search",
        status="completed",
        wave_number=1,
        reason="Judge score 0.62 below 0.75: Needs valuation evidence.",
    )
    claim = Claim(
        run_id=run.id,
        task_id=task.id,
        statement="SAP valuation has improved versus peers.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    verification = Verification(
        run_id=run.id,
        claim_id=claim.id,
        verdict="verified",
        rationale="Supported.",
        confidence=0.7,
    )
    old_final = FinalReport(
        run_id=run.id,
        answer="Buy with caution.",
        verified_claim_ids=[],
        sources=[],
        wave_number=0,
        judge_score=0.62,
        judge_feedback="Needs stronger valuation evidence.",
    )
    rt = runtime(run, QueueLLM({"answer": "Updated answer with valuation evidence."}))
    rt.blackboard.tasks.append(task)
    rt.blackboard.claims.append(claim)
    rt.blackboard.verifications.append(verification)
    rt.blackboard.final = old_final

    asyncio.run(
        aggregate(rt, EventEnvelope(type=EventType.CLAIM_VERIFIED, run_id=run.id, producer="test"))
    )

    assert rt.blackboard.final.answer == "Updated answer with valuation evidence."
    assert rt.blackboard.final.wave_number == 1
    assert rt.blackboard.final.judge_score is None
    assert any(event.type == EventType.FINAL_CREATED for _topic, event in rt.published)


def test_second_low_judge_score_stops_without_another_followup_wave() -> None:
    run = Run(
        question="Should I buy SAP?",
        status=RunStatus.RUNNING,
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    followup_task = ResearchTask(
        run_id=run.id,
        title="Follow-up: valuation",
        question="Find valuation evidence.",
        tool="web_search",
        status="completed",
        wave_number=1,
        reason="Judge score 0.62 below 0.75: Needs valuation evidence.",
    )
    final = FinalReport(
        run_id=run.id,
        answer="Still weak.",
        verified_claim_ids=[],
        sources=[],
        wave_number=1,
    )
    rt = runtime(run, QueueLLM({"score": 0.51, "feedback": "Still too thin."}))
    rt.blackboard.tasks.append(followup_task)
    rt.blackboard.actions.append(
        PrincipalAction(
            run_id=run.id,
            action_type=PrincipalActionType.REQUEST_FOLLOWUP,
            reason="Judge score 0.62 below 0.75; requesting targeted follow-up wave 1.",
        )
    )
    rt.blackboard.final = final

    asyncio.run(
        judge(rt, EventEnvelope(type=EventType.FINAL_CREATED, run_id=run.id, producer="test"))
    )

    assert rt.blackboard.run.status == RunStatus.COMPLETED
    assert len(rt.blackboard.tasks) == 1
    assert max(task.wave_number for task in rt.blackboard.tasks) == 1
    assert sum(
        action.action_type == PrincipalActionType.REQUEST_FOLLOWUP
        for action in rt.blackboard.actions
    ) == 1
    assert any(
        action.action_type == PrincipalActionType.STOP_RUN for action in rt.blackboard.actions
    )


def test_deterministic_partial_records_executed_aggregation_action() -> None:
    run = Run(
        question="Should I buy SAP?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    rt = runtime(run, QueueLLM())

    asyncio.run(deterministic_partial(rt, run.id, "budget exhausted"))

    assert rt.blackboard.final is not None
    assert rt.blackboard.run.status == RunStatus.PARTIAL_BUDGET_EXHAUSTED
    assert any(
        action.action_type == PrincipalActionType.REQUEST_AGGREGATION
        for action in rt.blackboard.actions
    )
    assert any(action.action_type == PrincipalActionType.STOP_RUN for action in rt.blackboard.actions)
    assert rt.blackboard.run.failure_reason == "budget exhausted"


def test_execute_tool_budget_block_marks_task_skipped_and_logs(caplog) -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget(tavily_max_credits=1)),
    )
    task = ResearchTask(
        run_id=run.id,
        title="SAP stock catalysts",
        question="SAP stock catalysts",
        tool="web_search",
    )
    rt = runtime(run, QueueLLM())
    rt.blackboard.tasks.append(task)

    class BudgetBlockedTools(FakeTools):
        async def web_search(self, _run_id, _query):
            raise BudgetExceeded(
                "Tavily credit budget reached: requested=1, used=1, max=1",
                budget_type="tool",
                provider="tavily",
            )

    rt.tools = BudgetBlockedTools()

    with caplog.at_level("WARNING"), pytest.raises(BudgetExceeded):
        asyncio.run(
            execute_tool(
                rt,
                EventEnvelope(
                    type=EventType.TASK_CREATED,
                    run_id=run.id,
                    producer="test",
                    payload={"task_id": str(task.id)},
                ),
            )
        )

    assert rt.blackboard.tasks[0].status == "skipped_budget"
    assert "Budget blocked tool_execution" in (rt.blackboard.tasks[0].reason or "")
    assert any(
        action.status == ActionStatus.REJECTED
        and action.action_type == PrincipalActionType.REQUEST_TOOL_CALL
        for action in rt.blackboard.actions
    )
    assert "budget skipped task" in caplog.text


def test_execute_tool_failure_does_not_record_executed_tool_action() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="SAP stock catalysts",
        question="SAP stock catalysts",
        tool="web_search",
    )
    rt = runtime(run, QueueLLM({"summary": "unused"}))
    rt.blackboard.tasks.append(task)

    class FailingTools(FakeTools):
        async def web_search(self, _run_id, _query):
            raise RuntimeError("tool failed")

    rt.tools = FailingTools()

    asyncio.run(
        execute_tool(
            rt,
            EventEnvelope(
                type=EventType.TASK_CREATED,
                run_id=run.id,
                producer="test",
                payload={"task_id": str(task.id)},
            ),
        )
    )

    assert rt.blackboard.tasks[0].status == "failed"
    assert rt.blackboard.observations == []
    assert not any(
        artifact.artifact_type == ArtifactType.OBSERVATION for artifact in rt.blackboard.artifacts
    )
    assert not any(
        action.action_type == PrincipalActionType.REQUEST_TOOL_CALL
        and action.status == ActionStatus.EXECUTED
        for action in rt.blackboard.actions
    )
    assert any(
        action.action_type == PrincipalActionType.REQUEST_TOOL_CALL
        and action.status == ActionStatus.FAILED
        and "Task failed during tool_execution" in action.reason
        for action in rt.blackboard.actions
    )


def test_evidence_engine_failure_falls_back_to_legacy_web_search(monkeypatch) -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="SAP stock catalysts",
        question="SAP stock catalysts",
        tool="web_search",
    )
    rt = runtime(run, QueueLLM({"summary": "Legacy summary kept."}))
    rt.blackboard.tasks.append(task)

    async def failing_evidence_search(_runtime, _request):
        raise ValueError("bundle conversion failed")

    monkeypatch.setattr(
        "src.agents.workflow.search_evidence",
        failing_evidence_search,
    )

    asyncio.run(
        execute_tool(
            rt,
            EventEnvelope(
                type=EventType.TASK_CREATED,
                run_id=run.id,
                producer="test",
                payload={"task_id": str(task.id)},
            ),
        )
    )

    observation = rt.blackboard.observations[0]
    assert "EvidenceEngine failed" in observation.summary
    assert "Legacy summary kept." in observation.summary
    assert "legacy_web_search_fallback" in rt.blackboard.artifacts[-1].tags
    assert any(
        action.action_type == PrincipalActionType.REQUEST_TOOL_CALL
        and action.status == ActionStatus.EXECUTED
        and "EvidenceEngine failure" in action.reason
        for action in rt.blackboard.actions
    )


def test_evidence_engine_failure_without_observation_records_no_success(monkeypatch) -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="SAP stock catalysts",
        question="SAP stock catalysts",
        tool="web_search",
    )
    rt = runtime(run, QueueLLM({"summary": "unused"}))
    rt.blackboard.tasks.append(task)

    async def failing_evidence_search(_runtime, _request):
        raise ValueError("bundle conversion failed")

    class FailingFallbackTools(FakeTools):
        async def web_search(self, _run_id, _query):
            raise RuntimeError("fallback failed")

    monkeypatch.setattr(
        "src.agents.workflow.search_evidence",
        failing_evidence_search,
    )
    rt.tools = FailingFallbackTools()

    asyncio.run(
        execute_tool(
            rt,
            EventEnvelope(
                type=EventType.TASK_CREATED,
                run_id=run.id,
                producer="test",
                payload={"task_id": str(task.id)},
            ),
        )
    )

    assert rt.blackboard.tasks[0].status == "failed"
    assert rt.blackboard.observations == []
    assert not any(
        action.action_type == PrincipalActionType.REQUEST_TOOL_CALL
        and action.status == ActionStatus.EXECUTED
        for action in rt.blackboard.actions
    )
    failed_action = next(
        action for action in rt.blackboard.actions if action.status == ActionStatus.FAILED
    )
    assert "EvidenceEngine failed" in failed_action.reason
    assert "fallback failed" in failed_action.reason


def test_dispatch_retries_transient_tool_failure_with_metadata() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="SAP stock catalysts",
        question="SAP stock catalysts",
        tool="web_search",
    )
    rt = runtime(run, QueueLLM({"summary": "unused"}))
    rt.blackboard.tasks.append(task)

    class TimeoutTools(FakeTools):
        async def web_search(self, _run_id, _query):
            raise TimeoutError("temporary tool timeout")

    rt.tools = TimeoutTools()

    asyncio.run(
        dispatch(
            rt,
            "tool-runner",
            EventEnvelope(
                type=EventType.TASK_CREATED,
                run_id=run.id,
                producer="test",
                payload={"task_id": str(task.id)},
            ),
        )
    )

    retry_events = [event for _topic, event in rt.published if event.type == EventType.TASK_CREATED]
    assert len(retry_events) == 1
    assert retry_events[0].payload["_retry_count"] == 1
    assert retry_events[0].payload["_max_retries"] == MAX_HANDLER_RETRIES
    assert retry_events[0].payload["_suggested_backoff_seconds"] == 1
    assert retry_events[0].payload["_original_event_id"]
    assert rt.blackboard.dead_letters == []
    assert rt.blackboard.tasks[0].status == "created"
    assert rt.blackboard.observations == []


def test_dispatch_dead_letters_permanent_malformed_event_without_retry() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    rt = runtime(run, QueueLLM())

    asyncio.run(
        dispatch(
            rt,
            "tool-runner",
            EventEnvelope(type=EventType.TASK_CREATED, run_id=run.id, producer="test"),
        )
    )

    retry_events = [event for _topic, event in rt.published if event.type == EventType.TASK_CREATED]
    assert retry_events == []
    assert len(rt.blackboard.dead_letters) == 1
    assert rt.blackboard.dead_letters[0].classification == "permanent"
    assert rt.blackboard.dead_letters[0].retry_count == 0
    assert rt.blackboard.run.status == RunStatus.FAILED
    assert "Dead-lettered task.created" in rt.blackboard.run.failure_reason


def test_dispatch_dead_letters_transient_after_max_retries_and_marks_task_failed() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="SAP stock catalysts",
        question="SAP stock catalysts",
        tool="web_search",
    )
    rt = runtime(run, QueueLLM({"summary": "unused"}))
    rt.blackboard.tasks.append(task)

    class TimeoutTools(FakeTools):
        async def web_search(self, _run_id, _query):
            raise TimeoutError("temporary tool timeout")

    rt.tools = TimeoutTools()

    asyncio.run(
        dispatch(
            rt,
            "tool-runner",
            EventEnvelope(
                type=EventType.TASK_CREATED,
                run_id=run.id,
                producer="test",
                payload={"task_id": str(task.id), "_retry_count": MAX_HANDLER_RETRIES},
            ),
        )
    )

    retry_events = [event for _topic, event in rt.published if event.type == EventType.TASK_CREATED]
    assert retry_events == []
    assert len(rt.blackboard.dead_letters) == 1
    assert rt.blackboard.dead_letters[0].classification == "transient"
    assert rt.blackboard.dead_letters[0].retry_count == MAX_HANDLER_RETRIES
    assert rt.blackboard.tasks[0].status == "failed"
    assert rt.blackboard.run.status == RunStatus.FAILED
    assert rt.blackboard.observations == []
    assert not any(
        action.action_type == PrincipalActionType.REQUEST_TOOL_CALL
        and action.status == ActionStatus.EXECUTED
        for action in rt.blackboard.actions
    )


def test_one_failed_task_does_not_prevent_another_task_from_completing() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    failing_task = ResearchTask(
        run_id=run.id,
        title="SAP failed query",
        question="fail SAP query",
        tool="web_search",
    )
    successful_task = ResearchTask(
        run_id=run.id,
        title="SAP stock catalysts",
        question="SAP stock catalysts",
        tool="web_search",
    )
    rt = runtime(run, QueueLLM({"summary": "SAP has cloud catalysts."}))
    rt.blackboard.tasks.extend([failing_task, successful_task])

    class PartlyFailingTools(FakeTools):
        async def web_search(self, _run_id, query):
            if "fail" in query:
                raise RuntimeError("tool failed")
            return await super().web_search(_run_id, query)

    rt.tools = PartlyFailingTools()

    for task in [failing_task, successful_task]:
        asyncio.run(
            execute_tool(
                rt,
                EventEnvelope(
                    type=EventType.TASK_CREATED,
                    run_id=run.id,
                    producer="test",
                    payload={"task_id": str(task.id)},
                ),
            )
        )

    statuses = {task.id: task.status for task in rt.blackboard.tasks}
    assert statuses[failing_task.id] == "failed"
    assert statuses[successful_task.id] == "completed"
    assert len(rt.blackboard.observations) == 1
    assert rt.blackboard.observations[0].task_id == successful_task.id
    assert rt.blackboard.run.status == RunStatus.CREATED


def test_claim_generation_failure_marks_only_that_task_failed() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id,
        title="SAP stock catalysts",
        question="SAP stock catalysts",
        tool="web_search",
        status="completed",
    )
    observation_id = uuid4()
    observation = SimpleNamespace(
        id=observation_id,
        run_id=run.id,
        task_id=task.id,
        summary="SAP cloud backlog is rising.",
        sources=["https://example.com/sap"],
    )
    rt = runtime(run, QueueLLM(RuntimeError("claim model failed")))
    rt.blackboard.tasks.append(task)
    rt.blackboard.observations.append(observation)

    asyncio.run(
        create_claim(
            rt,
            EventEnvelope(
                type=EventType.OBSERVATION_CREATED,
                run_id=run.id,
                producer="test",
                payload={"observation_id": str(observation_id)},
            ),
        )
    )

    assert rt.blackboard.tasks[0].status == "failed"
    assert rt.blackboard.claims == []
    assert any(
        action.status == ActionStatus.FAILED and "claim_generation" in action.reason
        for action in rt.blackboard.actions
    )


def test_verify_claim_search_failure_marks_claim_uncertain_without_failing_run() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    claim = Claim(
        run_id=run.id,
        task_id=uuid4(),
        statement="SAP backlog is rising.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    claim_artifact = Artifact(
        run_id=run.id,
        artifact_type=ArtifactType.CLAIM,
        branch="market/equities",
        text_or_summary=claim.statement,
        legacy_object_type="claim",
        legacy_object_id=claim.id,
    )
    rt = runtime(run, QueueLLM())
    rt.blackboard.claims.append(claim)
    rt.blackboard.artifacts.append(claim_artifact)

    class FailingTools(FakeTools):
        async def web_search(self, _run_id, _query):
            raise RuntimeError("verification search failed")

    rt.tools = FailingTools()

    asyncio.run(
        verify_claim(
            rt,
            EventEnvelope(
                type=EventType.CLAIM_CREATED,
                run_id=run.id,
                producer="test",
                payload={"claim_id": str(claim.id)},
            ),
        )
    )

    assert rt.blackboard.verifications[-1].verdict == "uncertain"
    assert any(
        action.action_type == PrincipalActionType.REQUEST_VERIFICATION
        for action in rt.blackboard.actions
    )
    promoted = next(item for item in rt.blackboard.artifacts if item.id == claim_artifact.id)
    assert promoted.status == ArtifactStatus.DISPUTED
    assert rt.blackboard.artifacts[-1].artifact_type == ArtifactType.VERIFICATION
    assert rt.blackboard.artifacts[-1].status == ArtifactStatus.DISPUTED


def test_aggregate_failure_does_not_record_executed_aggregation_action() -> None:
    run = Run(
        question="Should I buy SAP?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    task = ResearchTask(
        run_id=run.id, title="SAP", question="SAP stock", tool="web_search", status="completed"
    )
    claim = Claim(
        run_id=run.id,
        task_id=task.id,
        statement="SAP has positive momentum.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    verification = Verification(
        run_id=run.id, claim_id=claim.id, verdict="verified", rationale="Supported.", confidence=0.7
    )
    rt = runtime(run, QueueLLM())
    rt.blackboard.tasks.append(task)
    rt.blackboard.claims.append(claim)
    rt.blackboard.verifications.append(verification)

    try:
        asyncio.run(
            aggregate(
                rt, EventEnvelope(type=EventType.CLAIM_VERIFIED, run_id=run.id, producer="test")
            )
        )
    except IndexError:
        pass

    assert not any(
        action.action_type == PrincipalActionType.REQUEST_AGGREGATION
        for action in rt.blackboard.actions
    )


def test_aggregation_proceeds_with_verified_partial_evidence_after_task_failure() -> None:
    run = Run(
        question="Should I buy SAP?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    completed_task = ResearchTask(
        run_id=run.id, title="SAP", question="SAP stock", tool="web_search", status="completed"
    )
    failed_task = ResearchTask(
        run_id=run.id,
        title="SAP macro risk",
        question="SAP macro risk",
        tool="web_search",
        status="failed",
    )
    claim = Claim(
        run_id=run.id,
        task_id=completed_task.id,
        statement="SAP has positive momentum.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    verification = Verification(
        run_id=run.id, claim_id=claim.id, verdict="verified", rationale="Supported.", confidence=0.7
    )
    rt = runtime(run, QueueLLM({"answer": "Partial evidence supports SAP."}))
    rt.blackboard.tasks.extend([completed_task, failed_task])
    rt.blackboard.claims.append(claim)
    rt.blackboard.verifications.append(verification)

    asyncio.run(
        aggregate(rt, EventEnvelope(type=EventType.CLAIM_VERIFIED, run_id=run.id, producer="test"))
    )

    assert rt.blackboard.final is not None
    assert rt.blackboard.final.verified_claim_ids == [claim.id]
    assert "Partial evidence supports SAP." in rt.blackboard.final.answer


def test_all_tasks_failing_produces_clear_terminal_status() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    tasks = [
        ResearchTask(
            run_id=run.id,
            title="SAP catalysts",
            question="SAP catalysts",
            tool="web_search",
        ),
        ResearchTask(
            run_id=run.id,
            title="SAP valuation",
            question="SAP valuation",
            tool="web_search",
        ),
    ]
    rt = runtime(run, QueueLLM())
    rt.blackboard.tasks.extend(tasks)

    class FailingTools(FakeTools):
        async def web_search(self, _run_id, _query):
            raise RuntimeError("tool failed")

    rt.tools = FailingTools()

    for task in tasks:
        asyncio.run(
            execute_tool(
                rt,
                EventEnvelope(
                    type=EventType.TASK_CREATED,
                    run_id=run.id,
                    producer="test",
                    payload={"task_id": str(task.id)},
                ),
            )
        )

    assert all(task.status == "failed" for task in rt.blackboard.tasks)
    assert rt.blackboard.run.status == RunStatus.FAILED
    assert rt.blackboard.run.failure_reason
    assert "All 2 research task(s) failed" in rt.blackboard.run.failure_reason
    assert any(
        action.action_type == PrincipalActionType.STOP_RUN for action in rt.blackboard.actions
    )


def test_verify_claim_promotes_original_claim_artifact_status_and_visibility() -> None:
    run = Run(
        question="Should I buy SAP stock?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    claim = Claim(
        run_id=run.id,
        task_id=uuid4(),
        statement="SAP backlog is rising.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    claim_artifact = Artifact(
        run_id=run.id,
        artifact_type=ArtifactType.CLAIM,
        branch="market/equities",
        text_or_summary=claim.statement,
        legacy_object_type="claim",
        legacy_object_id=claim.id,
    )
    rt = runtime(
        run, QueueLLM({"verdict": "verified", "rationale": "Supported.", "confidence": 0.7})
    )
    rt.blackboard.claims.append(claim)
    rt.blackboard.artifacts.append(claim_artifact)

    asyncio.run(
        verify_claim(
            rt,
            EventEnvelope(
                type=EventType.CLAIM_CREATED,
                run_id=run.id,
                producer="test",
                payload={"claim_id": str(claim.id)},
            ),
        )
    )

    promoted = next(item for item in rt.blackboard.artifacts if item.id == claim_artifact.id)
    assert promoted.status == ArtifactStatus.VERIFIED
    assert promoted.visibility == "public_verified"


def test_research_budget_is_divided_across_worker_branches() -> None:
    run = Run(
        question="Will gold and oil rise if Fed cuts rates?",
        models=model_policy(),
        budget=Budget(limit_usd=1, tools=ToolBudget()),
    )
    rt = runtime(run, QueueLLM({"tasks": []}))

    asyncio.run(plan(rt, EventEnvelope(type=EventType.RUN_CREATED, run_id=run.id, producer="test")))

    workers = [
        agent
        for agent in rt.blackboard.agent_specs
        if agent.id in rt.blackboard.organization_plan.worker_agents
    ]
    assert len(workers) >= 3
    assert sum(agent.local_budget_usd for agent in workers) == role_policy(0.3).cap_usd
    assert all(agent.local_budget_usd < role_policy(0.3).cap_usd for agent in workers)
