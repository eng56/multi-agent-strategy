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
    def put_json(self, run_id, kind, _raw):
        return ArtifactPointer(
            uri=f"gs://bucket/runs/{run_id}/{kind}/raw.json",
            size_bytes=2,
            sha256="0" * 64,
        )


class FakeTools:
    async def web_search(self, _run_id, query):
        return {"results": [{"url": f"https://example.com/{query[:8]}"}]}

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


def test_shadow_policy_persists_proposed_action_without_side_effect_event() -> None:
    run, _task, rt = pending_policy_runtime()

    selected = asyncio.run(evaluate_principal_policy(rt, run.id, trigger="test"))

    assert selected is not None
    assert len(rt.blackboard.actions) == 1
    action = rt.blackboard.actions[0]
    assert action.status == ActionStatus.PROPOSED
    assert action.producer == SHADOW_POLICY_PRODUCER
    assert action.reason.startswith("shadow policy proposed:")
    assert action.action_type == PrincipalActionType.REQUEST_TOOL_CALL
    assert action.idempotency_key
    assert rt.published == []


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


def test_plan_hook_writes_shadow_policy_action_without_action_created_event() -> None:
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

    shadow_actions = [
        action
        for action in rt.blackboard.actions
        if action.producer == SHADOW_POLICY_PRODUCER
    ]
    assert len(shadow_actions) == 1
    assert shadow_actions[0].status == ActionStatus.PROPOSED
    assert shadow_actions[0].action_type == PrincipalActionType.REQUEST_TOOL_CALL
    published_action_ids = {
        event.payload.get("action_id")
        for _topic, event in rt.published
        if event.type == EventType.PRINCIPAL_ACTION_CREATED
    }
    assert str(shadow_actions[0].id) not in published_action_ids


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


def test_tool_summary_prompt_includes_gold_agent_spec_context() -> None:
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

    prompt = llm.calls[-1]["prompt"]
    assert llm.calls[-1]["name"] == "tool-summary"
    assert "branch: market/gold" in prompt
    assert "domain: market" in prompt
    assert "objective: Research gold evidence relevant to Fed cuts." in prompt
    assert "allowed_tools: web_search, market_data" in prompt
    assert "retrieval_tags: market:gold, macro:rates" in prompt
    assert "visibility_scope: team" in prompt
    assert "local_budget_usd: $0.0700" in prompt
    assert "You own gold evidence for branch market/gold" in prompt
    assert "do not overreach into equities" in prompt
    assert 'Return {"summary":"..."}' in prompt
    assert rt.blackboard.observations[0].summary == "Gold context kept."


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
    assert rt.blackboard.final.answer == "Gold has the cleaner verified setup."


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
