import asyncio
from types import SimpleNamespace
from uuid import uuid4

from src.agents.workflow import (
    HANDLERS,
    aggregate,
    create_claim,
    deterministic_partial,
    execute_tool,
    judge,
    plan,
    semantic_branches,
    verify_claim,
)
from src.integrations.llm import LLMOutputError
from src.common.models import (
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

    async def json(self, *_args, **_kwargs):
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def runtime(run: Run, llm: QueueLLM | None = None):
    published = []
    blackboard = FakeBlackboard(run)
    return SimpleNamespace(
        blackboard=blackboard,
        llm=llm or QueueLLM(),
        tools=FakeTools(),
        artifacts=FakeArtifacts(),
        settings=SimpleNamespace(runtime_topic="agent-runtime"),
        events=SimpleNamespace(publish=lambda topic, event: published.append((topic, event))),
        published=published,
    )


def test_every_workflow_stage_has_an_event_handler() -> None:
    assert HANDLERS["planner-agent"] == {
        EventType.RUN_CREATED: HANDLERS["planner-agent"][EventType.RUN_CREATED]
    }
    assert EventType.TASK_CREATED in HANDLERS["tool-runner"]
    assert EventType.OBSERVATION_CREATED in HANDLERS["worker-agents"]
    assert EventType.CLAIM_CREATED in HANDLERS["verifier-agent"]
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

    try:
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
    except RuntimeError:
        pass

    assert not any(
        action.action_type == PrincipalActionType.REQUEST_TOOL_CALL
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
