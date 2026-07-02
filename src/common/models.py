from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class AgentRole(StrEnum):
    PLANNER = "planner"
    UTILITY = "utility"
    RESEARCH = "research"
    VERIFIER = "verifier"
    AGGREGATOR = "aggregator"
    JUDGE = "judge"


class EventType(StrEnum):
    RUN_CREATED = "run.created"
    TASK_CREATED = "task.created"
    CLAIM_CREATED = "claim.created"
    OBSERVATION_CREATED = "observation.created"
    CLAIM_VERIFIED = "claim.verified"
    FINAL_CREATED = "final.created"
    PRINCIPAL_ACTION_CREATED = "principal.action.created"
    AGENT_SPEC_CREATED = "agent.spec.created"
    ORGANIZATION_PLAN_CREATED = "organization.plan.created"
    ARTIFACT_CREATED = "artifact.created"
    FOLLOWUP_REQUESTED = "followup.requested"
    SKEPTIC_REVIEW_REQUESTED = "skeptic.review.requested"


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL_BUDGET_EXHAUSTED = "partial_budget_exhausted"
    FAILED = "failed"


class PrincipalActionType(StrEnum):
    SPAWN_AGENT = "spawn_agent"
    ASSIGN_TASK = "assign_task"
    REQUEST_TOOL_CALL = "request_tool_call"
    REQUEST_VERIFICATION = "request_verification"
    REQUEST_SKEPTIC_REVIEW = "request_skeptic_review"
    REQUEST_AGGREGATION = "request_aggregation"
    REQUEST_FOLLOWUP = "request_followup"
    STOP_RUN = "stop_run"


class InformationGain(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"


class AgentStatus(StrEnum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class VisibilityScope(StrEnum):
    PRIVATE = "private"
    TEAM = "team"
    PUBLIC_UNVERIFIED = "public_unverified"
    PUBLIC_VERIFIED = "public_verified"


class ArtifactType(StrEnum):
    CLAIM = "claim"
    OBSERVATION = "observation"
    FORECAST = "forecast"
    COUNTERARGUMENT = "counterargument"
    OPEN_QUESTION = "open_question"
    BACKTEST_RESULT = "backtest_result"
    VERIFICATION = "verification"
    JUDGE_FEEDBACK = "judge_feedback"
    PRINCIPAL_ACTION = "principal_action"
    FINAL_REPORT = "final_report"


class ArtifactStatus(StrEnum):
    DRAFT = "draft"
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    REJECTED = "rejected"
    DISPUTED = "disputed"


class RunPhase(StrEnum):
    INTAKE = "intake"
    PLANNING = "planning"
    RESEARCHING = "researching"
    VERIFYING = "verifying"
    SYNTHESIZING = "synthesizing"
    JUDGING = "judging"
    COMPLETED = "completed"
    FAILED = "failed"


class RoleModelPolicy(BaseModel):
    model: str = Field(min_length=1)
    cap_usd: float | None = Field(default=None, gt=0)
    max_output_tokens: int = Field(default=2000, ge=1, le=100_000)
    max_call_cost_usd: float = Field(default=0.25, gt=0)
    protected_usd: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_role_budget(self) -> "RoleModelPolicy":
        if self.cap_usd is not None and self.max_call_cost_usd > self.cap_usd:
            raise ValueError("maximum call cost cannot exceed role cap")
        if self.cap_usd is not None and self.protected_usd > self.cap_usd:
            raise ValueError("protected budget cannot exceed role cap")
        return self


class ModelPolicy(BaseModel):
    planner: RoleModelPolicy
    utility: RoleModelPolicy
    research: RoleModelPolicy
    verifier: RoleModelPolicy
    aggregator: RoleModelPolicy
    judge: RoleModelPolicy | None = None


class ToolBudget(BaseModel):
    tavily_max_credits: int = Field(default=2, ge=1)
    market_data_max_requests: int = Field(default=1, ge=1)
    tavily_credits_used: int = Field(default=0, ge=0)
    market_data_requests_used: int = Field(default=0, ge=0)


class Budget(BaseModel):
    limit_usd: float = Field(gt=0)
    spent_usd: float = 0
    reserved_usd: float = 0
    role_spent_usd: dict[AgentRole, float] = Field(default_factory=dict)
    role_reserved_usd: dict[AgentRole, float] = Field(default_factory=dict)
    tools: ToolBudget
    max_parallel_agents: int = Field(default=10, gt=0)
    active_agents: int = 0


class RunRequest(BaseModel):
    question: str = Field(min_length=3, max_length=10_000)
    llm_budget_usd: float = Field(gt=0)
    models: ModelPolicy
    tool_budget: ToolBudget = Field(default_factory=ToolBudget)

    @model_validator(mode="after")
    def validate_caps(self) -> "RunRequest":
        policies = [getattr(self.models, role.value) for role in AgentRole]
        caps = [policy.cap_usd for policy in policies if policy and policy.cap_usd is not None]
        if sum(caps) > self.llm_budget_usd:
            raise ValueError("sum of role caps cannot exceed shared LLM budget")
        if self.models.aggregator.protected_usd < self.models.aggregator.max_call_cost_usd:
            raise ValueError("aggregator protected budget must fund its maximum call cost")
        protected = self.models.aggregator.protected_usd
        if self.models.judge:
            if self.models.judge.protected_usd < self.models.judge.max_call_cost_usd:
                raise ValueError("judge protected budget must fund its maximum call cost")
            protected += self.models.judge.protected_usd
        if protected > self.llm_budget_usd:
            raise ValueError("protected aggregator and judge budget exceeds shared LLM budget")
        return self


class Run(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    question: str
    status: RunStatus = RunStatus.CREATED
    budget: Budget
    models: ModelPolicy
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    final_answer: str | None = None
    failure_reason: str | None = None


class ResearchTask(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    title: str
    question: str
    tool: Literal["web_search", "market_data"]
    status: Literal["created", "completed", "failed", "skipped_budget"] = "created"


class ArtifactPointer(BaseModel):
    uri: str
    media_type: str = "application/json"
    size_bytes: int
    sha256: str


class Observation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    task_id: UUID
    tool: str
    summary: str
    artifact: ArtifactPointer
    sources: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Claim(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    task_id: UUID
    statement: str
    evidence_observation_ids: list[UUID]
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class Verification(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    claim_id: UUID
    verdict: Literal["verified", "rejected", "uncertain"]
    rationale: str
    confidence: float = Field(ge=0, le=1)
    sources: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    run_id: UUID
    answer: str
    verified_claim_ids: list[UUID]
    sources: list[str]
    partial: bool = False
    judge_score: float | None = Field(default=None, ge=0, le=1)
    judge_feedback: str | None = None


class PrincipalAction(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    action_type: PrincipalActionType
    reason: str = Field(min_length=1)
    expected_information_gain: InformationGain = InformationGain.MEDIUM
    estimated_cost: float = Field(default=0, ge=0)
    target_branch: str | None = None
    required_role: str | None = None
    priority: int = Field(default=5, ge=1, le=10)
    status: ActionStatus = ActionStatus.PROPOSED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentSpec(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    parent_id: UUID | None = None
    name: str = Field(min_length=1)
    role_template: str = Field(min_length=1)
    branch: str = Field(min_length=1)
    domain: str | None = None
    objective: str = Field(min_length=1)
    allowed_tools: list[str] = Field(default_factory=list)
    retrieval_tags: list[str] = Field(default_factory=list)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    local_budget_usd: float = Field(default=0, ge=0)
    visibility_scope: VisibilityScope = VisibilityScope.TEAM
    can_request_spawn: bool = False
    status: AgentStatus = AgentStatus.PROPOSED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class OrganizationPlan(BaseModel):
    run_id: UUID
    root_agent_id: UUID
    agent_specs: list[AgentSpec] = Field(default_factory=list)
    branches: list[str] = Field(default_factory=list)
    manager_agents: list[UUID] = Field(default_factory=list)
    worker_agents: list[UUID] = Field(default_factory=list)
    verifier_agents: list[UUID] = Field(default_factory=list)
    aggregator_agents: list[UUID] = Field(default_factory=list)
    judge_agents: list[UUID] = Field(default_factory=list)
    dependencies: dict[UUID, list[UUID]] = Field(default_factory=dict)
    budget_allocation: dict[str, float] = Field(default_factory=dict)
    stop_conditions: list[str] = Field(default_factory=list)


class Artifact(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    artifact_type: ArtifactType
    created_by_agent_id: UUID | None = None
    branch: str | None = None
    text_or_summary: str
    tags: list[str] = Field(default_factory=list)
    visibility: VisibilityScope = VisibilityScope.PUBLIC_UNVERIFIED
    status: ArtifactStatus = ArtifactStatus.UNVERIFIED
    confidence: float | None = Field(default=None, ge=0, le=1)
    source_refs: list[str] = Field(default_factory=list)
    parent_artifact_ids: list[UUID] = Field(default_factory=list)
    depends_on_artifact_ids: list[UUID] = Field(default_factory=list)
    contradicts_artifact_ids: list[UUID] = Field(default_factory=list)
    supports_artifact_ids: list[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RunState(BaseModel):
    run_id: UUID
    iteration: int = 0
    question: str
    objective: str
    current_phase: RunPhase = RunPhase.INTAKE
    active_branches: list[str] = Field(default_factory=list)
    known_facts: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    verified_claim_count: int = 0
    rejected_claim_count: int = 0
    disputed_claim_count: int = 0
    coverage_by_topic: dict[str, str] = Field(default_factory=dict)
    budget_remaining: float = 0
    tool_budget_remaining: dict[str, int] = Field(default_factory=dict)
    agent_count: int = 0
    last_judge_score: float | None = None
    last_judge_feedback: str | None = None
    stop_reasons: list[str] = Field(default_factory=list)
    next_action_candidates: list[PrincipalAction] = Field(default_factory=list)


class RunDetail(BaseModel):
    run: Run
    tasks: list[ResearchTask] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    verifications: list[Verification] = Field(default_factory=list)
    final: FinalReport | None = None
    run_state: RunState | None = None
    principal_actions: list[PrincipalAction] = Field(default_factory=list)
    agent_specs: list[AgentSpec] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    organization_plan: OrganizationPlan | None = None


class EventEnvelope(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    type: EventType
    run_id: UUID
    producer: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
