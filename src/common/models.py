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


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL_BUDGET_EXHAUSTED = "partial_budget_exhausted"
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


class RunDetail(BaseModel):
    run: Run
    tasks: list[ResearchTask] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    verifications: list[Verification] = Field(default_factory=list)
    final: FinalReport | None = None


class EventEnvelope(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    type: EventType
    run_id: UUID
    producer: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
