import asyncio

import pytest

from src.common.budget import BudgetExceeded, PersistentBudget
from src.common.models import AgentRole, Budget, ModelPolicy, RoleModelPolicy, Run, ToolBudget


def policy() -> ModelPolicy:
    def role(cap: float, protected: float = 0) -> RoleModelPolicy:
        return RoleModelPolicy(model="provider/model", cap_usd=cap, protected_usd=protected, max_call_cost_usd=protected or 0.25)

    return ModelPolicy(planner=role(1), utility=role(1), research=role(4), verifier=role(2), aggregator=role(1, 1), judge=role(1, 1))


class Blackboard:
    def __init__(self, run: Run) -> None:
        self.run = run

    async def command(self, *parts: object) -> str:
        return "lock"

    async def get_run(self, _: object) -> Run:
        return self.run

    async def put_run(self, run: Run) -> None:
        self.run = run


def test_protected_finalization_budget_blocks_research() -> None:
    run = Run(question="q", models=policy(), budget=Budget(limit_usd=10, spent_usd=7, tools=ToolBudget()))
    budget = PersistentBudget(Blackboard(run))  # type: ignore[arg-type]
    with pytest.raises(BudgetExceeded, match="shared budget") as raised:
        asyncio.run(budget.reserve_llm(run.id, AgentRole.RESEARCH, 1.01))
    assert raised.value.role == AgentRole.RESEARCH
    assert raised.value.protected == 2


def test_aggregator_and_judge_can_use_their_protected_caps() -> None:
    run = Run(question="q", models=policy(), budget=Budget(limit_usd=10, spent_usd=8, tools=ToolBudget()))
    budget = PersistentBudget(Blackboard(run))  # type: ignore[arg-type]
    asyncio.run(budget.reserve_llm(run.id, AgentRole.AGGREGATOR, 1))
    asyncio.run(budget.reserve_llm(run.id, AgentRole.JUDGE, 1))
    assert run.budget.reserved_usd == 2


def test_tool_budget_exhaustion_reports_provider_usage() -> None:
    run = Run(
        question="q",
        models=policy(),
        budget=Budget(
            limit_usd=10,
            tools=ToolBudget(tavily_max_credits=1, tavily_credits_used=1),
        ),
    )
    budget = PersistentBudget(Blackboard(run))  # type: ignore[arg-type]

    with pytest.raises(BudgetExceeded, match="Tavily credit budget reached") as raised:
        asyncio.run(budget.consume_tool(run.id, "tavily"))

    assert raised.value.budget_type == "tool"
    assert raised.value.provider == "tavily"
    assert raised.value.used == 1
    assert raised.value.limit == 1
