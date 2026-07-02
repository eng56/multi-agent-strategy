import asyncio
from uuid import UUID, uuid4

from src.common.models import AgentRole, Run
from src.integrations.upstash import UpstashBlackboard


class BudgetExceeded(RuntimeError):
    pass


class PersistentBudget:
    """Serializes reservations in Upstash and protects finalization role budgets."""

    def __init__(self, blackboard: UpstashBlackboard) -> None:
        self.blackboard = blackboard

    async def _lock(self, run_id: UUID) -> str:
        token = str(uuid4())
        for _ in range(50):
            if await self.blackboard.command("SET", f"run:{run_id}:budget-lock", token, "NX", "EX", 10):
                return token
            await asyncio.sleep(0.05)
        raise RuntimeError("could not acquire budget lock")

    async def _unlock(self, run_id: UUID, token: str) -> None:
        script = "if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('del',KEYS[1]) end return 0"
        await self.blackboard.command("EVAL", script, 1, f"run:{run_id}:budget-lock", token)

    @staticmethod
    def _protected_for_other_roles(run: Run, role: AgentRole) -> float:
        protected = 0.0
        if role != AgentRole.AGGREGATOR:
            used = run.budget.role_spent_usd.get(AgentRole.AGGREGATOR, 0) + run.budget.role_reserved_usd.get(AgentRole.AGGREGATOR, 0)
            protected += max(0, run.models.aggregator.protected_usd - used)
        if run.models.judge and role != AgentRole.JUDGE:
            used = run.budget.role_spent_usd.get(AgentRole.JUDGE, 0) + run.budget.role_reserved_usd.get(AgentRole.JUDGE, 0)
            protected += max(0, run.models.judge.protected_usd - used)
        return protected

    async def reserve_llm(self, run_id: UUID, role: AgentRole, estimate: float) -> None:
        token = await self._lock(run_id)
        try:
            run = await self.blackboard.get_run(run_id)
            if not run:
                raise RuntimeError(f"run {run_id} not found")
            budget = run.budget
            available = budget.limit_usd - budget.spent_usd - budget.reserved_usd
            if available - self._protected_for_other_roles(run, role) < estimate:
                raise BudgetExceeded(f"insufficient shared budget for {role}")
            policy = getattr(run.models, role.value)
            role_used = budget.role_spent_usd.get(role, 0) + budget.role_reserved_usd.get(role, 0)
            if policy and policy.cap_usd is not None and role_used + estimate > policy.cap_usd:
                raise BudgetExceeded(f"{role} cap reached")
            budget.reserved_usd += estimate
            budget.role_reserved_usd[role] = budget.role_reserved_usd.get(role, 0) + estimate
            await self.blackboard.put_run(run)
        finally:
            await self._unlock(run_id, token)

    async def reconcile_llm(self, run_id: UUID, role: AgentRole, estimate: float, actual: float) -> None:
        token = await self._lock(run_id)
        try:
            run = await self.blackboard.get_run(run_id)
            if not run:
                return
            budget = run.budget
            budget.reserved_usd = max(0, budget.reserved_usd - estimate)
            budget.role_reserved_usd[role] = max(0, budget.role_reserved_usd.get(role, 0) - estimate)
            budget.spent_usd += actual
            budget.role_spent_usd[role] = budget.role_spent_usd.get(role, 0) + actual
            await self.blackboard.put_run(run)
        finally:
            await self._unlock(run_id, token)

    async def consume_tool(self, run_id: UUID, provider: str, units: int = 1) -> None:
        token = await self._lock(run_id)
        try:
            run = await self.blackboard.get_run(run_id)
            if not run:
                raise RuntimeError(f"run {run_id} not found")
            tools = run.budget.tools
            if provider == "tavily":
                if tools.tavily_credits_used + units > tools.tavily_max_credits:
                    raise BudgetExceeded("Tavily credit budget reached")
                tools.tavily_credits_used += units
            else:
                if tools.market_data_requests_used + units > tools.market_data_max_requests:
                    raise BudgetExceeded("market-data request budget reached")
                tools.market_data_requests_used += units
            await self.blackboard.put_run(run)
        finally:
            await self._unlock(run_id, token)
