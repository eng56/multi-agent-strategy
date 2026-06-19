"""Role-specific event worker; all infrastructure dependencies remain managed services."""
import asyncio
import os

from src.agents.workflow import HANDLERS, aggregate, deterministic_partial
from src.common.budget import BudgetExceeded
from src.runtime import build_runtime


async def dispatch(runtime, role, event) -> None:
    handler = HANDLERS.get(role, {}).get(event.type)
    if not handler:
        return
    try:
        await handler(runtime, event)
    except BudgetExceeded as exc:
        if role not in {"aggregator-agent", "judge-agent"}:
            try:
                event.payload["force"] = True
                await aggregate(runtime, event)
                return
            except BudgetExceeded:
                pass
        await deterministic_partial(runtime, event.run_id, str(exc))


def main() -> None:
    role = os.getenv("AGENT_ROLE", "worker-agents")
    runtime = build_runtime()
    for event in runtime.events.consume(runtime.settings.runtime_topic, role):
        asyncio.run(dispatch(runtime, role, event))


if __name__ == "__main__":
    main()
