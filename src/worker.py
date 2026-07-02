"""Role-specific event worker; all infrastructure dependencies remain managed services."""
import asyncio
import logging
import os

from src.agents.workflow import HANDLERS, aggregate, deterministic_partial, stop_run
from src.common.budget import BudgetExceeded
from src.runtime import build_runtime

logger = logging.getLogger(__name__)


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
    except Exception as exc:
        logger.exception("agent handler failed role=%s event_type=%s run_id=%s", role, event.type, event.run_id)
        await stop_run(runtime, event.run_id, f"{role} handler failed: {type(exc).__name__}: {exc}")


def main() -> None:
    role = os.getenv("AGENT_ROLE", "worker-agents")
    runtime = build_runtime()
    for event in runtime.events.consume(runtime.settings.runtime_topic, role):
        asyncio.run(dispatch(runtime, role, event))


if __name__ == "__main__":
    main()
