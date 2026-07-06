"""Role-specific event worker; all infrastructure dependencies remain managed services."""
import asyncio
import json
import logging
import os
import threading
from uuid import UUID

from src.agents.workflow import (
    HANDLERS,
    aggregate,
    deterministic_partial,
    fail_run_if_all_tasks_failed,
)
from src.common.budget import BudgetExceeded
from src.common.failures import (
    PermanentEventError,
    concise_exception,
    is_permanent_failure,
    is_transient_failure,
    sanitize_payload,
    sanitize_text,
)
from src.common.models import (
    DeadLetterRecord,
    EventEnvelope,
    EventType,
    Observation,
    ResearchTask,
    RunStatus,
)
from src.common.health import (
    WORKER_HEARTBEAT_INTERVAL_SECONDS,
    WORKER_HEARTBEAT_TTL_SECONDS,
    worker_heartbeat_key,
    worker_heartbeat_payload,
)
from src.runtime import build_runtime

logger = logging.getLogger(__name__)

MAX_HANDLER_RETRIES = 2
RETRY_COUNT_KEY = "_retry_count"
MAX_RETRIES_KEY = "_max_retries"
BACKOFF_SECONDS_KEY = "_suggested_backoff_seconds"
LAST_ERROR_KEY = "_last_error"
ORIGINAL_EVENT_ID_KEY = "_original_event_id"
REQUIRED_PAYLOAD_FIELDS = {
    EventType.TASK_CREATED: ("task_id",),
    EventType.OBSERVATION_CREATED: ("observation_id",),
    EventType.CLAIM_CREATED: ("claim_id",),
}
TERMINAL_RUN_STATUSES = {
    RunStatus.COMPLETED,
    RunStatus.PARTIAL_BUDGET_EXHAUSTED,
    RunStatus.FAILED,
}


def _retry_count(event: EventEnvelope) -> int:
    raw_count = event.payload.get(RETRY_COUNT_KEY, 0)
    try:
        count = int(raw_count)
    except (TypeError, ValueError) as exc:
        raise PermanentEventError(f"malformed retry metadata: {RETRY_COUNT_KEY}={raw_count}") from exc
    if count < 0:
        raise PermanentEventError(f"malformed retry metadata: {RETRY_COUNT_KEY}={raw_count}")
    return count


def _require_uuid_payload(event: EventEnvelope) -> None:
    for field in REQUIRED_PAYLOAD_FIELDS.get(event.type, ()):
        value = event.payload.get(field)
        if not value:
            raise PermanentEventError(f"malformed {event.type} payload: missing {field}")
        try:
            UUID(str(value))
        except ValueError as exc:
            raise PermanentEventError(
                f"malformed {event.type} payload: {field} must be a UUID"
            ) from exc


def _retry_backoff_seconds(retry_count: int) -> int:
    return min(60, 2 ** max(0, retry_count - 1))


def _retry_event(runtime, role: str, event: EventEnvelope, exc: Exception, retry_count: int) -> None:
    next_retry_count = retry_count + 1
    payload = dict(event.payload)
    payload[RETRY_COUNT_KEY] = next_retry_count
    payload[MAX_RETRIES_KEY] = MAX_HANDLER_RETRIES
    payload[BACKOFF_SECONDS_KEY] = _retry_backoff_seconds(next_retry_count)
    payload[LAST_ERROR_KEY] = concise_exception(exc)
    payload.setdefault(ORIGINAL_EVENT_ID_KEY, str(event.id))
    retry_event = EventEnvelope(
        type=event.type,
        run_id=event.run_id,
        producer=event.producer,
        payload=payload,
    )
    runtime.events.publish(runtime.settings.runtime_topic, retry_event)
    logger.warning(
        "transient handler failure requeued role=%s event_type=%s run_id=%s retry=%s/%s "
        "suggested_backoff_seconds=%s",
        role,
        event.type,
        event.run_id,
        next_retry_count,
        MAX_HANDLER_RETRIES,
        payload[BACKOFF_SECONDS_KEY],
    )


async def _mark_task_failed(runtime, run_id, task_id: object) -> bool:
    tasks = await runtime.blackboard.list_models(run_id, "tasks", ResearchTask)
    task = next((value for value in tasks if str(value.id) == str(task_id)), None)
    if not task:
        return False
    task.status = "failed"
    await runtime.blackboard.put_task(task)
    await fail_run_if_all_tasks_failed(runtime, run_id)
    return True


async def _mark_affected_state(runtime, event: EventEnvelope, reason: str) -> None:
    if event.payload.get("task_id") and await _mark_task_failed(
        runtime, event.run_id, event.payload["task_id"]
    ):
        return

    if event.payload.get("observation_id"):
        observations = await runtime.blackboard.list_models(
            event.run_id, "observations", Observation
        )
        observation = next(
            (value for value in observations if str(value.id) == str(event.payload["observation_id"])),
            None,
        )
        if observation and await _mark_task_failed(runtime, event.run_id, observation.task_id):
            return

    run = await runtime.blackboard.get_run(event.run_id)
    if not run or run.status in TERMINAL_RUN_STATUSES:
        return
    run.status = RunStatus.FAILED
    run.failure_reason = reason
    run.final_answer = None
    await runtime.blackboard.put_run(run)


async def _dead_letter_event(
    runtime,
    role: str,
    event: EventEnvelope,
    exc: Exception,
    *,
    retry_count: int,
    classification: str,
) -> None:
    error_message = sanitize_text(exc)
    record = DeadLetterRecord(
        run_id=event.run_id,
        event_id=event.id,
        event_type=event.type,
        event_producer=event.producer,
        worker_role=role,
        retry_count=retry_count,
        max_retries=MAX_HANDLER_RETRIES,
        classification=classification,
        error_type=type(exc).__name__,
        error_message=error_message,
        event_payload=sanitize_payload(event.payload),
    )
    await runtime.blackboard.put_dead_letter(record)
    reason = (
        f"Dead-lettered {event.type} for role {role} after {retry_count} retry attempt(s): "
        f"{type(exc).__name__}: {error_message[:240]}"
    )
    await _mark_affected_state(runtime, event, reason)
    logger.error(
        "dead-lettered handler event role=%s event_type=%s run_id=%s classification=%s retry=%s/%s",
        role,
        event.type,
        event.run_id,
        classification,
        retry_count,
        MAX_HANDLER_RETRIES,
    )


async def dispatch(runtime, role, event) -> None:
    handler = HANDLERS.get(role, {}).get(event.type)
    if not handler:
        return
    try:
        _require_uuid_payload(event)
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
        handler_exc = exc
        try:
            retry_count = _retry_count(event)
        except PermanentEventError as metadata_exc:
            retry_count = 0
            handler_exc = metadata_exc
        transient = is_transient_failure(handler_exc)
        if transient and retry_count < MAX_HANDLER_RETRIES:
            _retry_event(runtime, role, event, handler_exc, retry_count)
            return
        classification = (
            "transient"
            if transient
            else "permanent"
            if is_permanent_failure(handler_exc)
            else "unknown"
        )
        logger.error(
            "agent handler failed role=%s event_type=%s run_id=%s classification=%s",
            role,
            event.type,
            event.run_id,
            classification,
        )
        await _dead_letter_event(
            runtime,
            role,
            event,
            handler_exc,
            retry_count=retry_count,
            classification=classification,
        )


async def write_worker_heartbeat(runtime, role: str) -> None:
    await runtime.blackboard.command(
        "SET",
        worker_heartbeat_key(role),
        json.dumps(worker_heartbeat_payload(role), separators=(",", ":")),
        "EX",
        WORKER_HEARTBEAT_TTL_SECONDS,
    )


def start_worker_heartbeat(runtime, role: str) -> threading.Event:
    stop_event = threading.Event()

    def loop() -> None:
        while not stop_event.is_set():
            try:
                asyncio.run(write_worker_heartbeat(runtime, role))
            except Exception:
                logger.exception("worker heartbeat failed role=%s", role)
            stop_event.wait(WORKER_HEARTBEAT_INTERVAL_SECONDS)

    thread = threading.Thread(target=loop, name=f"{role}-heartbeat", daemon=True)
    thread.start()
    return stop_event


def main() -> None:
    role = os.getenv("AGENT_ROLE", "worker-agents")
    runtime = build_runtime()
    heartbeat_stop = start_worker_heartbeat(runtime, role)
    try:
        for event in runtime.events.consume(runtime.settings.runtime_topic, role):
            asyncio.run(dispatch(runtime, role, event))
    finally:
        heartbeat_stop.set()


if __name__ == "__main__":
    main()
