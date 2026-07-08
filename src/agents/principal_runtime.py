from __future__ import annotations

import logging
from dataclasses import dataclass
import re
from uuid import UUID

from src.agents.principal_policy import (
    PrincipalSnapshot,
    build_principal_snapshot,
    select_principal_action,
)
from src.agents.state import (
    FOLLOWUP_JUDGE_SCORE_THRESHOLD,
    branch_for_task,
    build_run_state,
    max_task_wave,
)
from src.common.models import (
    ActionStatus,
    AgentSpec,
    Artifact,
    ArtifactType,
    Claim,
    DeadLetterRecord,
    EventEnvelope,
    EventType,
    Observation,
    PrincipalAction,
    PrincipalActionType,
    ResearchTask,
    RunPhase,
    RunStatus,
    Verification,
)
from src.runtime import Runtime

logger = logging.getLogger(__name__)

SHADOW_POLICY_PRODUCER = "principal-policy-shadow"
ACTIVE_POLICY_PRODUCER = "principal-policy-active"
SHADOW_REASON_PREFIX = "shadow policy proposed:"
ACTIVE_REASON_PREFIX = "active policy selected:"
ACTIVE_EXECUTOR_ACTIONS = {
    PrincipalActionType.ASSIGN_TASK,
    PrincipalActionType.REQUEST_TOOL_CALL,
    PrincipalActionType.REQUEST_VERIFICATION,
    PrincipalActionType.REQUEST_SKEPTIC_REVIEW,
    PrincipalActionType.REQUEST_AGGREGATION,
    PrincipalActionType.REQUEST_FOLLOWUP,
    PrincipalActionType.STOP_RUN,
}
SAFE_SHADOW_ACTIVE_ACTIONS = {
    PrincipalActionType.ASSIGN_TASK,
    PrincipalActionType.REQUEST_TOOL_CALL,
    PrincipalActionType.REQUEST_VERIFICATION,
    PrincipalActionType.REQUEST_AGGREGATION,
    PrincipalActionType.REQUEST_FOLLOWUP,
    PrincipalActionType.STOP_RUN,
}


@dataclass(frozen=True)
class RuntimePrincipalSnapshot:
    snapshot: PrincipalSnapshot
    all_actions: list[PrincipalAction]


async def evaluate_principal_policy(
    runtime: Runtime,
    run_id: UUID,
    *,
    trigger: str,
) -> PrincipalAction | None:
    mode = getattr(runtime.settings, "principal_policy_mode", "shadow")
    if mode == "off":
        return None

    runtime_snapshot = await load_principal_snapshot(runtime, run_id)
    if not runtime_snapshot:
        return None

    selected = select_principal_action(runtime_snapshot.snapshot)
    if not selected:
        return None
    if dispatch_already_executed(runtime_snapshot.snapshot, selected):
        return None

    phase = runtime_snapshot.snapshot.run_state.current_phase
    wave_number = policy_wave_number(selected, runtime_snapshot.snapshot)
    idempotency_key = policy_idempotency_key(
        selected,
        current_phase=phase,
        wave_number=wave_number,
    )
    safe_shadow_active = mode == "shadow" and is_safe_shadow_active_candidate(
        runtime_snapshot.snapshot, selected
    )
    has_equivalent_proposed = equivalent_proposed_action_exists(
        runtime_snapshot.all_actions,
        selected,
        current_phase=phase,
        wave_number=wave_number,
        idempotency_key=idempotency_key,
    )
    if has_equivalent_proposed and not safe_shadow_active:
        return None

    if mode == "active" or safe_shadow_active:
        return await persist_active_action(
            runtime,
            runtime_snapshot.snapshot,
            selected,
            trigger=trigger,
            current_phase=phase,
            wave_number=wave_number,
            idempotency_key=idempotency_key,
        )

    shadow_action = copy_policy_action(
        selected,
        reason=f"{SHADOW_REASON_PREFIX} {selected.reason}",
        status=ActionStatus.PROPOSED,
        producer=SHADOW_POLICY_PRODUCER,
        current_phase=phase,
        wave_number=wave_number,
        idempotency_key=idempotency_key,
    )
    await runtime.blackboard.put_principal_action(shadow_action)
    logger.info(
        "shadow principal policy proposed action run_id=%s trigger=%s action_type=%s",
        run_id,
        trigger,
        shadow_action.action_type,
    )
    return shadow_action


def is_safe_shadow_active_candidate(
    snapshot: PrincipalSnapshot, selected: PrincipalAction
) -> bool:
    if selected.action_type not in SAFE_SHADOW_ACTIVE_ACTIONS:
        return False
    capacity = snapshot.run_state.capacity
    if selected.action_type in {
        PrincipalActionType.REQUEST_VERIFICATION,
        PrincipalActionType.REQUEST_FOLLOWUP,
    } and capacity and capacity.search_exhausted:
        return False
    if selected.action_type == PrincipalActionType.ASSIGN_TASK:
        return selected.reason.startswith("No research tasks exist") or (
            "observation(s) have no extracted claim" in selected.reason
        )
    if selected.action_type == PrincipalActionType.REQUEST_TOOL_CALL:
        if snapshot.run.status in {
            RunStatus.COMPLETED,
            RunStatus.PARTIAL_BUDGET_EXHAUSTED,
            RunStatus.FAILED,
        }:
            return False
        task = matching_pending_task(snapshot, selected)
        if task is None:
            return False
        if task_dispatch_already_executed(snapshot, task, selected):
            return False
        if task.tool == "market_data":
            return capacity is None or not capacity.market_data_exhausted
        return capacity is None or not capacity.search_exhausted
    if selected.action_type == PrincipalActionType.REQUEST_VERIFICATION:
        return first_unverified_claim(snapshot) is not None
    if selected.action_type == PrincipalActionType.REQUEST_AGGREGATION:
        return should_dispatch_aggregation(snapshot)
    if selected.action_type == PrincipalActionType.REQUEST_FOLLOWUP:
        return True
    if selected.action_type == PrincipalActionType.STOP_RUN:
        return not useful_action_remains(snapshot)
    return False


async def load_principal_snapshot(
    runtime: Runtime, run_id: UUID
) -> RuntimePrincipalSnapshot | None:
    run = await runtime.blackboard.get_run(run_id)
    if not run:
        return None

    tasks = await runtime.blackboard.list_models(run_id, "tasks", ResearchTask)
    observations = await runtime.blackboard.list_models(run_id, "observations", Observation)
    claims = await runtime.blackboard.list_models(run_id, "claims", Claim)
    verifications = await runtime.blackboard.list_models(run_id, "verifications", Verification)
    all_actions = await runtime.blackboard.list_models(
        run_id, "principal_actions", PrincipalAction
    )
    policy_actions = [action for action in all_actions if not is_shadow_action(action)]
    agent_specs = await runtime.blackboard.list_models(run_id, "agent_specs", AgentSpec)
    artifacts = await runtime.blackboard.list_models(run_id, "artifacts", Artifact)
    dead_letters = await runtime.blackboard.list_models(
        run_id, "dead_letters", DeadLetterRecord
    )
    organization_plan = await runtime.blackboard.get_organization_plan(run_id)
    final = await runtime.blackboard.get_final(run_id)
    run_state = build_run_state(
        run,
        tasks,
        claims,
        verifications,
        final,
        agent_specs=agent_specs,
        artifacts=artifacts,
        organization_plan=organization_plan,
        dead_letters=dead_letters,
        principal_actions=policy_actions,
        observations=observations,
    )
    return RuntimePrincipalSnapshot(
        snapshot=build_principal_snapshot(
            run,
            run_state,
            organization_plan,
            agent_specs=agent_specs,
            tasks=tasks,
            observations=observations,
            claims=claims,
            verifications=verifications,
            artifacts=artifacts,
            principal_actions=policy_actions,
            dead_letters=dead_letters,
            final=final,
        ),
        all_actions=all_actions,
    )


def is_shadow_action(action: PrincipalAction) -> bool:
    return action.producer == SHADOW_POLICY_PRODUCER or action.reason.startswith(
        SHADOW_REASON_PREFIX
    )


def copy_policy_action(
    action: PrincipalAction,
    *,
    reason: str,
    status: ActionStatus,
    producer: str,
    current_phase: RunPhase,
    wave_number: int | None,
    idempotency_key: str,
) -> PrincipalAction:
    return PrincipalAction(
        run_id=action.run_id,
        action_type=action.action_type,
        reason=reason,
        expected_information_gain=action.expected_information_gain,
        estimated_cost=action.estimated_cost,
        target_branch=action.target_branch,
        required_role=action.required_role,
        priority=action.priority,
        status=status,
        producer=producer,
        policy_phase=current_phase,
        policy_wave_number=wave_number,
        idempotency_key=idempotency_key,
    )


def policy_idempotency_key(
    action: PrincipalAction,
    *,
    current_phase: RunPhase,
    wave_number: int | None,
) -> str:
    parts = [
        str(action.run_id),
        action.action_type.value,
        action.target_branch or "",
        action.required_role or "",
        current_phase.value,
        "" if wave_number is None else str(wave_number),
    ]
    if action.action_type == PrincipalActionType.REQUEST_TOOL_CALL:
        parts.append(_task_id_from_action_reason(action.reason) or action.reason)
    return "|".join(parts)


def policy_wave_number(action: PrincipalAction, snapshot: PrincipalSnapshot) -> int | None:
    if action.target_branch and snapshot.tasks:
        branch_waves = [
            task.wave_number
            for task in snapshot.tasks
            if branch_for_task(task, snapshot.agent_specs) == action.target_branch
        ]
        if branch_waves:
            return max(branch_waves)
    if snapshot.final:
        return snapshot.final.wave_number
    if snapshot.tasks:
        return max_task_wave(snapshot.tasks)
    return None


def equivalent_proposed_action_exists(
    actions: list[PrincipalAction],
    candidate: PrincipalAction,
    *,
    current_phase: RunPhase,
    wave_number: int | None,
    idempotency_key: str,
) -> bool:
    for action in actions:
        if action.status != ActionStatus.PROPOSED:
            continue
        if action.idempotency_key and action.idempotency_key == idempotency_key:
            return True
        action_phase = action.policy_phase or current_phase
        action_wave = action.policy_wave_number
        if action_wave is None:
            action_wave = wave_number
        action_key = policy_idempotency_key(
            action,
            current_phase=action_phase,
            wave_number=action_wave,
        )
        if action_key == idempotency_key:
            return True
    return False


async def persist_active_action(
    runtime: Runtime,
    snapshot: PrincipalSnapshot,
    selected: PrincipalAction,
    *,
    trigger: str,
    current_phase: RunPhase,
    wave_number: int | None,
    idempotency_key: str,
) -> PrincipalAction:
    dispatched = await execute_principal_decision(runtime, snapshot, selected)
    active_action = copy_policy_action(
        selected,
        reason=f"{ACTIVE_REASON_PREFIX} {selected.reason}",
        status=ActionStatus.EXECUTED if dispatched else ActionStatus.PROPOSED,
        producer=ACTIVE_POLICY_PRODUCER,
        current_phase=current_phase,
        wave_number=wave_number,
        idempotency_key=idempotency_key,
    )
    await runtime.blackboard.put_principal_action(active_action)
    runtime.events.publish(
        runtime.settings.runtime_topic,
        EventEnvelope(
            type=EventType.PRINCIPAL_ACTION_CREATED,
            run_id=selected.run_id,
            producer=ACTIVE_POLICY_PRODUCER,
            payload={"action_id": str(active_action.id), "trigger": trigger},
        ),
    )
    return active_action


async def execute_principal_decision(
    runtime: Runtime, snapshot: PrincipalSnapshot, action: PrincipalAction
) -> bool:
    if action.action_type not in ACTIVE_EXECUTOR_ACTIONS:
        return False

    if dispatch_already_executed(snapshot, action):
        return False

    if action.action_type == PrincipalActionType.ASSIGN_TASK:
        if "observation(s) have no extracted claim" in action.reason:
            return execute_claim_extraction_assignment(runtime, snapshot, action)
        return await execute_assign_task(runtime, snapshot, action)

    if (
        action.action_type == PrincipalActionType.REQUEST_TOOL_CALL
        and action.required_role == "tool_runner"
    ):
        task = matching_pending_task(snapshot, action)
        if not task:
            return False
        if task_dispatch_already_executed(snapshot, task, action):
            return False
        publish(runtime, EventType.TASK_CREATED, action.run_id, task_id=str(task.id))
        return True

    if (
        action.action_type == PrincipalActionType.REQUEST_TOOL_CALL
        and action.required_role == "judge_agent"
    ):
        if not snapshot.final or snapshot.final.judge_score is not None:
            return False
        if dispatch_already_executed(snapshot, action):
            return False
        publish(runtime, EventType.FINAL_CREATED, action.run_id)
        return True

    if action.action_type == PrincipalActionType.REQUEST_VERIFICATION:
        claim = first_unverified_claim(snapshot)
        if not claim:
            return False
        if dispatch_already_executed(snapshot, action):
            return False
        publish(runtime, EventType.CLAIM_CREATED, action.run_id, claim_id=str(claim.id))
        return True

    if action.action_type == PrincipalActionType.REQUEST_SKEPTIC_REVIEW:
        if has_counterargument(snapshot):
            return False
        if dispatch_already_executed(snapshot, action):
            return False
        publish(runtime, EventType.SKEPTIC_REVIEW_REQUESTED, action.run_id)
        return True

    if action.action_type == PrincipalActionType.REQUEST_AGGREGATION:
        if not should_dispatch_aggregation(snapshot):
            return False
        if dispatch_already_executed(snapshot, action):
            return False
        publish(runtime, EventType.CLAIM_VERIFIED, action.run_id, force=True)
        return True

    if action.action_type == PrincipalActionType.REQUEST_FOLLOWUP:
        return await execute_followup(runtime, snapshot, action)

    if action.action_type == PrincipalActionType.STOP_RUN:
        return await execute_stop_run(runtime, snapshot, action)

    return False


async def execute_assign_task(
    runtime: Runtime, snapshot: PrincipalSnapshot, action: PrincipalAction
) -> bool:
    if snapshot.tasks:
        return False
    if dispatch_already_executed(snapshot, action):
        return False

    from src.agents.workflow import deterministic_task_items, persist_tool_dispatch_marker

    created = 0
    for item in deterministic_task_items(snapshot.run.question, snapshot.organization_plan):
        task = ResearchTask(run_id=snapshot.run.id, **item)
        await runtime.blackboard.put_task(task)
        await persist_tool_dispatch_marker(
            runtime,
            task,
            branch_for_task(task, snapshot.agent_specs),
            producer=ACTIVE_POLICY_PRODUCER,
            reason_prefix="Principal dispatched tool task",
        )
        publish(
            runtime,
            EventType.TASK_CREATED,
            snapshot.run.id,
            task_id=str(task.id),
        )
        created += 1
    return created > 0


async def execute_followup(
    runtime: Runtime, snapshot: PrincipalSnapshot, action: PrincipalAction
) -> bool:
    if dispatch_already_executed(snapshot, action):
        return False

    if not snapshot.final:
        from src.agents.workflow import request_zero_verified_followup_wave

        created = await request_zero_verified_followup_wave(
            runtime,
            snapshot.run,
            tasks=snapshot.tasks,
            claims=snapshot.claims,
            verifications=snapshot.verifications,
        )
        return created > 0

    from src.agents.workflow import (
        followup_already_requested,
        followup_task_items,
        persist_tool_dispatch_marker,
    )

    score = snapshot.final.judge_score
    if score is None or score >= FOLLOWUP_JUDGE_SCORE_THRESHOLD:
        return False
    if followup_already_requested(snapshot.tasks, snapshot.principal_actions):
        return False

    wave_number = max_task_wave(snapshot.tasks) + 1
    created = 0
    publish(
        runtime,
        EventType.FOLLOWUP_REQUESTED,
        snapshot.run.id,
        wave_number=wave_number,
        judge_score=score,
    )
    for item in followup_task_items(snapshot.run, snapshot.final, wave_number):
        task = ResearchTask(run_id=snapshot.run.id, **item)
        await runtime.blackboard.put_task(task)
        await persist_tool_dispatch_marker(
            runtime,
            task,
            branch_for_task(task, snapshot.agent_specs),
            producer=ACTIVE_POLICY_PRODUCER,
            reason_prefix=f"Principal follow-up wave {wave_number} dispatched tool task",
        )
        publish(
            runtime,
            EventType.TASK_CREATED,
            snapshot.run.id,
            task_id=str(task.id),
            wave_number=wave_number,
            followup=True,
        )
        created += 1
    return created > 0


async def execute_stop_run(
    runtime: Runtime, snapshot: PrincipalSnapshot, action: PrincipalAction
) -> bool:
    if snapshot.run.status in {
        RunStatus.COMPLETED,
        RunStatus.PARTIAL_BUDGET_EXHAUSTED,
        RunStatus.FAILED,
    }:
        return False
    if useful_action_remains(snapshot):
        return False
    if dispatch_already_executed(snapshot, action):
        return False

    run = snapshot.run
    if snapshot.final:
        run.status = RunStatus.COMPLETED
        run.final_answer = snapshot.final.answer
        run.failure_reason = None
    else:
        run.status = RunStatus.FAILED
        run.failure_reason = action.reason
        run.final_answer = None
    await runtime.blackboard.put_run(run)
    return True


def matching_pending_task(
    snapshot: PrincipalSnapshot, action: PrincipalAction
) -> ResearchTask | None:
    task_id = _task_id_from_action_reason(action.reason)
    for task in sorted(snapshot.tasks, key=lambda value: (-value.wave_number, str(value.id))):
        if task.status != "created":
            continue
        if task_id and str(task.id) != task_id:
            continue
        if action.target_branch and branch_for_task(task, snapshot.agent_specs) != action.target_branch:
            continue
        return task
    return None


def execute_claim_extraction_assignment(
    runtime: Runtime, snapshot: PrincipalSnapshot, action: PrincipalAction
) -> bool:
    observation = first_observation_without_claim(snapshot)
    if not observation:
        return False
    publish(
        runtime,
        EventType.OBSERVATION_CREATED,
        action.run_id,
        observation_id=str(observation.id),
    )
    return True


def first_observation_without_claim(snapshot: PrincipalSnapshot) -> Observation | None:
    claimed_observation_ids = {
        observation_id
        for claim in snapshot.claims
        for observation_id in claim.evidence_observation_ids
    }
    return next(
        (
            observation
            for observation in snapshot.observations
            if observation.id not in claimed_observation_ids
            and observation.tool != "market_data"
            and not observation.summary.casefold().startswith("marketsnapshot for ")
        ),
        None,
    )


def first_unverified_claim(snapshot: PrincipalSnapshot) -> Claim | None:
    verified_claim_ids = {verification.claim_id for verification in snapshot.verifications}
    claim_artifact_by_id = {
        artifact.legacy_object_id: artifact
        for artifact in snapshot.artifacts
        if artifact.artifact_type == ArtifactType.CLAIM
        and artifact.legacy_object_type == "claim"
        and artifact.legacy_object_id is not None
    }
    return next(
        (
            claim
            for claim in snapshot.claims
            if claim.id not in verified_claim_ids
            and not is_market_snapshot_context_claim(claim, claim_artifact_by_id.get(claim.id))
        ),
        None,
    )


def is_market_snapshot_context_claim(claim: Claim, artifact: Artifact | None) -> bool:
    text = claim.statement.casefold()
    if text.startswith("market data snapshot") or text.startswith("marketsnapshot"):
        return True
    return bool(
        artifact
        and (
            "market_snapshot" in artifact.tags
            or "context_only" in artifact.tags
            or "tool:market_data" in artifact.tags
        )
    )


def has_counterargument(snapshot: PrincipalSnapshot) -> bool:
    return any(
        artifact.artifact_type == ArtifactType.COUNTERARGUMENT
        for artifact in snapshot.artifacts
    )


def should_dispatch_aggregation(snapshot: PrincipalSnapshot) -> bool:
    if not snapshot.tasks:
        return False
    if any(task.status == "created" for task in snapshot.tasks):
        return False
    if snapshot.final and not should_reaggregate_after_followup_snapshot(snapshot):
        return False
    verified_claim_ids = {
        verification.claim_id
        for verification in snapshot.verifications
        if verification.verdict == "verified"
    }
    if verified_claim_ids:
        return True
    return zero_verified_terminal_aggregation_ready(snapshot)


def zero_verified_terminal_aggregation_ready(snapshot: PrincipalSnapshot) -> bool:
    if snapshot.final is not None:
        return False
    if any(verification.verdict == "verified" for verification in snapshot.verifications):
        return False
    if snapshot.run_state.capacity and not snapshot.run_state.capacity.terminal_final_allowed:
        return False
    verified_or_checked_claim_ids = {verification.claim_id for verification in snapshot.verifications}
    claim_artifact_by_id = {
        artifact.legacy_object_id: artifact
        for artifact in snapshot.artifacts
        if artifact.artifact_type == ArtifactType.CLAIM
        and artifact.legacy_object_type == "claim"
        and artifact.legacy_object_id is not None
    }
    claim_ids = {
        claim.id
        for claim in snapshot.claims
        if not is_market_snapshot_context_claim(claim, claim_artifact_by_id.get(claim.id))
    }
    return claim_ids <= verified_or_checked_claim_ids


def should_reaggregate_after_followup_snapshot(snapshot: PrincipalSnapshot) -> bool:
    return bool(
        snapshot.final
        and snapshot.final.judge_score is not None
        and snapshot.final.judge_score < FOLLOWUP_JUDGE_SCORE_THRESHOLD
        and max_task_wave(snapshot.tasks) > snapshot.final.wave_number
    )


def useful_action_remains(snapshot: PrincipalSnapshot) -> bool:
    if snapshot.run_state.capacity and snapshot.run_state.capacity.useful_capacity_remaining:
        return True
    if any(task.status == "created" for task in snapshot.tasks):
        return True
    if first_unverified_claim(snapshot):
        return True
    if should_dispatch_aggregation(snapshot):
        return True
    return bool(
        snapshot.final
        and not snapshot.final.partial
        and snapshot.run.models.judge
        and snapshot.final.judge_score is None
    )


def dispatch_already_executed(snapshot: PrincipalSnapshot, action: PrincipalAction) -> bool:
    return any(actions_match(existing, action) for existing in snapshot.principal_actions)


def task_dispatch_already_executed(
    snapshot: PrincipalSnapshot, task: ResearchTask, action: PrincipalAction
) -> bool:
    branch = branch_for_task(task, snapshot.agent_specs)
    return any(
        existing.status == ActionStatus.EXECUTED
        and existing.action_type == action.action_type
        and existing.required_role == action.required_role
        and (
            _task_id_from_action_reason(existing.reason) == str(task.id)
            if _task_id_from_action_reason(existing.reason)
            else existing.target_branch == branch
        )
        for existing in snapshot.principal_actions
    )


def actions_match(existing: PrincipalAction, action: PrincipalAction) -> bool:
    existing_task_id = _task_id_from_action_reason(existing.reason)
    action_task_id = _task_id_from_action_reason(action.reason)
    if (
        existing.action_type == PrincipalActionType.REQUEST_TOOL_CALL
        and action.action_type == PrincipalActionType.REQUEST_TOOL_CALL
        and existing_task_id
        and action_task_id
    ):
        return bool(
            existing.status == ActionStatus.EXECUTED
            and existing.required_role == action.required_role
            and existing_task_id == action_task_id
        )
    return bool(
        existing.status == ActionStatus.EXECUTED
        and existing.action_type == action.action_type
        and existing.required_role == action.required_role
        and existing.target_branch == action.target_branch
    )


def _task_id_from_action_reason(reason: str) -> str | None:
    match = re.search(r"task_id=([0-9a-fA-F-]{36})", reason)
    return match.group(1) if match else None


def publish(runtime: Runtime, event_type: EventType, run_id: UUID, **payload: object) -> None:
    runtime.events.publish(
        runtime.settings.runtime_topic,
        EventEnvelope(
            type=event_type,
            run_id=run_id,
            producer=ACTIVE_POLICY_PRODUCER,
            payload=payload,
        ),
    )
