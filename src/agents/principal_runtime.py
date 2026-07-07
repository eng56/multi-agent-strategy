from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from src.agents.principal_policy import (
    PrincipalSnapshot,
    build_principal_snapshot,
    select_principal_action,
)
from src.agents.state import branch_for_task, build_run_state, max_task_wave
from src.common.models import (
    ActionStatus,
    AgentSpec,
    Artifact,
    Claim,
    DeadLetterRecord,
    EventEnvelope,
    EventType,
    Observation,
    PrincipalAction,
    PrincipalActionType,
    ResearchTask,
    RunPhase,
    Verification,
)
from src.runtime import Runtime

logger = logging.getLogger(__name__)

SHADOW_POLICY_PRODUCER = "principal-policy-shadow"
ACTIVE_POLICY_PRODUCER = "principal-policy-active"
SHADOW_REASON_PREFIX = "shadow policy proposed:"
ACTIVE_REASON_PREFIX = "active policy selected:"


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

    phase = runtime_snapshot.snapshot.run_state.current_phase
    wave_number = policy_wave_number(selected, runtime_snapshot.snapshot)
    idempotency_key = policy_idempotency_key(
        selected,
        current_phase=phase,
        wave_number=wave_number,
    )
    if equivalent_proposed_action_exists(
        runtime_snapshot.all_actions,
        selected,
        current_phase=phase,
        wave_number=wave_number,
        idempotency_key=idempotency_key,
    ):
        return None

    if mode == "active":
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
    return "|".join(
        [
            str(action.run_id),
            action.action_type.value,
            action.target_branch or "",
            action.required_role or "",
            current_phase.value,
            "" if wave_number is None else str(wave_number),
        ]
    )


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
    dispatched = execute_active_action(runtime, snapshot, selected)
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


def execute_active_action(
    runtime: Runtime, snapshot: PrincipalSnapshot, action: PrincipalAction
) -> bool:
    if (
        action.action_type == PrincipalActionType.REQUEST_TOOL_CALL
        and action.required_role == "tool_runner"
    ):
        task = matching_pending_task(snapshot, action)
        if not task:
            return False
        publish(runtime, EventType.TASK_CREATED, action.run_id, task_id=str(task.id))
        return True

    if (
        action.action_type == PrincipalActionType.REQUEST_TOOL_CALL
        and action.required_role == "judge_agent"
    ):
        publish(runtime, EventType.FINAL_CREATED, action.run_id)
        return True

    if action.action_type == PrincipalActionType.REQUEST_VERIFICATION:
        claim = first_unverified_claim(snapshot)
        if not claim:
            return False
        publish(runtime, EventType.CLAIM_CREATED, action.run_id, claim_id=str(claim.id))
        return True

    if action.action_type == PrincipalActionType.REQUEST_SKEPTIC_REVIEW:
        publish(runtime, EventType.SKEPTIC_REVIEW_REQUESTED, action.run_id)
        return True

    if action.action_type == PrincipalActionType.REQUEST_AGGREGATION:
        publish(runtime, EventType.CLAIM_VERIFIED, action.run_id, force=True)
        return True

    return False


def matching_pending_task(
    snapshot: PrincipalSnapshot, action: PrincipalAction
) -> ResearchTask | None:
    for task in sorted(snapshot.tasks, key=lambda value: (-value.wave_number, str(value.id))):
        if task.status != "created":
            continue
        if action.target_branch and branch_for_task(task, snapshot.agent_specs) != action.target_branch:
            continue
        return task
    return None


def first_unverified_claim(snapshot: PrincipalSnapshot) -> Claim | None:
    verified_claim_ids = {verification.claim_id for verification in snapshot.verifications}
    return next(
        (claim for claim in snapshot.claims if claim.id not in verified_claim_ids),
        None,
    )


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
