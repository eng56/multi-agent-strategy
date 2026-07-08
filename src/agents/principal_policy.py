from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable

from src.common.models import (
    ActionStatus,
    AgentSpec,
    Artifact,
    ArtifactStatus,
    ArtifactType,
    Claim,
    DeadLetterRecord,
    FinalReport,
    InformationGain,
    Observation,
    OrganizationPlan,
    PrincipalAction,
    PrincipalActionType,
    ResearchTask,
    Run,
    RunState,
    RunStatus,
    Verification,
)

LOW_JUDGE_SCORE_THRESHOLD = 0.75
MAX_FOLLOWUP_WAVES = 1
MIN_USEFUL_ACTION_BUDGET_USD = 0.05
RECENT_ACTION_LIMIT = 12


class PolicyDecisionReason(StrEnum):
    CREATE_ORGANIZATION = "create_organization"
    ASSIGN_INITIAL_TASKS = "assign_initial_tasks"
    RUN_PENDING_TASK = "run_pending_task"
    EXTRACT_CLAIMS = "extract_claims"
    VERIFY_CLAIMS = "verify_claims"
    REQUEST_SKEPTIC = "request_skeptic"
    REQUEST_AGGREGATION = "request_aggregation"
    REQUEST_JUDGE = "request_judge"
    REQUEST_FOLLOWUP = "request_followup"
    PARTIAL_AGGREGATION = "partial_aggregation"
    STOP_RUN = "stop_run"
    RECOVER_DEAD_LETTER = "recover_dead_letter"


@dataclass(frozen=True)
class PrincipalSnapshot:
    run: Run
    run_state: RunState
    organization_plan: OrganizationPlan | None
    agent_specs: list[AgentSpec] = field(default_factory=list)
    tasks: list[ResearchTask] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    verifications: list[Verification] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    principal_actions: list[PrincipalAction] = field(default_factory=list)
    dead_letters: list[DeadLetterRecord] = field(default_factory=list)
    final: FinalReport | None = None


@dataclass(frozen=True)
class PrincipalActionCandidate:
    action_type: PrincipalActionType
    priority: int
    expected_information_gain: InformationGain
    estimated_cost: float
    target_branch: str | None
    required_role: str | None
    reason: str
    decision_reason: PolicyDecisionReason

    def to_action(self, run_id) -> PrincipalAction:
        return PrincipalAction(
            run_id=run_id,
            action_type=self.action_type,
            reason=self.reason,
            expected_information_gain=self.expected_information_gain,
            estimated_cost=self.estimated_cost,
            target_branch=self.target_branch,
            required_role=self.required_role,
            priority=self.priority,
        )


@dataclass(frozen=True)
class PrincipalDecision:
    accepted: list[PrincipalActionCandidate]
    rejected: list[tuple[PrincipalActionCandidate, str]] = field(default_factory=list)


def build_principal_snapshot(
    run: Run,
    run_state: RunState,
    organization_plan: OrganizationPlan | None = None,
    agent_specs: Iterable[AgentSpec] | None = None,
    tasks: Iterable[ResearchTask] | None = None,
    observations: Iterable[Observation] | None = None,
    claims: Iterable[Claim] | None = None,
    verifications: Iterable[Verification] | None = None,
    artifacts: Iterable[Artifact] | None = None,
    principal_actions: Iterable[PrincipalAction] | None = None,
    dead_letters: Iterable[DeadLetterRecord] | None = None,
    final: FinalReport | None = None,
) -> PrincipalSnapshot:
    return PrincipalSnapshot(
        run=run,
        run_state=run_state,
        organization_plan=organization_plan,
        agent_specs=list(agent_specs or []),
        tasks=list(tasks or []),
        observations=list(observations or []),
        claims=list(claims or []),
        verifications=list(verifications or []),
        artifacts=list(artifacts or []),
        principal_actions=list(principal_actions or []),
        dead_letters=list(dead_letters or []),
        final=final,
    )


def propose_principal_actions(snapshot: PrincipalSnapshot) -> list[PrincipalAction]:
    decision = _decide(snapshot)
    return [candidate.to_action(snapshot.run.id) for candidate in decision.accepted]


def select_principal_action(snapshot: PrincipalSnapshot) -> PrincipalAction | None:
    actions = propose_principal_actions(snapshot)
    return actions[0] if actions else None


def _decide(snapshot: PrincipalSnapshot) -> PrincipalDecision:
    accepted: list[PrincipalActionCandidate] = []
    rejected: list[tuple[PrincipalActionCandidate, str]] = []
    seen: set[tuple[PrincipalActionType, str | None, str | None]] = set()

    for candidate in _candidate_rules(snapshot):
        key = _candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        rejection = _validate_candidate(snapshot, candidate)
        if rejection:
            rejected.append((candidate, rejection))
            continue
        accepted.append(candidate)

    accepted.sort(key=_candidate_sort_key)
    return PrincipalDecision(accepted=accepted, rejected=rejected)


def _candidate_rules(snapshot: PrincipalSnapshot) -> list[PrincipalActionCandidate]:
    if snapshot.run.status in {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.PARTIAL_BUDGET_EXHAUSTED,
    }:
        return []

    candidates: list[PrincipalActionCandidate] = []
    pending_tasks = [task for task in snapshot.tasks if task.status == "created"]
    completed_tasks = [task for task in snapshot.tasks if task.status == "completed"]
    verified_count = _verified_knowledge_count(snapshot)
    has_verified_knowledge = verified_count > 0

    if _budget_too_low(snapshot):
        if has_verified_knowledge:
            candidates.append(_partial_aggregation_candidate(snapshot))
        else:
            candidates.append(_stop_candidate(snapshot, "Budget is exhausted and no evidence exists."))
        return candidates

    if snapshot.dead_letters:
        candidates.extend(_dead_letter_candidates(snapshot, has_verified_knowledge))

    if snapshot.organization_plan is None and snapshot.final is None and not snapshot.tasks:
        candidates.append(
            PrincipalActionCandidate(
                action_type=PrincipalActionType.SPAWN_AGENT,
                reason="No organization plan exists; create the deterministic agent organization.",
                expected_information_gain=InformationGain.HIGH,
                estimated_cost=0,
                target_branch="root",
                required_role="principal_policy",
                priority=2,
                decision_reason=PolicyDecisionReason.CREATE_ORGANIZATION,
            )
        )

    if not snapshot.tasks and snapshot.final is None:
        candidates.append(
            PrincipalActionCandidate(
                action_type=PrincipalActionType.ASSIGN_TASK,
                reason="No research tasks exist yet; decompose the objective into initial tasks.",
                expected_information_gain=InformationGain.HIGH,
                estimated_cost=_estimated_cost(snapshot.run, "planner_agent"),
                target_branch="research/general",
                required_role="research_agent",
                priority=9,
                decision_reason=PolicyDecisionReason.ASSIGN_INITIAL_TASKS,
            )
        )
    elif pending_tasks:
        task = _highest_priority_pending_task(pending_tasks)
        candidates.append(
            PrincipalActionCandidate(
                action_type=PrincipalActionType.REQUEST_TOOL_CALL,
                reason=f"Pending task needs tool evidence: {task.title} (task_id={task.id})",
                expected_information_gain=InformationGain.HIGH,
                estimated_cost=_tool_call_cost(task),
                target_branch=_branch_for_task(task, snapshot.agent_specs),
                required_role="tool_runner",
                priority=8,
                decision_reason=PolicyDecisionReason.RUN_PENDING_TASK,
            )
        )

    missing_claim_observations = _observations_without_claims(snapshot)
    if missing_claim_observations and snapshot.final is None:
        observation = missing_claim_observations[0]
        branch = _branch_for_observation(observation, snapshot)
        candidates.append(
            PrincipalActionCandidate(
                action_type=PrincipalActionType.ASSIGN_TASK,
                reason=(
                    f"{len(missing_claim_observations)} observation(s) have no extracted "
                    "claim; assign claim extraction before verification."
                ),
                expected_information_gain=InformationGain.MEDIUM,
                estimated_cost=_estimated_cost(snapshot.run, "research_agent"),
                target_branch=branch,
                required_role="research_agent",
                priority=7,
                decision_reason=PolicyDecisionReason.EXTRACT_CLAIMS,
            )
        )

    unverified_count = _unverified_claim_count(snapshot)
    if unverified_count:
        candidates.append(
            PrincipalActionCandidate(
                action_type=PrincipalActionType.REQUEST_VERIFICATION,
                reason=f"{unverified_count} claim(s) need trust-layer verification.",
                expected_information_gain=InformationGain.MEDIUM,
                estimated_cost=_estimated_cost(snapshot.run, "source_verifier_agent"),
                target_branch="trust/source_verifier",
                required_role="source_verifier_agent",
                priority=7,
                decision_reason=PolicyDecisionReason.VERIFY_CLAIMS,
            )
        )

    has_counterargument = _has_counterargument(snapshot.artifacts)
    if verified_count >= 2 and not has_counterargument:
        candidates.append(
            PrincipalActionCandidate(
                action_type=PrincipalActionType.REQUEST_SKEPTIC_REVIEW,
                reason=(
                    "At least two knowledge items are verified, but no counterargument "
                    "artifact exists."
                ),
                expected_information_gain=InformationGain.MEDIUM,
                estimated_cost=_estimated_cost(snapshot.run, "skeptic_agent"),
                target_branch="trust/skeptic",
                required_role="skeptic_agent",
                priority=6,
                decision_reason=PolicyDecisionReason.REQUEST_SKEPTIC,
            )
        )

    needs_followup_aggregation = _should_reaggregate_after_followup(snapshot)
    zero_verified_terminal_aggregation = _zero_verified_terminal_aggregation_ready(snapshot)
    zero_verified_repair = _zero_verified_repair_ready(snapshot)
    if has_verified_knowledge and has_counterargument:
        candidates.append(_aggregation_candidate(snapshot, PolicyDecisionReason.REQUEST_AGGREGATION))
    elif (
        snapshot.tasks
        and not pending_tasks
        and completed_tasks
        and has_verified_knowledge
        and not snapshot.final
    ) or needs_followup_aggregation:
        candidates.append(_aggregation_candidate(snapshot, PolicyDecisionReason.REQUEST_AGGREGATION))
    elif zero_verified_repair:
        candidates.append(
            PrincipalActionCandidate(
                action_type=PrincipalActionType.REQUEST_FOLLOWUP,
                reason=(
                    "Zero claims passed verification while search capacity remains; "
                    "launch branch-distributed evidence repair before any terminal final."
                ),
                expected_information_gain=InformationGain.HIGH,
                estimated_cost=_estimated_cost(snapshot.run, "research_agent"),
                target_branch="research/recovery",
                required_role="principal_policy",
                priority=8,
                decision_reason=PolicyDecisionReason.REQUEST_FOLLOWUP,
            )
        )
    elif zero_verified_terminal_aggregation:
        candidates.append(_aggregation_candidate(snapshot, PolicyDecisionReason.PARTIAL_AGGREGATION))

    if snapshot.final and snapshot.run.models.judge and snapshot.final.judge_score is None:
        candidates.append(
            PrincipalActionCandidate(
                action_type=PrincipalActionType.REQUEST_TOOL_CALL,
                reason=(
                    "A final or evidence-limited report exists without a judge score; "
                    "run the judge before stopping."
                ),
                expected_information_gain=InformationGain.MEDIUM,
                estimated_cost=_estimated_cost(snapshot.run, "judge_agent"),
                target_branch="synthesis/judge",
                required_role="judge_agent",
                priority=8,
                decision_reason=PolicyDecisionReason.REQUEST_JUDGE,
            )
        )
    elif snapshot.final and snapshot.final.partial:
        candidates.append(
            _stop_candidate(
                snapshot,
                "Evidence-limited partial final has completed required evaluation.",
            )
        )

    if snapshot.final and snapshot.final.judge_score is not None:
        if _can_request_followup(snapshot):
            candidates.append(
                PrincipalActionCandidate(
                    action_type=PrincipalActionType.REQUEST_FOLLOWUP,
                    reason=(
                        f"Judge score {snapshot.final.judge_score:.2f} is below "
                        f"{LOW_JUDGE_SCORE_THRESHOLD:.2f}; request one targeted follow-up wave."
                    ),
                    expected_information_gain=InformationGain.MEDIUM,
                    estimated_cost=_estimated_cost(snapshot.run, "research_agent"),
                    target_branch="synthesis/judge",
                    required_role="principal_policy",
                    priority=8,
                    decision_reason=PolicyDecisionReason.REQUEST_FOLLOWUP,
                )
            )
        elif not needs_followup_aggregation:
            candidates.append(
                _stop_candidate(
                    snapshot,
                    "The judge score is recorded and no further follow-up wave is available.",
                )
            )
    elif snapshot.final and not snapshot.run.models.judge:
        candidates.append(_stop_candidate(snapshot, "A final report exists and judge is disabled."))

    return candidates


def _validate_candidate(
    snapshot: PrincipalSnapshot, candidate: PrincipalActionCandidate
) -> str | None:
    if snapshot.run.status in {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.PARTIAL_BUDGET_EXHAUSTED,
    }:
        return "run is terminal"

    if snapshot.run_state.budget_remaining <= 0 and not _allowed_when_budget_empty(candidate):
        return "budget is exhausted"

    if (
        candidate.action_type == PrincipalActionType.REQUEST_FOLLOWUP
        and _max_task_wave(snapshot.tasks) >= MAX_FOLLOWUP_WAVES
        and not _zero_verified_repair_ready(snapshot)
    ):
        return "follow-up wave already used"

    if candidate.action_type in {
        PrincipalActionType.REQUEST_VERIFICATION,
        PrincipalActionType.REQUEST_FOLLOWUP,
    } and snapshot.run_state.capacity and snapshot.run_state.capacity.search_exhausted:
        return "search/tool budget is exhausted"

    if _duplicates_recent_action(snapshot, candidate):
        return "duplicates recent equivalent action"

    if _branch_has_pending_equivalent(snapshot, candidate):
        return "branch already has pending equivalent task/action"

    return None


def _candidate_sort_key(
    candidate: PrincipalActionCandidate,
) -> tuple[int, float, float, str, str, str]:
    gain_per_cost = _information_gain_score(candidate.expected_information_gain) / max(
        candidate.estimated_cost, 0.01
    )
    return (
        -candidate.priority,
        -gain_per_cost,
        candidate.estimated_cost,
        candidate.action_type.value,
        candidate.target_branch or "",
        candidate.required_role or "",
    )


def _candidate_key(
    candidate: PrincipalActionCandidate,
) -> tuple[PrincipalActionType, str | None, str | None]:
    return (candidate.action_type, candidate.target_branch, candidate.required_role)


def _recent_actions(actions: list[PrincipalAction]) -> list[PrincipalAction]:
    return sorted(actions, key=lambda action: action.created_at, reverse=True)[:RECENT_ACTION_LIMIT]


def _duplicates_recent_action(
    snapshot: PrincipalSnapshot, candidate: PrincipalActionCandidate
) -> bool:
    for action in _recent_actions(snapshot.principal_actions):
        if action.status != ActionStatus.EXECUTED:
            continue
        if (
            candidate.action_type == PrincipalActionType.REQUEST_FOLLOWUP
            and candidate.target_branch == "research/recovery"
            and action.action_type == PrincipalActionType.REQUEST_FOLLOWUP
            and action.target_branch == "research/recovery"
        ):
            action_wave = _wave_number_from_reason(action.reason)
            if action_wave is not None and action_wave < _max_task_wave(snapshot.tasks) + 1:
                continue
        if (
            action.action_type == candidate.action_type
            and action.target_branch == candidate.target_branch
            and action.required_role == candidate.required_role
        ):
            return True
    return False


def _wave_number_from_reason(reason: str) -> int | None:
    match = re.search(r"wave\s+(\d+)", reason, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _branch_has_pending_equivalent(
    snapshot: PrincipalSnapshot, candidate: PrincipalActionCandidate
) -> bool:
    for action in snapshot.principal_actions:
        if action.status not in {ActionStatus.PROPOSED, ActionStatus.APPROVED}:
            continue
        if (
            action.action_type == candidate.action_type
            and action.target_branch == candidate.target_branch
            and action.required_role == candidate.required_role
        ):
            return True

    if candidate.action_type != PrincipalActionType.ASSIGN_TASK or not candidate.target_branch:
        return False

    return any(
        task.status == "created"
        and _branch_for_task(task, snapshot.agent_specs) == candidate.target_branch
        for task in snapshot.tasks
    )


def _allowed_when_budget_empty(candidate: PrincipalActionCandidate) -> bool:
    return candidate.action_type == PrincipalActionType.STOP_RUN or (
        candidate.action_type == PrincipalActionType.REQUEST_AGGREGATION
        and candidate.decision_reason == PolicyDecisionReason.PARTIAL_AGGREGATION
    )


def _budget_too_low(snapshot: PrincipalSnapshot) -> bool:
    return snapshot.run_state.budget_remaining < MIN_USEFUL_ACTION_BUDGET_USD


def _partial_aggregation_candidate(snapshot: PrincipalSnapshot) -> PrincipalActionCandidate:
    return PrincipalActionCandidate(
        action_type=PrincipalActionType.REQUEST_AGGREGATION,
        reason="Budget is too low for more research; synthesize the verified evidence available.",
        expected_information_gain=InformationGain.LOW,
        estimated_cost=0,
        target_branch="synthesis/aggregator",
        required_role="aggregator_agent",
        priority=9,
        decision_reason=PolicyDecisionReason.PARTIAL_AGGREGATION,
    )


def _aggregation_candidate(
    snapshot: PrincipalSnapshot, decision_reason: PolicyDecisionReason
) -> PrincipalActionCandidate:
    reason = "Verified knowledge and counterargument context are available; synthesize a report."
    if decision_reason == PolicyDecisionReason.PARTIAL_AGGREGATION:
        reason = (
            "Follow-up evidence recovery is complete with no verified claims; "
            "produce an evidence-limited final with caveated evidence links."
        )
    elif _should_reaggregate_after_followup(snapshot):
        reason = "Follow-up evidence is complete; refresh synthesis before the next judge pass."
    elif snapshot.final is None:
        reason = "Trusted evidence is available and no final report exists; synthesize a report."
    return PrincipalActionCandidate(
        action_type=PrincipalActionType.REQUEST_AGGREGATION,
        reason=reason,
        expected_information_gain=InformationGain.MEDIUM,
        estimated_cost=_estimated_cost(snapshot.run, "aggregator_agent"),
        target_branch="synthesis/aggregator",
        required_role="aggregator_agent",
        priority=5,
        decision_reason=decision_reason,
    )


def _stop_candidate(snapshot: PrincipalSnapshot, reason: str) -> PrincipalActionCandidate:
    return PrincipalActionCandidate(
        action_type=PrincipalActionType.STOP_RUN,
        reason=reason,
        expected_information_gain=InformationGain.LOW,
        estimated_cost=0,
        target_branch="root" if snapshot.final is None else "synthesis/judge",
        required_role="principal_policy",
        priority=3,
        decision_reason=PolicyDecisionReason.STOP_RUN,
    )


def _dead_letter_candidates(
    snapshot: PrincipalSnapshot, has_verified_knowledge: bool
) -> list[PrincipalActionCandidate]:
    if has_verified_knowledge and _max_task_wave(snapshot.tasks) < MAX_FOLLOWUP_WAVES:
        return [
            PrincipalActionCandidate(
                action_type=PrincipalActionType.REQUEST_FOLLOWUP,
                reason=(
                    f"{len(snapshot.dead_letters)} dead-lettered event(s) exist; request a "
                    "targeted recovery wave using verified context."
                ),
                expected_information_gain=InformationGain.MEDIUM,
                estimated_cost=_estimated_cost(snapshot.run, "research_agent"),
                target_branch="research/recovery",
                required_role="principal_policy",
                priority=6,
                decision_reason=PolicyDecisionReason.RECOVER_DEAD_LETTER,
            )
        ]
    return [
        _stop_candidate(
            snapshot,
            f"{len(snapshot.dead_letters)} dead-lettered event(s) exist and no useful recovery "
            "context is available.",
        )
    ]


def _highest_priority_pending_task(tasks: list[ResearchTask]) -> ResearchTask:
    return sorted(
        tasks,
        key=lambda task: (
            -task.wave_number,
            0 if task.tool == "market_data" else 1,
            task.title.lower(),
            task.question.lower(),
            str(task.id),
        ),
    )[0]


def _observations_without_claims(snapshot: PrincipalSnapshot) -> list[Observation]:
    claimed_observation_ids = {
        observation_id
        for claim in snapshot.claims
        for observation_id in claim.evidence_observation_ids
    }
    return [
        observation
        for observation in snapshot.observations
        if observation.id not in claimed_observation_ids
        and not _is_market_snapshot_observation(observation, snapshot.artifacts)
        and not _is_data_gap_observation(observation, snapshot.artifacts)
    ]


def _unverified_claim_count(snapshot: PrincipalSnapshot) -> int:
    verified_claim_ids = {verification.claim_id for verification in snapshot.verifications}
    artifact_by_claim_id = _claim_artifacts_by_legacy_id(snapshot.artifacts)
    return len(
        [
            claim
            for claim in snapshot.claims
            if claim.id not in verified_claim_ids
            and not _is_market_snapshot_claim(claim, artifact_by_claim_id.get(claim.id))
        ]
    )


def _verified_knowledge_count(snapshot: PrincipalSnapshot) -> int:
    verified_claim_ids = {
        verification.claim_id
        for verification in snapshot.verifications
        if verification.verdict == "verified"
    }
    verified_artifact_ids = {
        artifact.legacy_object_id or artifact.id
        for artifact in snapshot.artifacts
        if artifact.status == ArtifactStatus.VERIFIED
        and artifact.artifact_type in {ArtifactType.CLAIM, ArtifactType.FORECAST}
    }
    return len(verified_claim_ids | verified_artifact_ids)


def _claim_artifacts_by_legacy_id(artifacts: list[Artifact]) -> dict[object, Artifact]:
    return {
        artifact.legacy_object_id: artifact
        for artifact in artifacts
        if artifact.artifact_type == ArtifactType.CLAIM
        and artifact.legacy_object_type == "claim"
        and artifact.legacy_object_id is not None
    }


def _is_market_snapshot_claim(claim: Claim, artifact: Artifact | None) -> bool:
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


def _is_market_snapshot_observation(
    observation: Observation, artifacts: list[Artifact]
) -> bool:
    if observation.tool == "market_data" or observation.summary.casefold().startswith(
        "marketsnapshot for "
    ):
        return True
    return any(
        artifact.legacy_object_type == "observation"
        and artifact.legacy_object_id == observation.id
        and ("market_snapshot" in artifact.tags or "tool:market_data" in artifact.tags)
        for artifact in artifacts
    )


def _is_data_gap_observation(observation: Observation, artifacts: list[Artifact]) -> bool:
    if "market data gap" in observation.summary.casefold() or "tool failure" in observation.summary.casefold():
        return True
    return any(
        artifact.legacy_object_type == "observation"
        and artifact.legacy_object_id == observation.id
        and (
            artifact.artifact_type == ArtifactType.DATA_GAP
            or "data_gap" in artifact.tags
            or "market_data_gap" in artifact.tags
            or "tool_failure" in artifact.tags
        )
        for artifact in artifacts
    )


def _has_counterargument(artifacts: list[Artifact]) -> bool:
    return any(artifact.artifact_type == ArtifactType.COUNTERARGUMENT for artifact in artifacts)


def _can_request_followup(snapshot: PrincipalSnapshot) -> bool:
    return bool(
        snapshot.final
        and snapshot.final.judge_score is not None
        and snapshot.final.judge_score < LOW_JUDGE_SCORE_THRESHOLD
        and _max_task_wave(snapshot.tasks) < MAX_FOLLOWUP_WAVES
    )


def _should_reaggregate_after_followup(snapshot: PrincipalSnapshot) -> bool:
    return bool(
        snapshot.final
        and snapshot.final.judge_score is not None
        and snapshot.final.judge_score < LOW_JUDGE_SCORE_THRESHOLD
        and _max_task_wave(snapshot.tasks) > snapshot.final.wave_number
    )


def _zero_verified_terminal_aggregation_ready(snapshot: PrincipalSnapshot) -> bool:
    if snapshot.final is not None:
        return False
    if not snapshot.tasks or any(task.status == "created" for task in snapshot.tasks):
        return False
    if (
        snapshot.run_state.capacity
        and not snapshot.run_state.capacity.terminal_final_allowed
    ):
        return False
    if any(verification.verdict == "verified" for verification in snapshot.verifications):
        return False
    checked_claim_ids = {verification.claim_id for verification in snapshot.verifications}
    artifact_by_claim_id = _claim_artifacts_by_legacy_id(snapshot.artifacts)
    claim_ids = {
        claim.id
        for claim in snapshot.claims
        if not _is_market_snapshot_claim(claim, artifact_by_claim_id.get(claim.id))
    }
    return claim_ids <= checked_claim_ids


def _zero_verified_repair_ready(snapshot: PrincipalSnapshot) -> bool:
    if snapshot.final is not None:
        return False
    if not snapshot.tasks or any(task.status == "created" for task in snapshot.tasks):
        return False
    if any(verification.verdict == "verified" for verification in snapshot.verifications):
        return False
    if snapshot.run_state.capacity and snapshot.run_state.capacity.search_exhausted:
        return False
    if snapshot.run_state.budget_remaining < MIN_USEFUL_ACTION_BUDGET_USD:
        return False
    if snapshot.run_state.capacity and not snapshot.run_state.capacity.should_continue_research:
        return False
    checked_claim_ids = {verification.claim_id for verification in snapshot.verifications}
    artifact_by_claim_id = _claim_artifacts_by_legacy_id(snapshot.artifacts)
    claim_ids = {
        claim.id
        for claim in snapshot.claims
        if not _is_market_snapshot_claim(claim, artifact_by_claim_id.get(claim.id))
    }
    if claim_ids and not claim_ids <= checked_claim_ids:
        return False
    failed_claim_tasks = [
        task
        for task in snapshot.tasks
        if task.status == "failed"
        and task.reason
        and "claim_generation" in task.reason.lower()
    ]
    return (
        bool(claim_ids and claim_ids <= checked_claim_ids)
        or bool(failed_claim_tasks)
        or bool(
            snapshot.run_state.capacity
            and snapshot.run_state.capacity.under_researched_branches
        )
    )


def _max_task_wave(tasks: list[ResearchTask]) -> int:
    return max((task.wave_number for task in tasks), default=0)


def _branch_for_observation(observation: Observation, snapshot: PrincipalSnapshot) -> str:
    task_by_id = {task.id: task for task in snapshot.tasks}
    task = task_by_id.get(observation.task_id)
    if task:
        return _branch_for_task(task, snapshot.agent_specs)
    return observation.tool or "research/general"


def _branch_for_task(task: ResearchTask, agent_specs: list[AgentSpec]) -> str:
    # Import lazily to keep state.py free to use this module for final candidate generation.
    from src.agents.state import branch_for_task

    return branch_for_task(task, agent_specs)


def _estimated_cost(run: Run, required_role: str | None) -> float:
    role_name = {
        "planner_agent": "planner",
        "research_agent": "research",
        "source_verifier_agent": "verifier",
        "skeptic_agent": "research",
        "aggregator_agent": "aggregator",
        "judge_agent": "judge",
    }.get(required_role or "")
    policy = getattr(run.models, role_name, None) if role_name else None
    if policy:
        return policy.max_call_cost_usd
    return 0


def _tool_call_cost(task: ResearchTask) -> float:
    return 0.01 if task.tool == "market_data" else 0.02


def _information_gain_score(value: InformationGain) -> int:
    return {
        InformationGain.HIGH: 3,
        InformationGain.MEDIUM: 2,
        InformationGain.LOW: 1,
    }[value]
