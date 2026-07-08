from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from src.common.models import (
    AgentRole,
    Artifact,
    ArtifactStatus,
    ArtifactType,
    BudgetSummary,
    Claim,
    FinalReport,
    Observation,
    ResearchTask,
    Run,
    RunStatus,
    Verification,
)

MIN_TAVILY_UTILIZATION_BEFORE_PARTIAL_FINAL = 0.60
MIN_LLM_UTILIZATION_BEFORE_PARTIAL_FINAL = 0.20
MIN_REPAIR_WAVES_BEFORE_PARTIAL_FINAL = 2
MIN_BRANCHES_REPAIRED_BEFORE_PARTIAL_FINAL = 3
MIN_USEFUL_LLM_REMAINING_USD = 0.05

RESEARCH_BRANCH_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("market/gold", ("gold", "xau")),
    (
        "market/fx",
        (
            "usd",
            "dxy",
            "fx",
            "eur/usd",
            "eurusd",
            "usdjpy",
            "usd/jpy",
            "currency",
            "u.s. dollar",
            "us dollar",
        ),
    ),
    ("market/oil", ("oil", "crude", "wti", "brent")),
    (
        "market/equities",
        ("equities", "equity", "stocks", "stock", "spx", "s&p", "nasdaq", "sap"),
    ),
    (
        "macro/rates",
        ("rates", "fed", "cut", "cuts", "hike", "hikes", "yield", "yields", "treasury"),
    ),
    ("macro/inflation", ("inflation", "cpi", "pce")),
    ("market/crypto", ("crypto", "bitcoin", "btc", "ethereum", "eth")),
    ("market/bonds", ("bond", "bonds", "duration", "long-duration", "long duration", "tlt")),
)


@dataclass(frozen=True)
class BudgetUtilizationSettings:
    min_tavily_utilization_before_partial_final: float = (
        MIN_TAVILY_UTILIZATION_BEFORE_PARTIAL_FINAL
    )
    min_llm_utilization_before_partial_final: float = MIN_LLM_UTILIZATION_BEFORE_PARTIAL_FINAL
    min_repair_waves_before_partial_final: int = MIN_REPAIR_WAVES_BEFORE_PARTIAL_FINAL
    min_branches_repaired_before_partial_final: int = (
        MIN_BRANCHES_REPAIRED_BEFORE_PARTIAL_FINAL
    )


@dataclass(frozen=True)
class BudgetUtilizationDecision:
    llm_utilization_ratio: float
    tavily_utilization_ratio: float
    market_data_utilization_ratio: float
    verifier_utilization_ratio: float
    aggregator_utilization_ratio: float
    judge_utilization_ratio: float
    useful_capacity_remaining: bool
    should_continue_research: bool
    terminal_final_allowed: bool
    terminal_final_reasons: list[str] = field(default_factory=list)
    terminal_final_blockers: list[str] = field(default_factory=list)
    repair_waves_attempted: int = 0
    min_repair_waves_before_partial_final: int = MIN_REPAIR_WAVES_BEFORE_PARTIAL_FINAL
    branches_repaired: list[str] = field(default_factory=list)
    under_researched_branches: list[str] = field(default_factory=list)
    requested_branches: list[str] = field(default_factory=list)


def settings_from_object(settings: object | None = None) -> BudgetUtilizationSettings:
    return BudgetUtilizationSettings(
        min_tavily_utilization_before_partial_final=_float_setting(
            settings,
            "min_tavily_utilization_before_partial_final",
            "MIN_TAVILY_UTILIZATION_BEFORE_PARTIAL_FINAL",
            MIN_TAVILY_UTILIZATION_BEFORE_PARTIAL_FINAL,
        ),
        min_llm_utilization_before_partial_final=_float_setting(
            settings,
            "min_llm_utilization_before_partial_final",
            "MIN_LLM_UTILIZATION_BEFORE_PARTIAL_FINAL",
            MIN_LLM_UTILIZATION_BEFORE_PARTIAL_FINAL,
        ),
        min_repair_waves_before_partial_final=_int_setting(
            settings,
            "min_repair_waves_before_partial_final",
            "MIN_REPAIR_WAVES_BEFORE_PARTIAL_FINAL",
            MIN_REPAIR_WAVES_BEFORE_PARTIAL_FINAL,
        ),
        min_branches_repaired_before_partial_final=_int_setting(
            settings,
            "min_branches_repaired_before_partial_final",
            "MIN_BRANCHES_REPAIRED_BEFORE_PARTIAL_FINAL",
            MIN_BRANCHES_REPAIRED_BEFORE_PARTIAL_FINAL,
        ),
    )


def evaluate_budget_utilization(
    run: Run,
    budget_summary: BudgetSummary,
    *,
    tasks: Iterable[ResearchTask] = (),
    claims: Iterable[Claim] = (),
    verifications: Iterable[Verification] = (),
    artifacts: Iterable[Artifact] = (),
    observations: Iterable[Observation] = (),
    final: FinalReport | None = None,
    active_branches: Iterable[str] = (),
    settings: object | None = None,
) -> BudgetUtilizationDecision:
    """Compute durable capacity and terminal-final policy from submitted budgets."""
    policy = settings_from_object(settings)
    task_list = list(tasks)
    claim_list = list(claims)
    verification_list = list(verifications)
    artifact_list = list(artifacts)
    observation_list = list(observations)

    tavily_used = run.budget.tools.tavily_credits_used
    tavily_max = run.budget.tools.tavily_max_credits
    market_used = run.budget.tools.market_data_requests_used
    market_max = run.budget.tools.market_data_max_requests
    llm_ratio = _ratio(run.budget.spent_usd, run.budget.limit_usd)
    tavily_ratio = _ratio(tavily_used, tavily_max)
    market_ratio = _ratio(market_used, market_max)
    verifier_ratio = _role_utilization(run, budget_summary, AgentRole.VERIFIER)
    aggregator_ratio = _protected_role_utilization(run, budget_summary, AgentRole.AGGREGATOR)
    judge_ratio = _protected_role_utilization(run, budget_summary, AgentRole.JUDGE)

    terminal_status = run.status in {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.PARTIAL_BUDGET_EXHAUSTED,
    }
    tavily_remaining = max(0, tavily_max - tavily_used)
    llm_remaining = budget_summary.remaining_usd
    search_exhausted = tavily_remaining <= 0
    normal_claims = _normal_claims(claim_list, artifact_list)
    normal_claim_ids = {claim.id for claim in normal_claims}
    verified_claim_ids = {
        verification.claim_id
        for verification in verification_list
        if verification.verdict == "verified" and verification.claim_id in normal_claim_ids
    }
    verified_artifact_ids = {
        artifact.legacy_object_id or artifact.id
        for artifact in artifact_list
        if artifact.status == ArtifactStatus.VERIFIED
        and artifact.artifact_type in {ArtifactType.CLAIM, ArtifactType.FORECAST}
        and not _is_market_snapshot_artifact(artifact)
    }
    verified_claim_count = len(verified_claim_ids | verified_artifact_ids)
    checked_claim_ids = {verification.claim_id for verification in verification_list}
    unverified_claims = [claim for claim in normal_claims if claim.id not in checked_claim_ids]
    pending_tasks = [task for task in task_list if task.status == "created"]
    repair_waves = max((task.wave_number for task in task_list), default=0)
    branches_repaired = sorted(
        {
            _task_branch(task)
            for task in task_list
            if task.wave_number > 0 and _task_branch(task).startswith(("macro/", "market/", "research/"))
        }
    )
    requested_branches = _requested_branches(run.question, active_branches, task_list)
    completed_or_pending_branches = {
        _task_branch(task)
        for task in task_list
        if task.status in {"created", "completed"} and _task_branch(task)
    }
    under_researched = [
        branch
        for branch in requested_branches
        if branch not in completed_or_pending_branches and branch not in branches_repaired
    ]
    if verified_claim_count == 0:
        min_branch_repairs = min(
            policy.min_branches_repaired_before_partial_final,
            max(1, len(requested_branches)),
        )
        for branch in requested_branches:
            if branch in branches_repaired:
                continue
            if len(branches_repaired) >= min_branch_repairs:
                break
            if branch not in under_researched:
                under_researched.append(branch)

    missing_claim_observations = _observations_without_claims(
        observation_list, normal_claims, artifact_list
    )
    source_quality_insufficient = _source_quality_insufficient(verification_list)
    aggregator_attempted = final is not None or any(
        artifact.artifact_type == ArtifactType.FINAL_REPORT for artifact in artifact_list
    )
    judge_attempted = bool(final and final.judge_score is not None) or any(
        artifact.artifact_type == ArtifactType.JUDGE_FEEDBACK for artifact in artifact_list
    )
    judge_budget_remains = _protected_remaining(
        run, budget_summary, AgentRole.JUDGE
    ) > 0
    aggregator_budget_remains = _protected_remaining(
        run, budget_summary, AgentRole.AGGREGATOR
    ) > 0
    verifier_budget_remains = _role_remaining(run, budget_summary, AgentRole.VERIFIER) > 0

    useful_reasons: list[str] = []
    if not terminal_status:
        if pending_tasks:
            useful_reasons.append("pending tasks exist")
        if missing_claim_observations:
            useful_reasons.append("normal observations need claim extraction")
        if unverified_claims and verifier_budget_remains:
            useful_reasons.append("normal claims need verification")
        if (
            verified_claim_count == 0
            and repair_waves < policy.min_repair_waves_before_partial_final
            and tavily_remaining > 0
            and llm_remaining > MIN_USEFUL_LLM_REMAINING_USD
        ):
            useful_reasons.append("zero verified claims and repair waves remain")
        if (
            verified_claim_count == 0
            and tavily_max > 0
            and tavily_ratio < policy.min_tavily_utilization_before_partial_final
            and tavily_remaining > 0
        ):
            useful_reasons.append("no verified claims and search budget is underused")
        if verified_claim_count == 0 and not aggregator_attempted and aggregator_budget_remains:
            useful_reasons.append("no verified claims and aggregator has not produced diagnostics")
        if (
            verified_claim_count == 0
            and aggregator_attempted
            and not judge_attempted
            and judge_budget_remains
        ):
            useful_reasons.append("no verified claims and judge/payoff has not run")
        if under_researched and tavily_remaining > 0:
            useful_reasons.append("branch coverage is incomplete")
        if source_quality_insufficient and tavily_remaining > 0:
            useful_reasons.append("source quality is insufficient and search remains")

    utilization_threshold_reached = (
        tavily_max == 0
        or tavily_ratio >= policy.min_tavily_utilization_before_partial_final
    ) and llm_ratio >= policy.min_llm_utilization_before_partial_final
    repair_waves_exhausted = repair_waves >= policy.min_repair_waves_before_partial_final
    branch_repairs_sufficient = len(branches_repaired) >= min(
        policy.min_branches_repaired_before_partial_final,
        max(1, len(requested_branches)),
    )

    terminal_reasons: list[str] = []
    terminal_blockers: list[str] = []
    if terminal_status:
        terminal_reasons.append("run is already terminal")
    elif verified_claim_count > 0 and aggregator_attempted:
        terminal_reasons.append("verified claims exist and aggregation is complete")
    elif verified_claim_count == 0:
        if search_exhausted:
            terminal_reasons.append("search budget is exhausted")
        if repair_waves_exhausted:
            terminal_reasons.append("minimum zero-verified repair waves were attempted")
        if utilization_threshold_reached and branch_repairs_sufficient:
            terminal_reasons.append("budget utilization and branch repair thresholds were reached")

        if not terminal_reasons:
            if not search_exhausted:
                terminal_blockers.append("search budget remains")
            if tavily_max > 0 and tavily_ratio < policy.min_tavily_utilization_before_partial_final:
                terminal_blockers.append(
                    "Tavily utilization is below partial-final threshold"
                )
            if llm_ratio < policy.min_llm_utilization_before_partial_final:
                terminal_blockers.append("LLM utilization is below partial-final threshold")
            if repair_waves < policy.min_repair_waves_before_partial_final:
                terminal_blockers.append("minimum repair waves have not been attempted")
            if len(branches_repaired) < min(
                policy.min_branches_repaired_before_partial_final,
                max(1, len(requested_branches)),
            ):
                terminal_blockers.append("useful branches remain under-researched")
            if verifier_budget_remains and normal_claims and unverified_claims:
                terminal_blockers.append("verifier budget remains for unchecked claims")
            if aggregator_budget_remains and not aggregator_attempted:
                terminal_blockers.append("aggregator protected budget remains")
            if judge_budget_remains and not judge_attempted and aggregator_attempted:
                terminal_blockers.append("judge protected budget remains")
    elif final is not None:
        terminal_reasons.append("final report exists")

    terminal_allowed = bool(terminal_reasons) and not terminal_blockers
    should_continue = bool(useful_reasons) and not terminal_allowed
    return BudgetUtilizationDecision(
        llm_utilization_ratio=llm_ratio,
        tavily_utilization_ratio=tavily_ratio,
        market_data_utilization_ratio=market_ratio,
        verifier_utilization_ratio=verifier_ratio,
        aggregator_utilization_ratio=aggregator_ratio,
        judge_utilization_ratio=judge_ratio,
        useful_capacity_remaining=should_continue,
        should_continue_research=should_continue,
        terminal_final_allowed=terminal_allowed,
        terminal_final_reasons=terminal_reasons or useful_reasons,
        terminal_final_blockers=terminal_blockers,
        repair_waves_attempted=repair_waves,
        min_repair_waves_before_partial_final=policy.min_repair_waves_before_partial_final,
        branches_repaired=branches_repaired,
        under_researched_branches=under_researched,
        requested_branches=requested_branches,
    )


def _float_setting(
    settings: object | None, lower_name: str, upper_name: str, default: float
) -> float:
    value = _setting_value(settings, lower_name, upper_name)
    if value is None:
        return default
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _int_setting(
    settings: object | None, lower_name: str, upper_name: str, default: int
) -> int:
    value = _setting_value(settings, lower_name, upper_name)
    if value is None:
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _setting_value(settings: object | None, lower_name: str, upper_name: str) -> object | None:
    if settings is None:
        return None
    for name in (lower_name, upper_name):
        if hasattr(settings, name):
            return getattr(settings, name)
    return None


def _ratio(used: float, limit: float) -> float:
    if limit <= 0:
        return 1.0
    return max(0.0, min(1.0, float(used) / float(limit)))


def _role_utilization(run: Run, budget_summary: BudgetSummary, role: AgentRole) -> float:
    cap = _role_cap(run, role)
    used = budget_summary.role_spent_usd.get(role.value, 0.0) + budget_summary.role_reserved_usd.get(
        role.value, 0.0
    )
    return _ratio(used, cap)


def _protected_role_utilization(
    run: Run, budget_summary: BudgetSummary, role: AgentRole
) -> float:
    protected = budget_summary.role_protected_usd.get(role.value, 0.0)
    if protected <= 0:
        return _role_utilization(run, budget_summary, role)
    used = budget_summary.role_spent_usd.get(role.value, 0.0) + budget_summary.role_reserved_usd.get(
        role.value, 0.0
    )
    return _ratio(used, protected)


def _role_remaining(run: Run, budget_summary: BudgetSummary, role: AgentRole) -> float:
    cap = _role_cap(run, role)
    used = budget_summary.role_spent_usd.get(role.value, 0.0) + budget_summary.role_reserved_usd.get(
        role.value, 0.0
    )
    return max(0.0, cap - used)


def _protected_remaining(run: Run, budget_summary: BudgetSummary, role: AgentRole) -> float:
    policy = getattr(run.models, role.value, None)
    if policy is None:
        return 0.0
    protected = budget_summary.role_protected_usd.get(role.value, 0.0)
    used = budget_summary.role_spent_usd.get(role.value, 0.0) + budget_summary.role_reserved_usd.get(
        role.value, 0.0
    )
    return max(0.0, protected - used)


def _role_cap(run: Run, role: AgentRole) -> float:
    policy = getattr(run.models, role.value, None)
    if policy and policy.cap_usd is not None:
        return float(policy.cap_usd)
    return float(run.budget.limit_usd)


def _text_matches_any(text: str, needles: tuple[str, ...]) -> bool:
    lower = text.lower()
    for needle in needles:
        pattern = rf"(?<![A-Za-z0-9]){re.escape(needle.lower())}(?![A-Za-z0-9])"
        if re.search(pattern, lower):
            return True
    return False


def semantic_research_branches(question: str) -> list[str]:
    branches = [
        branch
        for branch, needles in RESEARCH_BRANCH_PATTERNS
        if _text_matches_any(question, needles)
    ]
    return branches or ["research/general"]


def _requested_branches(
    question: str, active_branches: Iterable[str], tasks: Iterable[ResearchTask]
) -> list[str]:
    branches: list[str] = []
    branches.extend(semantic_research_branches(question))
    branches.extend(
        branch
        for branch in active_branches
        if branch.startswith(("macro/", "market/", "research/"))
    )
    branches.extend(
        _task_branch(task)
        for task in tasks
        if _task_branch(task).startswith(("macro/", "market/", "research/"))
    )
    return _unique(branches) or ["research/general"]


def _task_branch(task: ResearchTask) -> str:
    if task.branch:
        return task.branch
    text = f"{task.title} {task.question}"
    for branch, needles in RESEARCH_BRANCH_PATTERNS:
        if _text_matches_any(text, needles):
            return branch
    return "research/general"


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _normal_claims(claims: list[Claim], artifacts: list[Artifact]) -> list[Claim]:
    artifact_by_claim_id = {
        artifact.legacy_object_id: artifact
        for artifact in artifacts
        if artifact.artifact_type == ArtifactType.CLAIM
        and artifact.legacy_object_type == "claim"
        and artifact.legacy_object_id is not None
    }
    return [
        claim
        for claim in claims
        if not _is_context_only_claim(claim, artifact_by_claim_id.get(claim.id))
    ]


def _is_context_only_claim(claim: Claim, artifact: Artifact | None) -> bool:
    text = claim.statement.casefold()
    if _is_market_snapshot_claim(claim, artifact):
        return True
    if any(
        pattern in text
        for pattern in (
            "evidenceengine",
            "publication date",
            "provider",
            "retrieval",
            "source count",
            "source title",
            "source quality",
            "quality score",
        )
    ):
        return True
    if claim.derived_from_verification_id is not None:
        return True
    return bool(
        artifact
        and any(
            tag in artifact.tags
            for tag in (
                "context_only",
                "source_meta_claim_rejected",
                "supported_part_subclaim",
                "derived_from_supported_parts",
            )
        )
    )


def _is_market_snapshot_claim(claim: Claim, artifact: Artifact | None) -> bool:
    text = claim.statement.casefold()
    if text.startswith("market data snapshot") or text.startswith("marketsnapshot"):
        return True
    return bool(artifact and _is_market_snapshot_artifact(artifact))


def _is_market_snapshot_artifact(artifact: Artifact) -> bool:
    return (
        "market_snapshot" in artifact.tags
        or "context_only" in artifact.tags
        or "tool:market_data" in artifact.tags
    )


def _observations_without_claims(
    observations: list[Observation], normal_claims: list[Claim], artifacts: list[Artifact]
) -> list[Observation]:
    claimed_observation_ids = {
        observation_id
        for claim in normal_claims
        for observation_id in claim.evidence_observation_ids
    }
    return [
        observation
        for observation in observations
        if observation.id not in claimed_observation_ids
        and _is_original_research_observation(observation, artifacts)
        and not _is_market_snapshot_observation(observation, artifacts)
        and not _is_data_gap_observation(observation, artifacts)
    ]


def _is_original_research_observation(
    observation: Observation, artifacts: list[Artifact]
) -> bool:
    if observation.tool != "web_search":
        return False
    return not any(
        artifact.legacy_object_type == "observation"
        and artifact.legacy_object_id == observation.id
        and (
            artifact.artifact_type
            in {
                ArtifactType.VERIFICATION,
                ArtifactType.DATA_GAP,
                ArtifactType.FINAL_REPORT,
                ArtifactType.JUDGE_FEEDBACK,
            }
            or (artifact.branch or "").startswith(("trust/", "synthesis/"))
            or any(
                tag in artifact.tags
                for tag in (
                    "context_only",
                    "evidence_gap_planner",
                    "source_meta_claim_rejected",
                    "market_snapshot_claim_rejected",
                    "tool:market_data",
                )
            )
        )
        for artifact in artifacts
    )


def _is_market_snapshot_observation(observation: Observation, artifacts: list[Artifact]) -> bool:
    if observation.tool == "market_data" or observation.summary.casefold().startswith(
        "marketsnapshot for "
    ):
        return True
    return any(
        artifact.legacy_object_type == "observation"
        and artifact.legacy_object_id == observation.id
        and _is_market_snapshot_artifact(artifact)
        for artifact in artifacts
    )


def _is_data_gap_observation(observation: Observation, artifacts: list[Artifact]) -> bool:
    summary = observation.summary.casefold()
    if "market data gap" in summary or "tool failure" in summary:
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


def _source_quality_insufficient(verifications: list[Verification]) -> bool:
    weak_terms = (
        "weak",
        "unknown",
        "missing",
        "no dated",
        "no primary",
        "insufficient",
        "unavailable",
    )
    return any(
        verification.verdict != "verified"
        and any(term in verification.source_quality_summary.casefold() for term in weak_terms)
        for verification in verifications
    )
