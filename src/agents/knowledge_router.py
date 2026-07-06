from __future__ import annotations

from uuid import UUID

from src.common.models import (
    AgentSpec,
    Artifact,
    ArtifactStatus,
    ArtifactType,
    ResearchTask,
    RunState,
    VisibilityScope,
)

HIGH_CONFIDENCE_THRESHOLD = 0.75
LINK_FIELDS = (
    "depends_on_artifact_ids",
    "supports_artifact_ids",
    "contradicts_artifact_ids",
)
PRIVILEGED_ROLE_MARKERS = ("verifier", "judge", "skeptic", "counterargument")


def select_context_for_agent(
    *,
    agent_spec: AgentSpec,
    run_state: RunState,
    artifacts: list[Artifact],
    tasks: list[ResearchTask],
    max_items: int = 12,
) -> list[Artifact]:
    """Select a deterministic, bounded artifact context for an agent."""
    if max_items <= 0:
        return []

    del tasks
    agent_terms = _agent_relevance_terms(agent_spec)
    ranked: dict[UUID, tuple[tuple[int, float, int], Artifact]] = {}
    visible = [
        (index, artifact)
        for index, artifact in enumerate(artifacts)
        if artifact.run_id == run_state.run_id and _status_visible_to_agent(agent_spec, artifact)
    ]

    for index, artifact in visible:
        score = _public_verified_relevance_score(agent_terms, agent_spec, artifact)
        if score > 0:
            _add_ranked(ranked, artifact, category=0, score=score, index=index)

    for index, artifact in visible:
        if _same_branch_team_artifact(agent_spec, artifact):
            _add_ranked(ranked, artifact, category=1, score=1.0, index=index)

    seed_ids = set(ranked)
    if seed_ids:
        seed_artifacts = {artifact.id: artifact for _, artifact in ranked.values()}
        for index, artifact in visible:
            if artifact.id in seed_ids:
                continue
            if _linked_to_selected_artifact(artifact, seed_ids, seed_artifacts):
                _add_ranked(ranked, artifact, category=2, score=1.0, index=index)

    for index, artifact in visible:
        if _high_confidence_verified_knowledge(artifact):
            confidence = artifact.confidence or 0
            _add_ranked(ranked, artifact, category=3, score=confidence, index=index)

    return [
        artifact
        for _, artifact in sorted(ranked.values(), key=lambda item: item[0])
    ][:max_items]


def _add_ranked(
    ranked: dict[UUID, tuple[tuple[int, float, int], Artifact]],
    artifact: Artifact,
    *,
    category: int,
    score: float,
    index: int,
) -> None:
    key = (category, -score, index)
    current = ranked.get(artifact.id)
    if current is None or key < current[0]:
        ranked[artifact.id] = (key, artifact)


def _agent_relevance_terms(agent_spec: AgentSpec) -> set[str]:
    terms = _branch_terms(agent_spec.branch)
    for tag in agent_spec.retrieval_tags:
        terms.update(_term_variants(tag))
    return terms


def _public_verified_relevance_score(
    agent_terms: set[str], agent_spec: AgentSpec, artifact: Artifact
) -> float:
    if (
        artifact.visibility != VisibilityScope.PUBLIC_VERIFIED
        or artifact.status != ArtifactStatus.VERIFIED
    ):
        return 0
    if _same_branch(agent_spec.branch, artifact.branch):
        return 3
    artifact_terms = _artifact_relevance_terms(artifact)
    overlap = agent_terms & artifact_terms
    if not overlap:
        return 0
    return 2 + min(len(overlap), 3) / 10


def _same_branch_team_artifact(agent_spec: AgentSpec, artifact: Artifact) -> bool:
    return artifact.visibility == VisibilityScope.TEAM and _same_branch(
        agent_spec.branch, artifact.branch
    )


def _high_confidence_verified_knowledge(artifact: Artifact) -> bool:
    return (
        artifact.visibility == VisibilityScope.PUBLIC_VERIFIED
        and artifact.status == ArtifactStatus.VERIFIED
        and artifact.artifact_type in {ArtifactType.CLAIM, ArtifactType.FORECAST}
        and artifact.confidence is not None
        and artifact.confidence >= HIGH_CONFIDENCE_THRESHOLD
    )


def _linked_to_selected_artifact(
    artifact: Artifact,
    seed_ids: set[UUID],
    seed_artifacts: dict[UUID, Artifact],
) -> bool:
    artifact_links = _linked_artifact_ids(artifact)
    if artifact_links & seed_ids:
        return True
    return any(artifact.id in _linked_artifact_ids(seed) for seed in seed_artifacts.values())


def _linked_artifact_ids(artifact: Artifact) -> set[UUID]:
    linked: set[UUID] = set()
    for field in LINK_FIELDS:
        linked.update(getattr(artifact, field))
    return linked


def _status_visible_to_agent(agent_spec: AgentSpec, artifact: Artifact) -> bool:
    if artifact.status in {ArtifactStatus.REJECTED, ArtifactStatus.DISPUTED}:
        return _has_privileged_visibility(agent_spec)
    return True


def _has_privileged_visibility(agent_spec: AgentSpec) -> bool:
    text = " ".join(
        [
            agent_spec.name,
            agent_spec.role_template,
            agent_spec.branch,
            agent_spec.domain or "",
            agent_spec.objective,
        ]
    ).lower()
    return any(marker in text for marker in PRIVILEGED_ROLE_MARKERS)


def _same_branch(agent_branch: str, artifact_branch: str | None) -> bool:
    return artifact_branch is not None and _normalize_term(agent_branch) == _normalize_term(
        artifact_branch
    )


def _artifact_relevance_terms(artifact: Artifact) -> set[str]:
    terms: set[str] = set()
    if artifact.branch:
        terms.update(_branch_terms(artifact.branch))
    for tag in artifact.tags:
        terms.update(_term_variants(tag))
    return terms


def _branch_terms(branch: str) -> set[str]:
    terms = _term_variants(branch)
    leaf = branch.replace(":", "/").rsplit("/", 1)[-1]
    if leaf and leaf not in {"market", "macro", "research", "synthesis", "trust"}:
        terms.update(_term_variants(leaf))
    return terms


def _term_variants(value: str) -> set[str]:
    normalized = _normalize_term(value)
    variants = {normalized}
    variants.add(normalized.replace("/", ":"))
    variants.add(normalized.replace(":", "/"))
    return {variant for variant in variants if variant}


def _normalize_term(value: str) -> str:
    return value.strip().lower().replace(" ", "_")
