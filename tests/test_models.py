from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.common.models import (
    AgentSpec,
    Artifact,
    ArtifactType,
    EventEnvelope,
    EventType,
    ModelPolicy,
    OrganizationPlan,
    PrincipalAction,
    PrincipalActionType,
    RoleModelPolicy,
    RunRequest,
    VisibilityScope,
)


def models(cap: float = 1) -> ModelPolicy:
    def item(protected: float = 0) -> RoleModelPolicy:
        return RoleModelPolicy(
            model="any/provider-model",
            cap_usd=cap,
            protected_usd=protected,
            max_call_cost_usd=protected or 0.25,
        )

    return ModelPolicy(
        planner=item(),
        utility=item(),
        research=item(),
        verifier=item(),
        aggregator=item(1),
        judge=item(1),
    )


def test_provider_keys_are_not_coordination_event_payload() -> None:
    event = EventEnvelope(type=EventType.RUN_CREATED, run_id=uuid4(), producer="api")
    assert "api_key" not in event.model_dump_json()


def test_role_caps_cannot_exceed_shared_budget() -> None:
    with pytest.raises(ValidationError, match="role caps"):
        RunRequest(question="research", llm_budget_usd=5, models=models())


def test_control_loop_models_serialize_typed_state() -> None:
    run_id = uuid4()
    root = AgentSpec(
        run_id=run_id,
        name="root principal",
        role_template="principal_policy",
        branch="root",
        objective="Allocate budgeted research actions",
        visibility_scope=VisibilityScope.PUBLIC_VERIFIED,
        can_request_spawn=True,
    )
    action = PrincipalAction(
        run_id=run_id,
        action_type=PrincipalActionType.REQUEST_SKEPTIC_REVIEW,
        reason="High-confidence forecast needs adversarial review.",
        required_role="skeptic_agent",
    )
    artifact = Artifact(
        run_id=run_id,
        artifact_type=ArtifactType.PRINCIPAL_ACTION,
        created_by_agent_id=root.id,
        text_or_summary=action.reason,
        supports_artifact_ids=[action.id],
    )
    organization = OrganizationPlan(
        run_id=run_id,
        root_agent_id=root.id,
        agent_specs=[root],
        branches=["root"],
        manager_agents=[root.id],
    )

    payload = {
        "action": action.model_dump(mode="json"),
        "artifact": artifact.model_dump(mode="json"),
        "organization": organization.model_dump(mode="json"),
    }

    assert payload["action"]["action_type"] == "request_skeptic_review"
    assert payload["artifact"]["artifact_type"] == "principal_action"
    assert payload["organization"]["agent_specs"][0]["role_template"] == "principal_policy"


def test_future_control_loop_events_serialize() -> None:
    event = EventEnvelope(type=EventType.PRINCIPAL_ACTION_CREATED, run_id=uuid4(), producer="test")
    assert event.model_dump(mode="json")["type"] == "principal.action.created"
