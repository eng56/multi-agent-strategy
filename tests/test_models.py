from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.common.models import EventEnvelope, EventType, ModelPolicy, RoleModelPolicy, RunRequest


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
