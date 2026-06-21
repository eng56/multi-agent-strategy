from src.agents.workflow import HANDLERS
from src.common.models import EventType


def test_every_workflow_stage_has_an_event_handler() -> None:
    assert HANDLERS["planner-agent"] == {EventType.RUN_CREATED: HANDLERS["planner-agent"][EventType.RUN_CREATED]}
    assert EventType.TASK_CREATED in HANDLERS["tool-runner"]
    assert EventType.OBSERVATION_CREATED in HANDLERS["worker-agents"]
    assert EventType.CLAIM_CREATED in HANDLERS["verifier-agent"]
    assert EventType.CLAIM_VERIFIED in HANDLERS["aggregator-agent"]
    assert EventType.FINAL_CREATED in HANDLERS["judge-agent"]
