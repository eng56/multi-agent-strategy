import asyncio
from types import SimpleNamespace

from src.agents.workflow import HANDLERS, plan
from src.common.models import (
    Budget,
    EventEnvelope,
    EventType,
    ModelPolicy,
    RoleModelPolicy,
    Run,
    RunStatus,
    ToolBudget,
)


def role_policy(cap: float = 0.1, protected: float = 0) -> RoleModelPolicy:
    return RoleModelPolicy(
        model="provider/model",
        cap_usd=cap,
        max_call_cost_usd=min(cap, 0.05),
        protected_usd=protected,
    )


def model_policy() -> ModelPolicy:
    return ModelPolicy(
        planner=role_policy(),
        utility=role_policy(),
        research=role_policy(0.3),
        verifier=role_policy(0.2),
        aggregator=role_policy(0.25, 0.05),
        judge=role_policy(0.1, 0.05),
    )


def test_every_workflow_stage_has_an_event_handler() -> None:
    assert HANDLERS["planner-agent"] == {EventType.RUN_CREATED: HANDLERS["planner-agent"][EventType.RUN_CREATED]}
    assert EventType.TASK_CREATED in HANDLERS["tool-runner"]
    assert EventType.OBSERVATION_CREATED in HANDLERS["worker-agents"]
    assert EventType.CLAIM_CREATED in HANDLERS["verifier-agent"]
    assert EventType.CLAIM_VERIFIED in HANDLERS["aggregator-agent"]
    assert EventType.FINAL_CREATED in HANDLERS["judge-agent"]


def test_planner_stops_run_when_model_returns_no_tasks() -> None:
    run = Run(question="Should I buy SAP?", models=model_policy(), budget=Budget(limit_usd=1, tools=ToolBudget()))
    published = []

    class Blackboard:
        async def get_run(self, _run_id):
            return run

        async def put_run(self, value):
            self.run = value

        async def put_task(self, value):
            self.task = value


    class LLM:
        async def json(self, *_args, **_kwargs):
            return {"tasks": []}

    runtime = SimpleNamespace(
        blackboard=Blackboard(),
        llm=LLM(),
        settings=SimpleNamespace(runtime_topic="agent-runtime"),
        events=SimpleNamespace(publish=lambda topic, event: published.append((topic, event))),
    )

    asyncio.run(plan(runtime, EventEnvelope(type=EventType.RUN_CREATED, run_id=run.id, producer="test")))

    assert not published
    assert runtime.blackboard.run.status == RunStatus.FAILED
    assert "planner produced no usable tasks" in runtime.blackboard.run.failure_reason
