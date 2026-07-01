import asyncio
from uuid import uuid4

from src.common.models import ResearchTask
from src.integrations.upstash import UpstashBlackboard


class FakeBlackboard(UpstashBlackboard):
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    async def command(self, *parts: object) -> object:
        command, key = parts[0], parts[1]
        if command == "SMEMBERS":
            return ["valid", "invalid"]
        if command == "GET":
            return self.values.get(str(key))
        raise AssertionError(f"unexpected command {parts}")


def test_list_models_skips_invalid_records() -> None:
    run_id = uuid4()
    valid = ResearchTask(run_id=run_id, title="title", question="question", tool="web_search")
    blackboard = FakeBlackboard(
        {
            f"run:{run_id}:tasks:valid": valid.model_dump_json(),
            f"run:{run_id}:tasks:invalid": '{"tool":"not-supported"}',
        }
    )

    tasks = asyncio.run(blackboard.list_models(run_id, "tasks", ResearchTask))

    assert tasks == [valid]
