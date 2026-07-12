from src.api.main import scrub_jsonable


def test_scrub_jsonable_removes_raw_control_characters() -> None:
    value = {"items": [{"text": "valid\x00bad\x1ftext", "nested": ["ok\x08too"]}]}

    scrubbed = scrub_jsonable(value)

    assert scrubbed == {"items": [{"text": "valid bad text", "nested": ["ok too"]}]}
