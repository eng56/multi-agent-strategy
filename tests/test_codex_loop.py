from __future__ import annotations

import copy
import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


def load_codex_loop() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / ".codex-loop" / "codex_loop.py"
    spec = importlib.util.spec_from_file_location("codex_loop_under_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


codex_loop = load_codex_loop()


def config_with_context(context_file: str | None) -> dict:
    config = copy.deepcopy(codex_loop.DEFAULT_CONFIG)
    if context_file is not None:
        config["context"] = {
            "files": [context_file],
            "prepend_to_prompt": True,
        }
    return config


def write_prompt(repo_root: Path, body: str = "Task prompt body.\n") -> Path:
    prompt_dir = repo_root / ".codex-loop" / "prompts"
    prompt_dir.mkdir(parents=True)
    prompt_path = prompt_dir / "001-test.md"
    prompt_path.write_text(body, encoding="utf-8")
    return prompt_path


def write_context(repo_root: Path, body: str = "# PAPER_CONTEXT\n\nPaper context body.\n") -> Path:
    context_dir = repo_root / ".codex-loop" / "context"
    context_dir.mkdir(parents=True)
    context_path = context_dir / "PAPER_CONTEXT.md"
    context_path.write_text(body, encoding="utf-8")
    return context_path


def test_dry_run_shows_context_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    repo_root = tmp_path
    loop_dir = repo_root / ".codex-loop"
    prompt_path = write_prompt(repo_root)
    context_path = write_context(repo_root)
    config = config_with_context(".codex-loop/context/PAPER_CONTEXT.md")
    config_path = loop_dir / "config.example.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("context: {}\n", encoding="utf-8")

    monkeypatch.setattr(codex_loop, "default_branch", lambda _repo_root: "main")

    codex_loop.print_dry_run(
        prompt_path,
        repo_root,
        loop_dir,
        config,
        config_path,
        "code",
        False,
    )

    output = capsys.readouterr().out
    assert "Context files:" in output
    assert codex_loop.repo_relative(context_path, repo_root) in output


def test_composed_prompt_includes_paper_context_before_task_prompt(tmp_path: Path) -> None:
    repo_root = tmp_path
    context_path = write_context(repo_root, "# PAPER_CONTEXT\n\nResearch framing.\n")
    prompt_body = "Implement the task.\n"
    config = config_with_context(".codex-loop/context/PAPER_CONTEXT.md")

    entries = codex_loop.load_context_entries(config, repo_root)
    composed = codex_loop.compose_prompt_with_context(prompt_body, entries, repo_root)

    assert f"### Context file: {codex_loop.repo_relative(context_path, repo_root)}" in composed
    assert "BEGIN GLOBAL PROJECT CONTEXT" in composed
    assert "END GLOBAL PROJECT CONTEXT" in composed
    assert "BEGIN TASK PROMPT" in composed
    assert "END TASK PROMPT" in composed
    assert composed.index("Research framing.") < composed.index("Implement the task.")


def test_missing_context_file_fails_clearly(tmp_path: Path) -> None:
    config = config_with_context(".codex-loop/context/MISSING.md")

    with pytest.raises(codex_loop.LoopError, match="Context file not found"):
        codex_loop.load_context_entries(config, tmp_path)


def test_run_one_preserves_original_prompt_and_writes_composed_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path
    loop_dir = repo_root / ".codex-loop"
    prompt_path = write_prompt(repo_root, "Task prompt body.\n")
    write_context(repo_root, "# PAPER_CONTEXT\n\nPaper context body.\n")
    config = config_with_context(".codex-loop/context/PAPER_CONTEXT.md")
    config["git"]["create_branch"] = False
    config["git"]["commit_on_success"] = False
    config["git"]["push_on_success"] = False
    config["git"]["create_pr_on_success"] = False
    config["git"]["merge_pr_on_success"] = False
    state: dict = {"version": 1, "prompts": {}}
    state_path = loop_dir / "state.json"
    seen: dict[str, str] = {}

    monkeypatch.setattr(codex_loop, "ensure_clean_worktree", lambda _repo_root: None)
    monkeypatch.setattr(
        codex_loop,
        "git",
        lambda _repo_root, _arguments, check=True: subprocess.CompletedProcess(
            ["git"], 0, stdout="main\n", stderr=""
        ),
    )

    def fake_execute_codex(
        _command,
        _pass_prompt_as,
        prompt_text,
        _repo_root,
        _environment,
        log_path,
        _secrets,
        _redact_enabled,
    ) -> int:
        seen["prompt_text"] = prompt_text
        log_path.write_text("codex ok\n", encoding="utf-8")
        return 0

    def fake_execute_checks(
        checks,
        _repo_root,
        _environment,
        log_path,
        _secrets,
        _redact_enabled,
    ) -> list[dict]:
        log_path.write_text("checks ok\n", encoding="utf-8")
        return [{"command": " ".join(command), "exit_code": 0} for command in checks]

    def fake_capture_git_artifacts(
        _repo_root,
        diff_path,
        status_path,
        _secrets,
        _redact_enabled,
    ) -> None:
        diff_path.write_text("", encoding="utf-8")
        status_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(codex_loop, "execute_codex", fake_execute_codex)
    monkeypatch.setattr(codex_loop, "execute_checks", fake_execute_checks)
    monkeypatch.setattr(codex_loop, "capture_git_artifacts", fake_capture_git_artifacts)

    status = codex_loop.run_one(
        prompt_path,
        repo_root,
        loop_dir,
        config,
        state,
        state_path,
        "code",
        False,
    )

    assert status == "READY_FOR_REVIEW"
    run_dirs = list((loop_dir / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "prompt.md").read_text(encoding="utf-8") == "Task prompt body.\n"
    composed = (run_dir / "composed_prompt.md").read_text(encoding="utf-8")
    assert "Paper context body." in composed
    assert "Task prompt body." in composed
    assert composed.index("Paper context body.") < composed.index("Task prompt body.")
    assert "BEGIN GLOBAL PROJECT CONTEXT" in seen["prompt_text"]
    assert "BEGIN TASK PROMPT" in seen["prompt_text"]


def test_config_with_no_context_still_composes_original_prompt(tmp_path: Path) -> None:
    repo_root = tmp_path
    config = copy.deepcopy(codex_loop.DEFAULT_CONFIG)
    config.pop("context", None)

    entries = codex_loop.load_context_entries(config, repo_root)
    composed = codex_loop.compose_prompt_with_context("Task only.\n", entries, repo_root)

    assert entries == []
    assert composed == "Task only.\n"
