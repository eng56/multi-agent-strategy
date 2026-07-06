#!/usr/bin/env python3
"""Run queued markdown prompts through Codex CLI, one at a time."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUSES = {
    "PENDING",
    "RUNNING_CODEX",
    "CODEX_FAILED",
    "TESTING",
    "FAILED_CHECKS",
    "READY_FOR_REVIEW",
    "COMPLETED_MANUALLY",
}
REVIEW_GATE_STATUSES = {"RUNNING_CODEX", "TESTING", "READY_FOR_REVIEW"}
DEFAULT_CONFIG: dict[str, Any] = {
    "codex": {
        "command": ["codex", "exec"],
        "args": ["--full-auto"],
        "pass_prompt_as": "stdin",
    },
    "checks": [["ruff", "check", "."], ["pytest"]],
    "git": {
        "create_branch": True,
        "commit_on_success": False,
        "push_on_success": False,
    },
    "security": {
        "env_file_by_mode": {
            "code": None,
            "staging": ".codex-loop/secrets/staging.env",
            "prod": ".codex-loop/secrets/prod.env",
        },
        "redact_env_values_in_logs": True,
    },
}


class LoopError(RuntimeError):
    """An expected, user-actionable orchestration error."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value[0] in {'"', "'", "[", "{"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                return value[1:-1]
            raise LoopError(f"Unsupported YAML value: {value}")
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def parse_minimal_yaml(text: str) -> dict[str, Any]:
    """Parse the mapping/list subset used by the example config and frontmatter."""
    lines: list[tuple[int, str, int]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if "\t" in raw_line[: len(raw_line) - len(raw_line.lstrip())]:
            raise LoopError(f"Tabs are not supported in YAML indentation (line {line_number})")
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip(), line_number))

    if not lines:
        return {}

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        is_list = lines[index][1].startswith("- ")
        container: Any = [] if is_list else {}

        while index < len(lines):
            current_indent, content, line_number = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise LoopError(f"Unexpected indentation in YAML at line {line_number}")

            if is_list:
                if not content.startswith("- "):
                    break
                item = content[2:].strip()
                if not item:
                    if index + 1 >= len(lines) or lines[index + 1][0] <= indent:
                        container.append(None)
                        index += 1
                    else:
                        child_indent = lines[index + 1][0]
                        child, index = parse_block(index + 1, child_indent)
                        container.append(child)
                else:
                    container.append(parse_scalar(item))
                    index += 1
                continue

            if content.startswith("- "):
                break
            if ":" not in content:
                raise LoopError(f"Expected a YAML mapping at line {line_number}")
            key, raw_value = content.split(":", 1)
            key = key.strip()
            if not key:
                raise LoopError(f"Empty YAML key at line {line_number}")
            raw_value = raw_value.strip()
            if raw_value:
                container[key] = parse_scalar(raw_value)
                index += 1
            elif index + 1 < len(lines) and lines[index + 1][0] > indent:
                child_indent = lines[index + 1][0]
                child, index = parse_block(index + 1, child_indent)
                container[key] = child
            else:
                container[key] = {}
                index += 1

        return container, index

    parsed, next_index = parse_block(0, lines[0][0])
    if next_index != len(lines) or not isinstance(parsed, dict):
        raise LoopError("The YAML document must contain a top-level mapping")
    return parsed


def load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return parse_minimal_yaml(text)

    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise LoopError(f"Invalid YAML in {path}: {error}") from error
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise LoopError(f"{path} must contain a top-level mapping")
    return value


def find_repo_root(script_path: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(script_path.parent), "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise LoopError(f"Could not find git repository: {result.stderr.strip()}")
    return Path(result.stdout.strip()).resolve()


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return str(path.resolve())


def validate_command_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise LoopError(f"{label} must be a non-empty list of strings")
    return value


def validate_checks(value: Any, label: str = "checks") -> list[list[str]]:
    if not isinstance(value, list):
        raise LoopError(f"{label} must be a list of command lists")
    checks: list[list[str]] = []
    for index, command in enumerate(value):
        checks.append(validate_command_list(command, f"{label}[{index}]"))
    return checks


def load_config(loop_dir: Path, explicit_path: str | None) -> tuple[dict[str, Any], Path]:
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
    else:
        local_path = loop_dir / "config.yaml"
        path = local_path if local_path.exists() else loop_dir / "config.example.yaml"
    if not path.is_file():
        raise LoopError(f"Config file not found: {path}")

    config = deep_merge(DEFAULT_CONFIG, load_yaml(path))
    codex = config.get("codex")
    git_config = config.get("git")
    security = config.get("security")
    if not isinstance(codex, dict) or not isinstance(git_config, dict):
        raise LoopError("Config sections 'codex' and 'git' must be mappings")
    if not isinstance(security, dict):
        raise LoopError("Config section 'security' must be a mapping")

    validate_command_list(codex.get("command"), "codex.command")
    args = codex.get("args", [])
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise LoopError("codex.args must be a list of strings")
    if codex.get("pass_prompt_as") not in {"stdin", "argument"}:
        raise LoopError("codex.pass_prompt_as must be 'stdin' or 'argument'")
    validate_checks(config.get("checks"))
    if git_config.get("commit_on_success") is not False:
        raise LoopError("git.commit_on_success must remain false in v1")
    if git_config.get("push_on_success") is not False:
        raise LoopError("git.push_on_success must remain false in v1")
    if not isinstance(security.get("env_file_by_mode"), dict):
        raise LoopError("security.env_file_by_mode must be a mapping")
    if security.get("redact_env_values_in_logs") is not True:
        raise LoopError("security.redact_env_values_in_logs must remain true")
    return config, path


def parse_prompt(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}, text
    lines = text.splitlines(keepends=True)
    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing_index is None:
        raise LoopError(f"Prompt frontmatter is missing its closing delimiter: {path}")
    frontmatter_text = "".join(lines[1:closing_index])
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        metadata = parse_minimal_yaml(frontmatter_text)
    else:
        try:
            metadata = yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError as error:
            raise LoopError(f"Invalid prompt frontmatter in {path}: {error}") from error
    if not isinstance(metadata, dict):
        raise LoopError(f"Prompt frontmatter must be a mapping: {path}")
    return metadata, "".join(lines[closing_index + 1 :]).lstrip("\n")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "prompts": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise LoopError(f"Could not read state file {path}: {error}") from error
    if not isinstance(state, dict) or not isinstance(state.get("prompts"), dict):
        raise LoopError(f"Invalid state file structure: {path}")
    return state


def set_status(
    state: dict[str, Any],
    state_path: Path,
    prompt_key: str,
    status: str,
    run_dir: str | None = None,
) -> None:
    if status not in STATUSES:
        raise LoopError(f"Unknown prompt status: {status}")
    prompts = state.setdefault("prompts", {})
    record = prompts.setdefault(prompt_key, {"history": []})
    timestamp = isoformat()
    record["status"] = status
    record["updated_at"] = timestamp
    if run_dir is not None:
        record["last_run"] = run_dir
    record.setdefault("history", []).append({"status": status, "at": timestamp})
    atomic_write_json(state_path, state)


def select_prompt(
    prompts_dir: Path,
    state: dict[str, Any],
    repo_root: Path,
    explicit_prompt: str | None,
    processed: set[str],
) -> Path | None:
    if explicit_prompt:
        path = Path(explicit_prompt)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            raise LoopError(f"Prompt file not found: {path}")
        if path.suffix.lower() != ".md":
            raise LoopError(f"Prompt must be a markdown file: {path}")
        return path

    prompt_paths = sorted(prompts_dir.glob("*.md"), key=lambda item: item.name)
    if not prompt_paths:
        raise LoopError(f"No markdown prompts found in {prompts_dir}")

    for path in prompt_paths:
        key = repo_relative(path, repo_root)
        if key in processed:
            continue
        status = state.get("prompts", {}).get(key, {}).get("status")
        if status == "COMPLETED_MANUALLY":
            continue
        if status in REVIEW_GATE_STATUSES:
            raise LoopError(
                f"{key} is {status}; review or recover it before starting another prompt"
            )
        return path
    return None


def safe_stem(path: Path) -> str:
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", path.stem).strip(".-_").lower()
    if not stem:
        raise LoopError(f"Prompt filename does not produce a safe branch name: {path.name}")
    return stem[:60]


def build_names(prompt_path: Path, started_at: datetime) -> tuple[str, str]:
    stem = safe_stem(prompt_path)
    run_timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    branch_timestamp = started_at.strftime("%Y%m%d%H%M%S")
    return f"{stem}-{run_timestamp}", f"codex-loop/{stem}-{branch_timestamp}"


def git(repo_root: Path, arguments: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise LoopError(f"git {shlex.join(arguments)} failed: {detail}")
    return result


def ensure_clean_worktree(repo_root: Path) -> None:
    status = git(repo_root, ["status", "--porcelain", "--untracked-files=all"]).stdout
    if status.strip():
        raise LoopError(
            "Git worktree is not clean. Commit, stash, or otherwise resolve existing changes "
            "before running Codex."
        )


def load_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise LoopError(f"Environment file not found: {path}")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise LoopError(f"Invalid environment assignment in {path} at line {line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise LoopError(f"Invalid environment variable name in {path} at line {line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def env_path_for_mode(config: dict[str, Any], mode: str, repo_root: Path) -> Path | None:
    mapping = config["security"]["env_file_by_mode"]
    if mode not in mapping:
        raise LoopError(f"No environment file configuration exists for mode '{mode}'")
    raw_path = mapping[mode]
    if raw_path is None:
        return None
    if not isinstance(raw_path, str):
        raise LoopError(f"Environment path for mode '{mode}' must be a string or null")
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def redact(text: str, secret_values: list[str], enabled: bool) -> str:
    if not enabled:
        return text
    for value in sorted({item for item in secret_values if item}, key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    return text


def display_command(command: list[str], pass_prompt_as: str) -> str:
    shown = list(command)
    if pass_prompt_as == "argument":
        shown.append("<PROMPT>")
    return shlex.join(shown) + (" < prompt.md" if pass_prompt_as == "stdin" else "")


def execute_codex(
    command: list[str],
    pass_prompt_as: str,
    prompt_body: str,
    repo_root: Path,
    environment: dict[str, str],
    log_path: Path,
    secrets: list[str],
    redact_enabled: bool,
) -> int:
    actual_command = list(command)
    input_text: str | None = None
    if pass_prompt_as == "stdin":
        input_text = prompt_body
    else:
        actual_command.append(prompt_body)

    try:
        result = subprocess.run(
            actual_command,
            cwd=repo_root,
            env=environment,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        output = result.stdout or ""
        exit_code = result.returncode
    except OSError as error:
        output = f"Could not start Codex CLI: {error}\n"
        exit_code = 127

    content = (
        f"$ {display_command(command, pass_prompt_as)}\n\n"
        f"{redact(output, secrets, redact_enabled)}"
    )
    log_path.write_text(content, encoding="utf-8")
    return exit_code


def execute_checks(
    checks: list[list[str]],
    repo_root: Path,
    environment: dict[str, str],
    log_path: Path,
    secrets: list[str],
    redact_enabled: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    sections: list[str] = []
    for command in checks:
        shown = shlex.join(command)
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            output = completed.stdout or ""
            exit_code = completed.returncode
        except OSError as error:
            output = f"Could not start check: {error}\n"
            exit_code = 127
        sections.append(
            redact(
                f"$ {shown}\nexit_code: {exit_code}\n\n{output}\n",
                secrets,
                redact_enabled,
            )
        )
        results.append({"command": shown, "exit_code": exit_code})
    log_path.write_text("\n".join(sections), encoding="utf-8")
    return results


def skip_untracked_diff(name: str) -> bool:
    path = Path(name)
    parts = path.parts
    basename = path.name.lower()
    if parts[:2] == (".codex-loop", "runs"):
        return True
    if parts[:2] == (".codex-loop", "secrets"):
        return True
    if name in {".codex-loop/state.json", ".codex-loop/config.yaml"}:
        return True
    return basename == ".env" or basename.startswith(".env.") or basename.endswith(".env")


def capture_git_artifacts(
    repo_root: Path,
    diff_path: Path,
    status_path: Path,
    secrets: list[str],
    redact_enabled: bool,
) -> None:
    status = git(
        repo_root,
        ["status", "--short", "--untracked-files=all"],
        check=False,
    )
    status_text = status.stdout + (status.stderr if status.returncode else "")
    status_path.write_text(redact(status_text, secrets, redact_enabled), encoding="utf-8")

    tracked = git(repo_root, ["diff", "--binary", "HEAD", "--"], check=False)
    diff_parts = [tracked.stdout]
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if untracked.returncode == 0:
        for raw_name in untracked.stdout.split(b"\0"):
            if not raw_name:
                continue
            name = raw_name.decode("utf-8", errors="surrogateescape")
            if skip_untracked_diff(name):
                continue
            item_diff = subprocess.run(
                ["git", "diff", "--no-index", "--binary", "--", "/dev/null", name],
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            diff_parts.append(item_diff.stdout.decode("utf-8", errors="replace"))
    else:
        diff_parts.append(untracked.stderr.decode("utf-8", errors="replace"))
    diff_path.write_text(
        redact("".join(diff_parts), secrets, redact_enabled),
        encoding="utf-8",
    )


def unique_run_dir(runs_dir: Path, base_name: str) -> Path:
    candidate = runs_dir / base_name
    counter = 2
    while candidate.exists():
        candidate = runs_dir / f"{base_name}-{counter}"
        counter += 1
    return candidate


def prepare_environment(
    config: dict[str, Any],
    metadata: dict[str, Any],
    mode: str,
    repo_root: Path,
) -> tuple[dict[str, str], list[str], Path | None]:
    env_file = env_path_for_mode(config, mode, repo_root)
    if env_file is not None and metadata.get("allow_secrets") is False:
        raise LoopError("This prompt sets allow_secrets: false and cannot run in a secrets mode")
    loaded = load_env_file(env_file) if env_file is not None else {}
    environment = os.environ.copy()
    environment.update(loaded)
    return environment, list(loaded.values()), env_file


def run_one(
    prompt_path: Path,
    repo_root: Path,
    loop_dir: Path,
    config: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    mode: str,
) -> str:
    ensure_clean_worktree(repo_root)
    metadata, prompt_body = parse_prompt(prompt_path)
    checks = (
        validate_checks(metadata["checks"], "prompt checks")
        if "checks" in metadata
        else validate_checks(config["checks"])
    )
    environment, secrets, _ = prepare_environment(config, metadata, mode, repo_root)
    redact_enabled = bool(config["security"].get("redact_env_values_in_logs", True))

    started_at = utc_now()
    run_name, branch_name = build_names(prompt_path, started_at)
    run_dir = unique_run_dir(loop_dir / "runs", run_name)
    run_dir.mkdir(parents=True)
    copied_prompt = run_dir / "prompt.md"
    shutil.copy2(prompt_path, copied_prompt)

    prompt_key = repo_relative(prompt_path, repo_root)
    run_key = repo_relative(run_dir, repo_root)
    codex_log = run_dir / "codex.log"
    checks_log = run_dir / "checks.log"
    diff_path = run_dir / "git.diff"
    status_path = run_dir / "git.status"
    result_path = run_dir / "result.json"
    result: dict[str, Any] = {
        "prompt": prompt_key,
        "branch": branch_name,
        "status": "PENDING",
        "started_at": isoformat(started_at),
        "finished_at": None,
        "codex_exit_code": None,
        "checks": [],
        "run_dir": run_key,
        "diff_path": repo_relative(diff_path, repo_root),
        "logs_path": repo_relative(codex_log, repo_root),
    }
    atomic_write_json(result_path, result)
    set_status(state, state_path, prompt_key, "PENDING", run_key)

    try:
        if config["git"].get("create_branch", True):
            git(repo_root, ["switch", "-c", branch_name])
        else:
            branch_name = git(repo_root, ["branch", "--show-current"]).stdout.strip()
            result["branch"] = branch_name

        set_status(state, state_path, prompt_key, "RUNNING_CODEX", run_key)
        result["status"] = "RUNNING_CODEX"
        atomic_write_json(result_path, result)

        codex_config = config["codex"]
        command = validate_command_list(codex_config["command"], "codex.command")
        command += list(codex_config.get("args", []))
        codex_exit_code = execute_codex(
            command,
            codex_config["pass_prompt_as"],
            prompt_body,
            repo_root,
            environment,
            codex_log,
            secrets,
            redact_enabled,
        )
        result["codex_exit_code"] = codex_exit_code
        if codex_exit_code != 0:
            result["status"] = "CODEX_FAILED"
            set_status(state, state_path, prompt_key, "CODEX_FAILED", run_key)
            checks_log.write_text("Checks skipped because Codex failed.\n", encoding="utf-8")
            return "CODEX_FAILED"

        result["status"] = "TESTING"
        set_status(state, state_path, prompt_key, "TESTING", run_key)
        atomic_write_json(result_path, result)
        check_results = execute_checks(
            checks,
            repo_root,
            environment,
            checks_log,
            secrets,
            redact_enabled,
        )
        result["checks"] = check_results
        if any(item["exit_code"] != 0 for item in check_results):
            result["status"] = "FAILED_CHECKS"
            set_status(state, state_path, prompt_key, "FAILED_CHECKS", run_key)
            return "FAILED_CHECKS"

        result["status"] = "READY_FOR_REVIEW"
        set_status(state, state_path, prompt_key, "READY_FOR_REVIEW", run_key)
        return "READY_FOR_REVIEW"
    except LoopError as error:
        codex_log.write_text(f"Orchestration error: {error}\n", encoding="utf-8")
        checks_log.write_text("Checks skipped because orchestration failed.\n", encoding="utf-8")
        result["status"] = "CODEX_FAILED"
        set_status(state, state_path, prompt_key, "CODEX_FAILED", run_key)
        return "CODEX_FAILED"
    finally:
        capture_git_artifacts(repo_root, diff_path, status_path, secrets, redact_enabled)
        result["finished_at"] = isoformat()
        atomic_write_json(result_path, result)
        print(f"Run directory: {run_key}")
        print(f"Status: {result['status']}")


def print_dry_run(
    prompt_path: Path,
    repo_root: Path,
    loop_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    mode: str,
) -> None:
    metadata, _ = parse_prompt(prompt_path)
    checks = (
        validate_checks(metadata["checks"], "prompt checks")
        if "checks" in metadata
        else validate_checks(config["checks"])
    )
    env_file = env_path_for_mode(config, mode, repo_root)
    if env_file is not None and metadata.get("allow_secrets") is False:
        raise LoopError("This prompt sets allow_secrets: false and cannot run in a secrets mode")
    _, branch_name = build_names(prompt_path, utc_now())
    codex_config = config["codex"]
    command = validate_command_list(codex_config["command"], "codex.command")
    command += list(codex_config.get("args", []))

    print("DRY RUN — no files, git state, Codex process, or checks will be changed/run.")
    print(f"Config: {repo_relative(config_path, repo_root)}")
    print(f"Prompt: {repo_relative(prompt_path, repo_root)}")
    print(f"Branch: {branch_name}")
    print(f"Mode: {mode}")
    print(
        "Environment file: "
        + (repo_relative(env_file, repo_root) if env_file is not None else "none")
    )
    print(f"Codex command: {display_command(command, codex_config['pass_prompt_as'])}")
    print("Checks:")
    if checks:
        for check in checks:
            print(f"  - {shlex.join(check)}")
    else:
        print("  - none")
    print(f"Run root: {repo_relative(loop_dir / 'runs', repo_root)}")


def mark_completed(
    prompt_argument: str,
    repo_root: Path,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    path = Path(prompt_argument)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    prompt_key = repo_relative(path, repo_root)
    existing = state.get("prompts", {}).get(prompt_key, {})
    set_status(state, state_path, prompt_key, "COMPLETED_MANUALLY", existing.get("last_run"))
    print(f"Marked {prompt_key} as COMPLETED_MANUALLY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local Codex CLI prompts one at a time with inspectable artifacts."
    )
    parser.add_argument("--dry-run", action="store_true", help="Show actions without changing state")
    parser.add_argument("--prompt", help="Run a specific markdown prompt")
    parser.add_argument("--config", help="Use a specific YAML config file")
    parser.add_argument("--mode", choices=("code", "staging", "prod"), default="code")
    parser.add_argument("--allow-prod", action="store_true", help="Required with --mode prod")
    parser.add_argument(
        "--continue-on-success",
        action="store_true",
        help="Continue only after success and only while the worktree remains clean",
    )
    parser.add_argument(
        "--mark-completed",
        metavar="PROMPT",
        help="Mark a prompt COMPLETED_MANUALLY without running Codex",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "prod" and not args.allow_prod:
        print("error: --mode prod requires --allow-prod", file=sys.stderr)
        return 2
    if args.prompt and args.mark_completed:
        print("error: --prompt and --mark-completed cannot be combined", file=sys.stderr)
        return 2

    try:
        script_path = Path(__file__).resolve()
        loop_dir = script_path.parent
        repo_root = find_repo_root(script_path)
        config, config_path = load_config(loop_dir, args.config)
        state_path = loop_dir / "state.json"
        state = load_state(state_path)

        if args.mark_completed:
            if args.dry_run:
                print(f"Would mark {args.mark_completed} as COMPLETED_MANUALLY")
            else:
                mark_completed(args.mark_completed, repo_root, state, state_path)
            return 0

        processed: set[str] = set()
        explicit_prompt = args.prompt
        while True:
            prompt_path = select_prompt(
                loop_dir / "prompts",
                state,
                repo_root,
                explicit_prompt,
                processed,
            )
            if prompt_path is None:
                print("No pending prompts.")
                return 0

            if args.dry_run:
                print_dry_run(
                    prompt_path,
                    repo_root,
                    loop_dir,
                    config,
                    config_path,
                    args.mode,
                )
                return 0

            prompt_key = repo_relative(prompt_path, repo_root)
            status = run_one(
                prompt_path,
                repo_root,
                loop_dir,
                config,
                state,
                state_path,
                args.mode,
            )
            processed.add(prompt_key)
            if status != "READY_FOR_REVIEW" or not args.continue_on_success:
                return 0 if status == "READY_FOR_REVIEW" else 1

            explicit_prompt = None
            if git(repo_root, ["status", "--porcelain", "--untracked-files=all"]).stdout.strip():
                print(
                    "Stopped after success: the worktree has changes to review, so another task "
                    "cannot safely receive its own branch."
                )
                return 0
    except LoopError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
