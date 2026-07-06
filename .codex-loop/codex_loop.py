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
    "NO_CHANGES",
    "MERGING",
    "MERGED",
    "COMPLETED_MANUALLY",
}
REVIEW_GATE_STATUSES = {"RUNNING_CODEX", "TESTING", "READY_FOR_REVIEW", "MERGING"}
FINAL_STATUSES = {"COMPLETED_MANUALLY", "MERGED", "NO_CHANGES"}
CONTINUE_SUCCESS_STATUSES = {"MERGED", "NO_CHANGES"}
DEFAULT_CONFIG: dict[str, Any] = {
    "codex": {
        "command": ["codex", "exec"],
        "args": ["--full-auto"],
        "pass_prompt_as": "stdin",
    },
    "checks": [["ruff", "check", "."], ["pytest"]],
    "git": {
        "create_branch": True,
        "commit_on_success": True,
        "push_on_success": True,
        "create_pr_on_success": True,
        "merge_pr_on_success": True,
        "delete_branch_on_merge": True,
        "sync_base_branch_after_merge": True,
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
    for key in (
        "commit_on_success",
        "push_on_success",
        "create_pr_on_success",
        "merge_pr_on_success",
        "delete_branch_on_merge",
        "sync_base_branch_after_merge",
    ):
        if key not in git_config:
            raise LoopError(f"git.{key} must be set")
        if not isinstance(git_config[key], bool):
            raise LoopError(f"git.{key} must be a boolean")
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
    queue_order: list[str] | None,
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

    if queue_order is None:
        queue_order = [path.name for path in sorted(prompts_dir.glob("*.md"), key=lambda item: item.name)]
    if not queue_order:
        raise LoopError(f"No markdown prompts found in {prompts_dir}")

    for name in queue_order:
        path = prompts_dir / name
        if path.suffix.lower() != ".md" or not path.is_file():
            continue
        key = repo_relative(path, repo_root)
        if key in processed:
            continue
        status = state.get("prompts", {}).get(key, {}).get("status")
        if status in FINAL_STATUSES:
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


def load_queue_order(loop_dir: Path) -> list[str] | None:
    path = loop_dir / "QUEUE_ORDER.md"
    if not path.is_file():
        return None
    order: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*\d+\.\s+`([^`]+)`", raw_line)
        if match:
            order.append(match.group(1))
    return order or None


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


def git_stdout(repo_root: Path, arguments: list[str]) -> str:
    return git(repo_root, arguments, check=True).stdout.strip()


def ensure_clean_worktree(repo_root: Path) -> None:
    status = git(repo_root, ["status", "--porcelain", "--untracked-files=all"]).stdout
    if status.strip():
        raise LoopError(
            "Git worktree is not clean. Commit, stash, or otherwise resolve existing changes "
            "before running Codex."
        )


def current_branch(repo_root: Path) -> str:
    return git_stdout(repo_root, ["branch", "--show-current"])


def default_branch(repo_root: Path) -> str:
    result = subprocess.run(
        ["gh", "repo", "view", "--json", "defaultBranchRef", "--jq", ".defaultBranchRef.name"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        local = git(repo_root, ["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"], check=False)
        if local.returncode == 0:
            ref = local.stdout.strip()
            if ref:
                return ref.rsplit("/", 1)[-1]
        for candidate in ("main", "master", "trunk"):
            if git(repo_root, ["show-ref", "--verify", "--quiet", f"refs/heads/{candidate}"], check=False).returncode == 0:
                return candidate
            if git(repo_root, ["show-ref", "--verify", "--quiet", f"refs/remotes/origin/{candidate}"], check=False).returncode == 0:
                return candidate
        detail = result.stderr.strip() or result.stdout.strip()
        raise LoopError(f"Could not determine default branch with gh: {detail}")
    branch = result.stdout.strip()
    if not branch:
        local = git(repo_root, ["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"], check=False)
        if local.returncode == 0:
            ref = local.stdout.strip()
            if ref:
                return ref.rsplit("/", 1)[-1]
        for candidate in ("main", "master", "trunk"):
            if git(repo_root, ["show-ref", "--verify", "--quiet", f"refs/heads/{candidate}"], check=False).returncode == 0:
                return candidate
            if git(repo_root, ["show-ref", "--verify", "--quiet", f"refs/remotes/origin/{candidate}"], check=False).returncode == 0:
                return candidate
        raise LoopError("Could not determine default branch with gh")
    return branch


def require_gh() -> None:
    if shutil.which("gh") is None:
        raise LoopError("gh is required for commit/push/PR merge automation")


def gh(repo_root: Path, arguments: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["gh", *arguments],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise LoopError(f"gh {shlex.join(arguments)} failed: {detail}")
    return result


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


def bool_config(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise LoopError(f"{label} must be a boolean")
    return value


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


def prompt_title(metadata: dict[str, Any], prompt_path: Path) -> str:
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return prompt_path.stem.replace("-", " ").replace("_", " ").strip().title()


def pr_body(prompt_key: str, prompt_body: str, checks: list[list[str]]) -> str:
    checks_text = "\n".join(f"- {shlex.join(command)}" for command in checks)
    body = [
        f"Automated Codex run for `{prompt_key}`.",
        "",
        f"Prompt file: `{prompt_key}`",
        "",
        "Checks:",
        checks_text if checks_text else "- none",
    ]
    if prompt_body.strip():
        body.extend(["", "Prompt body:", "```markdown", prompt_body.strip(), "```"])
    return "\n".join(body)


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


def git_status_text(repo_root: Path) -> str:
    status = git(repo_root, ["status", "--short", "--untracked-files=all"], check=False)
    return status.stdout.strip()


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


def stage_commit_push(
    repo_root: Path,
    branch_name: str,
    commit_message: str,
    prompt_key: str,
    state: dict[str, Any],
    state_path: Path,
    run_key: str,
) -> str | None:
    git(repo_root, ["add", "-A"])
    status = git_status_text(repo_root)
    if not status:
        return None
    git(repo_root, ["commit", "-m", commit_message])
    commit_sha = git_stdout(repo_root, ["rev-parse", "HEAD"])
    git(repo_root, ["push", "-u", "origin", branch_name])
    return commit_sha


def find_open_pr_number(repo_root: Path, branch_name: str) -> str | None:
    result = gh(
        repo_root,
        ["pr", "list", "--head", branch_name, "--state", "open", "--json", "number", "--jq", ".[0].number"],
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise LoopError(f"Could not inspect pull requests for {branch_name}: {detail}")
    return result.stdout.strip() or None


def create_pr(repo_root: Path, branch_name: str, title: str, body: str) -> tuple[str, str]:
    result = gh(
        repo_root,
        [
            "pr",
            "create",
            "--head",
            branch_name,
            "--title",
            title,
            "--body",
            body,
        ],
    )
    url = result.stdout.strip()
    if not url:
        raise LoopError("gh pr create did not return a PR URL")
    number = gh(
        repo_root,
        ["pr", "view", "--head", branch_name, "--json", "number", "--jq", ".number"],
    ).stdout.strip()
    if not number:
        raise LoopError("Could not determine created PR number")
    return number, url


def merge_pr(
    repo_root: Path,
    pr_number: str,
    delete_branch: bool,
) -> None:
    arguments = ["pr", "merge", pr_number, "--merge"]
    if delete_branch:
        arguments.append("--delete-branch")
    gh(repo_root, arguments)


def switch_to_base_branch(repo_root: Path, base_branch: str, sync: bool) -> None:
    git(repo_root, ["switch", base_branch])
    if sync:
        git(repo_root, ["pull", "--ff-only", "origin", base_branch])


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
    venv_bin = repo_root / ".venv" / "bin"
    if venv_bin.is_dir():
        current_path = environment.get("PATH", "")
        environment["PATH"] = f"{venv_bin}{os.pathsep}{current_path}" if current_path else str(venv_bin)
    return environment, list(loaded.values()), env_file


def git_publish_enabled(config: dict[str, Any], key: str) -> bool:
    return bool_config(config["git"].get(key), f"git.{key}")


def build_codex_input(
    prompt_body: str,
    mode: str,
    env_file: Path | None,
    repo_root: Path,
    run_dir: Path,
) -> str:
    env_description = repo_relative(env_file, repo_root) if env_file is not None else "none"
    return "\n".join(
        [
            "# Codex loop execution context",
            "",
            f"- Loop mode: `{mode}`.",
            f"- Environment file loaded: `{env_description}`; values are secrets and must never be printed.",
            f"- Run artifacts directory: `{repo_relative(run_dir, repo_root)}`.",
            "- If the prompt text mentions a different default environment, this loop mode is authoritative.",
            "- Do not print, rotate, or modify credentials.",
            "- Save requested run artifacts in the run artifacts directory above.",
            "",
            "# Original prompt",
            "",
            prompt_body,
        ]
    )


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
    git_config = config["git"]
    environment, secrets, env_file = prepare_environment(config, metadata, mode, repo_root)
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
        "mode": mode,
        "env_file": repo_relative(env_file, repo_root) if env_file is not None else None,
    }
    atomic_write_json(result_path, result)
    set_status(state, state_path, prompt_key, "PENDING", run_key)
    artifacts_captured = False

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
            build_codex_input(prompt_body, mode, env_file, repo_root, run_dir),
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

        capture_git_artifacts(repo_root, diff_path, status_path, secrets, redact_enabled)
        artifacts_captured = True
        result["status"] = "READY_FOR_REVIEW"
        set_status(state, state_path, prompt_key, "READY_FOR_REVIEW", run_key)

        publish_enabled = (
            git_publish_enabled(config, "commit_on_success")
            and git_publish_enabled(config, "push_on_success")
            and git_publish_enabled(config, "merge_pr_on_success")
        )
        if not publish_enabled:
            return "READY_FOR_REVIEW"

        require_gh()
        base_branch = default_branch(repo_root)
        commit_message = prompt_title(metadata, prompt_path)
        result["status"] = "MERGING"
        set_status(state, state_path, prompt_key, "MERGING", run_key)
        atomic_write_json(result_path, result)

        try:
            commit_sha = stage_commit_push(
                repo_root,
                branch_name,
                commit_message,
                prompt_key,
                state,
                state_path,
                run_key,
            )
            if commit_sha is None:
                result["status"] = "NO_CHANGES"
                result["completed_at"] = isoformat()
                result["base_branch"] = base_branch
                set_status(state, state_path, prompt_key, "NO_CHANGES", run_key)
                atomic_write_json(result_path, result)
                switch_to_base_branch(
                    repo_root,
                    base_branch,
                    bool_config(
                        git_config.get("sync_base_branch_after_merge"),
                        "git.sync_base_branch_after_merge",
                    ),
                )
                return "NO_CHANGES"
            result["commit_sha"] = commit_sha
            atomic_write_json(result_path, result)

            pr_number = find_open_pr_number(repo_root, branch_name)
            if pr_number is None:
                if not git_publish_enabled(config, "create_pr_on_success"):
                    raise LoopError("No open PR exists for the successful branch")
                pr_number, pr_url = create_pr(
                    repo_root,
                    branch_name,
                    prompt_title(metadata, prompt_path),
                    pr_body(prompt_key, prompt_body, checks),
                )
                result["pr_url"] = pr_url
            else:
                pr_url = gh(
                    repo_root,
                    ["pr", "view", pr_number, "--json", "url", "--jq", ".url"],
                ).stdout.strip()
                if pr_url:
                    result["pr_url"] = pr_url
            result["pr_number"] = pr_number
            atomic_write_json(result_path, result)

            merge_pr(
                repo_root,
                pr_number,
                bool_config(git_config.get("delete_branch_on_merge"), "git.delete_branch_on_merge"),
            )
            result["status"] = "MERGED"
            result["merged_at"] = isoformat()
            result["base_branch"] = base_branch
            set_status(state, state_path, prompt_key, "MERGED", run_key)
            atomic_write_json(result_path, result)

            switch_to_base_branch(
                repo_root,
                base_branch,
                bool_config(
                    git_config.get("sync_base_branch_after_merge"),
                    "git.sync_base_branch_after_merge",
                ),
            )
            return "MERGED"
        except LoopError as error:
            codex_log.write_text(
                codex_log.read_text(encoding="utf-8") + f"\nPublish error: {error}\n",
                encoding="utf-8",
            )
            result["status"] = "READY_FOR_REVIEW"
            set_status(state, state_path, prompt_key, "READY_FOR_REVIEW", run_key)
            atomic_write_json(result_path, result)
            raise
    except LoopError as error:
        if not artifacts_captured:
            codex_log.write_text(f"Orchestration error: {error}\n", encoding="utf-8")
            checks_log.write_text("Checks skipped because orchestration failed.\n", encoding="utf-8")
            result["status"] = "CODEX_FAILED"
            set_status(state, state_path, prompt_key, "CODEX_FAILED", run_key)
        raise
    finally:
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
    base_branch = default_branch(repo_root)
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
    print(f"Commit message: {prompt_title(metadata, prompt_path)}")
    print(f"Base branch: {base_branch}")
    print("Publish flow:")
    print("  - git add -A")
    print("  - git commit")
    print("  - git push -u origin <branch>")
    print("  - gh pr create (if needed)")
    print("  - gh pr merge --merge --delete-branch")
    print("  - or mark NO_CHANGES when checks pass without a commit-worthy diff")
    print("  - git switch <base-branch> && git pull --ff-only")
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
        queue_order = load_queue_order(loop_dir)

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
                queue_order,
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
            if status not in CONTINUE_SUCCESS_STATUSES or not args.continue_on_success:
                return 0 if status in {"READY_FOR_REVIEW", *CONTINUE_SUCCESS_STATUSES} else 1

            explicit_prompt = None
    except LoopError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
