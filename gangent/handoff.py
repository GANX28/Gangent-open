"""Generate timestamped handoff files for cross-model or cross-machine use."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .audit import default_audit_path
from .checkpoint import default_checkpoint_path, load_checkpoint
from .secret_guard import redact_secrets
from .session_store import default_session_path, load_session
from .state import state_summary


HANDOFF_BASENAME = "gangent-handoff"


def default_handoff_path(workspace_root: str, timestamp: str | None = None) -> Path:
    """Return the default handoff file path inside the workspace runtime state."""

    root = Path(workspace_root).resolve()
    stamp = timestamp or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return root / ".gangent" / "handoff" / f"{HANDOFF_BASENAME}-{stamp}.md"


def export_handoff_file(
    workspace_root: str,
    *,
    trigger: str = "manual",
    session_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    audit_log_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Build and write a compact Markdown handoff file."""

    workspace = Path(workspace_root).resolve()
    session_file = Path(session_path) if session_path is not None else default_session_path(workspace_root)
    checkpoint_file = (
        Path(checkpoint_path) if checkpoint_path is not None else default_checkpoint_path(workspace_root)
    )
    audit_file = Path(audit_log_path) if audit_log_path is not None else default_audit_path(workspace_root)
    target = Path(output_path) if output_path is not None else default_handoff_path(workspace_root)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    session = load_session(session_file) if session_file.exists() else None
    checkpoint = load_checkpoint(checkpoint_file) if checkpoint_file.exists() else None
    recent_audits = _read_recent_audit_records(audit_file, limit=5)

    lines: list[str] = [
        "# Gangent Handoff",
        "",
        f"- generated_at: {timestamp}",
        f"- trigger: {trigger}",
        f"- workspace_root: {workspace}",
        f"- session_file: {session_file}",
        f"- checkpoint_file: {checkpoint_file}",
        f"- audit_log: {audit_file}",
        f"- output_file: {target}",
        "",
        "## Read This First",
        "This file is a compact handoff for the next model. Read it before scanning the full workspace.",
        "",
        "## Workspace Snapshot",
    ]
    lines.extend(_workspace_tree_lines(workspace))
    lines.extend(["", "## Current Session"])

    if session is None:
        lines.append("- session: (missing)")
    else:
        lines.append(f"- session_id: {session.session_id}")
        lines.append(f"- turns: {len(session.turns)}")
        lines.append(f"- updated_at: {session.updated_at}")
        if session.context_summary:
            lines.append("- context_summary:")
            lines.extend(_indented_block(redact_secrets(session.context_summary)))
        else:
            lines.append("- context_summary: (empty)")
        if session.turns:
            lines.append("- recent_turns:")
            for index, turn in enumerate(session.turns[-4:], start=max(1, len(session.turns) - 3)):
                lines.append(f"  - turn_{index}:")
                lines.append(f"    - user_message: {redact_secrets(turn.user_message)}")
                lines.append(f"    - task_id: {turn.task_id}")
                lines.append(f"    - task_status: {turn.task_status}")
                if turn.final_answer:
                    lines.append(f"    - final_answer: {redact_secrets(turn.final_answer)}")
                if turn.tool_summaries:
                    lines.append("    - tool_summaries:")
                    for tool_summary in turn.tool_summaries[:3]:
                        lines.append(f"      - {redact_secrets(tool_summary)}")
                if turn.state_summary:
                    lines.append(f"    - state_summary: {turn.state_summary}")

    lines.extend(["", "## Active Checkpoint"])
    if checkpoint is None:
        lines.append("- checkpoint: (missing)")
    else:
        lines.append(f"- task_id: {checkpoint.task.task_id}")
        lines.append(f"- task_goal: {redact_secrets(checkpoint.task.goal)}")
        lines.append(f"- task_status: {checkpoint.task.status.value}")
        lines.append(f"- step_count: {len(checkpoint.steps)}")
        lines.append(f"- state_summary: {state_summary(checkpoint.state)}")
        if checkpoint.state.errors:
            lines.append("- errors:")
            for error in checkpoint.state.errors[-5:]:
                lines.append(f"  - {redact_secrets(error)}")
        if checkpoint.task_input.constraints:
            lines.append("- task_constraints:")
            for constraint in checkpoint.task_input.constraints[:8]:
                lines.append(f"  - {redact_secrets(constraint)}")
        if checkpoint.state.context_summary:
            lines.append("- checkpoint_context:")
            lines.extend(_indented_block(redact_secrets(checkpoint.state.context_summary)))

    lines.extend(["", "## Failure Focus"])
    failure_lines = _recent_failure_lines(session, checkpoint)
    if failure_lines:
        lines.extend(failure_lines)
    else:
        lines.append("- recent_failures: (none)")

    lines.extend(["", "## Recent Audit"])
    if not recent_audits:
        lines.append("- records: (missing or empty)")
    else:
        lines.append(f"- records: {len(recent_audits)}")
        for record in recent_audits:
            task = record.get("task", {})
            stats = record.get("stats", {})
            lines.append(
                f"  - task_id: {task.get('task_id', '(unknown)')}; "
                f"status: {task.get('status', '(unknown)')}; "
                f"steps: {stats.get('step_count', 0)}; "
                f"duration_seconds: {stats.get('duration_seconds', 0)}"
            )

    lines.extend(["", "## Suggested Next Step"])
    if checkpoint is not None and checkpoint.task.status.value not in {"completed", "failed"}:
        lines.append("- Continue the active checkpoint first, because it is the most recent unfinished task.")
    elif session is not None and session.turns:
        lines.append("- Use the latest session turn as the starting point for the next task.")
    else:
        lines.append("- Start from the workspace summary and read the project rules first.")
    lines.append("")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _read_recent_audit_records(path: Path, limit: int = 5) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for raw_line in lines[-limit:]:
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _indented_block(text: str, indent: str = "  ") -> list[str]:
    return [f"{indent}{line}" for line in text.splitlines()] or [f"{indent}(empty)"]


def _workspace_tree_lines(root: Path, max_depth: int = 2, max_entries: int = 60) -> list[str]:
    """Return a bounded tree summary suitable for handoff files."""

    lines = ["```text", f"{root.name}/"]
    count = 0
    ignored = {".git", ".gangent", "__pycache__", ".pytest_cache", ".mypy_cache"}

    def walk(path: Path, depth: int) -> None:
        nonlocal count
        if depth > max_depth or count >= max_entries:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        except OSError:
            return
        for entry in entries:
            if count >= max_entries:
                return
            if entry.name in ignored:
                continue
            prefix = "  " * depth + "- "
            label = entry.name + ("/" if entry.is_dir() else "")
            lines.append(prefix + label)
            count += 1
            if entry.is_dir():
                walk(entry, depth + 1)

    walk(root, 1)
    if count >= max_entries:
        lines.append("  ... (truncated)")
    lines.append("```")
    return lines


def _recent_failure_lines(session, checkpoint) -> list[str]:
    lines: list[str] = []
    if checkpoint is not None and checkpoint.state.errors:
        lines.append("- checkpoint_errors:")
        for error in checkpoint.state.errors[-5:]:
            lines.append(f"  - {redact_secrets(error)}")
    if session is not None:
        failed_turns = [turn for turn in session.turns if turn.task_status == "failed"]
        if failed_turns:
            lines.append("- failed_turns:")
            for turn in failed_turns[-3:]:
                lines.append(f"  - task_id={turn.task_id}; state={turn.state_summary}")
    return lines
