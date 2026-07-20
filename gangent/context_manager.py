"""Context Manager v1.

Context Manager（上下文管理器）负责把任务状态整理成模型可读的资料包。
第一版不做 embedding，也不调用额外模型做压缩；它用确定性规则选择
计划、错误、最近工具结果、session 摘要和 repo map。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess

from .budget_stats import default_budget_history_path, planner_feedback_for_task
from .models import AgentState, MessageRole, Task
from .memory_graph import memory_context_for_query
from .patch_editor import inspect_patch_paths
from .planner import format_plan_for_model, infer_task_kind
from .planner_budget import format_planner_budget_control
from .secret_guard import redact_secrets, secret_labels
from .state import state_summary


@dataclass(frozen=True)
class ContextBundle:
    """One compact model-facing context package."""

    text: str
    repo_map: str = ""
    recent_activity: str = ""
    errors: str = ""
    segments: tuple["ContextSegment", ...] = ()
    pollution_report: "ContextPollutionReport | None" = None


@dataclass(frozen=True)
class ContextSegment:
    """One source-aware context unit.

    ContextSegment（上下文片段）让 runtime 知道每段上下文从哪里来、适用范围
    是什么、优先级多高、是否可能敏感。后续动态上下文和中断重规划都应基于
    这种片段，而不是拼接一大段无来源文本。
    """

    title: str
    content: str
    source: str
    scope: str = "task"
    priority: int = 50
    confidence: float = 1.0
    sensitivity: str = "normal"


@dataclass(frozen=True)
class ContextPollutionReport:
    """Deterministic diagnostics for one assembled context bundle."""

    total_segments: int
    total_chars: int
    source_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    sensitive_segments: list[str] = field(default_factory=list)
    low_confidence_segments: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DynamicContextPack:
    """Selected context grouped by runtime purpose before prompt formatting."""

    must_include: tuple[ContextSegment, ...] = ()
    useful_background: tuple[ContextSegment, ...] = ()
    warnings: tuple[ContextSegment, ...] = ()
    excluded: tuple[str, ...] = ()


def build_context_bundle(task: Task, state: AgentState, max_chars: int = 8_000) -> ContextBundle:
    """Build bounded context for one model call."""

    segments = build_context_segments(task, state)
    report = analyze_context_segments(segments, max_chars=max_chars)
    text = format_context_segments(segments, report, max_chars=max_chars)
    repo_map = _segment_content(segments, "Repository Map")
    recent_activity = _segment_content(segments, "Recent Activity")
    errors = _segment_content(segments, "Recent Errors")
    return ContextBundle(
        text=text,
        repo_map=repo_map,
        recent_activity=recent_activity,
        errors=errors,
        segments=tuple(segments),
        pollution_report=report,
    )


def build_context_segments(task: Task, state: AgentState) -> list[ContextSegment]:
    """Build source-aware context segments before final formatting."""

    policy = _context_policy(task, state)
    repo_map = _repo_map(state.workspace_root) if state.workspace_root and policy["include_repo_map"] else ""
    recent_activity = _recent_activity(state)
    errors = _recent_errors(state)
    segments = [
        ContextSegment("Task", _task_section(task, state), source="runtime", priority=100),
        ContextSegment(
            "Planner Control",
            format_planner_budget_control(state),
            source="planner_budget",
            priority=96,
        ),
        ContextSegment("Current Plan", format_plan_for_model(state), source="planner", priority=90),
        ContextSegment("Recent Activity", recent_activity, source="runtime_messages", priority=70),
        ContextSegment("Recent Errors", errors, source="runtime_errors", priority=85),
    ]
    if state.event_summaries:
        segments.append(
            ContextSegment(
                "Runtime Events",
                _runtime_events(state),
                source="event_queue",
                scope="task",
                priority=88,
                confidence=0.9,
            )
        )
    memory_context = _memory_context(task, state)
    if memory_context:
        segments.append(
            ContextSegment(
                "Relevant Memory",
                memory_context,
                source="memory_graph",
                scope="workspace",
                priority=75,
                confidence=0.75,
            )
        )
    planner_feedback = _planner_history_feedback(task, state)
    if planner_feedback:
        segments.append(
            ContextSegment(
                "Planner History Feedback",
                planner_feedback,
                source="planner_history",
                scope="task_kind",
                priority=82,
                confidence=0.85,
            )
        )
    if policy["include_git_summary"]:
        segments.append(
            ContextSegment("Git Summary", _git_summary(state.workspace_root), source="git", scope="workspace", priority=55)
        )
    if policy["include_focused_files"]:
        focused = _focused_file_snippets(state)
        segments.append(
            ContextSegment(
                "Focused Files",
                focused,
                source="workspace_files",
                scope="recent_focus",
                priority=65,
                sensitivity=_sensitivity_for(focused),
            )
        )
    if repo_map:
        segments.append(
            ContextSegment("Repository Map", repo_map, source="workspace_scan", scope="workspace", priority=35)
        )
    return segments


def analyze_context_segments(segments: list[ContextSegment], max_chars: int = 8_000) -> ContextPollutionReport:
    """Return deterministic context pollution diagnostics."""

    source_counts: dict[str, int] = {}
    warnings: list[str] = []
    sensitive_segments: list[str] = []
    low_confidence_segments: list[str] = []
    total_chars = 0
    for segment in segments:
        source_counts[segment.source] = source_counts.get(segment.source, 0) + 1
        total_chars += len(segment.content)
        if segment.sensitivity != "normal" or secret_labels(segment.content):
            sensitive_segments.append(segment.title)
        if segment.confidence < 0.5:
            low_confidence_segments.append(segment.title)
    if total_chars > max_chars:
        warnings.append(f"context_chars_exceed_budget:{total_chars}>{max_chars}")
    if source_counts:
        dominant_source, dominant_count = max(source_counts.items(), key=lambda item: item[1])
        if len(segments) >= 5 and dominant_count / len(segments) >= 0.6:
            warnings.append(f"dominant_context_source:{dominant_source}")
    if not any(segment.priority >= 90 for segment in segments):
        warnings.append("missing_high_priority_task_or_plan_context")
    if sensitive_segments:
        warnings.append("sensitive_context_present")
    if low_confidence_segments:
        warnings.append("low_confidence_context_present")
    return ContextPollutionReport(
        total_segments=len(segments),
        total_chars=total_chars,
        source_counts=source_counts,
        warnings=warnings,
        sensitive_segments=sensitive_segments,
        low_confidence_segments=low_confidence_segments,
    )


def format_context_segments(
    segments: list[ContextSegment],
    report: ContextPollutionReport | None = None,
    max_chars: int = 8_000,
) -> str:
    """Format context segments into one bounded model-facing text."""

    pack = build_dynamic_context_pack(segments, max_chars=max_chars)
    selected = [*pack.must_include, *pack.warnings, *pack.useful_background]
    sections = [_format_segment(segment) for segment in selected]
    if report:
        sections.append(_format_pollution_report(report, list(pack.excluded)))
    text = redact_secrets("\n\n".join(sections))
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... context truncated"
    return text


def build_dynamic_context_pack(segments: list[ContextSegment], max_chars: int = 8_000) -> DynamicContextPack:
    """Select task-specific context under budget with explicit omission tracking."""

    must: list[ContextSegment] = []
    warnings: list[ContextSegment] = []
    useful: list[ContextSegment] = []
    excluded: list[str] = []
    used = 0

    ordered = sorted(segments, key=lambda item: (-item.priority, item.title))
    for segment in ordered:
        cost = len(segment.content) + 120
        target = must if segment.priority >= 90 else warnings if _is_warning_segment(segment) else useful
        if used + cost <= max_chars or segment.priority >= 90:
            target.append(segment)
            used += cost
        else:
            excluded.append(segment.title)
    return DynamicContextPack(
        must_include=tuple(must),
        warnings=tuple(warnings),
        useful_background=tuple(useful),
        excluded=tuple(excluded),
    )


def _task_section(task: Task, state: AgentState) -> str:
    context = state.context_summary or "(empty)"
    return (
        f"Task ID: {task.task_id}\n"
        f"Task goal: {task.goal}\n"
        f"Task status: {task.status.value}\n"
        f"Runtime state: {state_summary(state)}\n"
        f"Context summary: {context}"
    )


def _format_segment(segment: ContextSegment) -> str:
    metadata = (
        f"source={segment.source}; scope={segment.scope}; priority={segment.priority}; "
        f"confidence={segment.confidence:.2f}; sensitivity={segment.sensitivity}"
    )
    return f"## {segment.title}\n[{metadata}]\n{segment.content}"


def _format_pollution_report(report: ContextPollutionReport, omitted: list[str]) -> str:
    source_counts = ", ".join(f"{source}={count}" for source, count in sorted(report.source_counts.items()))
    lines = [
        "## Context Pollution Report",
        f"segments={report.total_segments}; chars={report.total_chars}; sources={source_counts or '-'}",
    ]
    if report.warnings:
        lines.append("warnings=" + ", ".join(report.warnings))
    if report.sensitive_segments:
        lines.append("sensitive_segments=" + ", ".join(report.sensitive_segments))
    if report.low_confidence_segments:
        lines.append("low_confidence_segments=" + ", ".join(report.low_confidence_segments))
    if omitted:
        lines.append("omitted_low_priority_segments=" + ", ".join(omitted))
    return "\n".join(lines)


def _is_warning_segment(segment: ContextSegment) -> bool:
    return (
        segment.sensitivity != "normal"
        or segment.confidence < 0.5
        or segment.source in {"runtime_errors", "event_queue"}
        or "warning" in segment.title.lower()
        or "error" in segment.title.lower()
    )


def _segment_content(segments: list[ContextSegment], title: str) -> str:
    for segment in segments:
        if segment.title == title:
            return segment.content
    return ""


def _sensitivity_for(text: str) -> str:
    return "sensitive" if secret_labels(text) else "normal"


def _memory_context(task: Task, state: AgentState) -> str:
    if not state.workspace_root:
        return ""
    query = f"{task.goal}\n{state.context_summary}"
    return memory_context_for_query(query, state.workspace_root, top_k=5)


def _planner_history_feedback(task: Task, state: AgentState) -> str:
    if not state.workspace_root:
        return ""
    task_input_like = type(
        "TaskInputLike",
        (),
        {
            "goal": task.goal,
            "user_message": task.goal,
            "workspace_root": state.workspace_root,
            "constraints": [],
            "created_at": task.created_at,
        },
    )()
    return planner_feedback_for_task(task_input_like, default_budget_history_path(state.workspace_root), min_samples=2)


def _recent_activity(state: AgentState, max_messages: int = 6) -> str:
    if not state.messages:
        return "(none)"
    lines: list[str] = []
    for message in state.messages[-max_messages:]:
        content = message.content.replace("\n", " ").strip()
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"- {message.role.value}: {content}")
    if state.last_tool_result:
        result = state.last_tool_result
        status = "success" if result.success else "failed"
        content = result.output if result.success else result.error or ""
        content = content.replace("\n", " ")
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"- last_tool_result: {status}; {content}")
    return "\n".join(lines)


def _recent_errors(state: AgentState, max_errors: int = 4) -> str:
    if not state.errors:
        return "(none)"
    return "\n".join(f"- {error}" for error in state.errors[-max_errors:])


def _runtime_events(state: AgentState, max_events: int = 5) -> str:
    if not state.event_summaries:
        return "(none)"
    return "\n".join(f"- {event}" for event in state.event_summaries[-max_events:])


def _repo_map(workspace_root: str, max_entries: int = 80, max_depth: int = 3) -> str:
    root = Path(workspace_root).resolve()
    if not root.exists() or not root.is_dir():
        return "(workspace root does not exist)"

    entries: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item).lower()):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        parts = relative.parts
        if len(parts) > max_depth:
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in parts):
            continue
        kind = "dir" if path.is_dir() else "file"
        entries.append(f"[{kind}] {relative.as_posix()}")
        if len(entries) >= max_entries:
            entries.append("... repo map truncated")
            break
    return "\n".join(entries) if entries else "(empty workspace)"


def _context_policy(task: Task, state: AgentState) -> dict[str, bool]:
    text = f"{task.goal}\n{state.context_summary}".lower()
    task_kind = infer_task_kind(type("TaskInputLike", (), {"goal": task.goal, "user_message": task.goal})())
    include_git_summary = any(
        word in text for word in ["git", "commit", "diff", "status", "branch", "repo", "repository"]
    )
    include_repo_map = any(
        word in text
        for word in ["structure", "folder", "directory", "workspace", "project", "architecture", "module", "文件", "结构"]
    )
    if task_kind in {"build", "debug"} and state.step_index == 0:
        include_repo_map = True
    include_focused_files = bool(_recent_focus_paths(state))
    return {
        "include_git_summary": include_git_summary,
        "include_repo_map": include_repo_map,
        "include_focused_files": include_focused_files,
    }


def _git_summary(workspace_root: str) -> str:
    root = Path(workspace_root).resolve()
    if not (root / ".git").exists():
        return "(not a git repository)"
    commands = [
        ("status", ["git", "status", "--short"]),
        ("recent commits", ["git", "log", "-n3", "--oneline", "--decorate"]),
    ]
    sections: list[str] = []
    for label, command in commands:
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
        except Exception as exc:
            sections.append(f"{label}: unavailable ({exc})")
            continue
        content = (completed.stdout or completed.stderr or "").strip()
        if not content:
            content = "(clean)"
        content = redact_secrets(content)
        if len(content) > 1_200:
            content = content[:1_200] + "\n... truncated"
        sections.append(f"{label}:\n{content}")
    return "\n\n".join(sections)


def _focused_file_snippets(state: AgentState, max_chars_per_file: int = 600) -> str:
    root = Path(state.workspace_root).resolve()
    paths = _recent_focus_paths(state)
    if not paths:
        return "(none)"

    snippets: list[str] = []
    for relative in paths[:3]:
        try:
            target = (root / relative).resolve()
            target.relative_to(root)
        except Exception:
            continue
        if not target.exists() or not target.is_file():
            continue
        try:
            text = target.read_text(encoding="utf-8")
        except Exception:
            continue
        text = redact_secrets(text)
        if len(text) > max_chars_per_file:
            text = text[:max_chars_per_file] + "\n... truncated"
        snippets.append(f"[file] {relative.as_posix()}\n{text}")
    return "\n\n".join(snippets) if snippets else "(none)"


def _recent_focus_paths(state: AgentState) -> list[Path]:
    if not state.last_decision or not state.last_decision.tool_args:
        return []
    args = state.last_decision.tool_args
    values: list[str] = []
    if isinstance(args.get("path"), str):
        values.append(args["path"])
    if isinstance(args.get("paths"), list):
        values.extend(item for item in args["paths"] if isinstance(item, str))
    if isinstance(args.get("patch"), str):
        try:
            values.extend(inspect_patch_paths(args["patch"]))
        except Exception:
            pass
    cleaned: list[Path] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip().replace("\\", "/")
        if not normalized or normalized in seen or normalized.startswith("."):
            continue
        seen.add(normalized)
        cleaned.append(Path(normalized))
    return cleaned
