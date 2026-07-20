"""Task execution profile selection.

This module keeps the first routing decision deterministic: before the model is
called, classify whether the task can use a narrow plan, smaller context, and a
smaller tool surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .models import TaskInput


@dataclass(frozen=True)
class TaskExecutionProfile:
    """Runtime routing profile for prompt, context, and tool budget."""

    name: str
    context_tier: str
    context_max_chars: int
    tool_names: tuple[str, ...] | None
    reason: str


STANDARD_PROFILE = TaskExecutionProfile(
    name="standard",
    context_tier="repo",
    context_max_chars=6_000,
    tool_names=None,
    reason="general task needs the full runtime surface",
)

LONG_TASK_PROFILE = TaskExecutionProfile(
    name="standard",
    context_tier="long_task",
    context_max_chars=10_000,
    tool_names=None,
    reason="broad analysis or architecture task may need a larger context window",
)


READ_MARKERS = (
    "read",
    "open",
    "inspect",
    "look at",
    "check",
    "\u8bfb\u53d6",
    "\u67e5\u770b",
    "\u770b\u4e00\u4e0b",
)

ANALYSIS_MARKERS = (
    "summarize",
    "summary",
    "analyze",
    "explain",
    "review",
    "compare",
    "describe",
    "\u603b\u7ed3",
    "\u5206\u6790",
    "\u89e3\u91ca",
    "\u4ecb\u7ecd",
    "\u8bf4\u660e",
)

WRITE_MARKERS = (
    "write",
    "create",
    "save",
    "edit",
    "generate",
    "output",
    "modify",
    "patch",
    "\u5199\u5165",
    "\u521b\u5efa",
    "\u4fdd\u5b58",
    "\u4fee\u6539",
    "\u751f\u6210",
    "\u8f93\u51fa",
)

NO_WRITE_MARKERS = (
    "do not modify",
    "don't modify",
    "do not edit",
    "don't edit",
    "do not write",
    "don't write",
    "do not save",
    "do not create",
    "without modifying",
    "read-only",
    "no file changes",
    "\u4e0d\u8981\u4fee\u6539",
    "\u4e0d\u4fee\u6539",
    "\u4e0d\u8981\u5199\u5165",
    "\u4e0d\u5199\u5165",
    "\u4e0d\u8981\u4fdd\u5b58",
    "\u4e0d\u4fdd\u5b58",
    "\u4e0d\u8981\u521b\u5efa",
    "\u4e0d\u521b\u5efa",
    "\u4e0d\u8981\u6539",
    "\u4e0d\u6539",
    "\u53ea\u8bfb",
)

LIMITED_NO_WRITE_MARKERS = (
    "do not modify other files",
    "don't modify other files",
    "do not edit other files",
    "don't edit other files",
    "do not write other files",
    "don't write other files",
    "do not change other files",
    "don't change other files",
    "\u4e0d\u8981\u4fee\u6539\u5176\u4ed6\u6587\u4ef6",
    "\u4e0d\u4fee\u6539\u5176\u4ed6\u6587\u4ef6",
    "\u4e0d\u8981\u6539\u5176\u4ed6\u6587\u4ef6",
    "\u4e0d\u6539\u5176\u4ed6\u6587\u4ef6",
)

OUTPUT_FILE_MARKERS = (
    "save as",
    "save to",
    "write to",
    "summarize to",
    "summary to",
    "convert to",
    "export to",
    "\u4fdd\u5b58\u4e3a",
    "\u4fdd\u5b58\u5230",
    "\u5199\u5165\u5230",
    "\u5199\u6210",
    "\u751f\u6210\u4e3a",
    "\u751f\u6210\u5230",
    "\u8f93\u51fa\u5230",
    "\u6574\u7406\u4e3a",
    "\u603b\u7ed3\u4e3a",
)

GIT_MARKERS = (
    "git status",
    "git diff",
    "git log",
    "git show",
    "\u5f53\u524d git",
    "\u67e5\u770b git",
    "\u672a\u63d0\u4ea4",
    "\u5de5\u4f5c\u6811",
)


def task_execution_profile(task_input: TaskInput) -> TaskExecutionProfile:
    """Return the smallest safe execution profile for one task."""

    texts = _task_texts(task_input)

    if _is_explicit_direct_task(texts):
        return TaskExecutionProfile(
            name="direct",
            context_tier="direct",
            context_max_chars=900,
            tool_names=(),
            reason="explicit direct-answer task; no workspace tools or function-call wrapper needed",
        )

    if _is_read_write_task(texts):
        return TaskExecutionProfile(
            name="read_write",
            context_tier="tool",
            context_max_chars=2_400,
            tool_names=("read_file", "read_many_files", "write_file", "edit_file", "finish_task"),
            reason="read source file and write a derived output file",
        )

    if _is_git_analysis_task(texts):
        return TaskExecutionProfile(
            name="git_analysis",
            context_tier="tool",
            context_max_chars=2_400,
            tool_names=("git_status", "git_diff", "git_log", "git_show", "finish_task"),
            reason="read-only git state analysis should use git tools and finish without file changes",
        )

    if _is_code_read_analysis_task(texts):
        return TaskExecutionProfile(
            name="read_analysis",
            context_tier="tool",
            context_max_chars=4_000,
            tool_names=("list_files", "file_info", "grep_files", "read_file", "read_many_files", "search_context", "finish_task"),
            reason="read code or runtime files and synthesize an answer without modifying files",
        )

    if _is_read_analysis_task(texts):
        return TaskExecutionProfile(
            name="read_analysis",
            context_tier="tool",
            context_max_chars=4_000,
            tool_names=("list_files", "file_info", "grep_files", "read_file", "read_many_files", "search_context", "finish_task"),
            reason="read one or more source files and synthesize an answer without modifying files",
        )

    if _is_single_file_write_task(texts):
        return TaskExecutionProfile(
            name="single_write",
            context_tier="tool",
            context_max_chars=1_600,
            tool_names=("read_file", "read_many_files", "write_file", "edit_file", "file_info", "finish_task"),
            reason="single file write task; expose only file write and finish tools",
        )

    if _is_single_file_read_task(texts):
        return TaskExecutionProfile(
            name="single_read",
            context_tier="tool",
            context_max_chars=1_600,
            tool_names=("read_file", "file_info", "list_files", "finish_task"),
            reason="single file read task; expose only read-oriented tools",
        )

    if _is_long_context_task(texts):
        return LONG_TASK_PROFILE
    return STANDARD_PROFILE


def infer_profile_name(task_input: TaskInput) -> str:
    """Convenience wrapper used by planner and budget statistics."""

    return task_execution_profile(task_input).name


def _task_texts(task_input: TaskInput) -> tuple[str, ...]:
    raw = f"{task_input.goal}\n{task_input.user_message}".lstrip("\ufeff").lower()
    return _text_candidates(raw)


def _text_candidates(text: str) -> tuple[str, ...]:
    candidates = [text]
    repaired = _repair_common_mojibake(text)
    if repaired and repaired != text:
        candidates.append(repaired.lower())
    return tuple(dict.fromkeys(candidates))


def _repair_common_mojibake(text: str) -> str:
    try:
        return text.encode("gbk").decode("utf-8")
    except UnicodeError:
        return ""


def _is_explicit_direct_task(texts: tuple[str, ...]) -> bool:
    direct_markers = (
        "do not call tools",
        "no tools",
        "answer directly",
        "finish_task",
        "\u4e0d\u8981\u8c03\u7528\u5de5\u5177",
        "\u4e0d\u8c03\u7528\u5de5\u5177",
        "\u76f4\u63a5\u56de\u7b54",
        "\u53ea\u56de\u7b54",
    )
    if _contains_any(texts, direct_markers):
        return True
    workspace_markers = (
        "read",
        "write",
        "create",
        "edit",
        "file",
        "workspace",
        "git",
        "\u8bfb\u53d6",
        "\u5199\u5165",
        "\u521b\u5efa",
        "\u4fee\u6539",
        "\u6587\u4ef6",
        "\u5de5\u4f5c\u533a",
    )
    has_workspace_marker = _contains_any(texts, workspace_markers)
    return _contains_any(texts, ("\u56de\u7b54",)) and not has_workspace_marker and max(map(len, texts)) < 160


def _is_single_file_read_task(texts: tuple[str, ...]) -> bool:
    if _has_write_intent(texts):
        return False
    if not _contains_any(texts, READ_MARKERS):
        return False
    return len(_probable_file_paths(texts)) == 1


def _is_single_file_write_task(texts: tuple[str, ...]) -> bool:
    if not _has_write_intent(texts):
        return False
    return _mentions_probable_file_path(texts)


def _is_read_write_task(texts: tuple[str, ...]) -> bool:
    if not _contains_any(texts, READ_MARKERS):
        return False
    paths = _probable_file_paths(texts)
    if _has_output_file_intent(texts) and len(paths) >= 2:
        return True
    if not _has_write_intent(texts):
        return False
    return len(paths) >= 2 or (_contains_any(texts, ("\u4fdd\u5b58\u4e3a", "save as")) and bool(paths))


def _is_read_analysis_task(texts: tuple[str, ...]) -> bool:
    if _has_write_intent(texts):
        return False
    paths = _probable_file_paths(texts)
    if not paths:
        return False
    return len(paths) >= 2 or _contains_any(texts, ANALYSIS_MARKERS)


def _has_write_intent(texts: tuple[str, ...]) -> bool:
    if _contains_any(texts, NO_WRITE_MARKERS):
        if _contains_any(texts, LIMITED_NO_WRITE_MARKERS) and _contains_any(texts, WRITE_MARKERS):
            return True
        return False
    return _contains_any(texts, WRITE_MARKERS)


def _has_output_file_intent(texts: tuple[str, ...]) -> bool:
    return _contains_any(texts, OUTPUT_FILE_MARKERS)


def _is_git_analysis_task(texts: tuple[str, ...]) -> bool:
    if _has_write_intent(texts):
        return False
    return _contains_any(texts, GIT_MARKERS)


def _is_code_read_analysis_task(texts: tuple[str, ...]) -> bool:
    if _has_write_intent(texts):
        return False
    if not _contains_any(texts, READ_MARKERS):
        return False
    code_markers = (
        "code",
        "source",
        "source code",
        "runtime",
        "planner",
        "module",
        "implementation",
        "代码",
        "源码",
        "实现",
        "模块",
        "运行时",
        "规划器",
    )
    return _contains_any(texts, code_markers)


def _mentions_probable_file_path(texts: tuple[str, ...]) -> bool:
    return bool(_probable_file_paths(texts))


def _probable_file_paths(texts: tuple[str, ...]) -> list[str]:
    paths: list[str] = []
    for text in texts:
        matches = re.findall(r"[\w./\\-]+\.(?:txt|md|json|py|csv|yaml|yml|toml|ini|log)\b", text)
        generic = matches or re.findall(r"\b[a-z0-9_-]+\.[a-z0-9]{1,8}\b", text)
        for item in generic:
            normalized = item.strip(".,;:()[]{}")
            if normalized and normalized not in paths:
                paths.append(normalized)
    return paths


def _is_long_context_task(texts: tuple[str, ...]) -> bool:
    markers = (
        "architecture",
        "commercial",
        "compare",
        "analyze",
        "review",
        "overall",
        "\u5168\u9762",
        "\u6574\u4f53",
        "\u67b6\u6784",
        "\u5546\u7528",
        "\u7efc\u5408",
        "\u6df1\u5165",
    )
    return any(len(text) > 320 for text in texts) or _contains_any(texts, markers)


def _contains_any(texts: tuple[str, ...], markers: tuple[str, ...]) -> bool:
    return any(marker in text for text in texts for marker in markers)
