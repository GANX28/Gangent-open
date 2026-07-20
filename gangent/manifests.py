"""Source/output manifest and deterministic validation helpers.

Manifest v1 is intentionally small and cheap:
- Source Manifest tracks files the user asked the agent to read or use.
- Output Manifest tracks files the user asked the agent to create.
- Validator Layer blocks task finish when requested outputs are missing or malformed.

This layer does not call an LLM. It is designed to reduce unstable model behavior
without increasing token cost or runtime latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re

from .models import TaskInput


TEXT_EXTENSIONS = {".md", ".txt", ".csv", ".yaml", ".yml"}
JSON_EXTENSIONS = {".json"}
PATH_PATTERN = re.compile(
    r"(?<![\w./\\:-])((?:[A-Za-z]:)?[A-Za-z0-9_.-]+(?:[\\/][A-Za-z0-9_.-]+)*\."
    r"(?:md|json|txt|csv|yaml|yml|pdf|xlsx))",
    re.IGNORECASE,
)
OUTPUT_MARKERS = (
    "保存",
    "写入",
    "写",
    "输出",
    "生成",
    "创建",
    "写成",
    "总结为",
    "整理为",
    "保存",
    "写入",
    "输出",
    "生成",
    "创建",
    "save",
    "write",
    "create",
    "output",
    "generate",
)


@dataclass(frozen=True)
class ManifestEntry:
    """One file entry in a source or output manifest."""

    path: str
    exists: bool
    kind: str
    status: str
    note: str = ""


@dataclass(frozen=True)
class ValidationIssue:
    """One validator finding."""

    severity: str
    path: str
    message: str


@dataclass(frozen=True)
class ExecutionManifest:
    """Source Manifest + Output Manifest for one task."""

    sources: list[ManifestEntry] = field(default_factory=list)
    outputs: list[ManifestEntry] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def blocking_issues(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]


@dataclass(frozen=True)
class PathMentions:
    """Raw path mentions split into source and output intent."""

    sources: list[str]
    outputs: list[str]


def build_execution_manifest(task_input: TaskInput) -> ExecutionManifest:
    """Build source/output manifests from one natural-language task."""

    mentions = extract_path_mentions(f"{task_input.goal}\n{task_input.user_message}")
    root = Path(task_input.workspace_root)
    sources = [_entry_for_path(root, path, role="source") for path in mentions.sources]
    outputs = [_entry_for_path(root, path, role="output") for path in mentions.outputs]
    issues = _validate_outputs(root, mentions.outputs)
    return ExecutionManifest(sources=sources, outputs=outputs, issues=issues)


def extract_path_mentions(text: str) -> PathMentions:
    """Extract likely source and output paths from task text.

    A path is considered output when it appears after a write/create marker, or
    when a write/create marker immediately follows the path. Other paths are
    treated as sources.
    """

    matches = _path_matches(text)
    if not matches:
        return PathMentions(sources=[], outputs=[])

    lowered = text.lower()
    marker_positions = [lowered.find(marker.lower()) for marker in OUTPUT_MARKERS if lowered.find(marker.lower()) >= 0]
    first_marker = min(marker_positions) if marker_positions else None

    outputs: list[str] = []
    sources: list[str] = []
    for path, start, end in matches:
        is_output = False
        if first_marker is not None and start >= first_marker:
            is_output = True
        if not is_output:
            follow_text = lowered[end : min(len(lowered), end + 32)]
            is_output = any(
                re.match(rf"^\s*{re.escape(marker.lower())}", follow_text) is not None
                for marker in OUTPUT_MARKERS
            )
        target = outputs if is_output else sources
        if path not in target:
            target.append(path)

    return PathMentions(sources=sources, outputs=outputs)


def format_manifest_prompt(manifest: ExecutionManifest) -> str:
    """Format a compact model-facing manifest message."""

    if not manifest.sources and not manifest.outputs:
        return ""

    lines = ["Execution Manifest"]
    if manifest.sources:
        lines.append("Source Manifest:")
        for entry in manifest.sources:
            lines.append(f"- {entry.path} [{entry.status}]")
    if manifest.outputs:
        lines.append("Output Manifest:")
        for entry in manifest.outputs:
            lines.append(f"- {entry.path} [{entry.status}]")
        missing = [entry.path for entry in manifest.outputs if not entry.exists]
        if missing:
            lines.append(f"Next required output: {missing[0]}")
            lines.append(
                "Hard rule: if the next required output is missing and write_file is available, "
                "the next action should create it with write_file. Do not finish by merely reporting that it is missing."
            )
        lines.append(
            "Validator rule: do not finish until every Output Manifest file exists and passes basic validation."
        )
    return "\n".join(lines)


def format_blocking_validation_hint(manifest: ExecutionManifest) -> str:
    """Return a deterministic finish guard hint for blocking validation issues."""

    if not manifest.blocking_issues:
        return ""
    issue_text = "; ".join(f"{issue.path}: {issue.message}" for issue in manifest.blocking_issues)
    return (
        "Finish guard / Validator Layer: the task cannot finish because required outputs are not valid. "
        f"{issue_text}. If an output file is missing and write_file is available, create it now with write_file. "
        "Continue with write_file/edit_file/apply_patch until the Output Manifest is valid."
    )


def _path_matches(text: str) -> list[tuple[str, int, int]]:
    matches: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for match in PATH_PATTERN.finditer(text):
        path = _normalize_path(match.group(1))
        if not path or path in seen:
            continue
        seen.add(path)
        matches.append((path, match.start(1), match.end(1)))
    return matches


def _normalize_path(raw: str) -> str:
    path = raw.strip().strip(".,;:，。；：)）]】")
    path = path.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if " " in path:
        # The regex can over-capture Chinese/English prose before the path.
        path = path.split()[-1]
    return path.strip().strip(".,;:，。；：)）]】")


def _entry_for_path(root: Path, path: str, *, role: str) -> ManifestEntry:
    target = root / path
    exists = target.exists()
    status = "exists" if exists else "missing"
    return ManifestEntry(path=path, exists=exists, kind=_kind_for_path(path), status=status, note=role)


def _validate_outputs(root: Path, outputs: list[str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for path in outputs:
        target = root / path
        if not target.exists():
            issues.append(ValidationIssue("error", path, "missing output file"))
            continue
        if not target.is_file():
            issues.append(ValidationIssue("error", path, "output path is not a file"))
            continue
        issues.extend(_validate_existing_file(target, path))
    return issues


def _validate_existing_file(target: Path, display_path: str) -> list[ValidationIssue]:
    suffix = target.suffix.lower()
    if suffix in JSON_EXTENSIONS:
        try:
            json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:
            return [ValidationIssue("error", display_path, f"invalid JSON: {exc}")]
        return []

    if suffix in TEXT_EXTENSIONS:
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            return [ValidationIssue("error", display_path, f"text file is not valid UTF-8: {exc}")]
        if not text.strip():
            return [ValidationIssue("error", display_path, "text output is empty")]
    return []


def _kind_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in JSON_EXTENSIONS:
        return "json"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".xlsx":
        return "spreadsheet"
    return "file"
