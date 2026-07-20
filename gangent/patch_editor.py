"""Patch Editor v1.

Patch Editor（补丁编辑器）让模型用局部 diff 修改文件，而不是整文件
覆盖。第一版支持 Add File 和 Update File，不支持 Delete File。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .permissions import (
    AccessMode,
    PermissionError,
    default_permission_profile,
    ensure_content_size_allowed,
    ensure_file_size_allowed,
    resolve_allowed_path,
)
from .secret_guard import is_sensitive_path, secret_labels


class PatchError(ValueError):
    """Patch cannot be parsed or safely applied."""


@dataclass
class PatchHunk:
    old_text: str = ""
    new_text: str = ""


@dataclass
class PatchOperation:
    op_type: str
    path: str
    content: str = ""
    hunks: list[PatchHunk] = field(default_factory=list)


def apply_text_patch(patch: str, workspace_root: str) -> str:
    """Parse and apply a restricted patch."""

    operations = parse_patch(patch)
    profile = default_permission_profile(workspace_root)
    summaries: list[str] = []
    for operation in operations:
        target = resolve_allowed_path(operation.path, profile, AccessMode.WRITE)
        _ensure_patch_target_allowed(operation.path, target)
        if operation.op_type == "add":
            summaries.append(_apply_add_file(operation, target, profile.max_file_bytes))
        elif operation.op_type == "update":
            summaries.append(_apply_update_file(operation, target, profile.max_file_bytes))
        else:
            raise PatchError(f"Unsupported patch operation: {operation.op_type}")
    return "\n".join(summaries)


def inspect_patch_paths(patch: str) -> list[str]:
    """Return paths touched by a restricted patch without applying it."""

    return [operation.path for operation in parse_patch(patch)]


def summarize_patch(patch: str) -> str:
    """Return a short human-readable patch summary for approval and audit."""

    operations = parse_patch(patch)
    parts: list[str] = []
    for operation in operations:
        if operation.op_type == "add":
            line_count = operation.content.count("\n")
            parts.append(f"add {operation.path} ({line_count} line(s))")
        elif operation.op_type == "update":
            parts.append(f"update {operation.path} ({len(operation.hunks)} hunk(s))")
        else:
            parts.append(f"{operation.op_type} {operation.path}")
    return "; ".join(parts)


def parse_patch(patch: str) -> list[PatchOperation]:
    """Parse a small apply_patch-like format.

    Supported format:
    *** Begin Patch
    *** Add File: path
    +content
    *** Update File: path
    @@
     context
    -old
    +new
    *** End Patch
    """

    lines = patch.splitlines()
    if not lines or lines[0].strip() != "*** Begin Patch":
        raise PatchError("Patch must start with *** Begin Patch")
    if lines[-1].strip() != "*** End Patch":
        raise PatchError("Patch must end with *** End Patch")

    operations: list[PatchOperation] = []
    current: PatchOperation | None = None
    current_hunk: PatchHunk | None = None

    for line in lines[1:-1]:
        if line.startswith("*** Add File: "):
            _finish_hunk(current, current_hunk)
            current_hunk = None
            current = PatchOperation(op_type="add", path=line.removeprefix("*** Add File: ").strip())
            operations.append(current)
            continue
        if line.startswith("*** Update File: "):
            _finish_hunk(current, current_hunk)
            current_hunk = None
            current = PatchOperation(op_type="update", path=line.removeprefix("*** Update File: ").strip())
            operations.append(current)
            continue
        if line.startswith("*** Delete File: "):
            raise PatchError("Delete File is not supported in Patch Editor v1.")
        if current is None:
            raise PatchError("Patch line appears before an operation header.")

        if current.op_type == "add":
            if not line.startswith("+"):
                raise PatchError("Add File lines must start with '+'.")
            current.content += line[1:] + "\n"
            continue

        if current.op_type == "update":
            if line.startswith("@@"):
                _finish_hunk(current, current_hunk)
                current_hunk = PatchHunk()
                continue
            if current_hunk is None:
                raise PatchError("Update File content must appear inside @@ hunk.")
            if line.startswith(" "):
                current_hunk.old_text += line[1:] + "\n"
                current_hunk.new_text += line[1:] + "\n"
            elif line.startswith("-"):
                current_hunk.old_text += line[1:] + "\n"
            elif line.startswith("+"):
                current_hunk.new_text += line[1:] + "\n"
            else:
                raise PatchError("Update hunk lines must start with ' ', '-', or '+'.")

    _finish_hunk(current, current_hunk)
    if not operations:
        raise PatchError("Patch contains no operations.")
    return operations


def _finish_hunk(current: PatchOperation | None, hunk: PatchHunk | None) -> None:
    if current is None or hunk is None:
        return
    if current.op_type == "update":
        if not hunk.old_text:
            raise PatchError("Update hunk old_text cannot be empty.")
        if hunk.old_text == hunk.new_text:
            raise PatchError("Update hunk does not change anything.")
        current.hunks.append(hunk)


def _apply_add_file(operation: PatchOperation, target: Path, max_file_bytes: int) -> str:
    if target.exists():
        raise PatchError(f"Add File target already exists: {operation.path}")
    labels = secret_labels(operation.content)
    if labels:
        raise PatchError(f"Refusing to write possible secrets: {', '.join(labels)}")
    ensure_content_size_allowed(operation.content, max_file_bytes)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(operation.content, encoding="utf-8")
    return f"Added {operation.path}"


def _apply_update_file(operation: PatchOperation, target: Path, max_file_bytes: int) -> str:
    if not target.exists():
        raise PatchError(f"Update File target does not exist: {operation.path}")
    if not target.is_file():
        raise PatchError(f"Update File target is not a file: {operation.path}")
    ensure_file_size_allowed(target, max_file_bytes, "Patch target")
    data = target.read_bytes()
    if b"\x00" in data:
        raise PatchError(f"Binary file is not supported: {operation.path}")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PatchError(f"File is not valid UTF-8: {operation.path}") from exc
    text = target.read_text(encoding="utf-8")

    for hunk in operation.hunks:
        count = text.count(hunk.old_text)
        if count == 0:
            raise PatchError(f"Patch context was not found: {operation.path}")
        if count > 1:
            raise PatchError(f"Patch context is ambiguous: {operation.path}")
        text = text.replace(hunk.old_text, hunk.new_text, 1)

    labels = secret_labels(text)
    if labels:
        raise PatchError(f"Refusing to write possible secrets: {', '.join(labels)}")
    ensure_content_size_allowed(text, max_file_bytes)
    target.write_text(text, encoding="utf-8")
    return f"Updated {operation.path}: {len(operation.hunks)} hunk(s)"


def _ensure_patch_target_allowed(requested_path: str, target: Path) -> None:
    if is_sensitive_path(requested_path) or is_sensitive_path(target):
        raise PermissionError(f"Sensitive path is blocked: {requested_path}")
