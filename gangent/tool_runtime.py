"""Tool Runtime（工具运行层）。

这一层负责把模型提出的工具调用意图真正执行成结果。
模型只能产生 ActionDecision，不能直接读文件、写文件或执行命令。
Tool Runtime 会检查工具名、解析路径、限制访问范围，然后返回标准化 ToolResult。
"""

from __future__ import annotations

import sys
from pathlib import Path
import re
import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING

from .command_policy import CommandRisk, assess_run_command_args
from .memory_graph import JsonMemoryGraphStore, MemoryLayer, MemoryNodeType, default_memory_graph_path
from .models import ActionDecision, DecisionType, ToolResult, new_id
from .patch_editor import PatchError, apply_text_patch
from .permissions import (
    AccessMode,
    PermissionError,
    default_permission_profile,
    ensure_content_size_allowed,
    ensure_file_size_allowed,
    resolve_allowed_path,
    resolve_workspace_path,
)
from .secret_guard import is_sensitive_path, redact_secrets, secret_labels
from .runner import LocalRunner, SandboxCommand
from .text_guard import ensure_no_encoding_artifacts

if TYPE_CHECKING:
    from .tool_registry import ToolRegistry


class ToolRuntimeError(ValueError):
    """工具无法安全执行时使用的错误类型。"""


def execute_tool_call(
    decision: ActionDecision,
    workspace_root: str,
    tool_registry: "ToolRegistry | None" = None,
) -> ToolResult:
    """执行一个工具调用决策，并统一返回 ToolResult。

    专业说法：这是 tool dispatch（工具分发）和 result normalization（结果标准化）。
    通俗说法：模型说要调哪个工具以后，这里负责找到对应 Python 函数，执行它，
    然后不管成功失败都包装成统一结果。
    """

    call_id = new_id("call")
    try:
        if decision.decision_type != DecisionType.TOOL_CALL:
            raise ToolRuntimeError("Decision is not a tool call.")
        if not decision.tool_name:
            raise ToolRuntimeError("Tool name is missing.")
        if decision.tool_args is None:
            raise ToolRuntimeError("Tool args are missing.")

        if tool_registry is None:
            from .tool_registry import default_tool_registry

            tool_registry = default_tool_registry()

        output = tool_registry.dispatch(decision, workspace_root)

        return ToolResult(call_id=call_id, success=True, output=output)
    except KeyError as exc:
        message = str(exc.args[0]) if exc.args else str(exc)
        return ToolResult(call_id=call_id, success=False, error=message)
    except (ToolRuntimeError, PermissionError, PatchError, ValueError) as exc:
        return ToolResult(call_id=call_id, success=False, error=str(exc))


def list_files(path: str, workspace_root: str, max_entries: int = 100) -> str:
    """列出 workspace root 内某个目录下的文件。

    第一版只返回名字和类型，不递归扫描，避免输出太大。
    """

    target = resolve_workspace_path(path, workspace_root)
    if not target.exists():
        raise ToolRuntimeError(f"Directory does not exist: {path}")
    if not target.is_dir():
        raise ToolRuntimeError(f"Path is not a directory: {path}")

    entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    if not entries:
        return "(empty directory)"

    lines: list[str] = []
    for entry in entries[:max_entries]:
        kind = "dir" if entry.is_dir() else "file"
        lines.append(f"[{kind}] {entry.name}")
    if len(entries) > max_entries:
        lines.append(f"... {len(entries) - max_entries} more entries")
    return "\n".join(lines)


def file_info(path: str, workspace_root: str) -> str:
    """Return path type and lightweight file metadata before reading or editing."""

    target = resolve_workspace_path(path, workspace_root)
    if is_sensitive_path(path) or is_sensitive_path(target):
        raise ToolRuntimeError(f"Sensitive path is blocked: {path}")
    if not target.exists():
        raise ToolRuntimeError(f"Path does not exist: {path}")
    if target.is_dir():
        entries = list(target.iterdir())
        return f"path={path}\ntype=directory\nentries={len(entries)}"
    if not target.is_file():
        return f"path={path}\ntype=other"

    size = target.stat().st_size
    binary = b"\x00" in target.read_bytes()[:1024]
    line_count = "unknown"
    if not binary and size <= 2_000_000:
        try:
            line_count = str(len(_decode_text_file(target.read_bytes(), path).splitlines()))
        except ToolRuntimeError:
            line_count = "undecodable"
    return (
        f"path={path}\n"
        "type=file\n"
        f"size_bytes={size}\n"
        f"binary={str(binary).lower()}\n"
        f"line_count={line_count}"
    )


def read_file(
    path: str,
    workspace_root: str,
    max_bytes: int = 20_000,
    start_line: int | None = None,
    max_lines: int | None = None,
) -> str:
    """读取 workspace root 内的 UTF-8 文本文件。

    第一版限制大小并拒绝明显的二进制文件，避免一次把大文件或不可读内容塞进状态。
    """

    target = resolve_workspace_path(path, workspace_root)
    if is_sensitive_path(path) or is_sensitive_path(target):
        raise ToolRuntimeError(f"Sensitive path is blocked: {path}")
    if not target.exists():
        raise ToolRuntimeError(f"File does not exist: {path}")
    if not target.is_file():
        raise ToolRuntimeError(
            f"Path is a directory, not a file: {path}. "
            "Use list_files for directories, then call read_file on a concrete file."
        )
    if start_line is not None and start_line < 1:
        raise ToolRuntimeError("read_file start_line must be >= 1.")
    if max_lines is not None and max_lines < 1:
        raise ToolRuntimeError("read_file max_lines must be >= 1.")

    file_size = target.stat().st_size
    data = target.read_bytes()
    if b"\x00" in data:
        raise ToolRuntimeError(f"Binary file is not supported: {path}")

    text = _decode_text_file(data, path)

    if start_line is None and max_lines is None:
        if file_size <= max_bytes:
            return redact_secrets(text)
        start_line = 1
        max_lines = 200

    lines = text.splitlines()
    slice_start = 0 if start_line is None else start_line - 1
    slice_end = len(lines) if max_lines is None else slice_start + max_lines
    selected_lines = lines[slice_start:slice_end]
    if not selected_lines:
        raise ToolRuntimeError("Requested read_file chunk is empty.")

    numbered_lines = [
        f"{line_number} | {line}"
        for line_number, line in enumerate(selected_lines, start=slice_start + 1)
    ]
    chunk = "\n".join(numbered_lines)
    encoded = chunk.encode("utf-8")
    if len(encoded) > max_bytes:
        chunk = encoded[:max_bytes].decode("utf-8", errors="ignore")
        chunk = chunk.rstrip() + "\n[truncated chunk]"

    partial_hint = ""
    if file_size > max_bytes and start_line == 1 and max_lines == 200:
        partial_hint = " partial=true reason=file_too_large"
    header = (
        f"[chunk path={path} start_line={slice_start + 1} "
        f"line_count={len(selected_lines)} total_lines={len(lines)}{partial_hint}]\n"
    )
    return redact_secrets(header + chunk)


def read_many_files(paths: list[str], workspace_root: str, max_files: int = 8, max_bytes_per_file: int = 12_000) -> str:
    """Read several small text files with citations in one call."""

    if not isinstance(paths, list) or not paths:
        raise ToolRuntimeError("read_many_files requires a non-empty paths list.")
    if len(paths) > max_files:
        raise ToolRuntimeError(f"read_many_files supports at most {max_files} files.")

    sections: list[str] = []
    for requested_path in paths:
        if not isinstance(requested_path, str) or not requested_path.strip():
            raise ToolRuntimeError("read_many_files paths must be non-empty strings.")
        content = read_file(requested_path, workspace_root, max_bytes=max_bytes_per_file)
        sections.append(f"--- {requested_path} ---\n{content}")
    return "\n\n".join(sections)


def grep_files(pattern: str, path: str, workspace_root: str, max_results: int = 50) -> str:
    """Search workspace text files with a bounded regex pattern."""

    if not isinstance(pattern, str) or not pattern.strip():
        raise ToolRuntimeError("grep_files requires a non-empty pattern.")
    if len(pattern) > 200:
        raise ToolRuntimeError("grep_files pattern is too long.")
    if max_results < 1 or max_results > 100:
        raise ToolRuntimeError("grep_files max_results must be between 1 and 100.")
    try:
        regex = re.compile(pattern)
    except re.error:
        regex = re.compile(re.escape(pattern))

    root = Path(workspace_root).resolve()
    target = resolve_workspace_path(path or ".", workspace_root)
    if is_sensitive_path(path) or is_sensitive_path(target):
        raise ToolRuntimeError(f"Sensitive path is blocked: {path}")
    if not target.exists():
        raise ToolRuntimeError(f"grep_files path does not exist: {path}")

    files = [target] if target.is_file() else sorted(target.rglob("*"))
    matches: list[str] = []
    for file_path in files:
        if len(matches) >= max_results:
            break
        if not _is_searchable_text_file(file_path, root):
            continue
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        relative = file_path.relative_to(root).as_posix()
        for line_number, line in enumerate(lines, start=1):
            if regex.search(line):
                matches.append(f"{relative}:{line_number}: {redact_secrets(line)}")
                if len(matches) >= max_results:
                    break

    return "\n".join(matches) if matches else "No matches found."


def export_artifact(name: str, content: str, workspace_root: str, overwrite: bool = False) -> str:
    """Export a generated artifact under artifacts/."""

    if not isinstance(name, str) or not name.strip():
        raise ToolRuntimeError("export_artifact requires a non-empty name.")
    if "/" in name or "\\" in name or name.startswith("."):
        raise ToolRuntimeError("export_artifact name must be a simple non-hidden file name.")
    return write_file(f"artifacts/{name}", content, workspace_root, overwrite=overwrite)


def scratchpad_note(note: str, workspace_root: str) -> str:
    """Append one internal note to the task scratchpad."""

    if not isinstance(note, str) or not note.strip():
        raise ToolRuntimeError("scratchpad_note requires a non-empty note.")
    labels = secret_labels(note)
    if labels:
        raise ToolRuntimeError(f"Refusing to write possible secrets: {', '.join(labels)}")
    if len(note.encode("utf-8")) > 4_000:
        raise ToolRuntimeError("scratchpad_note is too large.")

    root = Path(workspace_root).resolve()
    target_dir = root / ".gangent" / "scratchpad"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "latest.md"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(f"- {redact_secrets(note.strip())}\n")
    return "Scratchpad note appended."


def memory_add(
    node_type: str,
    content: str,
    workspace_root: str,
    summary: str = "",
    project_scope: str = "",
    source: str = "runtime",
    tags: list[str] | None = None,
    importance: float = 0.5,
    confidence: float = 0.8,
    layer: str = "",
) -> str:
    """Add one node to the local memory graph."""

    if not isinstance(content, str) or not content.strip():
        raise ToolRuntimeError("memory_add requires non-empty content.")
    if len(content.encode("utf-8")) > 4_000:
        raise ToolRuntimeError("memory_add content is too large.")
    labels = secret_labels("\n".join([content, summary]))
    if labels:
        raise ToolRuntimeError(f"Refusing to store possible secrets: {', '.join(labels)}")
    try:
        resolved_type = MemoryNodeType(node_type)
    except ValueError as exc:
        raise ToolRuntimeError(f"Unsupported memory node_type: {node_type}") from exc
    resolved_layer = None
    if layer:
        try:
            resolved_layer = MemoryLayer(layer)
        except ValueError as exc:
            raise ToolRuntimeError(f"Unsupported memory layer: {layer}") from exc
    store = JsonMemoryGraphStore(default_memory_graph_path(workspace_root))
    node = store.add_node(
        resolved_type,
        content=content,
        summary=summary,
        project_scope=project_scope,
        source=source,
        tags=tags or [],
        importance=float(importance),
        confidence=float(confidence),
        layer=resolved_layer,
    )
    store.save()
    return (
        f"memory_node_added id={node.node_id}; type={node.node_type.value}; "
        f"layer={node.layer.value}; project={node.project_scope or '-'}"
    )


def fetch_url(url: str, workspace_root: str, max_bytes: int = 60_000, timeout_seconds: int = 15) -> str:
    """Fetch a public HTTP/HTTPS URL as bounded text.

    workspace_root 参数保留是为了和 ToolRegistry handler 签名一致。该函数
    不读取本地文件，也不执行 JavaScript。
    """

    if not isinstance(url, str) or not url.strip():
        raise ToolRuntimeError("fetch_url requires a non-empty URL.")
    if max_bytes < 1_000 or max_bytes > 200_000:
        raise ToolRuntimeError("fetch_url max_bytes must be between 1,000 and 200,000.")

    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ToolRuntimeError("fetch_url only supports http and https URLs.")
    if not parsed.hostname:
        raise ToolRuntimeError("fetch_url URL must include a hostname.")
    _ensure_public_hostname(parsed.hostname)

    request = urllib.request.Request(
        urllib.parse.urlunparse(parsed),
        headers={"User-Agent": "Gangent/0.1 (+local-agent-runtime)"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            content_type = response.headers.get("content-type", "")
            data = response.read(max_bytes + 1)
    except urllib.error.URLError as exc:
        raise ToolRuntimeError(f"fetch_url request failed: {exc}") from exc

    if len(data) > max_bytes:
        raise ToolRuntimeError("fetch_url response exceeded max_bytes.")
    if b"\x00" in data:
        raise ToolRuntimeError("fetch_url binary response is not supported.")
    encoding = "utf-8"
    match = re.search(r"charset=([^;]+)", content_type, flags=re.IGNORECASE)
    if match:
        encoding = match.group(1).strip()
    try:
        text = data.decode(encoding, errors="replace")
    except LookupError:
        text = data.decode("utf-8", errors="replace")
    return redact_secrets(_strip_html(text))


def write_file(
    path: str,
    content: str,
    workspace_root: str,
    overwrite: bool = False,
) -> str:
    """在 workspace root 内写入 UTF-8 文本文件。

    第一版不支持写隐藏文件，不支持写大文件。
    覆盖已有文件必须显式传 overwrite=true。
    """

    profile = default_permission_profile(workspace_root)
    target = resolve_allowed_path(path, profile, AccessMode.WRITE)
    if is_sensitive_path(path) or is_sensitive_path(target):
        raise ToolRuntimeError(f"Sensitive path is blocked: {path}")
    labels = secret_labels(content)
    if labels:
        raise ToolRuntimeError(f"Refusing to write possible secrets: {', '.join(labels)}")
    try:
        ensure_no_encoding_artifacts(content)
    except ValueError as exc:
        raise ToolRuntimeError(str(exc)) from exc
    ensure_content_size_allowed(content, profile.max_file_bytes)

    if target.exists() and not overwrite:
        raise ToolRuntimeError(f"File already exists; set overwrite=true to replace: {path}")
    if target.exists() and not target.is_file():
        raise ToolRuntimeError(f"Path is not a file: {path}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content.encode('utf-8'))} bytes to {path}"


def edit_file(
    path: str,
    old_text: str,
    new_text: str,
    workspace_root: str,
) -> str:
    """用精确文本替换修改 workspace root 内的 UTF-8 文本文件。

    第一版只支持 old_text 唯一匹配时替换，避免模型模糊修改导致误伤。
    """

    if not old_text:
        raise ToolRuntimeError("old_text must not be empty.")

    profile = default_permission_profile(workspace_root)
    target = resolve_allowed_path(path, profile, AccessMode.WRITE)
    if is_sensitive_path(path) or is_sensitive_path(target):
        raise ToolRuntimeError(f"Sensitive path is blocked: {path}")
    if not target.exists():
        raise ToolRuntimeError(f"File does not exist: {path}")
    if not target.is_file():
        raise ToolRuntimeError(f"Path is not a file: {path}")
    ensure_file_size_allowed(target, profile.max_file_bytes, "File")

    data = target.read_bytes()
    if b"\x00" in data:
        raise ToolRuntimeError(f"Binary file is not supported: {path}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolRuntimeError(f"File is not valid UTF-8: {path}") from exc

    count = text.count(old_text)
    if count == 0:
        raise ToolRuntimeError("old_text was not found.")
    if count > 1:
        raise ToolRuntimeError("old_text matched multiple locations; edit is ambiguous.")

    updated = text.replace(old_text, new_text, 1)
    labels = secret_labels(updated)
    if labels:
        raise ToolRuntimeError(f"Refusing to write possible secrets: {', '.join(labels)}")
    try:
        ensure_no_encoding_artifacts(updated)
    except ValueError as exc:
        raise ToolRuntimeError(str(exc)) from exc
    ensure_content_size_allowed(updated, profile.max_file_bytes)
    target.write_text(updated, encoding="utf-8")
    return f"Edited {path}"


def apply_patch(patch: str, workspace_root: str) -> str:
    """Apply a restricted text patch inside the workspace."""

    return apply_text_patch(patch, workspace_root)


def git_status(workspace_root: str, timeout_seconds: int = 10, max_output: int = 20_000) -> str:
    """运行只读 git status。"""

    return _run_git_command(
        ["git", "status", "--short"],
        workspace_root,
        timeout_seconds,
        max_output,
    )


def git_diff(workspace_root: str, timeout_seconds: int = 10, max_output: int = 20_000) -> str:
    """运行只读 git diff。"""

    return _run_git_command(
        ["git", "diff", "--"],
        workspace_root,
        timeout_seconds,
        max_output,
    )


def git_log(
    workspace_root: str,
    limit: int = 5,
    timeout_seconds: int = 10,
    max_output: int = 20_000,
) -> str:
    """Run read-only git log with a bounded commit count."""

    if limit < 1 or limit > 20:
        raise ToolRuntimeError("git_log limit must be between 1 and 20.")
    return _run_git_command(
        ["git", "log", f"-n{limit}", "--oneline", "--decorate"],
        workspace_root,
        timeout_seconds,
        max_output,
    )


def git_show(
    revision: str,
    workspace_root: str,
    timeout_seconds: int = 10,
    max_output: int = 20_000,
) -> str:
    """Run read-only git show for one revision."""

    if not _is_safe_git_revision(revision):
        raise ToolRuntimeError("git_show revision contains unsupported characters.")
    return _run_git_command(
        ["git", "show", "--stat", "--oneline", "--no-ext-diff", revision],
        workspace_root,
        timeout_seconds,
        max_output,
    )


def git_add(paths: list[str], workspace_root: str, timeout_seconds: int = 10, max_output: int = 20_000) -> str:
    """Stage specific workspace paths with git add --."""

    if not paths:
        raise ToolRuntimeError("git_add requires at least one path.")
    validated_paths = _validated_git_paths(paths, workspace_root)
    _run_git_command(
        ["git", "add", "--", *validated_paths],
        workspace_root,
        timeout_seconds,
        max_output,
    )
    return f"Staged {len(validated_paths)} path(s): {', '.join(validated_paths)}"


def git_commit(
    message: str,
    workspace_root: str,
    timeout_seconds: int = 20,
    max_output: int = 20_000,
) -> str:
    """Create a local git commit with a plain-text message."""

    if not isinstance(message, str) or not message.strip():
        raise ToolRuntimeError("git_commit requires non-empty message.")
    normalized = " ".join(message.strip().split())
    if len(normalized) > 120:
        raise ToolRuntimeError("git_commit message is too long; keep it under 120 characters.")
    if any(token in normalized for token in ["--", ";", "&&", "||", "|", "`"]):
        raise ToolRuntimeError("git_commit message contains unsupported shell-like tokens.")
    return _run_git_command(
        ["git", "commit", "-m", normalized],
        workspace_root,
        timeout_seconds,
        max_output,
    )


def run_tests(workspace_root: str, timeout_seconds: int = 60, max_output: int = 30_000) -> str:
    """运行受控 unittest 测试命令。

    注意：测试会执行项目代码，所以 policy 默认要求人工确认。
    """

    root = Path(workspace_root).resolve()
    tests_dir = root / "tests"
    if not tests_dir.exists() or not tests_dir.is_dir():
        raise ToolRuntimeError("tests directory does not exist.")

    return _run_command(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"],
        workspace_root=root,
        timeout_seconds=timeout_seconds,
        max_output=max_output,
    )


def run_command(
    args: list[str],
    workspace_root: str,
    cwd: str = ".",
    timeout_seconds: int = 30,
    max_output: int = 30_000,
) -> str:
    """Run a structured development command through LocalRunner.

    This tool still revalidates the argv shape and workspace cwd. Policy decides
    whether approval is required before this function is reached.
    """

    assessment = assess_run_command_args({"args": args, "cwd": cwd})
    if assessment.risk == CommandRisk.BLOCK:
        raise ToolRuntimeError(assessment.reason)
    if timeout_seconds < 1 or timeout_seconds > 120:
        raise ToolRuntimeError("timeout_seconds must be between 1 and 120.")

    target_cwd = resolve_workspace_path(assessment.cwd, workspace_root)
    if not target_cwd.exists() or not target_cwd.is_dir():
        raise ToolRuntimeError(f"Command cwd does not exist or is not a directory: {cwd}")

    command = list(assessment.args)
    if command[0].lower() in {"python", "py"}:
        command[0] = sys.executable

    return _run_command(
        command,
        workspace_root=target_cwd,
        timeout_seconds=timeout_seconds,
        max_output=max_output,
    )


def compile_python(workspace_root: str, max_files: int = 500) -> str:
    """只编译 Python 源码语法，不执行项目代码，不写 __pycache__。

    这比 compileall 更适合第一版权限环境，因为它不会向项目目录写缓存文件。
    """

    root = Path(workspace_root).resolve()
    python_files = [
        path
        for path in root.rglob("*.py")
        if not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(root).parts)
    ]
    if len(python_files) > max_files:
        raise ToolRuntimeError(f"Too many Python files to compile safely: {len(python_files)}")

    errors: list[str] = []
    for path in python_files:
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except Exception as exc:
            relative = path.relative_to(root)
            errors.append(f"{relative}: {exc}")

    if errors:
        raise ToolRuntimeError("Python syntax check failed:\n" + "\n".join(errors[:20]))
    return f"Python syntax ok: {len(python_files)} files checked"


def ensure_workspace(path: str, workspace_root: str) -> str:
    """创建普通 workspace 目录和 README.md。

    不使用 .gitkeep，因为第一版把点开头路径视为隐藏/元数据路径。
    """

    profile = default_permission_profile(workspace_root)
    target = resolve_allowed_path(path, profile, AccessMode.WRITE)
    if is_sensitive_path(path) or is_sensitive_path(target):
        raise ToolRuntimeError(f"Sensitive path is blocked: {path}")
    if target.exists() and not target.is_dir():
        raise ToolRuntimeError(f"Path exists but is not a directory: {path}")

    target.mkdir(parents=True, exist_ok=True)
    readme = target / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Workspace\n\nThis folder is reserved for Gangent-generated work files.\n",
            encoding="utf-8",
        )
    return f"Workspace ready: {path}"


def _run_git_command(
    command: list[str],
    workspace_root: str,
    timeout_seconds: int,
    max_output: int,
) -> str:
    """运行受控 git 命令。

    这里不用 shell=True，避免模型注入额外命令。
    """

    result = _run_command(command, Path(workspace_root).resolve(), timeout_seconds, max_output)
    return result


def _validated_git_paths(paths: list[str], workspace_root: str) -> list[str]:
    validated: list[str] = []
    root = Path(workspace_root).resolve()
    for requested_path in paths:
        if not isinstance(requested_path, str) or not requested_path.strip():
            raise ToolRuntimeError("git_add paths must be non-empty strings.")
        target = resolve_workspace_path(requested_path, workspace_root)
        if is_sensitive_path(requested_path) or is_sensitive_path(target):
            raise ToolRuntimeError(f"Sensitive path is blocked: {requested_path}")
        if not target.exists():
            raise ToolRuntimeError(f"git_add path does not exist: {requested_path}")
        validated.append(target.relative_to(root).as_posix())
    return validated


def _is_searchable_text_file(path: Path, root: Path) -> bool:
    if not path.is_file():
        return False
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if any(part.startswith(".") for part in relative.parts):
        return False
    if is_sensitive_path(relative) or is_sensitive_path(path):
        return False
    if path.suffix.lower() not in {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".ini", ".cfg"}:
        return False
    if path.stat().st_size > 80_000:
        return False
    try:
        return b"\x00" not in path.read_bytes()[:1024]
    except OSError:
        return False


def _is_safe_git_revision(revision: str) -> bool:
    if not isinstance(revision, str) or not revision.strip():
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9._/\-~^:@]+", revision.strip()))


def _decode_text_file(data: bytes, path: str) -> str:
    """Decode common text encodings without silently accepting binary files."""

    encodings = ["utf-8-sig"]
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        encodings.insert(0, "utf-16")
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ToolRuntimeError(
        f"File is not valid UTF-8 text: {path}. Convert it to UTF-8 or read it with a specialized tool."
    )


def _ensure_public_hostname(hostname: str) -> None:
    if hostname.lower() in {"localhost", "localhost.localdomain"}:
        raise ToolRuntimeError("fetch_url blocks localhost.")
    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ToolRuntimeError(f"fetch_url cannot resolve hostname: {hostname}") from exc
    for item in addresses:
        address = item[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ToolRuntimeError(f"fetch_url blocks non-public address: {address}")


def _strip_html(text: str) -> str:
    if "<" not in text or ">" not in text:
        return text
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    return text.strip()


def _run_command(
    command: list[str],
    workspace_root: Path,
    timeout_seconds: int,
    max_output: int,
) -> str:
    """运行固定参数命令。

    只接收程序内部构造的 list[str]，不接收模型给出的整段 shell 文本。
    """

    result = LocalRunner().run(
        SandboxCommand(
            name=command[0] if command else "command",
            args=command,
            cwd=str(workspace_root),
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output,
        )
    )
    if not result.success:
        if result.timed_out:
            raise ToolRuntimeError(result.error or "Command timed out.")
        raise ToolRuntimeError(f"Command failed with code {result.exit_code}: {result.output}")
    return result.output
