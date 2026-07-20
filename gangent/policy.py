"""Policy Check（策略检查层）。

这一层回答一个问题：模型提出的动作能不能执行。
它不负责调用模型，也不负责执行工具，只负责把 ActionDecision 转成 PolicyDecision。
"""

from __future__ import annotations

from pathlib import Path

from .command_policy import CommandRisk, assess_run_command_args
from .models import ActionDecision, DecisionType, PolicyDecision, PolicyMode
from .patch_editor import PatchError, inspect_patch_paths
from .permissions import (
    AccessMode,
    PermissionError,
    default_permission_profile,
    resolve_allowed_path,
    resolve_workspace_path,
)
from .secret_guard import is_sensitive_path, secret_labels
from .text_guard import encoding_artifact_reason


READ_ONLY_TOOLS = {
    "list_files",
    "file_info",
    "read_file",
    "read_many_files",
    "grep_files",
    "git_status",
    "git_diff",
    "git_log",
    "git_show",
    "search_context",
}
WRITE_TOOLS = {
    "write_file",
    "edit_file",
    "apply_patch",
    "export_artifact",
    "scratchpad_note",
    "memory_add",
    "ensure_workspace",
    "git_add",
    "git_commit",
}
COMMAND_TOOLS = {"compile_python", "run_tests", "run_command"}
EXTERNAL_TOOLS = {"fetch_url"}
BLOCKED_TOOLS = {"delete_file", "shell", "git", "network"}


def check_policy(
    decision: ActionDecision,
    workspace_root: str,
) -> PolicyDecision:
    """检查一个动作决策是否允许执行。

    专业说法：这是 policy enforcement point（策略执行点）之前的决策逻辑。
    通俗说法：模型说要干活以后，这里先拦一下，看看这个活能不能干。
    """

    if decision.decision_type != DecisionType.TOOL_CALL:
        return PolicyDecision(
            mode=PolicyMode.ALLOW,
            allowed=True,
            reason="Non-tool decision does not require tool permission.",
        )

    if not decision.tool_name:
        return _block("Tool name is missing.")

    if decision.tool_name in BLOCKED_TOOLS:
        return _block(f"Tool is blocked in version one: {decision.tool_name}")

    if (
        decision.tool_name not in READ_ONLY_TOOLS
        and decision.tool_name not in WRITE_TOOLS
        and decision.tool_name not in COMMAND_TOOLS
        and decision.tool_name not in EXTERNAL_TOOLS
    ):
        return _block(f"Unknown tool: {decision.tool_name}")

    if decision.tool_args is None:
        return _block("Tool args are missing.")

    if decision.tool_name in {"git_status", "git_diff"}:
        return PolicyDecision(
            mode=PolicyMode.ALLOW,
            allowed=True,
            reason=f"Read-only git tool is allowed: {decision.tool_name}",
        )

    if decision.tool_name == "git_log":
        limit = decision.tool_args.get("limit") if decision.tool_args else None
        if not isinstance(limit, int) or limit < 1 or limit > 20:
            return _block("git_log requires integer limit between 1 and 20.")
        return PolicyDecision(
            mode=PolicyMode.ALLOW,
            allowed=True,
            reason="Read-only git log is allowed.",
        )

    if decision.tool_name == "git_show":
        revision = decision.tool_args.get("revision") if decision.tool_args else None
        if not isinstance(revision, str) or not revision.strip():
            return _block("git_show requires non-empty revision.")
        return PolicyDecision(
            mode=PolicyMode.ALLOW,
            allowed=True,
            reason="Read-only git show is allowed.",
        )

    if decision.tool_name == "git_add":
        return _policy_for_git_add(decision, workspace_root)

    if decision.tool_name == "git_commit":
        return _policy_for_git_commit(decision)

    if decision.tool_name == "search_context":
        query = decision.tool_args.get("query") if decision.tool_args else None
        top_k = decision.tool_args.get("top_k") if decision.tool_args else None
        if not isinstance(query, str) or not query.strip():
            return _block("search_context requires non-empty string query.")
        if not isinstance(top_k, int):
            return _block("search_context requires integer top_k.")
        if top_k < 1 or top_k > 8:
            return _block("search_context top_k must be between 1 and 8.")
        return PolicyDecision(
            mode=PolicyMode.ALLOW,
            allowed=True,
            reason="Workspace RAG search is allowed with access filtering and redaction.",
        )

    if decision.tool_name == "fetch_url":
        return _policy_for_fetch_url(decision)

    if decision.tool_name == "read_many_files":
        return _policy_for_read_many_files(decision, workspace_root)

    if decision.tool_name == "grep_files":
        return _policy_for_grep_files(decision, workspace_root)

    if decision.tool_name == "compile_python":
        return PolicyDecision(
            mode=PolicyMode.ALLOW,
            allowed=True,
            reason="Python syntax check is allowed because it does not execute project code.",
        )

    if decision.tool_name == "run_tests":
        return PolicyDecision(
            mode=PolicyMode.ESCALATE,
            allowed=False,
            reason="Running tests executes workspace code and requires user confirmation.",
        )

    if decision.tool_name == "run_command":
        return _policy_for_run_command(decision, workspace_root)

    if decision.tool_name == "apply_patch":
        return _policy_for_apply_patch(decision, workspace_root)

    if decision.tool_name == "export_artifact":
        return _policy_for_export_artifact(decision)

    if decision.tool_name == "scratchpad_note":
        return _policy_for_scratchpad_note(decision)

    if decision.tool_name == "memory_add":
        return _policy_for_memory_add(decision)

    path = str(decision.tool_args.get("path", ""))
    if not path:
        return _block("Tool path is missing.")

    try:
        target = resolve_workspace_path(path, workspace_root)
    except PermissionError as exc:
        return _block(str(exc))

    if decision.tool_name == "file_info":
        return _policy_for_file_info(target, path)

    if decision.tool_name == "read_file":
        return _policy_for_read_file(decision, target, path)

    if decision.tool_name == "list_files":
        return _policy_for_list_files(target, path)

    if decision.tool_name == "write_file":
        return _policy_for_write_file(decision, workspace_root, path)

    if decision.tool_name == "edit_file":
        return _policy_for_edit_file(decision, workspace_root, path)

    if decision.tool_name == "ensure_workspace":
        return _policy_for_ensure_workspace(workspace_root, path)

    return _block(f"Unhandled tool: {decision.tool_name}")


def _policy_for_list_files(target: Path, requested_path: str) -> PolicyDecision:
    """list_files 的策略：只允许目录，且必须存在。"""

    if is_sensitive_path(requested_path) or is_sensitive_path(target):
        return _block(f"Sensitive path is blocked: {requested_path}")
    if not target.exists():
        return _block(f"Directory does not exist: {requested_path}")
    if not target.is_dir():
        return _block(f"Path is not a directory: {requested_path}")
    if any(part.startswith(".") for part in target.parts):
        return PolicyDecision(
            mode=PolicyMode.ESCALATE,
            allowed=False,
            reason=f"Listing hidden or metadata directory requires confirmation: {requested_path}",
        )
    return PolicyDecision(
        mode=PolicyMode.ALLOW,
        allowed=True,
        reason=f"Read-only directory listing is allowed: {requested_path}",
    )


def _policy_for_file_info(target: Path, requested_path: str) -> PolicyDecision:
    """file_info is read-only metadata inspection for both files and directories."""

    if is_sensitive_path(requested_path) or is_sensitive_path(target):
        return _block(f"Sensitive path is blocked: {requested_path}")
    if not target.exists():
        return _block(f"Path does not exist: {requested_path}")
    if any(part.startswith(".") for part in target.parts):
        return PolicyDecision(
            mode=PolicyMode.ESCALATE,
            allowed=False,
            reason=f"Inspecting hidden or metadata path requires confirmation: {requested_path}",
        )
    return PolicyDecision(
        mode=PolicyMode.ALLOW,
        allowed=True,
        reason=f"Read-only path metadata inspection is allowed: {requested_path}",
    )


def _policy_for_read_file(decision: ActionDecision, target: Path, requested_path: str) -> PolicyDecision:
    """read_file 的策略：允许小型文本文件，敏感/隐藏文件升级确认。"""

    if is_sensitive_path(requested_path) or is_sensitive_path(target):
        return _block(f"Sensitive path is blocked: {requested_path}")
    if not target.exists():
        return _block(f"File does not exist: {requested_path}")
    if not target.is_file():
        return _block(f"Path is not a file: {requested_path}")

    if any(part.startswith(".") for part in target.parts):
        return PolicyDecision(
            mode=PolicyMode.ESCALATE,
            allowed=False,
            reason=f"Reading hidden or metadata file requires confirmation: {requested_path}",
        )

    start_line = decision.tool_args.get("start_line") if decision.tool_args else None
    max_lines = decision.tool_args.get("max_lines") if decision.tool_args else None
    chunked_read = start_line is not None or max_lines is not None

    if chunked_read:
        if start_line is not None and (not isinstance(start_line, int) or start_line < 1):
            return _block("read_file start_line must be an integer >= 1.")
        if max_lines is not None and (not isinstance(max_lines, int) or max_lines < 1 or max_lines > 400):
            return _block("read_file max_lines must be between 1 and 400.")
        return PolicyDecision(
            mode=PolicyMode.ALLOW,
            allowed=True,
            reason=f"Chunked file read is allowed: {requested_path}",
        )

    if target.stat().st_size > 20_000:
        return PolicyDecision(
            mode=PolicyMode.ALLOW,
            allowed=True,
            reason=f"Large file read is allowed as an automatic partial chunk: {requested_path}",
        )

    return PolicyDecision(
        mode=PolicyMode.ALLOW,
        allowed=True,
        reason=f"Read-only file access is allowed: {requested_path}",
    )


def _policy_for_read_many_files(decision: ActionDecision, workspace_root: str) -> PolicyDecision:
    paths = decision.tool_args.get("paths") if decision.tool_args else None
    if not isinstance(paths, list) or not paths:
        return _block("read_many_files requires non-empty paths list.")
    if len(paths) > 8:
        return _block("read_many_files supports at most 8 paths.")
    for requested_path in paths:
        if not isinstance(requested_path, str) or not requested_path.strip():
            return _block("read_many_files paths must be non-empty strings.")
        try:
            target = resolve_workspace_path(requested_path, workspace_root)
        except PermissionError as exc:
            return _block(str(exc))
        policy = _policy_for_read_file(decision, target, requested_path)
        if not policy.allowed:
            return policy
    return PolicyDecision(
        mode=PolicyMode.ALLOW,
        allowed=True,
        reason=f"Batch read is allowed for {len(paths)} file(s).",
    )


def _policy_for_grep_files(decision: ActionDecision, workspace_root: str) -> PolicyDecision:
    pattern = decision.tool_args.get("pattern") if decision.tool_args else None
    path = str(decision.tool_args.get("path", ".")) if decision.tool_args else "."
    max_results = decision.tool_args.get("max_results") if decision.tool_args else None
    if not isinstance(pattern, str) or not pattern.strip():
        return _block("grep_files requires non-empty pattern.")
    if len(pattern) > 200:
        return _block("grep_files pattern is too long.")
    if not isinstance(max_results, int) or max_results < 1 or max_results > 100:
        return _block("grep_files max_results must be between 1 and 100.")
    try:
        target = resolve_workspace_path(path, workspace_root)
    except PermissionError as exc:
        return _block(str(exc))
    if is_sensitive_path(path) or is_sensitive_path(target):
        return _block(f"Sensitive path is blocked: {path}")
    if not target.exists():
        return _block(f"grep_files path does not exist: {path}")
    return PolicyDecision(
        mode=PolicyMode.ALLOW,
        allowed=True,
        reason=f"Workspace grep is allowed: {path}",
    )


def _policy_for_write_file(
    decision: ActionDecision,
    workspace_root: str,
    requested_path: str,
) -> PolicyDecision:
    """write_file 的策略：只允许写入普通小型文本文件。"""

    profile = default_permission_profile(workspace_root)
    try:
        target = resolve_allowed_path(requested_path, profile, AccessMode.WRITE)
    except PermissionError as exc:
        return _block(str(exc))

    content = decision.tool_args.get("content") if decision.tool_args else None
    if not isinstance(content, str):
        return _block("write_file requires string content.")
    labels = secret_labels(content)
    if labels:
        return _block(f"write_file content contains possible secrets: {', '.join(labels)}")
    artifact_reason = encoding_artifact_reason(content)
    if artifact_reason:
        return _block(artifact_reason)
    if len(content.encode("utf-8")) > profile.max_file_bytes:
        return _block("write_file content is too large for version one.")

    if target.exists() and not target.is_file():
        return _block(f"Path is not a file: {requested_path}")

    if target.exists() and not bool(decision.tool_args.get("overwrite", False)):
        return PolicyDecision(
            mode=PolicyMode.ESCALATE,
            allowed=False,
            reason=f"Overwriting an existing file requires confirmation: {requested_path}",
        )

    return PolicyDecision(
        mode=PolicyMode.ALLOW,
        allowed=True,
        reason=f"Workspace file write is allowed: {requested_path}",
    )


def _policy_for_edit_file(
    decision: ActionDecision,
    workspace_root: str,
    requested_path: str,
) -> PolicyDecision:
    """edit_file 的策略：只允许精确替换普通小型文本文件。"""

    profile = default_permission_profile(workspace_root)
    try:
        target = resolve_allowed_path(requested_path, profile, AccessMode.WRITE)
    except PermissionError as exc:
        return _block(str(exc))

    if not target.exists():
        return _block(f"File does not exist: {requested_path}")
    if not target.is_file():
        return _block(f"Path is not a file: {requested_path}")
    if target.stat().st_size > profile.max_file_bytes:
        return PolicyDecision(
            mode=PolicyMode.ESCALATE,
            allowed=False,
            reason=f"Editing a large file requires confirmation: {requested_path}",
        )

    old_text = decision.tool_args.get("old_text") if decision.tool_args else None
    new_text = decision.tool_args.get("new_text") if decision.tool_args else None
    if not isinstance(old_text, str) or not old_text:
        return _block("edit_file requires non-empty old_text.")
    if not isinstance(new_text, str):
        return _block("edit_file requires string new_text.")
    labels = secret_labels(new_text)
    if labels:
        return _block(f"edit_file new_text contains possible secrets: {', '.join(labels)}")
    artifact_reason = encoding_artifact_reason(new_text)
    if artifact_reason:
        return _block(artifact_reason)

    return PolicyDecision(
        mode=PolicyMode.ALLOW,
        allowed=True,
        reason=f"Exact workspace file edit is allowed: {requested_path}",
    )


def _policy_for_ensure_workspace(workspace_root: str, requested_path: str) -> PolicyDecision:
    """ensure_workspace 的策略：允许创建普通目录和 README.md。"""

    profile = default_permission_profile(workspace_root)
    try:
        target = resolve_allowed_path(requested_path, profile, AccessMode.WRITE)
    except PermissionError as exc:
        return _block(str(exc))

    if is_sensitive_path(requested_path) or is_sensitive_path(target):
        return _block(f"Sensitive path is blocked: {requested_path}")
    if target.exists() and not target.is_dir():
        return _block(f"Path exists but is not a directory: {requested_path}")

    return PolicyDecision(
        mode=PolicyMode.ALLOW,
        allowed=True,
        reason=f"Workspace directory creation is allowed: {requested_path}",
    )


def _policy_for_apply_patch(decision: ActionDecision, workspace_root: str) -> PolicyDecision:
    """apply_patch policy: parse paths before execution."""

    patch = decision.tool_args.get("patch") if decision.tool_args else None
    if not isinstance(patch, str) or not patch.strip():
        return _block("apply_patch requires non-empty patch string.")
    try:
        paths = inspect_patch_paths(patch)
    except PatchError as exc:
        return _block(str(exc))
    if len(paths) > 8:
        return _block("apply_patch touches too many files for version one.")

    profile = default_permission_profile(workspace_root)
    for requested_path in paths:
        try:
            target = resolve_allowed_path(requested_path, profile, AccessMode.WRITE)
        except PermissionError as exc:
            return _block(str(exc))
        if is_sensitive_path(requested_path) or is_sensitive_path(target):
            return _block(f"Sensitive path is blocked: {requested_path}")

    return PolicyDecision(
        mode=PolicyMode.ALLOW,
        allowed=True,
        reason=f"Restricted patch is allowed for {len(paths)} file(s).",
    )


def _policy_for_export_artifact(decision: ActionDecision) -> PolicyDecision:
    name = decision.tool_args.get("name") if decision.tool_args else None
    content = decision.tool_args.get("content") if decision.tool_args else None
    if not isinstance(name, str) or not name.strip():
        return _block("export_artifact requires non-empty name.")
    if "/" in name or "\\" in name or name.startswith("."):
        return _block("export_artifact name must be a simple non-hidden file name.")
    if not isinstance(content, str):
        return _block("export_artifact requires string content.")
    labels = secret_labels(content)
    if labels:
        return _block(f"export_artifact content contains possible secrets: {', '.join(labels)}")
    return PolicyDecision(
        mode=PolicyMode.ALLOW,
        allowed=True,
        reason=f"Artifact export is allowed: artifacts/{name}",
    )


def _policy_for_scratchpad_note(decision: ActionDecision) -> PolicyDecision:
    note = decision.tool_args.get("note") if decision.tool_args else None
    if not isinstance(note, str) or not note.strip():
        return _block("scratchpad_note requires non-empty note.")
    labels = secret_labels(note)
    if labels:
        return _block(f"scratchpad_note contains possible secrets: {', '.join(labels)}")
    if len(note.encode("utf-8")) > 4_000:
        return _block("scratchpad_note is too large.")
    return PolicyDecision(
        mode=PolicyMode.ALLOW,
        allowed=True,
        reason="Scratchpad note is allowed.",
    )


def _policy_for_memory_add(decision: ActionDecision) -> PolicyDecision:
    node_type = decision.tool_args.get("node_type") if decision.tool_args else None
    content = decision.tool_args.get("content") if decision.tool_args else None
    summary = decision.tool_args.get("summary", "") if decision.tool_args else ""
    if not isinstance(node_type, str) or not node_type.strip():
        return _block("memory_add requires non-empty node_type.")
    if not isinstance(content, str) or not content.strip():
        return _block("memory_add requires non-empty content.")
    if len(content.encode("utf-8")) > 4_000:
        return _block("memory_add content is too large.")
    labels = secret_labels("\n".join([content, str(summary)]))
    if labels:
        return _block(f"memory_add content contains possible secrets: {', '.join(labels)}")
    return PolicyDecision(
        mode=PolicyMode.ALLOW,
        allowed=True,
        reason="Memory graph write is allowed.",
    )


def _policy_for_fetch_url(decision: ActionDecision) -> PolicyDecision:
    url = decision.tool_args.get("url") if decision.tool_args else None
    max_bytes = decision.tool_args.get("max_bytes") if decision.tool_args else None
    if not isinstance(url, str) or not url.strip():
        return _block("fetch_url requires non-empty URL.")
    if not (url.startswith("http://") or url.startswith("https://")):
        return _block("fetch_url only supports http and https URLs.")
    if not isinstance(max_bytes, int) or max_bytes < 1_000 or max_bytes > 200_000:
        return _block("fetch_url max_bytes must be between 1,000 and 200,000.")
    return PolicyDecision(
        mode=PolicyMode.ESCALATE,
        allowed=False,
        reason="fetch_url accesses the external network and requires user confirmation.",
    )


def _policy_for_run_command(decision: ActionDecision, workspace_root: str) -> PolicyDecision:
    """run_command policy: structured argv + risk classifier."""

    assessment = assess_run_command_args(decision.tool_args or {})
    if assessment.risk == CommandRisk.BLOCK:
        return _block(assessment.reason)

    try:
        target_cwd = resolve_workspace_path(assessment.cwd, workspace_root)
    except PermissionError as exc:
        return _block(str(exc))
    if not target_cwd.exists():
        return _block(f"Command cwd does not exist: {assessment.cwd}")
    if not target_cwd.is_dir():
        return _block(f"Command cwd is not a directory: {assessment.cwd}")
    if is_sensitive_path(assessment.cwd) or is_sensitive_path(target_cwd):
        return _block(f"Sensitive cwd is blocked: {assessment.cwd}")

    if assessment.risk == CommandRisk.ALLOW:
        return PolicyDecision(
            mode=PolicyMode.ALLOW,
            allowed=True,
            reason=assessment.reason,
        )

    return PolicyDecision(
        mode=PolicyMode.ESCALATE,
        allowed=False,
        reason=assessment.reason,
    )


def _policy_for_git_add(decision: ActionDecision, workspace_root: str) -> PolicyDecision:
    paths = decision.tool_args.get("paths") if decision.tool_args else None
    if not isinstance(paths, list) or not paths:
        return _block("git_add requires non-empty paths list.")
    profile = default_permission_profile(workspace_root)
    validated: list[str] = []
    for requested_path in paths:
        if not isinstance(requested_path, str) or not requested_path.strip():
            return _block("git_add paths must be non-empty strings.")
        try:
            target = resolve_allowed_path(requested_path, profile, AccessMode.WRITE)
        except PermissionError as exc:
            return _block(str(exc))
        if is_sensitive_path(requested_path) or is_sensitive_path(target):
            return _block(f"Sensitive path is blocked: {requested_path}")
        validated.append(requested_path)
    return PolicyDecision(
        mode=PolicyMode.ESCALATE,
        allowed=False,
        reason=f"git_add stages {len(validated)} path(s) and requires user confirmation.",
    )


def _policy_for_git_commit(decision: ActionDecision) -> PolicyDecision:
    message = decision.tool_args.get("message") if decision.tool_args else None
    if not isinstance(message, str) or not message.strip():
        return _block("git_commit requires non-empty message.")
    normalized = " ".join(message.strip().split())
    if len(normalized) > 120:
        return _block("git_commit message is too long; keep it under 120 characters.")
    return PolicyDecision(
        mode=PolicyMode.ESCALATE,
        allowed=False,
        reason="git_commit creates a repository commit and requires user confirmation.",
    )


def _block(reason: str) -> PolicyDecision:
    """生成阻止执行的策略结果。"""

    return PolicyDecision(mode=PolicyMode.BLOCK, allowed=False, reason=reason)
