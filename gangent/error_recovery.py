"""Error Recovery v1.

Error Recovery（错误恢复）把失败结果转换成下一轮可执行的恢复提示。
第一版不自动重写计划，只把 deterministic hint 放回 AgentState。
"""

from __future__ import annotations

from .models import ActionDecision, AgentState, Message, MessageRole, PolicyDecision, ToolResult, utc_now


def recovery_hint_for_tool_result(decision: ActionDecision, result: ToolResult) -> str | None:
    """Create a recovery hint for a failed tool result."""

    if result.success:
        return None
    error = result.error or "Tool failed."
    lower = error.lower()
    tool_name = decision.tool_name or "(unknown tool)"

    if '"error_type":"tool_argument_validation"' in lower:
        return (
            f"Recovery hint: {tool_name} arguments do not match its JSON Schema. "
            "Read the structured issues in the tool error, correct only those argument fields, "
            "and retry once without changing tools unless the task itself requires replanning."
        )
    if "file does not exist" in lower or "directory does not exist" in lower:
        return (
            f"Recovery hint: {tool_name} failed because the path does not exist. "
            "Use list_files or search_context to locate the correct workspace-relative path before retrying."
        )
    if "patch context was not found" in lower or "old_text was not found" in lower:
        return (
            f"Recovery hint: {tool_name} failed because the file content did not match the proposed edit. "
            "Read the target file, rebuild the patch from current content, then retry a smaller update."
        )
    if "patch context is ambiguous" in lower or "matched multiple" in lower:
        return (
            f"Recovery hint: {tool_name} matched multiple locations. "
            "Read more surrounding context and retry with a more specific patch or exact edit."
        )
    if "command failed" in lower:
        return (
            f"Recovery hint: {tool_name} command failed. "
            "Use the command output to identify the failing file or test, inspect that context, then make the smallest fix."
        )
    if "timed out" in lower:
        return (
            f"Recovery hint: {tool_name} timed out. "
            "Retry with a narrower command or inspect the relevant files before running a broader check."
        )
    return (
        f"Recovery hint: {tool_name} failed with: {error}. "
        "Inspect current context, choose a safer smaller step, and retry only after grounding the next action."
    )


def recovery_hint_for_policy(decision: ActionDecision, policy: PolicyDecision) -> str:
    """Create a recovery hint when policy blocks or approval is unavailable."""

    tool_name = decision.tool_name or "(unknown tool)"
    lower_reason = policy.reason.lower()
    if tool_name == "run_command" and (
        "executable is blocked" in lower_reason or "shell metacharacters" in lower_reason
    ):
        return (
            f"Recovery hint: policy prevented run_command: {policy.reason}. "
            "Do not use powershell, cmd, bash, shell redirection, pipes, or raw shell text. "
            "For file creation use write_file with path/content/overwrite. "
            "For existing-file changes use edit_file or apply_patch. "
            "For project inspection use list_files, read_file, read_many_files, grep_files, or git_status."
        )
    return (
        f"Recovery hint: policy prevented {tool_name}: {policy.reason}. "
        "Choose an allowed tool, narrow the action, or ask the user for explicit approval when escalation is required."
    )


def attach_recovery_hint(state: AgentState, hint: str | None) -> AgentState:
    """Attach a recovery hint to messages and context summary."""

    if not hint:
        return state
    state.messages.append(Message(role=MessageRole.SYSTEM, content=hint))
    if "Recovery hints:" not in state.context_summary:
        state.context_summary += "\nRecovery hints:"
    state.context_summary += f"\n- {hint}"
    state.updated_at = utc_now()
    return state
