"""Session State（会话状态层）。

Session State 负责在同一个 CLI 进程里保存跨任务上下文。
它不是长期记忆，也不写磁盘；它只让用户在一次 gangent.cli 运行中可以连续交流。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import DecisionType, TaskInput, new_id, utc_now
from .runtime import RuntimeResult
from .secret_guard import redact_secrets
from .state import state_summary


@dataclass
class SessionTurn:
    """一次用户输入和 runtime 结果的压缩记录。"""

    user_message: str
    task_id: str
    task_status: str
    final_answer: str | None
    tool_summaries: list[str] = field(default_factory=list)
    state_summary: str = ""


@dataclass
class SessionState:
    """一个 CLI 会话的短期上下文。

    context_summary 是给下一轮模型看的压缩摘要。
    turns 是给程序和调试使用的结构化历史。
    """

    session_id: str
    workspace_root: str
    context_summary: str = ""
    turns: list[SessionTurn] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


def create_session(workspace_root: str) -> SessionState:
    """创建一个新的短期会话。"""

    return SessionState(session_id=new_id("session"), workspace_root=workspace_root)


def build_task_input_from_session(
    session: SessionState,
    user_message: str,
) -> TaskInput:
    """把当前会话状态合并进新任务输入。

    关键点：Runtime 仍然一次只执行一个 Task。
    Session 只是在 TaskInput.constraints 里注入上一轮压缩上下文。
    """

    constraints = [
        "obey runtime safety boundaries",
        f"session_id={session.session_id}",
    ]
    if session.context_summary:
        constraints.append(f"Session context:\n{session.context_summary}")

    return TaskInput(
        goal=redact_secrets(user_message),
        user_message=redact_secrets(user_message),
        workspace_root=session.workspace_root,
        constraints=constraints,
    )


def update_session_from_result(
    session: SessionState,
    user_message: str,
    result: RuntimeResult,
    max_turns: int = 4,
) -> SessionState:
    """把一次 runtime 结果压缩回 SessionState。

    默认行为是确定性摘要，不做每轮模型压缩。
    只有后续显式开启相关能力时，才考虑模型辅助压缩。
    """

    turn = SessionTurn(
        user_message=redact_secrets(user_message),
        task_id=result.task.task_id,
        task_status=result.task.status.value,
        final_answer=redact_secrets(_final_answer(result) or "") or None,
        tool_summaries=_tool_summaries(result),
        state_summary=state_summary(result.state),
    )
    session.turns.append(turn)
    if len(session.turns) > max_turns:
        session.turns = session.turns[-max_turns:]

    session.context_summary = _build_context_summary(session.turns)
    session.updated_at = utc_now()
    return session


def reset_session(session: SessionState) -> SessionState:
    """清空当前会话历史，但保留同一个 workspace root。"""

    return create_session(session.workspace_root)


def _final_answer(result: RuntimeResult) -> str | None:
    decision = result.state.last_decision
    if not decision:
        return None
    if decision.decision_type in {DecisionType.FINISH, DecisionType.DIRECT_RESPONSE}:
        return decision.response_text
    return None


def _tool_summaries(result: RuntimeResult) -> list[str]:
    summaries: list[str] = []
    for step in result.steps:
        if not step.tool_result:
            continue
        tool_name = step.decision.tool_name or "(unknown tool)"
        status = "success" if step.tool_result.success else "failed"
        content = step.tool_result.output if step.tool_result.success else step.tool_result.error or ""
        content = redact_secrets(content)
        content = content.replace("\n", " ")
        if len(content) > 240:
            content = content[:240] + "..."
        summaries.append(f"{tool_name}: {status}; {content}")
    return summaries


def _build_context_summary(turns: list[SessionTurn]) -> str:
    lines: list[str] = []
    for index, turn in enumerate(turns, start=1):
        lines.append(f"Turn {index}: user={turn.user_message}")
        if turn.final_answer:
            lines.append(f"  final_answer={turn.final_answer}")
        for tool_summary in turn.tool_summaries[:3]:
            lines.append(f"  tool={tool_summary}")
        lines.append(f"  state={turn.state_summary}")
    return "\n".join(lines)
