"""Task 和 AgentState 的生命周期辅助函数。

这里把状态变化集中成函数，而不是散落在 demo 或未来 runtime loop 里。
原因是：状态迁移越集中，后续越容易审计、测试和替换成更正式的状态机。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .models import (
    ActionDecision,
    AgentPhase,
    AgentState,
    Message,
    MessageRole,
    Task,
    TaskInput,
    TaskStatus,
    ToolResult,
    new_id,
    utc_now,
)
from .context_maintenance import snip_tool_result_for_state


def create_task(task_input: TaskInput) -> Task:
    """从 TaskInput 创建 Task。

    这里做最小输入校验：目标不能为空。
    Task 只保存任务档案，不保存运行过程细节。
    """

    goal = task_input.goal.strip()
    if not goal:
        raise ValueError("Task goal cannot be empty.")

    return Task(task_id=new_id("task"), goal=goal)


def create_initial_state(task: Task, task_input: TaskInput) -> AgentState:
    """创建初始 AgentState。

    初始状态绑定 task_id，并把用户原始输入放进 messages。
    这样 runtime 第一轮决策时能看到用户最初说了什么。
    """

    context_lines = [f"Goal: {task.goal}"]
    if task_input.constraints:
        context_lines.append("Context and constraints:")
        context_lines.extend(f"- {item}" for item in task_input.constraints)

    return AgentState(
        task_id=task.task_id,
        workspace_root=task_input.workspace_root,
        context_summary="\n".join(context_lines),
        messages=[
            Message(role=MessageRole.USER, content=task_input.user_message),
        ],
    )


def set_task_status(task: Task, status: TaskStatus) -> Task:
    """更新任务整体状态。

    TaskStatus 表示任务生命周期，所以每次修改时同时更新时间戳。
    """

    task.status = status
    task.updated_at = utc_now()
    return task


def set_phase(state: AgentState, phase: AgentPhase) -> AgentState:
    """更新 agent 当前运行阶段。

    AgentPhase 表示 runtime loop 内部阶段，比如 thinking 或 executing_tool。
    它比 TaskStatus 更细，用于调试和后续审计。
    """

    state.phase = phase
    state.updated_at = utc_now()
    return state


def start_task(task: Task, state: AgentState) -> tuple[Task, AgentState]:
    """让任务进入运行状态。

    任务整体从 pending 变成 running；
    agent 内部阶段进入 thinking，表示下一步要开始做决策。
    """

    set_task_status(task, TaskStatus.RUNNING)
    set_phase(state, AgentPhase.THINKING)
    return task, state


def attach_decision(state: AgentState, decision: ActionDecision) -> AgentState:
    """把最近一次动作决策写回 AgentState。

    同时把决策原因追加到 messages，方便后续模型上下文和审计记录复用。
    """

    state.last_decision = decision
    state.messages.append(Message(role=MessageRole.ASSISTANT, content=decision.reason))
    state.updated_at = utc_now()
    return state


def attach_tool_result(state: AgentState, result: ToolResult) -> AgentState:
    """把工具执行结果写回 AgentState。

    成功时记录 output，失败时记录 error。
    第一版只更新状态，后续审计层会把它写入 AuditEntry。
    """

    state.last_tool_result = result
    content = snip_tool_result_for_state(result)
    if result.success and content != result.output:
        result.snipped = True
    state.messages.append(Message(role=MessageRole.TOOL, content=content))
    state.updated_at = utc_now()
    return state


def advance_step(state: AgentState) -> AgentState:
    """推进一步执行计数。

    step_index 是 runtime loop 的最小进度标记。
    后续可以用它防止无限循环，也能定位审计日志中的具体轮次。
    """

    state.step_index += 1
    state.updated_at = utc_now()
    return state


def add_error(state: AgentState, message: str) -> AgentState:
    """记录运行过程错误。

    错误先留在 AgentState，后续可以决定是 failed、waiting_user，还是继续恢复。
    """

    state.errors.append(message)
    state.updated_at = utc_now()
    return state


def state_summary(state: AgentState) -> str:
    """生成给人看的状态摘要。

    这个摘要不是完整存档，而是便于日志、终端输出和调试快速理解当前状态。
    """

    error_count = len(state.errors)
    message_count = len(state.messages)
    event_count = len(state.event_summaries)
    return (
        f"task_id={state.task_id}; phase={state.phase.value}; "
        f"step={state.step_index}; messages={message_count}; events={event_count}; errors={error_count}"
    )


def state_snapshot(state: AgentState) -> dict[str, Any]:
    """生成完整状态快照。

    第一版用 dataclass 转 dict。后续如果接持久化或 checkpoint，可以从这里扩展。
    """

    return asdict(state)
