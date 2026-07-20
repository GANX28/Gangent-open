"""第一版 runtime 骨架的核心数据结构。

这里先只定义“对象长什么样”，不写复杂行为。
原因是第一版要先把任务、状态、决策、工具、审计这些边界固定下来，
后面再逐步接入模型、工具和策略逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    """返回统一的 UTC 时间字符串，方便后续日志和审计按同一时间标准记录。"""
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    """生成带前缀的短 ID，用于区分 task、tool call 等不同类型对象。"""
    return f"{prefix}_{uuid4().hex[:12]}"


class TaskStatus(str, Enum):
    """任务整体生命周期状态。

    它描述的是“这个任务整体做到哪了”，不是 agent 内部当前小步骤。
    """

    PENDING = "pending"
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentPhase(str, Enum):
    """Agent 在一次 runtime loop 内部的运行阶段。

    这里不用 AgentStatus，是为了避免和 TaskStatus 混淆。
    TaskStatus 管任务整体，AgentPhase 管 agent 此刻正在干什么。
    """

    IDLE = "idle"
    THINKING = "thinking"
    VALIDATING = "validating"
    EXECUTING_TOOL = "executing_tool"
    UPDATING_STATE = "updating_state"
    AUDITING = "auditing"


class MessageRole(str, Enum):
    """消息来源类型，用来统一保存用户、模型、工具和系统消息。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class PlanStepStatus(str, Enum):
    """轻量计划步骤的状态。

    第一版不做复杂 planner，但先保留计划步骤对象，方便后续扩展。
    """

    TODO = "todo"
    RUNNING = "running"
    DONE = "done"
    BLOCKED = "blocked"


class DecisionType(str, Enum):
    """下一步动作类型。

    模型或规划层以后不能只输出自由文本，而要落到这些结构化动作之一。
    """

    DIRECT_RESPONSE = "direct_response"
    TOOL_CALL = "tool_call"
    ASK_USER = "ask_user"
    FINISH = "finish"
    FAIL = "fail"


class PolicyMode(str, Enum):
    """策略判断结果。

    allow/block/escalate 是最小控制层，后续所有工具执行都应该先经过它。
    """

    ALLOW = "allow"
    BLOCK = "block"
    ESCALATE = "escalate"


@dataclass
class TaskInput:
    """用户原始输入的标准包装。

    不直接把自然语言丢进 runtime，是为了让入口参数稳定、可检查。
    """

    goal: str
    user_message: str
    workspace_root: str
    constraints: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)


@dataclass
class Task:
    """任务档案。

    Task 记录“任务是什么”和“整体状态是什么”。
    执行过程中的动态细节不放这里，而是放在 AgentState。
    """

    task_id: str
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass
class Message:
    """一条上下文消息。

    用统一结构保存用户输入、模型输出、工具返回，后续喂给模型或写日志都方便。
    """

    role: MessageRole
    content: str
    timestamp: str = field(default_factory=utc_now)


@dataclass
class PlanStep:
    """轻量计划步骤。

    第一版不追求复杂长程规划，但保留这个对象可以支持后续逐步规划。
    """

    step_id: str
    title: str
    status: PlanStepStatus = PlanStepStatus.TODO
    description: str = ""
    purpose: str = ""
    expected_output: str = ""
    tool_hint: str | None = None
    result_summary: str = ""


@dataclass
class Plan:
    """Structured execution plan for one task."""

    plan_id: str
    task_id: str
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass
class ActionDecision:
    """下一步动作决策。

    这是 Planning Layer 和 Agent Core 之间的接口。
    后续接模型时，模型输出应该被解析成这个结构，而不是直接执行文本。
    """

    decision_type: DecisionType
    reason: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    response_text: str | None = None


@dataclass
class PolicyDecision:
    """策略检查结果。

    它回答“这个动作能不能执行、为什么”。
    这是第一版控制层和后续 harness engineering 的入口。
    """

    mode: PolicyMode
    allowed: bool
    reason: str


@dataclass
class ToolCallRecord:
    """工具调用请求记录。

    工具执行前先记录请求和策略判断，方便后续审计和排错。
    """

    call_id: str
    tool_name: str
    args: dict[str, Any]
    policy_decision: PolicyDecision
    created_at: str = field(default_factory=utc_now)


@dataclass
class ToolResult:
    """工具执行结果。

    工具不能只返回散乱字符串，要统一记录成功、输出和错误。
    """

    call_id: str
    success: bool
    output: str = ""
    error: str | None = None
    reused: bool = False
    snipped: bool = False
    finished_at: str = field(default_factory=utc_now)


@dataclass
class RuntimeStats:
    """一次 runtime 任务的基础统计信息。"""

    duration_seconds: float = 0.0
    step_count: int = 0
    tool_call_count: int = 0
    error_count: int = 0
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Agent 当前运行状态。

    AgentState 是 runtime 的工作台：保存当前步数、上下文、消息、最近决策、
    最近工具结果和错误。它让 agent 能持续执行，而不是每轮重新开始。
    """

    task_id: str
    workspace_root: str = ""
    phase: AgentPhase = AgentPhase.IDLE
    step_index: int = 0
    context_summary: str = ""
    plan_id: str | None = None
    messages: list[Message] = field(default_factory=list)
    plan_steps: list[PlanStep] = field(default_factory=list)
    event_cursor: int = 0
    event_summaries: list[str] = field(default_factory=list)
    event_runtime_state: str = "idle"
    event_count: int = 0
    replan_count: int = 0
    interrupt_count: int = 0
    pending_event_count: int = 0
    stabilization_required: bool = False
    stale_outputs: list[str] = field(default_factory=list)
    plan_patch_summaries: list[str] = field(default_factory=list)
    budget_profile: str = ""
    runtime_step_limit: int = 0
    runtime_remaining_steps: int = 0
    total_step_budget: int = 0
    total_remaining_steps: int = 0
    last_decision: ActionDecision | None = None
    last_tool_result: ToolResult | None = None
    errors: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now)


@dataclass
class AuditEntry:
    """一轮执行的审计记录。

    它保存本轮发生了什么、策略如何判断、工具结果是什么、当时状态摘要是什么。
    第一版先定义结构，后续再接入实际写入 JSONL 的审计日志。
    """

    task_id: str
    step_index: int
    decision: ActionDecision | None
    policy: PolicyDecision | None
    tool_result: ToolResult | None
    state_summary: str
    timestamp: str = field(default_factory=utc_now)
