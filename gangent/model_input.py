"""把 Task 和 AgentState 转换成模型输入。

这一层解决的问题是：runtime 内部对象不能直接丢给大模型。
需要先压成稳定、清楚、可审计的消息结构，让模型知道任务目标、当前状态、
可用工具和输出要求。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .context_manager import build_context_bundle
from .context_maintenance import compact_old_tool_message, estimate_context_budget, stable_prefix_hash
from .manifests import build_execution_manifest, format_manifest_prompt
from .models import AgentState, MessageRole, Task
from .secret_guard import redact_secrets
from .task_profile import task_execution_profile


@dataclass
class ModelInput:
    """一次模型调用所需要的最小输入包。

    messages 是给模型看的上下文；tools 是模型可以选择调用的工具定义。
    这里不直接绑定某个厂商 SDK，后面 OpenAI、Claude 或本地模型都可以复用。
    """

    messages: list[dict[str, str]]
    tools: list[dict[str, Any]]
    diagnostics: dict[str, Any] = field(default_factory=dict)


SYSTEM_PROMPT = """You are the decision layer of a local agent runtime.
Choose exactly one next action: call one available tool, finish_task, ask_user, fail, or give a direct final answer when no tools are available.
Planner Control is a hard budget contract; prefer the smallest action that can finish the current plan phase.
Use tools only when needed. If the answer is already known, call finish_task immediately.
For simple direct tasks, call finish_task with answer only; omit reason unless it is needed for audit.
Never request raw shell commands for file or git work. Use constrained tools only.
For file content questions, prefer read_file over file_info. For new files use write_file; for exact existing edits use edit_file; for patches use apply_patch.
If the user asks you to create/save/write a document but does not provide exact text, draft a concise useful version yourself and call write_file.
Use workspace-relative paths. Do not invent absolute Unix paths.
Do not push, reset, clean, or run destructive git operations.
Return structured tool/function calls when possible; do not describe a tool request in plain text.
When citing files in final answers, use exact workspace-relative paths from successful tool results only.
Never claim that a file was read if a tool result said the path does not exist.
When writing files, use plain UTF-8 Chinese/English text. Avoid decorative glyphs; use ASCII "->" for arrows.
"""

DIRECT_SYSTEM_PROMPT = """Direct-answer mode.
Return only the final answer text requested by the user.
No tools are available. Do not output tool names, JSON, labels, reasoning, or explanations.
"""


def build_model_messages(task: Task, state: AgentState) -> list[dict[str, str]]:
    """构造发给模型的消息列表。

    专业说法：这是 context assembly（上下文组装），把任务、状态和历史消息
    映射成模型 API 能接收的 message list。

    通俗说法：就是把 agent 当前知道的东西整理成一段清楚的话，喂给大模型。
    """

    profile = task_execution_profile(
        type(
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
    )
    system_prompt = DIRECT_SYSTEM_PROMPT if profile.name == "direct" else SYSTEM_PROMPT
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_context_bundle(task, state, max_chars=profile.context_max_chars).text,
        },
    ]
    recent_limit = 2 if profile.name == "direct" else 4 if profile.name in {"single_read", "single_write"} else 8
    recent_messages = state.messages[-recent_limit:]
    for index, message in enumerate(recent_messages):
        role = _message_role_for_model(message.role)
        content = redact_secrets(message.content)
        if message.role == MessageRole.TOOL and index < len(recent_messages) - 2:
            content = compact_old_tool_message(content)
        messages.append({"role": role, "content": content})

    manifest_message = _execution_manifest_message(task, state, profile.name)
    if manifest_message:
        messages.append({"role": "system", "content": manifest_message})

    return messages


def _execution_manifest_message(task: Task, state: AgentState, profile_name: str) -> str:
    if profile_name == "direct":
        return ""
    task_input = type(
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
    return format_manifest_prompt(build_execution_manifest(task_input))


def build_model_input(
    task: Task,
    state: AgentState,
    tools: list[dict[str, Any]],
) -> ModelInput:
    """生成一次完整模型请求需要的输入。

    version one 只做最小拼装：messages + tools。
    不做 token 压缩、RAG 检索、长期记忆或多轮对话裁剪策略。
    """

    messages = build_model_messages(task, state)
    budget = estimate_context_budget(messages)
    diagnostics = {
        "stable_prefix_hash": stable_prefix_hash(messages[0]["content"], tools),
        "prefix_cache_strategy": "stable_system_prompt_and_tool_schema_first_dynamic_context_second",
        "prefix_cache_note": "Keep system prompt, tool schema order, and stable environment summary unchanged to improve DeepSeek prefix cache hits.",
        "execution_profile": task_execution_profile(
            type(
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
        ).name,
        "context_tier": task_execution_profile(
            type(
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
        ).context_tier,
        "context_char_count": budget.char_count,
        "estimated_context_tokens": budget.estimated_tokens,
        "context_level": budget.level,
    }
    return ModelInput(messages=messages, tools=tools, diagnostics=diagnostics)


def _message_role_for_model(role: MessageRole) -> str:
    """把内部消息角色映射成模型 API 的角色。

    工具消息在不同 API 中有更细的格式。第一版先把它作为 user 上下文补充，
    避免过早绑定复杂的工具结果协议。
    """

    if role == MessageRole.ASSISTANT:
        return "assistant"
    if role == MessageRole.SYSTEM:
        return "system"
    return "user"
