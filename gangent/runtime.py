"""Runtime Loop（运行时循环）。

这一层把前面做好的零件串成一个可重复执行的 agent loop。
demo.py 只负责命令行参数和打印，真正的执行流程放在这里。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Callable
from time import monotonic

from .llm_client import LLMClient
from .error_recovery import (
    attach_recovery_hint,
    recovery_hint_for_policy,
    recovery_hint_for_tool_result,
)
from .checkpoint import (
    TaskCheckpoint,
    decision_from_dict,
    policy_from_dict,
    tool_result_from_dict,
)
from .decision import DecisionParseError
from .events import (
    EventRuntimeState,
    InterruptAction,
    InterruptDecision,
    JsonlEventQueue,
    evaluate_interrupts,
    transition_from_interrupt,
)
from .failure import FailureReason, failure_message
from .failure import RECOVERABLE_FAILURE_REASONS, failure_reason_from_error, is_recoverable_failure
from .hooks import HookContext, HookEvent, HookManager
from .idempotency import find_reusable_tool_result
from .manifests import build_execution_manifest, format_blocking_validation_hint
from .model_input import build_model_input
from .models import (
    ActionDecision,
    AgentPhase,
    AgentState,
    DecisionType,
    Message,
    MessageRole,
    PlanStep,
    PlanStepStatus,
    PolicyDecision,
    RuntimeStats,
    Task,
    TaskInput,
    TaskStatus,
    ToolResult,
    new_id,
    utc_now,
)
from .policy import check_policy
from .models import PolicyMode
from .planner import (
    attach_plan,
    block_current_plan_step,
    complete_current_plan_step,
    create_initial_plan,
    current_plan_step,
    start_current_plan_step,
)
from .planner_budget import update_planner_budget_state
from .replanning import (
    PlanPatchAction,
    apply_plan_patch,
    build_replan_context,
    format_plan_patch,
    format_replan_context,
    plan_patch_from_events,
)
from .state import (
    add_error,
    advance_step,
    attach_decision,
    attach_tool_result,
    create_initial_state,
    create_task,
    set_phase,
    set_task_status,
    start_task,
)
from .resume import ResumeReport, build_resume_report
from .runtime_checkpoint import save_runtime_checkpoint
from .schema_validator import ToolArgumentsValidationError
from .tool_schema import available_tool_schemas
from .task_profile import task_execution_profile
from .tool_runtime import execute_tool_call

if TYPE_CHECKING:
    from .tool_registry import ToolRegistry


READ_ONLY_RECOVERY_TOOLS = {
    "list_files",
    "search_context",
    "read_file",
    "read_many_files",
    "file_info",
    "grep_files",
    "git_status",
    "git_diff",
    "git_log",
    "git_show",
}

BARE_CODE_CLAIM_TERMS = {
    "dependency",
    "dependencies",
    "pending",
    "in_progress",
    "completed",
    "failed",
    "blocked",
    "current_step_index",
}


@dataclass
class RuntimeStepTrace:
    """一轮 runtime loop 的执行记录。

    Trace（链路记录）不是完整审计系统，但它记录了每一步的核心事实：
    模型做了什么决策、策略怎么判断、工具有没有执行。
    """

    step_index: int
    decision: ActionDecision
    policy: PolicyDecision | None = None
    tool_result: ToolResult | None = None
    usage: dict | None = None
    approval_required: bool = False
    approved: bool | None = None
    reused_tool_result: bool = False


@dataclass
class RuntimeResult:
    """一次任务运行后的结果包。"""

    task: Task
    state: AgentState
    steps: list[RuntimeStepTrace]
    stats: RuntimeStats
    task_input: TaskInput | None = None
    resume_report: ResumeReport | None = None


def run_task(
    task_input: TaskInput,
    client: LLMClient,
    max_steps: int = 3,
    max_seconds: float | None = None,
    approval_callback: Callable[[ActionDecision, PolicyDecision], bool] | None = None,
    checkpoint_path: str | None = None,
    resume_checkpoint: TaskCheckpoint | None = None,
    hook_manager: HookManager | None = None,
    tool_registry: "ToolRegistry | None" = None,
    event_queue_path: str | None = None,
    budget_profile: str | None = None,
    total_step_budget: int | None = None,
    total_remaining_steps: int | None = None,
) -> RuntimeResult:
    """运行一个最小 agent loop。

    专业说法：这是 stateful step loop（有状态分步循环）。
    通俗说法：让 agent 不只做一步，而是最多连续想几轮、执行几轮。
    """

    if max_steps <= 0:
        raise ValueError("max_steps must be greater than 0.")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be greater than 0.")

    restored_step_count = 0
    restored_plan_done: set[str] = set()
    if resume_checkpoint is not None:
        task = resume_checkpoint.task
        state = resume_checkpoint.state
        _prepare_resumed_state(task, state)
        restored_plan_done = {
            step.step_id for step in state.plan_steps if step.status.value == "done"
        }
        steps = [
            RuntimeStepTrace(
                step_index=step.step_index,
                decision=decision_from_dict(step.decision),
                policy=policy_from_dict(step.policy),
                tool_result=tool_result_from_dict(step.tool_result),
                usage=step.usage,
                approval_required=step.approval_required,
                approved=step.approved,
                reused_tool_result=bool(getattr(tool_result_from_dict(step.tool_result), "reused", False)),
            )
            for step in resume_checkpoint.steps
        ]
        restored_step_count = len(steps)
    else:
        task = create_task(task_input)
        state = create_initial_state(task, task_input)
        plan = create_initial_plan(task, task_input)
        attach_plan(state, plan)
        task, state = start_task(task, state)
        steps = []
    execution_profile = task_execution_profile(task_input)
    profile_tool_names = execution_profile.tool_names
    started_at = monotonic()
    _save_runtime_checkpoint(checkpoint_path, task_input, task, state, steps, started_at)
    _emit_hook(hook_manager, HookEvent.TASK_START, task_input=task_input, task=task, state=state)
    _emit_hook(hook_manager, HookEvent.CHECKPOINT_SAVE, task_input=task_input, task=task, state=state)

    for _ in range(max_steps):
        if _deadline_exceeded(started_at, max_seconds):
            message = failure_message(
                FailureReason.DEADLINE_EXCEEDED,
                f"Runtime deadline exceeded: max_seconds={max_seconds}",
            )
            set_task_status(task, TaskStatus.FAILED)
            add_error(state, message)
            block_current_plan_step(state, message)
            _save_runtime_checkpoint(checkpoint_path, task_input, task, state, steps, started_at)
            break

        interrupt = _handle_pending_events(task_input, task, state, event_queue_path)
        if interrupt and interrupt.action in {InterruptAction.PAUSE, InterruptAction.ASK_USER, InterruptAction.FORK}:
            set_task_status(task, TaskStatus.WAITING_USER)
            block_current_plan_step(state, interrupt.reason)
            _save_runtime_checkpoint(checkpoint_path, task_input, task, state, steps, started_at)
            _emit_hook(hook_manager, HookEvent.CHECKPOINT_SAVE, task_input=task_input, task=task, state=state)
            break
        if interrupt and interrupt.action == InterruptAction.REPLAN:
            attach_recovery_hint(
                state,
                "Runtime event requested replanning. Treat the event as the newest requirement, "
                "avoid stale assumptions, and choose the next action under the current tool and budget constraints.",
            )

        _ensure_output_repair_plan_step(state)
        set_phase(state, AgentPhase.THINKING)
        start_current_plan_step(state)
        segment_used_steps = max(0, len(steps) - restored_step_count)
        update_planner_budget_state(
            state,
            segment_step_limit=max_steps,
            segment_remaining_steps=max(0, max_steps - segment_used_steps),
            profile=budget_profile,
            total_step_budget=total_step_budget,
            total_remaining_steps=max(0, total_remaining_steps - segment_used_steps)
            if total_remaining_steps is not None
            else None,
        )
        tools = _visible_tool_schemas_for_current_phase(profile_tool_names, state)
        model_input = build_model_input(task, state, tools)
        _emit_hook(
            hook_manager,
            HookEvent.BEFORE_MODEL_CALL,
            task_input=task_input,
            task=task,
            state=state,
            model_input=model_input,
        )
        try:
            decision = client.decide(model_input)
        except DecisionParseError as exc:
            message = f"Model output parse failed: {exc}"
            add_error(state, message)
            attach_recovery_hint(state, _recovery_hint_for_model_parse_error(str(exc)))
            _save_runtime_checkpoint(checkpoint_path, task_input, task, state, steps, started_at)
            _emit_hook(hook_manager, HookEvent.CHECKPOINT_SAVE, task_input=task_input, task=task, state=state)
            continue
        attach_decision(state, decision)
        _emit_hook(
            hook_manager,
            HookEvent.AFTER_MODEL_CALL,
            task_input=task_input,
            task=task,
            state=state,
            decision=decision,
            model_input=model_input,
        )

        trace = RuntimeStepTrace(step_index=state.step_index, decision=decision)
        usage = getattr(client, "last_usage", None)
        if isinstance(usage, dict):
            trace.usage = usage

        if decision.decision_type == DecisionType.DIRECT_RESPONSE:
            finish_guard = _finish_missing_outputs_hint(task_input, state)
            if finish_guard:
                salvage_trace = _try_write_single_missing_output_from_finish(
                    task_input,
                    state,
                    decision,
                    steps,
                    approval_callback,
                    hook_manager,
                    tool_registry=tool_registry,
                )
                if salvage_trace is not None:
                    steps.append(salvage_trace)
                    set_task_status(task, TaskStatus.COMPLETED)
                    complete_current_plan_step(
                        state,
                        salvage_trace.tool_result.output if salvage_trace.tool_result else "Runtime wrote requested output.",
                    )
                    advance_step(state)
                    _save_runtime_checkpoint(checkpoint_path, task_input, task, state, steps, started_at)
                    _emit_hook(hook_manager, HookEvent.CHECKPOINT_SAVE, task_input=task_input, task=task, state=state)
                    break
                add_error(state, finish_guard)
                attach_recovery_hint(state, finish_guard)
                steps.append(trace)
                advance_step(state)
                _save_runtime_checkpoint(checkpoint_path, task_input, task, state, steps, started_at)
                _emit_hook(hook_manager, HookEvent.CHECKPOINT_SAVE, task_input=task_input, task=task, state=state)
                continue
        if decision.decision_type == DecisionType.TOOL_CALL:
            _handle_tool_call(
                task_input,
                state,
                decision,
                trace,
                steps,
                approval_callback,
                hook_manager,
                tool_registry=tool_registry,
            )
            if _tool_call_should_auto_finish(task_input, state, decision, trace):
                answer = _auto_finish_answer(decision, trace)
                attach_decision(
                    state,
                    ActionDecision(
                        decision_type=DecisionType.FINISH,
                        reason="Runtime auto-finished after successful validated tool execution.",
                        response_text=answer,
                    ),
                )
                set_task_status(task, TaskStatus.COMPLETED)
                complete_current_plan_step(state, answer)
        elif decision.decision_type == DecisionType.FINISH:
            finish_guard = _finish_missing_outputs_hint(task_input, state)
            if finish_guard:
                salvage_trace = _try_write_single_missing_output_from_finish(
                    task_input,
                    state,
                    decision,
                    steps,
                    approval_callback,
                    hook_manager,
                    tool_registry=tool_registry,
                )
                if salvage_trace is not None:
                    steps.append(salvage_trace)
                    set_task_status(task, TaskStatus.COMPLETED)
                    complete_current_plan_step(
                        state,
                        salvage_trace.tool_result.output if salvage_trace.tool_result else "Runtime wrote requested output.",
                    )
                    advance_step(state)
                    _save_runtime_checkpoint(checkpoint_path, task_input, task, state, steps, started_at)
                    _emit_hook(hook_manager, HookEvent.CHECKPOINT_SAVE, task_input=task_input, task=task, state=state)
                    break
            if not finish_guard:
                finish_guard = _finish_unsupported_path_hint(task_input, decision, steps)
            if not finish_guard:
                finish_guard = _finish_unsupported_symbol_hint(decision, steps)
            if finish_guard:
                if _should_use_guarded_fallback(state, finish_guard):
                    fallback_decision = ActionDecision(
                        decision_type=DecisionType.FINISH,
                        reason="Runtime produced a guarded fallback after repeated final-answer guard failures.",
                        response_text=_guarded_fallback_answer(finish_guard, steps),
                    )
                    attach_decision(state, fallback_decision)
                    set_task_status(task, TaskStatus.COMPLETED)
                    complete_current_plan_step(state, fallback_decision.response_text)
                    steps.append(trace)
                    advance_step(state)
                    _save_runtime_checkpoint(checkpoint_path, task_input, task, state, steps, started_at)
                    _emit_hook(hook_manager, HookEvent.CHECKPOINT_SAVE, task_input=task_input, task=task, state=state)
                    break
                add_error(state, finish_guard)
                attach_recovery_hint(state, finish_guard)
                steps.append(trace)
                advance_step(state)
                _save_runtime_checkpoint(checkpoint_path, task_input, task, state, steps, started_at)
                _emit_hook(hook_manager, HookEvent.CHECKPOINT_SAVE, task_input=task_input, task=task, state=state)
                continue
            set_task_status(task, TaskStatus.COMPLETED)
            complete_current_plan_step(state, decision.response_text or decision.reason)
            steps.append(trace)
            advance_step(state)
            _save_runtime_checkpoint(checkpoint_path, task_input, task, state, steps, started_at)
            _emit_hook(hook_manager, HookEvent.CHECKPOINT_SAVE, task_input=task_input, task=task, state=state)
            break
        elif decision.decision_type == DecisionType.FAIL:
            set_task_status(task, TaskStatus.FAILED)
            add_error(state, decision.reason)
            block_current_plan_step(state, decision.reason)
            steps.append(trace)
            advance_step(state)
            _save_runtime_checkpoint(checkpoint_path, task_input, task, state, steps, started_at)
            _emit_hook(hook_manager, HookEvent.CHECKPOINT_SAVE, task_input=task_input, task=task, state=state)
            break
        elif decision.decision_type == DecisionType.ASK_USER:
            set_task_status(task, TaskStatus.WAITING_USER)
            block_current_plan_step(state, decision.reason)
            steps.append(trace)
            advance_step(state)
            _save_runtime_checkpoint(checkpoint_path, task_input, task, state, steps, started_at)
            _emit_hook(hook_manager, HookEvent.CHECKPOINT_SAVE, task_input=task_input, task=task, state=state)
            break
        else:
            # DIRECT_RESPONSE 在第一版里视为本轮可结束的回答。
            set_task_status(task, TaskStatus.COMPLETED)
            complete_current_plan_step(state, decision.response_text or decision.reason)
            steps.append(trace)
            advance_step(state)
            _save_runtime_checkpoint(checkpoint_path, task_input, task, state, steps, started_at)
            _emit_hook(hook_manager, HookEvent.CHECKPOINT_SAVE, task_input=task_input, task=task, state=state)
            break

        steps.append(trace)
        advance_step(state)
        _save_runtime_checkpoint(checkpoint_path, task_input, task, state, steps, started_at)
        _emit_hook(hook_manager, HookEvent.CHECKPOINT_SAVE, task_input=task_input, task=task, state=state)
        if task.status == TaskStatus.COMPLETED:
            break

    if task.status == TaskStatus.RUNNING:
        message = failure_message(
            FailureReason.MAX_STEPS,
            f"Runtime stopped after reaching max_steps={max_steps}.",
        )
        set_task_status(task, TaskStatus.FAILED)
        add_error(state, message)
        block_current_plan_step(state, message)
        _save_runtime_checkpoint(checkpoint_path, task_input, task, state, steps, started_at)

    _finalize_remaining_step_budget(
        state,
        max_steps=max_steps,
        restored_step_count=restored_step_count,
        used_steps=len(steps),
        total_step_budget=total_step_budget,
        total_remaining_steps=total_remaining_steps,
    )
    result = RuntimeResult(
        task=task,
        state=state,
        steps=steps,
        stats=_runtime_stats(started_at, state, steps),
        task_input=task_input,
        resume_report=build_resume_report(
            resume_checkpoint is not None,
            restored_step_count,
            restored_plan_done,
            state,
            steps,
        ),
    )
    _emit_hook(hook_manager, HookEvent.TASK_FINISH, task_input=task_input, task=task, state=state, result=result)
    return result


def _runtime_stats(
    started_at: float,
    state: AgentState,
    steps: list[RuntimeStepTrace],
) -> RuntimeStats:
    """汇总一次任务的基础统计。"""

    usage: dict = {}
    for step in steps:
        if not step.usage:
            continue
        for key, value in step.usage.items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
            else:
                usage[key] = value

    return RuntimeStats(
        duration_seconds=round(monotonic() - started_at, 6),
        step_count=len(steps),
        tool_call_count=sum(1 for step in steps if step.tool_result is not None),
        error_count=len(state.errors),
        usage=usage,
    )


def _recovery_hint_for_model_parse_error(error: str) -> str:
    """Build a deterministic retry instruction for malformed model output."""

    return (
        "Recovery hint: the previous model output could not be parsed into an "
        f"ActionDecision. Parser error: {error}. If a tool is needed, return a real "
        "structured tool call with valid JSON arguments according to the provided "
        "tool schema. Do not answer with plain text such as 'Model requested tool: ...'. "
        "If no tool is needed, use finish_task with a final answer."
    )


def _prepare_resumed_state(task: Task, state: AgentState) -> None:
    """Resume work state, but do not carry old recoverable failure markers forward."""

    if task.status != TaskStatus.FAILED or not is_recoverable_failure(state.errors):
        return

    set_task_status(task, TaskStatus.RUNNING)
    set_phase(state, AgentPhase.THINKING)
    state.errors = [
        error
        for error in state.errors
        if failure_reason_from_error(error) not in RECOVERABLE_FAILURE_REASONS
    ]
    for plan_step in state.plan_steps:
        if plan_step.status != PlanStepStatus.BLOCKED:
            continue
        if failure_reason_from_error(plan_step.result_summary) in RECOVERABLE_FAILURE_REASONS:
            plan_step.status = PlanStepStatus.TODO
            plan_step.result_summary = ""


def _handle_pending_events(
    task_input: TaskInput,
    task: Task,
    state: AgentState,
    event_queue_path: str | None,
):
    """Consume pending events at a safe runtime boundary."""

    if not event_queue_path:
        return None
    queue = JsonlEventQueue(event_queue_path)
    pending = queue.pending(cursor=state.event_cursor, task_id=task.task_id, created_after=task.created_at)
    if not pending:
        state.pending_event_count = 0
        return None

    decision = evaluate_interrupts(pending)
    replan_context = build_replan_context(task_input, task, state, pending)
    plan_patch = plan_patch_from_events(replan_context, pending)
    apply_plan_patch(state, plan_patch)
    if plan_patch.action == PlanPatchAction.STABILIZE:
        decision = InterruptDecision(
            action=InterruptAction.ASK_USER,
            reason=plan_patch.reason,
            events=decision.events,
            context_note=decision.context_note,
        )
    elif plan_patch.action == PlanPatchAction.PAUSE:
        decision = InterruptDecision(
            action=InterruptAction.PAUSE,
            reason=plan_patch.reason,
            events=decision.events,
            context_note=decision.context_note,
        )
    elif plan_patch.action == PlanPatchAction.ASK_USER:
        decision = InterruptDecision(
            action=InterruptAction.ASK_USER,
            reason=plan_patch.reason,
            events=decision.events,
            context_note=decision.context_note,
        )
    elif plan_patch.action == PlanPatchAction.REPLACE_PENDING_STEPS:
        decision = InterruptDecision(
            action=InterruptAction.REPLAN,
            reason=plan_patch.reason,
            events=decision.events,
            context_note=decision.context_note,
        )
    transition = transition_from_interrupt(EventRuntimeState(state.event_runtime_state), decision)
    state.event_runtime_state = transition.to_state.value
    state.event_cursor = max(event.index for event in pending)
    state.event_count += len(pending)
    state.pending_event_count = len(pending)
    if decision.action == InterruptAction.IGNORE:
        state.updated_at = utc_now()
        return decision

    note = (
        f"action={decision.action.value}; state={transition.from_state.value}->{transition.to_state.value}; "
        f"reason={decision.reason}\n"
        f"{decision.context_note}\n"
        f"{format_plan_patch(plan_patch)}\n"
        f"{format_replan_context(replan_context)}"
    ).strip()
    state.event_summaries.append(note)
    state.messages.append(Message(role=MessageRole.SYSTEM, content=f"Runtime event update:\n{note}"))
    state.context_summary = (state.context_summary + "\n\nRuntime event update:\n" + note).strip()
    state.updated_at = utc_now()
    return decision


def _deadline_exceeded(started_at: float, max_seconds: float | None) -> bool:
    """检查本次任务是否已经超过总耗时预算。"""

    if max_seconds is None:
        return False
    return monotonic() - started_at >= max_seconds


def _handle_tool_call(
    task_input: TaskInput,
    state: AgentState,
    decision: ActionDecision,
    trace: RuntimeStepTrace,
    prior_steps: list[RuntimeStepTrace],
    approval_callback: Callable[[ActionDecision, PolicyDecision], bool] | None = None,
    hook_manager: HookManager | None = None,
    tool_registry: "ToolRegistry | None" = None,
) -> None:
    """处理一轮工具调用：策略检查 -> 工具执行 -> 状态回写。"""

    set_phase(state, AgentPhase.VALIDATING)
    _normalize_tool_args_for_policy(decision, state)
    _repair_common_path_confusion(task_input, state, decision)
    reusable = find_reusable_tool_result(prior_steps, decision)
    if reusable is not None:
        trace.tool_result = reusable
        trace.reused_tool_result = True
        attach_tool_result(state, reusable)
        _update_plan_after_tool(task_input, state, reusable, decision, prior_steps)
        set_phase(state, AgentPhase.UPDATING_STATE)
        _emit_hook(
            hook_manager,
            HookEvent.AFTER_TOOL_CALL,
            task_input=task_input,
            state=state,
            decision=decision,
            policy=None,
            tool_result=reusable,
        )
        return
    plan_tool_hint = _plan_allowed_tool_hint(task_input, state, decision, prior_steps)
    if plan_tool_hint:
        add_error(state, plan_tool_hint)
        attach_recovery_hint(state, plan_tool_hint)
        return
    if tool_registry is None:
        from .tool_registry import default_tool_registry

        tool_registry = default_tool_registry()
    try:
        tool_registry.validate_arguments(decision)
    except ToolArgumentsValidationError as exc:
        tool_result = ToolResult(
            call_id=new_id("call"),
            success=False,
            error=str(exc),
        )
        trace.tool_result = tool_result
        attach_tool_result(state, tool_result)
        attach_recovery_hint(state, recovery_hint_for_tool_result(decision, tool_result))
        set_phase(state, AgentPhase.UPDATING_STATE)
        return
    duplicate_write_hint = _duplicate_successful_write_guard_hint(task_input, state, decision, prior_steps)
    if duplicate_write_hint:
        add_error(state, duplicate_write_hint)
        attach_recovery_hint(state, duplicate_write_hint)
        return
    _redirect_duplicate_source_read(task_input, state, decision, prior_steps)
    multi_source_hint = _multi_source_read_guard_hint(task_input, state, decision, prior_steps)
    if multi_source_hint:
        add_error(state, multi_source_hint)
        attach_recovery_hint(state, multi_source_hint)
        return
    repeated_read_hint = _repeated_successful_read_hint(decision, prior_steps)
    if repeated_read_hint:
        attach_recovery_hint(state, repeated_read_hint)
        complete_current_plan_step(state, repeated_read_hint)
        return
    repeat_hint = _repeat_guard_hint(decision, prior_steps)
    if repeat_hint:
        add_error(state, repeat_hint)
        attach_recovery_hint(state, repeat_hint)
        if not _guard_should_keep_current_phase(state, decision):
            block_current_plan_step(state, repeat_hint)
        return

    policy = check_policy(decision, workspace_root=task_input.workspace_root)
    trace.policy = policy
    _emit_hook(
        hook_manager,
        HookEvent.BEFORE_TOOL_CALL,
        task_input=task_input,
        state=state,
        decision=decision,
        policy=policy,
    )
    if not policy.allowed:
        if policy.mode == PolicyMode.ESCALATE:
            trace.approval_required = True
            approved = bool(approval_callback(decision, policy)) if approval_callback else False
            trace.approved = approved
            if approved:
                set_phase(state, AgentPhase.EXECUTING_TOOL)
                tool_result = execute_tool_call(
                    decision,
                    workspace_root=task_input.workspace_root,
                    tool_registry=tool_registry,
                )
                trace.tool_result = tool_result
                attach_tool_result(state, tool_result)
                _update_plan_after_tool(task_input, state, tool_result, decision, prior_steps)
                set_phase(state, AgentPhase.UPDATING_STATE)
                _emit_hook(
                    hook_manager,
                    HookEvent.AFTER_TOOL_CALL,
                    task_input=task_input,
                    state=state,
                    decision=decision,
                    policy=policy,
                    tool_result=tool_result,
                )
                return
            add_error(state, f"User approval denied or unavailable: {policy.reason}")
            attach_recovery_hint(state, recovery_hint_for_policy(decision, policy))
            block_current_plan_step(state, f"User approval denied or unavailable: {policy.reason}")
            return
        add_error(state, policy.reason)
        attach_recovery_hint(state, recovery_hint_for_policy(decision, policy))
        if _policy_failure_should_keep_current_phase(state, decision, policy):
            return
        block_current_plan_step(state, policy.reason)
        return

    set_phase(state, AgentPhase.EXECUTING_TOOL)
    tool_result = execute_tool_call(
        decision,
        workspace_root=task_input.workspace_root,
        tool_registry=tool_registry,
    )
    trace.tool_result = tool_result
    attach_tool_result(state, tool_result)
    _update_plan_after_tool(task_input, state, tool_result, decision, prior_steps)
    set_phase(state, AgentPhase.UPDATING_STATE)
    _emit_hook(
        hook_manager,
        HookEvent.AFTER_TOOL_CALL,
        task_input=task_input,
        state=state,
        decision=decision,
        policy=policy,
        tool_result=tool_result,
    )


def _update_plan_after_tool(
    task_input: TaskInput,
    state: AgentState,
    tool_result: ToolResult,
    decision: ActionDecision,
    prior_steps: list[RuntimeStepTrace],
) -> None:
    """根据工具结果更新当前计划步骤。

    第一版采用简单规则：工具成功则当前步骤完成；工具失败则阻塞。
    后续可以升级为基于错误类型的 replan。
    """

    if tool_result.success:
        if _write_phase_support_read_should_continue(state, decision):
            attach_recovery_hint(
                state,
                "Runtime hint: evidence lookup succeeded during the write phase. "
                "Use the retrieved evidence to write the requested output file; do not keep rereading the same source.",
            )
            return
        if _read_phase_needs_more_sources(task_input, state, decision, prior_steps):
            attach_recovery_hint(
                state,
                "Runtime hint: one requested source file was read, but the task asks for additional source files. "
                "Stay in the read phase and read the remaining requested source files before writing output.",
            )
            return
        if _write_phase_needs_more_outputs(task_input, state, decision):
            attach_recovery_hint(
                state,
                "Runtime hint: one requested output file was written, but the task asks for additional output files. "
                "Stay in the write phase and create the remaining requested files before finishing.",
            )
            return
        if _evidence_phase_should_continue(state, decision):
            attach_recovery_hint(
                state,
                "Runtime hint: evidence was collected in an analysis phase. "
                "Use finish_task if the evidence is enough; otherwise continue with a targeted read-only evidence tool.",
            )
            return
        complete_current_plan_step(state, tool_result.output)
        _attach_post_read_write_hint(state, decision)
        _attach_post_write_finish_hint(state, decision)
    else:
        if state.last_decision:
            attach_recovery_hint(state, recovery_hint_for_tool_result(state.last_decision, tool_result))
        if _tool_failure_should_keep_current_phase(state, decision, tool_result):
            return
        block_current_plan_step(state, tool_result.error or "Tool failed.")


def _normalize_tool_args_for_policy(decision: ActionDecision, state: AgentState) -> None:
    """Normalize harmless model argument overshoots before policy/tool execution."""

    if decision.decision_type != DecisionType.TOOL_CALL or decision.tool_name != "read_file":
        return
    if not isinstance(decision.tool_args, dict):
        return
    max_lines = decision.tool_args.get("max_lines")
    if isinstance(max_lines, int) and max_lines > 400:
        decision.tool_args["max_lines"] = 400
        attach_recovery_hint(
            state,
            "Runtime normalized read_file max_lines to 400, the maximum supported chunk size. "
            "Continue with chunked reads if more content is needed.",
        )


def _repair_common_path_confusion(
    task_input: TaskInput,
    state: AgentState,
    decision: ActionDecision,
) -> None:
    """Repair obvious read-only path mistakes before policy and plan guards run."""

    if decision.decision_type != DecisionType.TOOL_CALL:
        return
    if decision.tool_name not in {"list_files", "read_file", "file_info"}:
        return
    if not isinstance(decision.tool_args, dict):
        return
    raw_path = decision.tool_args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return
    path = raw_path.strip().replace("\\", "/")
    if Path(path).is_absolute() or ".." in Path(path).parts or Path(path).suffix:
        return

    root = Path(task_input.workspace_root)
    current = root / path
    if current.exists():
        return

    for suffix in (".py", ".md", ".json", ".txt", ".yaml", ".yml"):
        candidate = root / f"{path}{suffix}"
        if not candidate.is_file():
            continue
        repaired_path = f"{path}{suffix}"
        original_tool = decision.tool_name
        if original_tool == "list_files":
            decision.tool_name = "read_file"
            decision.tool_args = {"path": repaired_path}
        else:
            decision.tool_args["path"] = repaired_path
        attach_recovery_hint(
            state,
            "Runtime repaired an obvious path/tool mismatch: "
            f"{original_tool}({raw_path}) -> {decision.tool_name}({repaired_path}). "
            "The requested path is not a directory/file, but a same-name source file exists.",
        )
        return


def _tool_call_should_auto_finish(
    task_input: TaskInput,
    state: AgentState,
    decision: ActionDecision,
    trace: RuntimeStepTrace,
) -> bool:
    if trace.tool_result is None or not trace.tool_result.success:
        return False
    if decision.tool_name not in {"write_file", "edit_file", "apply_patch"}:
        return False
    if _finish_missing_outputs_hint(task_input, state):
        return False
    allowed = _current_allowed_tools(state)
    return allowed == {"finish_task"}


def _auto_finish_answer(decision: ActionDecision, trace: RuntimeStepTrace) -> str:
    tool_result = trace.tool_result.output if trace.tool_result else "Tool completed."
    if decision.tool_name in {"write_file", "edit_file", "apply_patch"}:
        path = ""
        if isinstance(decision.tool_args, dict):
            path_value = decision.tool_args.get("path")
            if isinstance(path_value, str) and path_value:
                path = f" `{path_value}`"
        return f"Completed file change{path}. {tool_result}"
    return tool_result


def _attach_post_write_finish_hint(state: AgentState, decision: ActionDecision) -> None:
    """Tell the model to finish after a successful final write phase."""

    if decision.tool_name not in {"write_file", "edit_file", "apply_patch"}:
        return
    current = current_plan_step(state)
    if current is None or "allowed_tools=finish_task" not in current.description:
        return
    attach_recovery_hint(
        state,
        "Runtime hint: the requested file change has succeeded and the current plan phase allows only finish_task. "
        "Do not call write_file/edit_file again unless the previous tool result failed; finish with a concise summary.",
    )


def _attach_post_read_write_hint(state: AgentState, decision: ActionDecision) -> None:
    """Tell the model to transform already-read content instead of rereading."""

    if decision.tool_name not in {"read_file", "read_many_files"}:
        return
    allowed = _current_allowed_tools(state)
    if not {"write_file", "edit_file"}.issubset(allowed):
        return
    attach_recovery_hint(
        state,
        "Runtime hint: the source content has already been read and is available in the conversation context. "
        "The current phase is the write phase; do not call read_file again for the same source. "
        "Create the requested derived output with write_file or edit_file.",
    )


def _write_phase_needs_more_outputs(
    task_input: TaskInput,
    state: AgentState,
    decision: ActionDecision,
) -> bool:
    """Keep multi-output write tasks in the write phase until all files exist."""

    if decision.tool_name not in {"write_file", "edit_file"}:
        return False
    allowed = _current_allowed_tools(state)
    if not {"write_file", "edit_file"}.issubset(allowed):
        return False
    output_paths = _requested_output_paths(task_input)
    if not output_paths:
        return False
    root = Path(task_input.workspace_root)
    return any(not (root / path).exists() for path in output_paths)


def _finish_missing_outputs_hint(task_input: TaskInput, state: AgentState) -> str:
    return format_blocking_validation_hint(build_execution_manifest(task_input))


def _try_write_single_missing_output_from_finish(
    task_input: TaskInput,
    state: AgentState,
    decision: ActionDecision,
    prior_steps: list[RuntimeStepTrace],
    approval_callback: Callable[[PolicyDecision], bool] | None,
    hook_manager: HookManager | None,
    *,
    tool_registry: "ToolRegistry | None" = None,
) -> RuntimeStepTrace | None:
    """Salvage a common model failure: final text instead of write_file.

    This is deliberately narrow. It only runs when the user clearly requested
    one output file, that file is still missing, and the current plan phase
    allows write tools. The model's final answer becomes the file content.
    """

    allowed = _current_allowed_tools(state)
    if not {"write_file", "edit_file"}.intersection(allowed):
        return None
    manifest = build_execution_manifest(task_input)
    missing = [entry.path for entry in manifest.outputs if not entry.exists]
    if len(missing) != 1:
        return None
    content = (decision.response_text or decision.reason or "").strip()
    if len(content) < 20:
        return None
    path = missing[0]
    write_decision = ActionDecision(
        decision_type=DecisionType.TOOL_CALL,
        reason="Runtime salvaged final text into the single missing requested output file.",
        tool_name="write_file",
        tool_args={"path": path, "content": content, "overwrite": False},
    )
    write_trace = RuntimeStepTrace(step_index=state.step_index, decision=write_decision)
    _handle_tool_call(
        task_input,
        state,
        write_decision,
        write_trace,
        prior_steps,
        approval_callback,
        hook_manager,
        tool_registry=tool_registry,
    )
    if write_trace.tool_result is None or not write_trace.tool_result.success:
        return None
    attach_recovery_hint(
        state,
        "Runtime salvage: model returned final text for a missing single output file, "
        f"so the runtime wrote it to {path} with write_file.",
    )
    return write_trace


def _finish_unsupported_path_hint(
    task_input: TaskInput,
    decision: ActionDecision,
    prior_steps: list[RuntimeStepTrace],
) -> str:
    response = decision.response_text or decision.reason or ""
    if not response:
        return ""
    refs = _path_refs_from_text(response)
    if not refs:
        return ""

    failed_paths = _failed_tool_paths(prior_steps)
    bad_refs = []
    for ref in refs:
        normalized = ref.replace("\\", "/").strip("`'\"")
        if normalized in failed_paths:
            if _path_ref_is_absence_statement(response, normalized):
                continue
            bad_refs.append(normalized)
            continue
        if _is_workspace_external_absolute_path(normalized, task_input.workspace_root):
            bad_refs.append(normalized)
            continue
        if _relative_ref_matches_successful_read(normalized, prior_steps):
            continue
        if _is_missing_workspace_relative_reference(normalized, task_input.workspace_root):
            if _path_ref_is_absence_statement(response, normalized):
                continue
            bad_refs.append(normalized)

    if not bad_refs:
        return ""
    return (
        "Final answer guard: the answer cites path(s) that were not successfully read "
        f"or are outside the workspace: {', '.join(sorted(set(bad_refs)))}. "
        "Rewrite the final answer using only successful tool evidence and exact workspace-relative paths. "
        "If evidence is insufficient, state the gap explicitly instead of inventing a path."
    )


def _finish_unsupported_symbol_hint(
    decision: ActionDecision,
    prior_steps: list[RuntimeStepTrace],
) -> str:
    response = decision.response_text or decision.reason or ""
    if not response:
        return ""
    evidence = _successful_python_read_evidence(prior_steps)
    if not evidence:
        return ""
    symbols = _backticked_code_symbols(response)
    bad_symbols = sorted(symbol for symbol in symbols if symbol not in evidence)
    if not bad_symbols:
        return ""
    allowed = ", ".join(_python_symbols_from_evidence(evidence)[:40])
    return (
        "Final answer guard: the answer cites code symbol(s) that were not found in the successfully read Python evidence: "
        f"{', '.join(bad_symbols)}. Rewrite the answer using only identifiers present in the read code, "
        "or state that the symbol was not found. "
        f"Allowed evidence symbols include: {allowed}."
    )


def _should_use_guarded_fallback(state: AgentState, finish_guard: str) -> bool:
    if "Final answer guard:" not in finish_guard:
        return False
    prior_final_guard_errors = sum(1 for error in state.errors if "Final answer guard:" in error)
    return prior_final_guard_errors >= 2


def _guarded_fallback_answer(finish_guard: str, prior_steps: list[RuntimeStepTrace]) -> str:
    paths = sorted(_successful_read_paths(prior_steps))
    evidence = _successful_python_read_evidence(prior_steps)
    symbols = _python_symbols_from_evidence(evidence)[:30] if evidence else []
    lines = [
        "I could not safely accept the model's full final answer because it repeatedly cited unverified paths or code identifiers.",
    ]
    if paths:
        lines.append(f"Verified files actually read: {', '.join(paths)}.")
    if symbols:
        lines.append(f"Verified code identifiers seen in the read evidence include: {', '.join(symbols)}.")
    lines.append("Unverified claims were blocked by the final-answer guard instead of being presented as facts.")
    lines.append(f"Last guard reason: {finish_guard}")
    return "\n".join(lines)


def _successful_python_read_evidence(prior_steps: list[RuntimeStepTrace]) -> str:
    chunks: list[str] = []
    for step in prior_steps:
        if step.decision.tool_name not in {"read_file", "read_many_files"}:
            continue
        if not step.tool_result or not step.tool_result.success or not step.tool_result.output:
            continue
        paths = _paths_from_read_decision(step.decision)
        if not any(path.lower().endswith(".py") for path in paths):
            continue
        chunks.append(step.tool_result.output)
    return "\n".join(chunks)


def _backticked_code_symbols(text: str) -> set[str]:
    symbols: set[str] = set()
    for match in re.finditer(r"`([^`\n]+)`", text):
        raw = match.group(1).strip()
        if any(separator in raw for separator in ["/", "\\", "."]):
            continue
        symbol_match = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw)
        if not symbol_match:
            continue
        if len(raw) < 4:
            continue
        symbols.add(raw)
    lowered = text.lower()
    for term in BARE_CODE_CLAIM_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            symbols.add(term)
    for match in re.finditer(r"\b[A-Za-z_]+_[A-Za-z0-9_]+\b", text):
        raw = match.group(0)
        if len(raw) >= 4:
            symbols.add(raw)
    return symbols


def _python_symbols_from_evidence(evidence: str) -> list[str]:
    symbols = {
        match.group(0)
        for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", evidence)
        if len(match.group(0)) >= 4
    }
    return sorted(symbols)


def _path_refs_from_text(text: str) -> set[str]:
    suffixes = "py|md|json|txt|csv|yaml|yml|toml|html|pdf|xlsx"
    pattern = re.compile(
        rf"(?<![\w])((?:[A-Za-z]:[\\/]|/)?[\w./\\-]+\.(?:{suffixes}))",
        re.IGNORECASE,
    )
    refs = {match.group(1).strip("`'\".,;:，。；：)）]】") for match in pattern.finditer(text)}
    quoted_path_pattern = re.compile(r"`([^`\n]*[/\\][^`\n]*)`")
    for match in quoted_path_pattern.finditer(text):
        value = match.group(1).strip("`'\".,;:，。；：)）]】")
        if value in {"/", "\\"}:
            continue
        if not re.match(r"^(?:[A-Za-z]:[\\/]|/|\.{1,2}[\\/]|[A-Za-z0-9_.-]+[\\/])", value):
            continue
        if value and " " not in value and not value.startswith(("http://", "https://")):
            refs.add(value)
    return refs


def _failed_tool_paths(prior_steps: list[RuntimeStepTrace]) -> set[str]:
    failed: set[str] = set()
    for step in prior_steps:
        if not isinstance(step.decision.tool_args, dict):
            continue
        path = step.decision.tool_args.get("path")
        if not isinstance(path, str) or not path:
            continue
        if step.policy is not None and not step.policy.allowed:
            failed.add(path.replace("\\", "/"))
        if step.tool_result is not None and not step.tool_result.success:
            failed.add(path.replace("\\", "/"))
    return failed


def _path_ref_is_absence_statement(text: str, path: str) -> bool:
    normalized_text = text.replace("\\", "/")
    index = normalized_text.find(path)
    if index < 0:
        return False
    window = normalized_text[max(0, index - 80) : index + len(path) + 120].lower()
    absence_markers = (
        "does not exist",
        "do not exist",
        "not found",
        "missing",
        "不存在",
        "没有找到",
        "未找到",
        "缺失",
    )
    return any(marker in window for marker in absence_markers)


def _is_workspace_external_absolute_path(path: str, workspace_root: str) -> bool:
    if path.startswith("/"):
        return True


def _is_missing_workspace_relative_reference(path: str, workspace_root: str) -> bool:
    if not path or Path(path).is_absolute():
        return False
    if "/" not in path and "\\" not in path:
        return False
    if "://" in path:
        return False
    try:
        root = Path(workspace_root).resolve()
        target = (root / path).resolve()
        target.relative_to(root)
    except Exception:
        return False
    return not target.exists()


def _relative_ref_matches_successful_read(path: str, prior_steps: list[RuntimeStepTrace]) -> bool:
    if not path or Path(path).is_absolute():
        return False
    if path.endswith(("/", "\\")):
        return False
    normalized = path.rstrip("/\\").replace("\\", "/")
    if not normalized:
        return False
    for read_path in _successful_read_paths(prior_steps):
        read_normalized = read_path.rstrip("/\\").replace("\\", "/")
        if read_normalized == normalized:
            return True
        if read_normalized.startswith(normalized + "."):
            return True
    return False
    if not re.match(r"^[A-Za-z]:/", path):
        return False
    try:
        root = Path(workspace_root).resolve()
        target = Path(path).resolve()
        target.relative_to(root)
        return False
    except Exception:
        return True


def _read_phase_needs_more_sources(
    task_input: TaskInput,
    state: AgentState,
    decision: ActionDecision,
    prior_steps: list[RuntimeStepTrace],
) -> bool:
    """Keep multi-source read tasks in the read phase until all files are read."""

    if decision.tool_name not in {"read_file", "read_many_files"}:
        return False
    if not _current_phase_allows_multi_source_reads(state):
        return False
    source_paths = _requested_source_paths(task_input)
    if len(source_paths) <= 1:
        return False
    read_paths = _successful_read_paths(prior_steps)
    read_paths.update(_paths_from_read_decision(decision))
    return any(path not in read_paths for path in source_paths)


def _write_phase_support_read_should_continue(state: AgentState, decision: ActionDecision) -> bool:
    if decision.tool_name not in {"read_file", "read_many_files"}:
        return False
    allowed = _current_allowed_tools(state)
    if not {"read_file", "read_many_files", "write_file", "edit_file"}.issubset(allowed):
        return False
    return True


def _evidence_phase_should_continue(state: AgentState, decision: ActionDecision) -> bool:
    evidence_tools = {
        "list_files",
        "file_info",
        "grep_files",
        "read_file",
        "read_many_files",
        "search_context",
        "git_status",
        "git_diff",
        "git_log",
        "git_show",
    }
    if decision.tool_name not in evidence_tools:
        return False
    allowed = _current_allowed_tools(state)
    if "finish_task" not in allowed:
        return False
    if not evidence_tools.intersection(allowed):
        return False
    return not {"write_file", "edit_file", "apply_patch"}.intersection(allowed)


def _policy_failure_should_keep_current_phase(
    state: AgentState,
    decision: ActionDecision,
    policy: PolicyDecision,
) -> bool:
    if decision.tool_name not in {"read_file", "read_many_files"}:
        return False
    if "File does not exist" not in policy.reason:
        return False
    allowed = _current_allowed_tools(state)
    if not {"write_file", "edit_file"}.intersection(allowed):
        return False
    attach_recovery_hint(
        state,
        "Runtime hint: a read target is missing, but the current phase can still write files. "
        "Create the missing requested artifact or continue from available evidence; do not abandon the phase.",
    )
    return True


def _tool_failure_should_keep_current_phase(
    state: AgentState,
    decision: ActionDecision,
    tool_result: ToolResult,
) -> bool:
    if decision.tool_name not in {"read_file", "read_many_files", "write_file", "edit_file"}:
        return False
    error = tool_result.error or tool_result.output or ""
    if not any(marker in error for marker in ["File does not exist", "already exists", "No such file"]):
        return False
    allowed = _current_allowed_tools(state)
    if not {"write_file", "edit_file"}.intersection(allowed):
        return False
    attach_recovery_hint(
        state,
        "Runtime hint: the tool failed on a recoverable file-state issue. "
        "Stay in the current phase and repair the missing or duplicate file action.",
    )
    return True


def _current_allowed_tools(state: AgentState) -> set[str]:
    current = current_plan_step(state)
    if current is None or "allowed_tools=" not in current.description:
        return set()
    allowed_text = current.description.split("allowed_tools=", 1)[1].split(";", 1)[0].strip()
    if allowed_text in {"", "(none)"}:
        return set()
    return {item.strip() for item in allowed_text.split(",") if item.strip()}


def _visible_tool_schemas_for_current_phase(
    profile_tool_names: tuple[str, ...] | None,
    state: AgentState,
) -> list[dict]:
    """Expose only tools that are both task-appropriate and phase-allowed."""

    if profile_tool_names == ():
        return []

    allowed = _current_allowed_tools(state)
    if not allowed and state.plan_steps:
        allowed = {"finish_task"}

    if profile_tool_names is None:
        return available_tool_schemas(tuple(sorted(allowed))) if allowed else available_tool_schemas(None)

    if not allowed:
        return available_tool_schemas(profile_tool_names)

    filtered = tuple(name for name in profile_tool_names if name in allowed)
    if _needs_output_repair_tools(state):
        repair_tools = {"read_file", "read_many_files", "write_file", "edit_file", "file_info", "finish_task"}
        repaired = tuple(name for name in profile_tool_names if name in repair_tools)
        if repaired:
            filtered = repaired
    if not filtered and "finish_task" in allowed:
        filtered = ("finish_task",)
    return available_tool_schemas(filtered)


def _needs_output_repair_tools(state: AgentState) -> bool:
    """Expose repair tools after validator detects missing required outputs."""

    allowed = _current_allowed_tools(state)
    if allowed != {"finish_task"}:
        return False
    if not state.errors:
        return False
    recent = "\n".join(state.errors[-4:]).lower()
    return "finish guard / validator layer" in recent and "missing output file" in recent


def _ensure_output_repair_plan_step(state: AgentState) -> None:
    """Insert a write-capable repair step when finish is blocked by missing outputs."""

    if not _needs_output_repair_tools(state):
        return
    if any(step.title == "Repair missing output" and step.status in {PlanStepStatus.TODO, PlanStepStatus.RUNNING} for step in state.plan_steps):
        return
    current = current_plan_step(state)
    if current is not None and _current_allowed_tools(state) == {"finish_task"}:
        current.status = PlanStepStatus.BLOCKED
        current.result_summary = "Blocked by missing output validator; runtime inserted a repair step."
    state.plan_steps.append(
        PlanStep(
            step_id=new_id("step"),
            title="Repair missing output",
            description=(
                "phase=repair_output; max_steps=3; "
                "allowed_tools=read_file,read_many_files,write_file,edit_file,file_info,finish_task; "
                "exit_criteria=Missing required output files are created or a precise blocker is reported."
            ),
            purpose="Recover from a validator failure where finish was attempted before required files existed.",
            expected_output="Required output files exist or the runtime has a specific unresolved blocker.",
            tool_hint="write_file",
        )
    )
    state.updated_at = utc_now()


def _current_phase_allows_multi_source_reads(state: AgentState) -> bool:
    allowed = _current_allowed_tools(state)
    return {"read_file", "read_many_files"}.issubset(allowed) and not {"write_file", "edit_file"}.intersection(allowed)


def _remaining_phase_output_paths(
    task_input: TaskInput,
    state: AgentState,
    prior_steps: list[RuntimeStepTrace],
) -> list[str]:
    output_paths = _requested_output_paths(task_input)
    if not output_paths:
        return []
    written = _successful_write_paths(prior_steps)
    root = Path(task_input.workspace_root)
    return [path for path in output_paths if path not in written and not (root / path).exists()]


def _guard_should_keep_current_phase(state: AgentState, decision: ActionDecision) -> bool:
    if decision.tool_name not in {"read_file", "read_many_files"}:
        return False
    allowed = _current_allowed_tools(state)
    return {"write_file", "edit_file"}.issubset(allowed)


def _successful_read_paths(steps: list[RuntimeStepTrace]) -> set[str]:
    paths: set[str] = set()
    for step in steps:
        if not step.tool_result or not step.tool_result.success:
            continue
        paths.update(_paths_from_read_decision(step.decision))
    return paths


def _paths_from_read_decision(decision: ActionDecision) -> set[str]:
    if not isinstance(decision.tool_args, dict):
        return set()
    if decision.tool_name == "read_file":
        path = decision.tool_args.get("path")
        return {path} if isinstance(path, str) and path else set()
    if decision.tool_name == "read_many_files":
        paths = decision.tool_args.get("paths")
        if isinstance(paths, list):
            return {path for path in paths if isinstance(path, str) and path}
    return set()


def _requested_output_paths(task_input: TaskInput) -> list[str]:
    text = _task_request_text(task_input)
    first_write_marker = _first_write_marker(text)
    if first_write_marker is None:
        return []
    return _file_paths_from_text(text, start_at=first_write_marker)


def _requested_source_paths(task_input: TaskInput) -> list[str]:
    text = _task_request_text(task_input)
    first_write_marker = _first_write_marker(text)
    return _file_paths_from_text(text, end_before=first_write_marker)


def _task_request_text(task_input: TaskInput) -> str:
    goal = task_input.goal.strip()
    user_message = task_input.user_message.strip()
    if goal == user_message:
        return goal
    return f"{goal}\n{user_message}"


def _first_write_marker(text: str) -> int | None:
    lowered = text.lower()
    markers = (
        "\u4fdd\u5b58",
        "\u5199\u5165",
        "\u8f93\u51fa",
        "\u751f\u6210",
        "\u521b\u5efa",
        "\u5199\u6210",
        "\u603b\u7ed3\u4e3a",
        "\u6574\u7406\u4e3a",
        "\u6dc7\u6fde\u74e8",
        "\u9340\u6b10\u53dd",
        "\u6769\u64b3\u5687",
        "\u9435\u71b8\u57c2",
        "save",
        "write",
        "create",
        "output",
        "generate",
    )
    marker_positions = [lowered.find(marker) for marker in markers if lowered.find(marker) >= 0]
    return min(marker_positions) if marker_positions else None

def _file_paths_from_text(text: str, *, start_at: int = 0, end_before: int | None = None) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(r"(?<![\w./\\-])([\w./\\-]+\.(?:md|json|txt|csv|yaml|yml|pdf|xlsx))", re.IGNORECASE)
    for match in pattern.finditer(text):
        if match.start() < start_at:
            continue
        if end_before is not None and match.start() >= end_before:
            continue
        path = _normalize_mentioned_path(match.group(1))
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def _normalize_mentioned_path(raw: str) -> str:
    path = raw.strip().strip("。；;，,、)")
    if " " in path:
        path = path.split()[-1]
    return path.strip().strip("。；;，,、)")


def _plan_allowed_tool_hint(
    task_input: TaskInput,
    state: AgentState,
    decision: ActionDecision,
    prior_steps: list[RuntimeStepTrace],
) -> str:
    """Block tool calls that are outside the current compiled plan phase."""

    if decision.decision_type != DecisionType.TOOL_CALL or not decision.tool_name:
        return ""
    current = start_current_plan_step(state)
    if current is None:
        if state.plan_steps:
            if _read_only_recovery_is_safe(state, decision, set()):
                attach_recovery_hint(
                    state,
                    "Runtime allowed a read-only recovery tool after all plan phases were exhausted. "
                    "Use this to resolve the earlier error, then finish or ask the user.",
                )
                return ""
            return (
                "Plan guard: no active plan phase remains for tool execution. "
                f"The model requested {decision.tool_name}; finish the task, ask the user, or replan instead."
            )
        return ""
    if "allowed_tools=" not in current.description:
        return ""
    allowed_text = current.description.split("allowed_tools=", 1)[1].split(";", 1)[0].strip()
    if allowed_text in {"", "(none)"}:
        allowed: set[str] = set()
    else:
        allowed = {item.strip() for item in allowed_text.split(",") if item.strip()}
    if decision.tool_name in allowed:
        return ""
    if _read_only_recovery_is_safe(state, decision, allowed):
        attach_recovery_hint(
            state,
            "Runtime allowed a read-only recovery tool outside the current plan phase after prior errors. "
            "Use this one step to gather missing evidence or locate the correct path, then finish or replan.",
        )
        return ""
    phase_hint = ""
    if decision.tool_name in {"read_file", "read_many_files"} and {"write_file", "edit_file"} & allowed:
        remaining = _remaining_phase_output_paths(task_input, state, prior_steps)
        remaining_text = f" Remaining files for this phase: {', '.join(remaining)}." if remaining else ""
        phase_hint = (
            " Source content should already be available from the read phase; "
            "use write_file/edit_file to create the requested output instead of rereading."
            f"{remaining_text}"
        )
    return (
        "Plan guard: tool call is outside the current plan phase. "
        f"Current phase allows [{', '.join(sorted(allowed)) or 'no tools'}], "
        f"but model requested {decision.tool_name}. Choose an allowed tool, finish, or replan."
        f"{phase_hint}"
    )


def _read_only_recovery_is_safe(
    state: AgentState,
    decision: ActionDecision,
    allowed: set[str],
) -> bool:
    """Allow bounded read-only evidence recovery after an earlier error."""

    if decision.tool_name not in READ_ONLY_RECOVERY_TOOLS:
        return False
    if not state.errors:
        return False
    if {"write_file", "edit_file", "apply_patch", "memory_add"} & allowed:
        return False
    if allowed <= {"finish_task"}:
        return True
    if allowed <= {"compile_python", "run_tests", "git_diff"}:
        return True
    if not allowed:
        return True
    return False


def _finalize_remaining_step_budget(
    state: AgentState,
    *,
    max_steps: int,
    restored_step_count: int,
    used_steps: int,
    total_step_budget: int | None,
    total_remaining_steps: int | None,
) -> None:
    segment_used_steps = max(0, used_steps - restored_step_count)
    state.runtime_step_limit = max_steps
    state.runtime_remaining_steps = max(0, max_steps - segment_used_steps)
    if total_step_budget is not None:
        state.total_step_budget = total_step_budget
    if total_remaining_steps is not None:
        state.total_remaining_steps = max(0, total_remaining_steps - segment_used_steps)
    state.updated_at = utc_now()


def _repeat_guard_hint(decision: ActionDecision, prior_steps: list[RuntimeStepTrace]) -> str:
    """Stop repeated identical failing tool calls before wasting more steps."""

    if decision.decision_type != DecisionType.TOOL_CALL:
        return ""
    signature = _tool_call_signature(decision)
    failures = 0
    last_error = ""
    for step in reversed(prior_steps):
        if _tool_call_signature(step.decision) != signature:
            continue
        if step.tool_result is not None and not step.tool_result.success:
            failures += 1
            last_error = step.tool_result.error or "Tool failed."
        elif step.policy is not None and not step.policy.allowed:
            failures += 1
            last_error = step.policy.reason
        else:
            continue
        if failures >= 2:
            return (
                "Repeat guard: the same tool call has failed twice. "
                f"Do not call {decision.tool_name} again with the same arguments. "
                f"Last error: {last_error}. Change strategy: inspect the path with file_info/list_files, "
                "use chunked read_file, or ask the user if the target is ambiguous."
            )
    return ""


def _repeated_successful_read_hint(decision: ActionDecision, prior_steps: list[RuntimeStepTrace]) -> str:
    if decision.tool_name not in {"read_file", "read_many_files"}:
        return ""
    requested_paths = _paths_from_read_decision(decision)
    if not requested_paths:
        return ""
    successful_reads = 0
    for step in prior_steps:
        if not step.tool_result or not step.tool_result.success:
            continue
        if not (_paths_from_read_decision(step.decision) & requested_paths):
            continue
        successful_reads += 1
    if successful_reads >= 2:
        paths = ", ".join(sorted(requested_paths))
        return (
            "Runtime hint: the requested source has already been read multiple times. "
            f"Do not read {paths} again. Use the available evidence to write the requested output file, "
            "or finish with an explicit gap if the evidence is insufficient."
        )
    return ""


def _multi_source_read_guard_hint(
    task_input: TaskInput,
    state: AgentState,
    decision: ActionDecision,
    prior_steps: list[RuntimeStepTrace],
) -> str:
    if decision.tool_name not in {"read_file", "read_many_files"}:
        return ""
    if not _current_phase_allows_multi_source_reads(state):
        return ""
    source_paths = _requested_source_paths(task_input)
    if len(source_paths) <= 1:
        return ""
    read_paths = _successful_read_paths(prior_steps)
    requested_paths = _paths_from_read_decision(decision)
    remaining = [path for path in source_paths if path not in read_paths]
    if not remaining:
        return ""
    if requested_paths and requested_paths.issubset(read_paths):
        return (
            "Multi-source read guard: this source file was already read, but other requested source files remain unread. "
            f"Read the remaining source file(s) next: {', '.join(remaining)}."
        )
    return ""


def _duplicate_successful_write_guard_hint(
    task_input: TaskInput,
    state: AgentState,
    decision: ActionDecision,
    prior_steps: list[RuntimeStepTrace],
) -> str:
    if decision.tool_name not in {"write_file", "edit_file"} or not isinstance(decision.tool_args, dict):
        return ""
    path = decision.tool_args.get("path")
    if not isinstance(path, str) or not path:
        return ""
    written_paths = _successful_write_paths(prior_steps)
    if path not in written_paths:
        return ""
    expected_paths = _requested_output_paths(task_input)
    if expected_paths and all(expected in written_paths for expected in expected_paths):
        complete_current_plan_step(state, "Requested files for this phase already exist; advancing to the next phase.")
        return (
            "Duplicate write guard: this file was already written successfully. "
            "All requested files for the current phase exist, so the runtime advanced to the next phase. "
            "Do not overwrite existing files unless the user explicitly asks."
        )
    remaining = [expected for expected in expected_paths if expected not in written_paths]
    remaining_text = f" Remaining files for this phase: {', '.join(remaining)}." if remaining else ""
    return (
        "Duplicate write guard: this file was already written successfully. "
        "Do not overwrite existing files in this model-loop run; continue with a missing file or the next phase."
        f"{remaining_text}"
    )


def _successful_write_paths(steps: list[RuntimeStepTrace]) -> set[str]:
    paths: set[str] = set()
    for step in steps:
        if step.decision.tool_name not in {"write_file", "edit_file"}:
            continue
        if not step.tool_result or not step.tool_result.success:
            continue
        if not isinstance(step.decision.tool_args, dict):
            continue
        path = step.decision.tool_args.get("path")
        if isinstance(path, str) and path:
            paths.add(path)
    return paths


def _redirect_duplicate_source_read(
    task_input: TaskInput,
    state: AgentState,
    decision: ActionDecision,
    prior_steps: list[RuntimeStepTrace],
) -> bool:
    if decision.tool_name != "read_file" or not isinstance(decision.tool_args, dict):
        return False
    if not _current_phase_allows_multi_source_reads(state):
        return False
    source_paths = _requested_source_paths(task_input)
    if len(source_paths) <= 1:
        return False
    read_paths = _successful_read_paths(prior_steps)
    requested_paths = _paths_from_read_decision(decision)
    remaining = [path for path in source_paths if path not in read_paths]
    if not remaining or not requested_paths or not requested_paths.issubset(read_paths):
        return False
    redirected_path = remaining[0]
    decision.tool_args = {"path": redirected_path}
    attach_recovery_hint(
        state,
        "Runtime redirected a duplicate read_file request to the next unread source file from the original task: "
        f"{redirected_path}.",
    )
    return True


def _tool_call_signature(decision: ActionDecision) -> str:
    if decision.decision_type != DecisionType.TOOL_CALL:
        return ""
    try:
        encoded_args = json.dumps(decision.tool_args or {}, sort_keys=True, ensure_ascii=False)
    except TypeError:
        encoded_args = repr(decision.tool_args)
    return f"{decision.tool_name}:{encoded_args}"


def _save_runtime_checkpoint(
    checkpoint_path: str | None,
    task_input: TaskInput,
    task: Task,
    state: AgentState,
    steps: list[RuntimeStepTrace],
    started_at: float,
) -> None:
    """Persist the current runtime state if a checkpoint path was provided."""

    if not checkpoint_path:
        return
    save_runtime_checkpoint(
        checkpoint_path=checkpoint_path,
        task_input=task_input,
        task=task,
        state=state,
        steps=steps,
        stats=_runtime_stats(started_at, state, steps),
    )


def _emit_hook(hook_manager: HookManager | None, event: HookEvent, **kwargs) -> None:
    """Emit one hook event. Hook failures are isolated from task execution."""

    if hook_manager is None:
        return
    try:
        hook_manager.emit(HookContext(event=event, **kwargs))
    except Exception as exc:
        state = kwargs.get("state")
        if state is not None:
            add_error(state, f"Hook failed: {event.value}: {exc}")
        return
