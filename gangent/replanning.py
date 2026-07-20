"""Deterministic event-aware replanning.

This module implements the first practical version of streaming state control:
new inputs are consumed at safe runtime boundaries, compared with the original
request and current progress, then converted into a small plan patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .events import AgentEventType, QueuedEvent
from .models import AgentState, PlanStep, PlanStepStatus, Task, TaskInput, new_id, utc_now


class PlanPatchAction(str, Enum):
    """How pending runtime events should affect the active plan."""

    CONTINUE = "continue"
    PAUSE = "pause"
    APPEND_STEPS = "append_steps"
    REPLACE_PENDING_STEPS = "replace_pending_steps"
    MARK_OUTPUTS_STALE = "mark_outputs_stale"
    ASK_USER = "ask_user"
    STABILIZE = "stabilize"


@dataclass(frozen=True)
class EventBudget:
    """Soft limits for event-driven replanning.

    These limits are intentionally hidden from the user-facing interaction. The
    runtime accepts events freely, but it enters stabilization mode when too many
    changes arrive before the current task can settle.
    """

    event_count: int
    replan_count: int
    interrupt_count: int
    pending_event_count: int
    max_auto_replans: int = 3
    max_interrupts: int = 5
    max_pending_events: int = 8

    @property
    def stabilization_required(self) -> bool:
        return (
            self.replan_count >= self.max_auto_replans
            or self.interrupt_count >= self.max_interrupts
            or self.pending_event_count > self.max_pending_events
        )


@dataclass(frozen=True)
class ReplanContext:
    """Snapshot used to compare old intent, new input, and current progress."""

    original_user_request: str
    latest_user_events: tuple[str, ...]
    current_runtime_phase: str
    current_plan_step_title: str
    completed_steps: tuple[str, ...]
    pending_steps: tuple[str, ...]
    intermediate_artifacts: tuple[str, ...]
    current_outputs: tuple[str, ...]
    constraints: tuple[str, ...]
    event_budget: EventBudget


@dataclass(frozen=True)
class PlanPatch:
    """Small deterministic change to the active plan."""

    action: PlanPatchAction
    reason: str
    affected_steps: tuple[str, ...] = ()
    stale_outputs: tuple[str, ...] = ()
    new_steps: tuple[PlanStep, ...] = ()
    need_user_confirmation: bool = False


def build_event_budget(state: AgentState, events: list[QueuedEvent]) -> EventBudget:
    """Build a soft event budget from persistent state and the current batch."""

    return EventBudget(
        event_count=state.event_count + len(events),
        replan_count=state.replan_count,
        interrupt_count=state.interrupt_count,
        pending_event_count=len(events),
    )


def build_replan_context(
    task_input: TaskInput,
    task: Task,
    state: AgentState,
    events: list[QueuedEvent],
) -> ReplanContext:
    """Package original request, new input, and runtime progress together."""

    del task
    latest_user_events = tuple(
        queued.event.content
        for queued in events
        if queued.event.event_type
        in {
            AgentEventType.USER_INPUT,
            AgentEventType.REPLAN_REQUEST,
            AgentEventType.USER_INTERRUPT,
            AgentEventType.APPROVAL,
        }
    )
    current_step = _current_plan_step(state)
    completed_steps = tuple(step.title for step in state.plan_steps if step.status == PlanStepStatus.DONE)
    pending_steps = tuple(
        step.title for step in state.plan_steps if step.status in {PlanStepStatus.TODO, PlanStepStatus.RUNNING}
    )
    intermediate_artifacts = tuple(
        step.result_summary
        for step in state.plan_steps
        if step.result_summary and step.status in {PlanStepStatus.DONE, PlanStepStatus.BLOCKED}
    )[-8:]
    current_outputs = tuple(_successful_written_paths(state))[-12:]
    return ReplanContext(
        original_user_request=task_input.user_message or task_input.goal,
        latest_user_events=latest_user_events,
        current_runtime_phase=state.event_runtime_state or state.phase.value,
        current_plan_step_title=current_step.title if current_step else "",
        completed_steps=completed_steps,
        pending_steps=pending_steps,
        intermediate_artifacts=intermediate_artifacts,
        current_outputs=current_outputs,
        constraints=tuple(task_input.constraints),
        event_budget=build_event_budget(state, events),
    )


def plan_patch_from_events(context: ReplanContext, events: list[QueuedEvent]) -> PlanPatch:
    """Convert a batch of events into a bounded plan patch."""

    if not events:
        return PlanPatch(PlanPatchAction.CONTINUE, "no pending runtime events")
    if context.event_budget.stabilization_required:
        return PlanPatch(
            PlanPatchAction.STABILIZE,
            "event pressure exceeded automatic replanning budget",
            affected_steps=context.pending_steps,
            need_user_confirmation=True,
        )

    ordered = sorted(events, key=lambda item: (-item.event.priority, item.index))
    highest = ordered[0].event
    event_types = {item.event.event_type for item in ordered}
    affected_steps = context.pending_steps

    if highest.event_type in {AgentEventType.USER_INTERRUPT, AgentEventType.SYSTEM_SIGNAL}:
        return PlanPatch(
            PlanPatchAction.PAUSE,
            "runtime was asked to pause before continuing",
            affected_steps=affected_steps,
            need_user_confirmation=True,
        )
    if highest.event_type == AgentEventType.ROLLBACK_REQUEST:
        return PlanPatch(
            PlanPatchAction.ASK_USER,
            "rollback changes external files and needs explicit confirmation",
            affected_steps=affected_steps,
            need_user_confirmation=True,
        )
    if AgentEventType.REPLAN_REQUEST in event_types or _has_requirement_change(event_types):
        return PlanPatch(
            PlanPatchAction.REPLACE_PENDING_STEPS,
            "new requirement should revise unfinished work while preserving completed evidence",
            affected_steps=affected_steps,
            stale_outputs=context.current_outputs,
            new_steps=(
                _event_step(
                    "Resolve event-driven requirement change",
                    "Compare the original request, current progress, and newest input before acting.",
                    "A revised next action that avoids stale assumptions.",
                    "search_context",
                ),
                _event_step(
                    "Validate revised outputs",
                    "Check that outputs still match the latest requirement and source evidence.",
                    "Validated outputs or a clear missing-information note.",
                    "finish_task",
                ),
            ),
        )
    if highest.event_type == AgentEventType.USER_INPUT and highest.priority >= 70:
        return PlanPatch(
            PlanPatchAction.REPLACE_PENDING_STEPS,
            "high-priority user input should revise unfinished work",
            affected_steps=affected_steps,
            stale_outputs=context.current_outputs,
            new_steps=(
                _event_step(
                    "Incorporate high-priority user input",
                    "Compare the original request with the newest user input, then choose the next safe action.",
                    "A revised next action under the current budget.",
                    "search_context",
                ),
            ),
        )
    if AgentEventType.FILE_CHANGE in event_types or _has_new_file_event(event_types):
        return PlanPatch(
            PlanPatchAction.APPEND_STEPS,
            "new file input should be read before final output",
            affected_steps=affected_steps,
            stale_outputs=context.current_outputs,
            new_steps=(
                _event_step(
                    "Read newly arrived source",
                    "Inspect the file-change event and update source evidence before continuing.",
                    "Updated source understanding with provenance.",
                    "read_file",
                ),
            ),
        )
    if highest.event_type == AgentEventType.APPROVAL and _looks_negative(highest.content):
        return PlanPatch(
            PlanPatchAction.APPEND_STEPS,
            "approval feedback rejected part of the current result",
            affected_steps=affected_steps,
            stale_outputs=context.current_outputs,
            new_steps=(
                _event_step(
                    "Repair after approval feedback",
                    "Use the approval feedback to revise only the affected unfinished result.",
                    "A repaired result or a precise follow-up question.",
                    "finish_task",
                ),
            ),
        )
    if event_types & {AgentEventType.USER_INPUT, AgentEventType.TOOL_RESULT, AgentEventType.APPROVAL}:
        return PlanPatch(
            PlanPatchAction.APPEND_STEPS,
            "event can be incorporated without replacing the current plan",
            affected_steps=affected_steps,
            new_steps=(
                _event_step(
                    "Incorporate runtime event",
                    "Use the new event as additional context without discarding completed work.",
                    "Continuation aligned with the latest event.",
                    None,
                ),
            ),
        )
    return PlanPatch(PlanPatchAction.CONTINUE, "event recorded without plan change", affected_steps=affected_steps)


def apply_plan_patch(state: AgentState, patch: PlanPatch) -> None:
    """Apply a deterministic plan patch to state."""

    if patch.action == PlanPatchAction.REPLACE_PENDING_STEPS:
        affected = set(patch.affected_steps)
        for step in state.plan_steps:
            if step.status in {PlanStepStatus.TODO, PlanStepStatus.RUNNING} and (
                not affected or step.title in affected
            ):
                step.status = PlanStepStatus.BLOCKED
                step.result_summary = _trim(f"Superseded by runtime event: {patch.reason}")
    if patch.action in {PlanPatchAction.APPEND_STEPS, PlanPatchAction.REPLACE_PENDING_STEPS}:
        state.plan_steps.extend(patch.new_steps)
    for output in patch.stale_outputs:
        if output not in state.stale_outputs:
            state.stale_outputs.append(output)
    if patch.action in {PlanPatchAction.REPLACE_PENDING_STEPS, PlanPatchAction.MARK_OUTPUTS_STALE}:
        state.replan_count += 1
    if patch.action in {PlanPatchAction.PAUSE, PlanPatchAction.ASK_USER, PlanPatchAction.STABILIZE}:
        state.interrupt_count += 1
    if patch.action == PlanPatchAction.STABILIZE:
        state.stabilization_required = True
    summary = format_plan_patch(patch)
    state.plan_patch_summaries.append(summary)
    state.updated_at = utc_now()


def format_replan_context(context: ReplanContext, max_chars: int = 1800) -> str:
    """Human/model-readable context package for event-aware replanning."""

    lines = [
        "ReplanContext:",
        f"- original_user_request: {_trim(context.original_user_request, 300)}",
        f"- latest_user_events: {_trim(' | '.join(context.latest_user_events), 500)}",
        f"- current_runtime_phase: {context.current_runtime_phase}",
        f"- current_plan_step: {context.current_plan_step_title or '-'}",
        f"- completed_steps: {_trim(' | '.join(context.completed_steps), 300)}",
        f"- pending_steps: {_trim(' | '.join(context.pending_steps), 300)}",
        f"- intermediate_artifacts: {_trim(' | '.join(context.intermediate_artifacts), 400)}",
        f"- stale/current_outputs: {_trim(' | '.join(context.current_outputs), 300)}",
        (
            "- event_budget: "
            f"events={context.event_budget.event_count}, "
            f"replans={context.event_budget.replan_count}, "
            f"interrupts={context.event_budget.interrupt_count}, "
            f"pending={context.event_budget.pending_event_count}"
        ),
    ]
    return _trim("\n".join(lines), max_chars)


def format_plan_patch(patch: PlanPatch) -> str:
    new_step_titles = ", ".join(step.title for step in patch.new_steps)
    stale = ", ".join(patch.stale_outputs)
    affected = ", ".join(patch.affected_steps)
    parts = [f"plan_patch={patch.action.value}", f"reason={patch.reason}"]
    if affected:
        parts.append(f"affected={affected}")
    if new_step_titles:
        parts.append(f"new_steps={new_step_titles}")
    if stale:
        parts.append(f"stale_outputs={stale}")
    if patch.need_user_confirmation:
        parts.append("need_user_confirmation=true")
    return "; ".join(parts)


def _event_step(title: str, description: str, expected_output: str, tool_hint: str | None) -> PlanStep:
    allowed_tools = "read_file,read_many_files,write_file,edit_file,file_info,finish_task"
    return PlanStep(
        step_id=new_id("step"),
        title=title,
        description=(
            f"phase=event_replan; max_steps=3; allowed_tools={allowed_tools}; "
            f"exit_criteria={expected_output}. {description}"
        ),
        purpose="Handle a cooperative runtime event at a safe boundary.",
        expected_output=expected_output,
        tool_hint=tool_hint,
    )


def _current_plan_step(state: AgentState) -> PlanStep | None:
    for step in state.plan_steps:
        if step.status in {PlanStepStatus.TODO, PlanStepStatus.RUNNING}:
            return step
    return None


def _successful_written_paths(state: AgentState) -> list[str]:
    result = state.last_tool_result
    decision = state.last_decision
    if not result or not decision or not result.success:
        return []
    if decision.tool_name not in {"write_file", "edit_file", "apply_patch"}:
        return []
    path = ""
    if decision.tool_args:
        path = str(decision.tool_args.get("path") or decision.tool_args.get("file") or "")
    return [path] if path else []


def _has_requirement_change(event_types: set[AgentEventType]) -> bool:
    return any(event_type.value in {"requirement_change"} for event_type in event_types)


def _has_new_file_event(event_types: set[AgentEventType]) -> bool:
    return any(event_type.value in {"new_file_added"} for event_type in event_types)


def _looks_negative(text: str) -> bool:
    lower = text.lower()
    return any(word in lower for word in ["reject", "rejected", "revise", "wrong", "拒绝", "不同意", "不通过", "修改"])


def _trim(text: str, max_chars: int = 240) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."
