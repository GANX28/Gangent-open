"""Planner budget controls.

This module keeps budget pressure outside the model's imagination. The model
still chooses the next action, but every call receives deterministic guidance
about remaining steps, plan progress, and action granularity.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import AgentState, PlanStepStatus, utc_now


@dataclass(frozen=True)
class PlannerBudgetControl:
    """Model-facing step budget state for one runtime boundary."""

    profile: str
    segment_step_limit: int
    segment_remaining_steps: int
    total_step_budget: int
    total_remaining_steps: int
    completed_plan_steps: int
    pending_plan_steps: int
    blocked_plan_steps: int
    pressure: str


def update_planner_budget_state(
    state: AgentState,
    *,
    segment_step_limit: int,
    segment_remaining_steps: int,
    profile: str | None = None,
    total_step_budget: int | None = None,
    total_remaining_steps: int | None = None,
) -> AgentState:
    """Attach current runtime budget numbers to AgentState."""

    state.runtime_step_limit = max(0, int(segment_step_limit))
    state.runtime_remaining_steps = max(0, int(segment_remaining_steps))
    if profile is not None:
        state.budget_profile = profile
    if total_step_budget is not None:
        state.total_step_budget = max(0, int(total_step_budget))
    if total_remaining_steps is not None:
        state.total_remaining_steps = max(0, int(total_remaining_steps))
    state.updated_at = utc_now()
    return state


def build_planner_budget_control(state: AgentState) -> PlannerBudgetControl:
    """Summarize plan and budget pressure for deterministic model guidance."""

    completed = sum(1 for step in state.plan_steps if step.status == PlanStepStatus.DONE)
    blocked = sum(1 for step in state.plan_steps if step.status == PlanStepStatus.BLOCKED)
    pending = sum(
        1
        for step in state.plan_steps
        if step.status in {PlanStepStatus.TODO, PlanStepStatus.RUNNING}
    )
    remaining = max(0, state.runtime_remaining_steps)
    pressure = _budget_pressure(remaining, pending)
    return PlannerBudgetControl(
        profile=state.budget_profile or "unknown",
        segment_step_limit=max(0, state.runtime_step_limit),
        segment_remaining_steps=remaining,
        total_step_budget=max(0, state.total_step_budget),
        total_remaining_steps=max(0, state.total_remaining_steps),
        completed_plan_steps=completed,
        pending_plan_steps=pending,
        blocked_plan_steps=blocked,
        pressure=pressure,
    )


def format_planner_budget_control(state: AgentState) -> str:
    """Format deterministic planner constraints for the model context."""

    control = build_planner_budget_control(state)
    lines = [
        "Planner Budget Control:",
        f"- profile: {control.profile}",
        f"- segment_step_limit: {control.segment_step_limit}",
        f"- segment_remaining_steps: {control.segment_remaining_steps}",
        f"- total_step_budget: {control.total_step_budget}",
        f"- total_remaining_steps: {control.total_remaining_steps}",
        f"- plan_progress: done={control.completed_plan_steps}; pending={control.pending_plan_steps}; blocked={control.blocked_plan_steps}",
        f"- budget_pressure: {control.pressure}",
        "",
        "Rules:",
        "- Treat one runtime step as one model decision, not as an entire project phase.",
        "- Prefer one purposeful tool call per step when tool evidence is needed.",
        "- Avoid micro-steps: do not split one simple read/search/edit into many tiny decisions.",
        "- Avoid giant steps: do not combine unrelated file edits, broad searches, tests, and final summary in one decision.",
        "- Finish as soon as the requested deliverable is complete and verified enough.",
    ]
    if control.segment_remaining_steps <= 2:
        lines.append("- Critical pressure: do not start broad exploration; finish, verify narrowly, or ask the user if blocked.")
    elif control.segment_remaining_steps <= max(4, control.pending_plan_steps):
        lines.append("- High pressure: choose the shortest action that advances the current plan step.")
    else:
        lines.append("- Normal pressure: gather or edit only the context needed for the current plan step.")
    return "\n".join(lines)


def _budget_pressure(remaining_steps: int, pending_plan_steps: int) -> str:
    if remaining_steps <= 2:
        return "critical"
    if pending_plan_steps and remaining_steps <= pending_plan_steps:
        return "high"
    if pending_plan_steps and remaining_steps <= pending_plan_steps * 2:
        return "medium"
    return "low"
