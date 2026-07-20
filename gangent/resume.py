"""Resume reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import AgentState


@dataclass
class ResumeReport:
    """恢复执行的简要报告。"""

    resumed: bool = False
    restored_step_count: int = 0
    new_step_count: int = 0
    reused_tool_call_count: int = 0
    restored_completed_steps: list[str] = field(default_factory=list)
    new_completed_steps: list[str] = field(default_factory=list)
    blocked_steps: list[str] = field(default_factory=list)
    summary: str = ""


def build_resume_report(
    resumed: bool,
    restored_step_count: int,
    restored_plan_done: set[str],
    state: AgentState,
    steps: list[Any],
) -> ResumeReport | None:
    """Build a compact diff between restored and newly completed work."""

    if not resumed:
        return None
    new_step_count = max(0, len(steps) - restored_step_count)
    reused_tool_call_count = sum(1 for step in steps[restored_step_count:] if step.reused_tool_result)
    restored_completed_steps = [
        step.title for step in state.plan_steps if step.step_id in restored_plan_done
    ]
    new_completed_steps = [
        step.title
        for step in state.plan_steps
        if step.status.value == "done" and step.step_id not in restored_plan_done
    ]
    blocked_steps = [step.title for step in state.plan_steps if step.status.value == "blocked"]
    summary = (
        f"resumed from checkpoint; restored_steps={restored_step_count}; "
        f"new_steps={new_step_count}; reused_tool_calls={reused_tool_call_count}; "
        f"new_completed_steps={len(new_completed_steps)}"
    )
    return ResumeReport(
        resumed=True,
        restored_step_count=restored_step_count,
        new_step_count=new_step_count,
        reused_tool_call_count=reused_tool_call_count,
        restored_completed_steps=restored_completed_steps,
        new_completed_steps=new_completed_steps,
        blocked_steps=blocked_steps,
        summary=summary,
    )
