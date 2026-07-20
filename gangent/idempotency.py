"""Tool idempotency helpers.

Idempotency（幂等性）means a resumed task can recognize a previously completed
side-effecting action and avoid repeating it.
"""

from __future__ import annotations

import json
from typing import Any

from .models import ActionDecision, ToolResult


REUSABLE_SIDE_EFFECT_TOOLS = {
    "write_file",
    "edit_file",
    "apply_patch",
    "git_add",
    "git_commit",
}


def find_reusable_tool_result(
    prior_steps: list[Any],
    decision: ActionDecision,
) -> ToolResult | None:
    """Return a prior successful result for the same side-effect action."""

    if decision.tool_name not in REUSABLE_SIDE_EFFECT_TOOLS:
        return None
    fingerprint = decision_fingerprint(decision)
    for step in reversed(prior_steps):
        if not step.tool_result or not step.tool_result.success:
            continue
        if decision_fingerprint(step.decision) != fingerprint:
            continue
        return ToolResult(
            call_id=step.tool_result.call_id,
            success=True,
            output=step.tool_result.output,
            error=step.tool_result.error,
            finished_at=step.tool_result.finished_at,
            reused=True,
        )
    return None


def decision_fingerprint(decision: ActionDecision) -> str:
    """Create a stable fingerprint for one tool call decision."""

    args = json.dumps(decision.tool_args or {}, ensure_ascii=True, sort_keys=True)
    return f"{decision.tool_name}|{args}"
