"""Adaptive Runtime v1.

This layer sits above the base runtime loop and adds:
- automatic task budget selection
- staged step/time allocation
- continuation when one segment exhausts its local step budget
- one-shot token retry for malformed or truncated tool-call JSON
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from .llm_client import LLMClient
from .models import RuntimeStats, TaskInput, TaskStatus
from .planner import infer_task_kind
from .checkpoint import checkpoint_from_runtime_result
from .failure import FailureReason, recoverable_failure_reason
from .runtime import RuntimeResult, run_task
from .task_profile import task_execution_profile

if TYPE_CHECKING:
    from .tool_registry import ToolRegistry


HISTORY_TOKEN_CAPS = {
    "light": 4_000,
    "medium": 8_000,
    "heavy": 12_000,
    "ultra": 42_000,
}

HISTORY_PROFILE_STEP_CAPS = {
    "direct": 2,
    "single_read": 6,
    "single_write": 6,
    "read_analysis": 12,
    "git_analysis": 12,
    "read_write": 12,
}

HISTORY_PROFILE_TOKEN_CAPS = {
    "direct": 320,
    "single_read": 1_200,
    "single_write": 1_200,
    "read_analysis": 4_000,
    "git_analysis": 3_000,
    "read_write": 3_000,
}


@dataclass(frozen=True)
class AdaptiveBudget:
    """Resolved task budget for one user request."""

    profile: str
    total_steps: int
    segment_steps: tuple[int, ...]
    total_seconds: float
    segment_seconds: tuple[float, ...]
    max_tokens: int
    retry_max_tokens: int


def resolve_budget(
    task_input: TaskInput,
    profile: str = "auto",
    max_steps: int | None = None,
    max_tokens: int | None = None,
    max_seconds: float | None = None,
) -> AdaptiveBudget:
    """Resolve the adaptive budget for one task.

    Manual overrides remain supported, but they are optional. When no override is
    provided, the runtime selects a profile from the task content.
    """

    selected_profile = profile if profile != "auto" else infer_budget_profile(task_input)
    budget = _profile_budget(selected_profile)
    manual_steps = max_steps is not None
    manual_tokens = max_tokens is not None
    manual_seconds = max_seconds is not None
    budget = _apply_execution_profile_budget(
        budget,
        task_input,
        apply_steps=not manual_steps,
        apply_tokens=not manual_tokens,
        apply_seconds=not manual_seconds,
    )
    if max_steps is not None:
        budget = AdaptiveBudget(
            profile=budget.profile,
            total_steps=max_steps,
            segment_steps=_segment_steps_for_override(max_steps),
            total_seconds=budget.total_seconds,
            segment_seconds=budget.segment_seconds,
            max_tokens=budget.max_tokens,
            retry_max_tokens=budget.retry_max_tokens,
        )
    if max_tokens is not None:
        budget = AdaptiveBudget(
            profile=budget.profile,
            total_steps=budget.total_steps,
            segment_steps=budget.segment_steps,
            total_seconds=budget.total_seconds,
            segment_seconds=budget.segment_seconds,
            max_tokens=max_tokens,
            retry_max_tokens=max(max_tokens + 800, int(max_tokens * 1.5)),
        )
    if max_seconds is not None:
        budget = AdaptiveBudget(
            profile=budget.profile,
            total_steps=budget.total_steps,
            segment_steps=budget.segment_steps,
            total_seconds=max_seconds,
            segment_seconds=_segment_seconds_for_override(max_seconds, len(budget.segment_steps)),
            max_tokens=budget.max_tokens,
            retry_max_tokens=budget.retry_max_tokens,
        )
    return budget


def _apply_execution_profile_budget(
    budget: AdaptiveBudget,
    task_input: TaskInput,
    *,
    apply_steps: bool,
    apply_tokens: bool,
    apply_seconds: bool,
) -> AdaptiveBudget:
    profile = task_execution_profile(task_input)
    if profile.name == "direct":
        total_seconds = min(budget.total_seconds, 60.0) if apply_seconds else budget.total_seconds
        total_steps = 2 if apply_steps else budget.total_steps
        return AdaptiveBudget(
            profile=f"{budget.profile}:direct",
            total_steps=total_steps,
            segment_steps=(total_steps,) if apply_steps else budget.segment_steps,
            total_seconds=total_seconds,
            segment_seconds=(total_seconds,) if apply_seconds else budget.segment_seconds,
            max_tokens=min(budget.max_tokens, 160) if apply_tokens else budget.max_tokens,
            retry_max_tokens=min(budget.retry_max_tokens, 320) if apply_tokens else budget.retry_max_tokens,
        )
    if profile.name == "read_analysis":
        total_seconds = min(budget.total_seconds, 240.0) if apply_seconds else budget.total_seconds
        total_steps = 12 if apply_steps else budget.total_steps
        return AdaptiveBudget(
            profile=f"{budget.profile}:read_analysis",
            total_steps=total_steps,
            segment_steps=(total_steps,) if apply_steps else budget.segment_steps,
            total_seconds=total_seconds,
            segment_seconds=(total_seconds,) if apply_seconds else budget.segment_seconds,
            max_tokens=min(budget.max_tokens, 2200) if apply_tokens else budget.max_tokens,
            retry_max_tokens=min(budget.retry_max_tokens, 3400) if apply_tokens else budget.retry_max_tokens,
        )
    if profile.name == "git_analysis":
        total_seconds = min(budget.total_seconds, 240.0) if apply_seconds else budget.total_seconds
        total_steps = 8 if apply_steps else budget.total_steps
        return AdaptiveBudget(
            profile=f"{budget.profile}:git_analysis",
            total_steps=total_steps,
            segment_steps=(total_steps,) if apply_steps else budget.segment_steps,
            total_seconds=total_seconds,
            segment_seconds=(total_seconds,) if apply_seconds else budget.segment_seconds,
            max_tokens=min(budget.max_tokens, 1800) if apply_tokens else budget.max_tokens,
            retry_max_tokens=min(budget.retry_max_tokens, 3000) if apply_tokens else budget.retry_max_tokens,
        )
    if profile.name == "single_read":
        total_seconds = min(budget.total_seconds, 120.0) if apply_seconds else budget.total_seconds
        total_steps = 4 if apply_steps else budget.total_steps
        return AdaptiveBudget(
            profile=f"{budget.profile}:{profile.name}",
            total_steps=total_steps,
            segment_steps=(total_steps,) if apply_steps else budget.segment_steps,
            total_seconds=total_seconds,
            segment_seconds=(total_seconds,) if apply_seconds else budget.segment_seconds,
            max_tokens=min(budget.max_tokens, 900) if apply_tokens else budget.max_tokens,
            retry_max_tokens=min(budget.retry_max_tokens, 1600) if apply_tokens else budget.retry_max_tokens,
        )
    if profile.name == "single_write":
        total_seconds = min(budget.total_seconds, 180.0) if apply_seconds else budget.total_seconds
        total_steps = 6 if apply_steps else budget.total_steps
        return AdaptiveBudget(
            profile=f"{budget.profile}:single_write",
            total_steps=total_steps,
            segment_steps=(total_steps,) if apply_steps else budget.segment_steps,
            total_seconds=total_seconds,
            segment_seconds=(total_seconds,) if apply_seconds else budget.segment_seconds,
            max_tokens=min(budget.max_tokens, 1400) if apply_tokens else budget.max_tokens,
            retry_max_tokens=min(budget.retry_max_tokens, 2400) if apply_tokens else budget.retry_max_tokens,
        )
    if profile.name == "read_write":
        total_seconds = min(budget.total_seconds, 240.0) if apply_seconds else budget.total_seconds
        total_steps = 6 if apply_steps else budget.total_steps
        return AdaptiveBudget(
            profile=f"{budget.profile}:read_write",
            total_steps=total_steps,
            segment_steps=(total_steps,) if apply_steps else budget.segment_steps,
            total_seconds=total_seconds,
            segment_seconds=(total_seconds,) if apply_seconds else budget.segment_seconds,
            max_tokens=min(budget.max_tokens, 1200) if apply_tokens else budget.max_tokens,
            retry_max_tokens=min(budget.retry_max_tokens, 2400) if apply_tokens else budget.retry_max_tokens,
        )
    return budget


def apply_budget_recommendation(budget: AdaptiveBudget, recommendation) -> AdaptiveBudget:
    """Raise a heuristic budget with historical percentile data when available."""

    if recommendation is None:
        return budget
    step_cap = _history_step_cap(budget.profile)
    token_cap = _history_token_cap(budget.profile)
    recommended_steps = min(step_cap, max(budget.total_steps, int(recommendation.steps_p80) + 2))
    recommended_seconds = max(budget.total_seconds, float(recommendation.seconds_p80) + 20.0)
    recommended_tokens = min(
        token_cap,
        max(budget.max_tokens, int(recommendation.tokens_p80) + 400),
    )
    if recommended_tokens == budget.max_tokens and recommended_steps == budget.total_steps and recommended_seconds == budget.total_seconds:
        return budget
    return AdaptiveBudget(
        profile=f"{budget.profile}+history",
        total_steps=recommended_steps,
        segment_steps=_segment_steps_for_override(recommended_steps),
        total_seconds=recommended_seconds,
        segment_seconds=_segment_seconds_for_override(recommended_seconds, len(_segment_steps_for_override(recommended_steps))),
        max_tokens=recommended_tokens,
        retry_max_tokens=max(budget.retry_max_tokens, recommended_tokens + 800, int(recommended_tokens * 1.5)),
    )


def _history_token_cap(profile: str) -> int:
    execution_profile = _execution_profile_from_budget_profile(profile)
    if execution_profile in HISTORY_PROFILE_TOKEN_CAPS:
        return HISTORY_PROFILE_TOKEN_CAPS[execution_profile]
    base_profile = profile.split("+", 1)[0].split(":", 1)[0]
    return HISTORY_TOKEN_CAPS.get(base_profile, HISTORY_TOKEN_CAPS["heavy"])


def _history_step_cap(profile: str) -> int:
    execution_profile = _execution_profile_from_budget_profile(profile)
    if execution_profile in HISTORY_PROFILE_STEP_CAPS:
        return HISTORY_PROFILE_STEP_CAPS[execution_profile]
    base_profile = profile.split("+", 1)[0].split(":", 1)[0]
    return {
        "light": 40,
        "medium": 72,
        "heavy": 120,
        "ultra": 220,
    }.get(base_profile, 96)


def _execution_profile_from_budget_profile(profile: str) -> str:
    base = profile.split("+", 1)[0]
    if ":" not in base:
        return ""
    return base.split(":", 1)[1]


def infer_budget_profile(task_input: TaskInput) -> str:
    """Infer one of light/medium/heavy from the user task."""

    text = f"{task_input.goal}\n{task_input.user_message}".lower()
    task_kind = infer_task_kind(task_input)
    complexity_score = 0

    if len(text) > 240:
        complexity_score += 1
    if any(word in text for word in [" and ", "compare", "optimize", "architecture", "structure"]):
        complexity_score += 1
    if any(word in text for word in ["test", "fix", "debug", "refactor", "multi-file", "commercial", "商用"]):
        complexity_score += 1
    if any(
        word in text
        for word in [
            "write",
            "save",
            "create file",
            "document",
            "markdown",
            "生成",
            "写",
            "写入",
            "保存",
            "文档",
            "文件",
        ]
    ):
        complexity_score += 1
    if task_kind == "analysis":
        complexity_score += 1
    if task_kind == "debug":
        complexity_score += 1
    if task_kind == "build":
        complexity_score += 1

    if complexity_score >= 3:
        return "heavy"
    if complexity_score >= 1:
        return "medium"
    return "light"


def run_task_adaptive(
    task_input: TaskInput,
    client_factory: Callable[[int], LLMClient],
    budget: AdaptiveBudget,
    approval_callback=None,
    checkpoint_path: str | None = None,
    resume_checkpoint=None,
    tool_registry: "ToolRegistry | None" = None,
    event_queue_path: str | None = None,
    hook_manager=None,
) -> RuntimeResult:
    """Run one task through the adaptive controller."""

    remaining_steps = budget.total_steps
    remaining_seconds = budget.total_seconds
    current_tokens = budget.max_tokens
    retries_used = 0
    segment_index = 0

    aggregate_steps = []
    aggregate_usage: dict = {}
    aggregate_duration_seconds = 0.0
    last_result: RuntimeResult | None = None
    current_task_input = task_input
    resume_checkpoint = resume_checkpoint

    while remaining_steps > 0 and remaining_seconds > 0:
        segment_steps = budget.segment_steps[min(segment_index, len(budget.segment_steps) - 1)]
        segment_seconds = budget.segment_seconds[min(segment_index, len(budget.segment_seconds) - 1)]
        segment_steps = min(segment_steps, remaining_steps)
        segment_seconds = min(segment_seconds, remaining_seconds)

        try:
            client = client_factory(current_tokens)
            result = run_task(
                current_task_input,
                client,
                max_steps=segment_steps,
                max_seconds=segment_seconds,
                approval_callback=approval_callback,
                checkpoint_path=checkpoint_path,
                resume_checkpoint=resume_checkpoint,
                tool_registry=tool_registry,
                event_queue_path=event_queue_path,
                hook_manager=hook_manager,
                budget_profile=budget.profile,
                total_step_budget=budget.total_steps,
                total_remaining_steps=remaining_steps,
            )
        except Exception as exc:
            if _is_retryable_json_error(exc) and retries_used < 1:
                retries_used += 1
                current_tokens = max(current_tokens + 800, min(budget.retry_max_tokens, int(current_tokens * 1.5)))
                continue
            raise

        last_result = result
        step_slice = _result_step_slice(result, aggregate_steps)
        aggregate_steps.extend(step_slice)
        _merge_usage(aggregate_usage, result.stats.usage)
        aggregate_duration_seconds += result.stats.duration_seconds

        consumed_steps = max(1, _consumed_step_count(result))
        remaining_steps = max(0, remaining_steps - consumed_steps)
        remaining_seconds = max(0.0, remaining_seconds - result.stats.duration_seconds)

        if result.task.status in {TaskStatus.COMPLETED, TaskStatus.WAITING_USER}:
            return _merged_result(result, aggregate_steps, aggregate_usage, aggregate_duration_seconds)

        if (
            _failed_due_to_max_steps(result)
            and remaining_steps > 0
            and remaining_seconds > 0
            and not _unstable_segment_should_stop(result)
        ):
            resume_checkpoint = checkpoint_from_runtime_result(result)
            segment_index += 1
            continue

        return _merged_result(result, aggregate_steps, aggregate_usage, aggregate_duration_seconds)

    if last_result is None:
        raise RuntimeError("Adaptive runtime could not start any task segment.")
    return _merged_result(last_result, aggregate_steps, aggregate_usage, aggregate_duration_seconds)


def format_budget_summary(budget: AdaptiveBudget) -> str:
    """Human-readable budget summary for CLI startup and debugging."""

    return (
        f"profile={budget.profile}; total_steps={budget.total_steps}; "
        f"segment_steps={list(budget.segment_steps)}; total_seconds={budget.total_seconds:g}; "
        f"max_tokens={budget.max_tokens}"
    )


def _profile_budget(profile: str) -> AdaptiveBudget:
    if profile == "light":
        return AdaptiveBudget(
            profile="light",
            total_steps=20,
            segment_steps=(10, 10),
            total_seconds=240.0,
            segment_seconds=(120.0, 120.0),
            max_tokens=2200,
            retry_max_tokens=3400,
        )
    if profile == "medium":
        return AdaptiveBudget(
            profile="medium",
            total_steps=48,
            segment_steps=(12, 12, 12, 12),
            total_seconds=720.0,
            segment_seconds=(180.0, 180.0, 180.0, 180.0),
            max_tokens=3600,
            retry_max_tokens=5400,
        )
    if profile == "heavy":
        return AdaptiveBudget(
            profile="heavy",
            total_steps=96,
            segment_steps=(16,) * 6,
            total_seconds=1440.0,
            segment_seconds=(240.0,) * 6,
            max_tokens=5600,
            retry_max_tokens=8400,
        )
    if profile == "ultra":
        return AdaptiveBudget(
            profile="ultra",
            total_steps=220,
            segment_steps=(20,) * 11,
            total_seconds=3600.0,
            segment_seconds=(327.0,) * 11,
            max_tokens=28_000,
            retry_max_tokens=42_000,
        )
    raise ValueError(f"Unknown budget profile: {profile}")


def _segment_steps_for_override(total_steps: int) -> tuple[int, ...]:
    if total_steps <= 0:
        raise ValueError("total_steps must be greater than 0.")
    if total_steps <= 16:
        segment_cap = total_steps
    elif total_steps <= 40:
        segment_cap = 20
    else:
        segment_cap = 20

    segments: list[int] = []
    remaining = total_steps
    while remaining > segment_cap:
        segments.append(segment_cap)
        remaining -= segment_cap
    segments.append(remaining)
    return tuple(segments)


def _segment_seconds_for_override(total_seconds: float, segments: int) -> tuple[float, ...]:
    safe_segments = max(1, segments)
    base = max(1.0, total_seconds / safe_segments)
    return tuple(base for _ in range(safe_segments))


def _is_retryable_json_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "function arguments" in message and "valid json" in message


def _failed_due_to_max_steps(result: RuntimeResult) -> bool:
    return recoverable_failure_reason(result.state.errors) == FailureReason.MAX_STEPS


def _unstable_segment_should_stop(result: RuntimeResult) -> bool:
    """Avoid automatic continuation after repeated guard/parser failures.

    Continuation is useful when work is progressing. When a segment is mostly
    blocked by schema, policy, or phase-guard failures, resuming tends to replay
    the same bad loop with a larger context bill.
    """

    unstable_markers = (
        "Model output parse failed",
        "Plan guard:",
        "Repeat guard:",
        "policy prevented",
        "requires confirmation",
        "is not allowed",
        "must be between",
        "Unknown tool requested",
    )
    unstable_errors = [
        error
        for error in result.state.errors
        if any(marker in error for marker in unstable_markers)
    ]
    if _has_repeated_successful_read_hint(result) and not _meaningful_segment_progress(result):
        return True
    if any("Repeat guard:" in error for error in unstable_errors) and not _meaningful_segment_progress(result):
        return True
    if unstable_errors and _meaningful_segment_progress(result):
        return False
    if len(unstable_errors) >= 2:
        return True
    token_total = result.stats.usage.get("total_tokens")
    if isinstance(token_total, (int, float)) and token_total >= 20_000 and unstable_errors:
        return True
    return False


def _has_repeated_successful_read_hint(result: RuntimeResult) -> bool:
    marker = "the requested source has already been read multiple times"
    return any(marker in message.content for message in result.state.messages)


def _meaningful_segment_progress(result: RuntimeResult) -> bool:
    successful_writes = 0
    for step in result.steps:
        if not step.tool_result or not step.tool_result.success:
            continue
        if step.decision.tool_name in {"write_file", "edit_file", "apply_patch"}:
            successful_writes += 1
    return successful_writes >= 2


def _continuation_task_input(
    original: TaskInput,
    result: RuntimeResult,
    remaining_steps: int,
    remaining_seconds: float,
) -> TaskInput:
    constraints = list(original.constraints)
    constraints.append(
        "Continuation mode: continue from the previous progress summary below. "
        "Do not restart completed work unless necessary."
    )
    constraints.append(_progress_summary(result, remaining_steps, remaining_seconds))
    return TaskInput(
        goal=original.goal,
        user_message=original.user_message,
        workspace_root=original.workspace_root,
        constraints=constraints,
    )


def _progress_summary(result: RuntimeResult, remaining_steps: int, remaining_seconds: float) -> str:
    last_decision = result.state.last_decision.reason if result.state.last_decision else "(none)"
    last_tool = "(none)"
    if result.state.last_tool_result:
        tool_output = result.state.last_tool_result.output or result.state.last_tool_result.error or "(empty)"
        tool_output = tool_output.replace("\n", " ")
        if len(tool_output) > 240:
            tool_output = tool_output[:240] + "..."
        last_tool = tool_output
    errors = "; ".join(result.state.errors[-3:]) if result.state.errors else "(none)"
    return (
        f"Previous segment status={result.task.status.value}; "
        f"steps_used={result.stats.step_count}; remaining_steps={remaining_steps}; "
        f"remaining_seconds={remaining_seconds:.2f}; last_decision={last_decision}; "
        f"last_tool_result={last_tool}; errors={errors}"
    )


def _merge_usage(target: dict, source: dict) -> None:
    for key, value in source.items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value
        else:
            target[key] = value


def _merged_result(result: RuntimeResult, steps: list, usage: dict, duration_seconds: float) -> RuntimeResult:
    merged_stats = RuntimeStats(
        duration_seconds=round(duration_seconds, 6),
        step_count=len(steps),
        tool_call_count=sum(1 for step in steps if step.tool_result is not None),
        error_count=len(result.state.errors),
        usage=usage,
    )
    return RuntimeResult(
        task=result.task,
        state=result.state,
        steps=list(steps),
        stats=merged_stats,
        task_input=result.task_input,
        resume_report=result.resume_report,
    )


def _consumed_step_count(result: RuntimeResult) -> int:
    if result.resume_report and result.resume_report.resumed:
        return max(0, result.resume_report.new_step_count)
    return result.stats.step_count


def _result_step_slice(result: RuntimeResult, aggregate_steps: list) -> list:
    if result.resume_report and result.resume_report.resumed and aggregate_steps:
        return result.steps[result.resume_report.restored_step_count :]
    return list(result.steps)
