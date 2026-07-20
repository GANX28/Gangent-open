"""Budget history and percentile recommendation.

This module records completed task resource usage and recommends future
budgets from historical percentiles. It does not train a model; it provides the
data layer needed before any learned budget policy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .models import AgentState, PlanStepStatus, RuntimeStats, TaskInput, TaskStatus, utc_now
from .planner import infer_task_kind


DEFAULT_BUDGET_HISTORY = Path(".gangent") / "budget" / "history.json"
WRITE_KEYWORDS = {
    "write",
    "save",
    "create",
    "edit",
    "patch",
    "document",
    "file",
    "生成",
    "写",
    "写入",
    "保存",
    "文档",
    "文件",
    "修改",
}


@dataclass(frozen=True)
class BudgetSample:
    """One completed task's resource usage."""

    task_kind: str
    success: bool
    status: str
    step_count: int
    tool_call_count: int
    duration_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    prompt_cache_hit_ratio: float = 0.0
    budget_profile: str = ""
    planned_step_count: int = 0
    completed_plan_step_count: int = 0
    blocked_plan_step_count: int = 0
    runtime_step_limit: int = 0
    total_step_budget: int = 0
    total_remaining_steps: int = 0
    avg_tokens_per_step: int = 0
    avg_tokens_per_tool_call: int = 0
    failure_reason: str | None = None
    created_at: str = ""


@dataclass(frozen=True)
class BudgetRecommendation:
    """Percentile-based budget recommendation for one task kind."""

    task_kind: str
    sample_count: int
    steps_p50: int
    steps_p80: int
    steps_p95: int
    seconds_p50: float
    seconds_p80: float
    seconds_p95: float
    tokens_p50: int
    tokens_p80: int
    tokens_p95: int
    planned_steps_p80: int = 0
    completed_plan_steps_p80: int = 0
    tokens_per_step_p80: int = 0
    cache_hit_ratio_p80: float = 0.0


@dataclass(frozen=True)
class PlannerQualityReport:
    """Deterministic planner quality assessment for one completed run."""

    task_kind: str
    outcome: str
    granularity: str
    budget_fit: str
    success: bool
    token_fit: str = "fit"
    findings: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()


def default_budget_history_path(workspace_root: str) -> Path:
    """Return the default budget history path for one workspace."""

    return Path(workspace_root).resolve() / DEFAULT_BUDGET_HISTORY


def classify_budget_task(task_input: TaskInput) -> str:
    """Classify a task for budget statistics."""

    task_kind = infer_task_kind(task_input)
    text = f"{task_input.goal}\n{task_input.user_message}".lower()
    if any(keyword in text for keyword in WRITE_KEYWORDS):
        return f"{task_kind}:write"
    return task_kind


def sample_from_result(
    task_input: TaskInput,
    status: TaskStatus,
    stats: RuntimeStats,
    errors: list[str],
    state: AgentState | None = None,
) -> BudgetSample:
    """Build one budget sample from a runtime result."""

    usage = stats.usage or {}
    prompt_tokens = _int_usage(usage, "prompt_tokens")
    completion_tokens = _int_usage(usage, "completion_tokens")
    total_tokens = _int_usage(usage, "total_tokens") or prompt_tokens + completion_tokens
    cache_hit_tokens = _int_usage(usage, "prompt_cache_hit_tokens")
    cache_miss_tokens = _int_usage(usage, "prompt_cache_miss_tokens")
    planned_steps = len(state.plan_steps) if state else 0
    completed_plan_steps = (
        sum(1 for step in state.plan_steps if step.status == PlanStepStatus.DONE)
        if state
        else 0
    )
    blocked_plan_steps = (
        sum(1 for step in state.plan_steps if step.status == PlanStepStatus.BLOCKED)
        if state
        else 0
    )
    return BudgetSample(
        task_kind=classify_budget_task(task_input),
        success=status == TaskStatus.COMPLETED,
        status=status.value,
        step_count=max(0, int(stats.step_count)),
        tool_call_count=max(0, int(stats.tool_call_count)),
        duration_seconds=max(0.0, float(stats.duration_seconds)),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_cache_hit_tokens=cache_hit_tokens,
        prompt_cache_miss_tokens=cache_miss_tokens,
        prompt_cache_hit_ratio=_cache_hit_ratio(cache_hit_tokens, cache_miss_tokens),
        budget_profile=state.budget_profile if state else "",
        planned_step_count=planned_steps,
        completed_plan_step_count=completed_plan_steps,
        blocked_plan_step_count=blocked_plan_steps,
        runtime_step_limit=state.runtime_step_limit if state else 0,
        total_step_budget=state.total_step_budget if state else 0,
        total_remaining_steps=state.total_remaining_steps if state else 0,
        avg_tokens_per_step=_safe_average(total_tokens, stats.step_count),
        avg_tokens_per_tool_call=_safe_average(total_tokens, stats.tool_call_count),
        failure_reason=errors[-1] if errors else None,
        created_at=utc_now(),
    )


def append_budget_sample(sample: BudgetSample, path: str | Path) -> Path:
    """Append a budget sample to the JSON history file."""

    target = Path(path)
    history = load_budget_history(target)
    history.append(sample)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([asdict(item) for item in history], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def load_budget_history(path: str | Path) -> list[BudgetSample]:
    """Load budget samples. Missing files return an empty history."""

    source = Path(path)
    if not source.exists():
        return []
    data = json.loads(source.read_text(encoding="utf-8"))
    return [_budget_sample_from_dict(item) for item in data]


def recommend_budget(task_input: TaskInput, path: str | Path, min_samples: int = 3) -> BudgetRecommendation | None:
    """Recommend a budget from historical successful samples."""

    task_kind = classify_budget_task(task_input)
    samples = [
        sample
        for sample in load_budget_history(path)
        if sample.task_kind == task_kind and sample.success
    ]
    if len(samples) < min_samples:
        return None
    return BudgetRecommendation(
        task_kind=task_kind,
        sample_count=len(samples),
        steps_p50=max(1, int(_percentile([sample.step_count for sample in samples], 50))),
        steps_p80=max(1, int(_percentile([sample.step_count for sample in samples], 80))),
        steps_p95=max(1, int(_percentile([sample.step_count for sample in samples], 95))),
        seconds_p50=round(_percentile([sample.duration_seconds for sample in samples], 50), 3),
        seconds_p80=round(_percentile([sample.duration_seconds for sample in samples], 80), 3),
        seconds_p95=round(_percentile([sample.duration_seconds for sample in samples], 95), 3),
        tokens_p50=max(0, int(_percentile([sample.total_tokens for sample in samples], 50))),
        tokens_p80=max(0, int(_percentile([sample.total_tokens for sample in samples], 80))),
        tokens_p95=max(0, int(_percentile([sample.total_tokens for sample in samples], 95))),
        planned_steps_p80=max(0, int(_percentile([sample.planned_step_count for sample in samples], 80))),
        completed_plan_steps_p80=max(0, int(_percentile([sample.completed_plan_step_count for sample in samples], 80))),
        tokens_per_step_p80=max(0, int(_percentile([sample.avg_tokens_per_step for sample in samples], 80))),
        cache_hit_ratio_p80=round(_percentile([sample.prompt_cache_hit_ratio for sample in samples], 80), 4),
    )


def recommendation_to_dict(recommendation: BudgetRecommendation | None) -> dict[str, Any] | None:
    """Convert recommendation to JSON-friendly data."""

    return asdict(recommendation) if recommendation else None


def evaluate_planner_quality(sample: BudgetSample) -> PlannerQualityReport:
    """Assess whether one planner run was too coarse, too fine, or budget-mismatched."""

    findings: list[str] = []
    recommendations: list[str] = []
    planned = max(0, sample.planned_step_count)
    completed = max(0, sample.completed_plan_step_count)
    runtime_limit = max(0, sample.runtime_step_limit)

    if sample.success:
        outcome = "success"
    elif sample.status == TaskStatus.WAITING_USER.value:
        outcome = "waiting_user"
    else:
        outcome = "failed"

    if planned <= 3 and (sample.tool_call_count >= 5 or sample.step_count >= 6):
        granularity = "too_coarse"
        findings.append("plan_had_few_steps_but_runtime_needed_many_actions")
        recommendations.append("split broad phases into smaller inspect / change / verify steps")
    elif planned >= 12 and completed <= max(2, planned // 3):
        granularity = "too_fine"
        findings.append("plan_had_many_steps_but_only_a_small_fraction_completed")
        recommendations.append("merge low-risk adjacent steps and keep only auditable boundaries")
    elif sample.blocked_plan_step_count:
        granularity = "blocked"
        findings.append("one_or_more_plan_steps_blocked")
        recommendations.append("make blocked phases narrower and add clearer exit criteria")
    else:
        granularity = "balanced"

    budget_pressure = runtime_limit > 0 and sample.step_count >= max(1, int(runtime_limit * 0.8))
    no_remaining_total = sample.total_step_budget > 0 and sample.total_remaining_steps <= 0
    if budget_pressure or no_remaining_total:
        budget_fit = "tight"
        findings.append("runtime_used_most_available_step_budget")
        recommendations.append("raise profile or reduce initial plan breadth for similar tasks")
    elif sample.success and sample.total_remaining_steps > sample.total_step_budget * 0.5 > 0:
        budget_fit = "loose"
        findings.append("task_finished_with_large_unused_budget")
        recommendations.append("try a smaller profile for similar tasks")
    else:
        budget_fit = "fit"

    if sample.avg_tokens_per_step > 2_000:
        token_fit = "high_context_cost"
        findings.append("high_tokens_per_step")
        recommendations.append("load narrower context and avoid repeated broad repo summaries")
    elif sample.total_tokens == 0:
        token_fit = "unknown"
    else:
        token_fit = "fit"
    if sample.failure_reason:
        findings.append("failure_reason_present")

    return PlannerQualityReport(
        task_kind=sample.task_kind,
        outcome=outcome,
        granularity=granularity,
        budget_fit=budget_fit,
        token_fit=token_fit,
        success=sample.success,
        findings=tuple(findings),
        recommendations=tuple(dict.fromkeys(recommendations)),
    )


def planner_feedback_for_task(task_input: TaskInput, path: str | Path, min_samples: int = 2) -> str:
    """Build compact model-facing planner feedback from similar historical runs."""

    task_kind = classify_budget_task(task_input)
    samples = [sample for sample in load_budget_history(path) if sample.task_kind == task_kind]
    if len(samples) < min_samples:
        return ""

    recent = samples[-12:]
    successes = [sample for sample in recent if sample.success]
    reports = [evaluate_planner_quality(sample) for sample in recent]
    success_rate = len(successes) / len(recent)
    recommendation = recommend_budget(task_input, path, min_samples=min_samples)
    common_findings = _top_counts(finding for report in reports for finding in report.findings)
    common_recommendations = _top_counts(
        recommendation
        for report in reports
        for recommendation in report.recommendations
    )

    lines = [
        "Planner History Feedback:",
        f"- task_kind: {task_kind}",
        f"- sample_count: {len(recent)}",
        f"- success_rate: {success_rate:.2f}",
    ]
    if recommendation is not None:
        lines.extend(
            [
                f"- successful_steps_p80: {recommendation.steps_p80}",
                f"- successful_tokens_p80: {recommendation.tokens_p80}",
                f"- successful_planned_steps_p80: {recommendation.planned_steps_p80}",
            ]
        )
    if common_findings:
        lines.append("- common_findings: " + ", ".join(common_findings))
    if common_recommendations:
        lines.append("- planner_guidance: " + "; ".join(common_recommendations))
    lines.append("- use history as a constraint, not as permission to skip verification")
    return "\n".join(lines)


def _percentile(values: list[int] | list[float], percentile: int) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percentile / 100)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _int_usage(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _safe_average(total: int, count: int) -> int:
    return int(total / count) if count > 0 else 0


def _cache_hit_ratio(hit_tokens: int, miss_tokens: int) -> float:
    total = hit_tokens + miss_tokens
    return round(hit_tokens / total, 4) if total > 0 else 0.0


def _budget_sample_from_dict(data: dict[str, Any]) -> BudgetSample:
    allowed = BudgetSample.__dataclass_fields__
    return BudgetSample(**{key: value for key, value in data.items() if key in allowed})


def _top_counts(values: Iterable[str], limit: int = 3) -> list[str]:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [f"{value}({count})" for value, count in ordered[:limit]]
