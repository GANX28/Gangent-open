"""Planner evaluation persistence and summaries."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .budget_stats import PlannerQualityReport, evaluate_planner_quality, sample_from_result
from .models import RuntimeStats, TaskInput, TaskStatus


DEFAULT_PLANNER_EVAL_LOG = Path(".gangent") / "planner" / "evaluation.jsonl"


def default_planner_eval_path(workspace_root: str) -> Path:
    return Path(workspace_root).resolve() / DEFAULT_PLANNER_EVAL_LOG


def append_planner_evaluation(report: PlannerQualityReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True) + "\n")
    return target


def load_planner_evaluations(path: str | Path) -> list[PlannerQualityReport]:
    target = Path(path)
    if not target.exists():
        return []
    reports: list[PlannerQualityReport] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            reports.append(_report_from_dict(json.loads(line)))
        except Exception:
            continue
    return reports


def planner_evaluation_from_result(
    task_input: TaskInput,
    status: TaskStatus,
    stats: RuntimeStats,
    errors: list[str],
    state,
) -> PlannerQualityReport:
    sample = sample_from_result(task_input, status, stats, errors, state)
    return evaluate_planner_quality(sample)


def summarize_planner_evaluations(path: str | Path, limit: int = 12) -> str:
    reports = load_planner_evaluations(path)
    if not reports:
        return "status=(none)"
    recent = reports[-max(1, limit):]
    success_count = sum(1 for report in recent if report.success)
    granularity_counts = _counts(report.granularity for report in recent)
    budget_counts = _counts(report.budget_fit for report in recent)
    token_counts = _counts(report.token_fit for report in recent)
    finding_counts = _counts(finding for report in recent for finding in report.findings)
    lines = [
        f"records={len(reports)}",
        f"recent={len(recent)}",
        f"success_rate={success_count / len(recent):.2f}",
        "granularity=" + _format_counts(granularity_counts),
        "budget_fit=" + _format_counts(budget_counts),
        "token_fit=" + _format_counts(token_counts),
    ]
    if finding_counts:
        lines.append("top_findings=" + _format_counts(finding_counts, limit=5))
    last = recent[-1]
    lines.append(
        f"last=task_kind={last.task_kind}; outcome={last.outcome}; "
        f"granularity={last.granularity}; budget_fit={last.budget_fit}; token_fit={last.token_fit}"
    )
    return "\n".join(lines)


def _report_from_dict(data: dict) -> PlannerQualityReport:
    return PlannerQualityReport(
        task_kind=str(data.get("task_kind", "")),
        outcome=str(data.get("outcome", "")),
        granularity=str(data.get("granularity", "")),
        budget_fit=str(data.get("budget_fit", "")),
        success=bool(data.get("success", False)),
        token_fit=str(data.get("token_fit", "fit")),
        findings=tuple(data.get("findings", [])),
        recommendations=tuple(data.get("recommendations", [])),
    )


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _format_counts(counts: dict[str, int], limit: int = 4) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{key}:{value}" for key, value in list(counts.items())[:limit])
