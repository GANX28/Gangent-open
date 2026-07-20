"""Metrics（指标计算）。

这里放本地可运行的最小指标函数。
第一版重点支持 policy/eval 里的 accuracy、precision、recall、F1，
以及从 audit JSONL 里汇总 runtime 基础指标。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ClassificationMetrics:
    """二分类指标。

    positive 表示我们关心的正类。对 policy 来说，正类通常是“不允许执行”，
    也就是 block 或 escalate。
    """

    total: int
    correct: int
    accuracy: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classification_metrics(
    expected: Iterable[str],
    predicted: Iterable[str],
    positive_labels: set[str],
) -> ClassificationMetrics:
    """计算 accuracy / precision / recall / F1。

    expected 是标准答案，predicted 是系统输出。
    positive_labels 定义哪些标签算“正类”。
    """

    expected_list = list(expected)
    predicted_list = list(predicted)
    if len(expected_list) != len(predicted_list):
        raise ValueError("expected and predicted must have the same length.")

    true_positive = false_positive = false_negative = true_negative = 0
    correct = 0

    for expected_label, predicted_label in zip(expected_list, predicted_list):
        expected_positive = expected_label in positive_labels
        predicted_positive = predicted_label in positive_labels
        if expected_label == predicted_label:
            correct += 1
        if expected_positive and predicted_positive:
            true_positive += 1
        elif not expected_positive and predicted_positive:
            false_positive += 1
        elif expected_positive and not predicted_positive:
            false_negative += 1
        else:
            true_negative += 1

    total = len(expected_list)
    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return ClassificationMetrics(
        total=total,
        correct=correct,
        accuracy=_safe_div(correct, total),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def summarize_audit_jsonl(path: str | Path) -> dict[str, Any]:
    """从本地 audit JSONL 汇总 runtime 指标。"""

    source = Path(path)
    records = []
    if source.exists():
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

    total_tasks = len(records)
    total_steps = sum(record.get("stats", {}).get("step_count", 0) for record in records)
    total_tool_calls = sum(
        record.get("stats", {}).get("tool_call_count", 0) for record in records
    )
    total_errors = sum(record.get("stats", {}).get("error_count", 0) for record in records)
    total_duration = sum(
        record.get("stats", {}).get("duration_seconds", 0.0) for record in records
    )

    return {
        "total_tasks": total_tasks,
        "total_steps": total_steps,
        "total_tool_calls": total_tool_calls,
        "total_errors": total_errors,
        "average_steps": _safe_div(total_steps, total_tasks),
        "average_tool_calls": _safe_div(total_tool_calls, total_tasks),
        "average_duration_seconds": _safe_div(total_duration, total_tasks),
        "error_rate": _safe_div(total_errors, total_tasks),
    }


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)
