"""CLI output modes（命令行输出模式）。"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pprint import pformat
from typing import Any

from .models import DecisionType
from .runtime import RuntimeResult
from .state import state_summary


def print_result(result: RuntimeResult, mode: str = "verbose") -> None:
    """根据输出模式打印 RuntimeResult。"""

    if mode == "quiet":
        print_quiet_result(result)
        return
    if mode == "json":
        print_json_result(result)
        return
    if mode == "verbose":
        print_verbose_result(result)
        return
    raise ValueError(f"Unknown output mode: {mode}")


def print_quiet_result(result: RuntimeResult) -> None:
    """只打印用户最关心的信息。"""

    if result.resume_report and result.resume_report.summary:
        _safe_print(f"Resume: {result.resume_report.summary}")
    final_answer = final_answer_from_result(result)
    if final_answer:
        _safe_print(final_answer)
    else:
        _safe_print(f"Task status: {result.task.status.value}; steps={len(result.steps)}")

    if result.state.errors and result.task.status.value != "completed":
        _safe_print("Errors:")
        for error in result.state.errors:
            _safe_print(f"- {error}")
    _safe_print()


def print_verbose_result(result: RuntimeResult) -> None:
    """打印完整调试信息。"""

    if result.resume_report:
        _safe_print("RESUME REPORT")
        _safe_pprint(asdict(result.resume_report))
    for step in result.steps:
        _safe_print(f"\nSTEP {step.step_index}")
        _safe_print("DECISION")
        _safe_pprint(asdict(step.decision))
        if step.policy:
            _safe_print("POLICY")
            _safe_pprint(asdict(step.policy))
        if step.tool_result:
            _safe_print("TOOL RESULT")
            _safe_pprint(asdict(step.tool_result))
        if step.usage:
            _safe_print("USAGE")
            _safe_pprint(step.usage)

    final_answer = final_answer_from_result(result)
    if final_answer:
        _safe_print("\nFINAL ANSWER")
        _safe_print(final_answer)

    _safe_print("\nTASK")
    _safe_pprint(asdict(result.task))
    _safe_print("\nSTATE SUMMARY")
    _safe_print(state_summary(result.state))
    _safe_print("\nRUNTIME STATS")
    _safe_pprint(asdict(result.stats))
    if result.state.errors:
        _safe_print("\nERRORS")
        for error in result.state.errors:
            _safe_print(f"- {error}")
    _safe_print()


def print_json_result(result: RuntimeResult) -> None:
    """打印机器可读 JSON。"""

    _safe_print(json.dumps(result_to_dict(result), ensure_ascii=False, indent=2))


def result_to_dict(result: RuntimeResult) -> dict[str, Any]:
    """把 RuntimeResult 转成可 JSON 序列化的 dict。"""

    return asdict(result)


def final_answer_from_result(result: RuntimeResult) -> str | None:
    """从最终决策里提取给用户看的答案。"""

    decision = result.state.last_decision
    if not decision:
        return None
    if decision.decision_type in {DecisionType.FINISH, DecisionType.DIRECT_RESPONSE}:
        return decision.response_text
    return None


def _safe_pprint(value: Any) -> None:
    """Pretty-print data without crashing on Windows console encoding limits."""

    _safe_print(pformat(value))


def _safe_print(text: str = "") -> None:
    """Print text after coercing unsupported console characters to escapes."""

    print(_console_safe_text(text))


def _console_safe_text(text: str) -> str:
    """Return a console-safe representation for the current stdout encoding."""

    if not isinstance(text, str):
        text = str(text)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return text
    except UnicodeEncodeError:
        return text.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")
