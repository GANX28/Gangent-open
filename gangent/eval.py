"""Local Evaluation Harness（本地评估框架）。

第一版评估两件事：
1. Policy Check 是否按预期 allow/block/escalate。
2. Tool Runtime 是否按预期 success/failure。
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .metrics import classification_metrics, summarize_audit_jsonl
from .models import ActionDecision, DecisionType
from .policy import check_policy
from .tool_runtime import execute_tool_call


DEFAULT_EVAL_DIR = Path(__file__).resolve().parent.parent / "evals"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 测试集。"""

    source = Path(path)
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_policy_eval(cases_path: str | Path | None = None) -> dict[str, Any]:
    """运行 policy 测试集并返回指标。"""

    path = Path(cases_path) if cases_path else DEFAULT_EVAL_DIR / "policy_cases.jsonl"
    cases = load_jsonl(path)
    expected: list[str] = []
    predicted: list[str] = []
    rows: list[dict[str, Any]] = []

    with _eval_workspace() as workspace_root:
        for case in cases:
            decision = _decision_from_case(case)
            policy = check_policy(decision, workspace_root=str(workspace_root))
            expected_mode = case["expected_mode"]
            predicted_mode = policy.mode.value
            expected.append(expected_mode)
            predicted.append(predicted_mode)
            rows.append(
                {
                    "name": case["name"],
                    "expected": expected_mode,
                    "predicted": predicted_mode,
                    "pass": expected_mode == predicted_mode,
                    "reason": policy.reason,
                }
            )

    metrics = classification_metrics(
        expected,
        predicted,
        positive_labels={"block", "escalate"},
    )
    return {"kind": "policy", "metrics": metrics.to_dict(), "cases": rows}


def run_tool_eval(cases_path: str | Path | None = None) -> dict[str, Any]:
    """运行 tool runtime 测试集并返回指标。"""

    path = Path(cases_path) if cases_path else DEFAULT_EVAL_DIR / "tool_cases.jsonl"
    cases = load_jsonl(path)
    expected: list[str] = []
    predicted: list[str] = []
    rows: list[dict[str, Any]] = []

    with _eval_workspace() as workspace_root:
        for case in cases:
            decision = _decision_from_case(case)
            result = execute_tool_call(decision, workspace_root=str(workspace_root))
            expected_label = "success" if case["expected_success"] else "failure"
            predicted_label = "success" if result.success else "failure"
            expected.append(expected_label)
            predicted.append(predicted_label)
            rows.append(
                {
                    "name": case["name"],
                    "expected": expected_label,
                    "predicted": predicted_label,
                    "pass": expected_label == predicted_label,
                    "result": asdict(result),
                }
            )

    metrics = classification_metrics(
        expected,
        predicted,
        positive_labels={"success"},
    )
    return {"kind": "tool", "metrics": metrics.to_dict(), "cases": rows}


def _decision_from_case(case: dict[str, Any]) -> ActionDecision:
    return ActionDecision(
        decision_type=DecisionType.TOOL_CALL,
        reason=case.get("reason", "eval case"),
        tool_name=case["tool_name"],
        tool_args=case.get("tool_args", {}),
    )


class _eval_workspace:
    """创建一个临时 workspace，保证评估可重复。"""

    def __enter__(self) -> Path:
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        (root / "README.md").write_text("hello from eval workspace", encoding="utf-8")
        (root / "large.txt").write_text("x" * 20_001, encoding="utf-8")
        (root / ".hidden").write_text("hidden", encoding="utf-8")
        (root / "src").mkdir()
        (root / "src" / "demo.py").write_text("VALUE = 1\n", encoding="utf-8")
        (root / "src" / "runtime_notes.py").write_text(
            "def runtime_policy_loop():\n"
            "    return 'runtime policy tool execution'\n",
            encoding="utf-8",
        )
        (root / "tests").mkdir()
        (root / "tests" / "test_ok.py").write_text(
            "import unittest\n\n"
            "class OkTest(unittest.TestCase):\n"
            "    def test_ok(self):\n"
            "        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        return root

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._temp.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Gangent evals.")
    parser.add_argument(
        "--kind",
        choices=["policy", "tool", "all", "audit"],
        default="all",
        help="Which eval to run.",
    )
    parser.add_argument("--policy-cases", help="Custom policy cases JSONL path.")
    parser.add_argument("--tool-cases", help="Custom tool cases JSONL path.")
    parser.add_argument("--audit-log", help="Audit JSONL path to summarize.")
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    if args.kind in {"policy", "all"}:
        results.append(run_policy_eval(args.policy_cases))
    if args.kind in {"tool", "all"}:
        results.append(run_tool_eval(args.tool_cases))
    if args.kind == "audit":
        if not args.audit_log:
            raise SystemExit("--audit-log is required for audit summary.")
        results.append({"kind": "audit", "metrics": summarize_audit_jsonl(args.audit_log)})

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
