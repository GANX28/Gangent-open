import tempfile
import unittest
from pathlib import Path

from gangent.audit import append_audit_record
from gangent.eval import run_policy_eval, run_tool_eval
from gangent.llm_client import FakeLLMClient
from gangent.metrics import classification_metrics, summarize_audit_jsonl
from gangent.models import ActionDecision, DecisionType, TaskInput
from gangent.runtime import run_task


class MetricsAndEvalTests(unittest.TestCase):
    def test_classification_metrics(self):
        metrics = classification_metrics(
            expected=["allow", "block", "block", "allow"],
            predicted=["allow", "block", "allow", "block"],
            positive_labels={"block"},
        )

        self.assertEqual(metrics.total, 4)
        self.assertEqual(metrics.correct, 2)
        self.assertEqual(metrics.true_positive, 1)
        self.assertEqual(metrics.false_positive, 1)
        self.assertEqual(metrics.false_negative, 1)
        self.assertEqual(metrics.true_negative, 1)
        self.assertEqual(metrics.precision, 0.5)
        self.assertEqual(metrics.recall, 0.5)

    def test_policy_eval_runs_default_cases(self):
        result = run_policy_eval()

        self.assertEqual(result["kind"], "policy")
        self.assertEqual(result["metrics"]["total"], 16)
        self.assertEqual(result["metrics"]["accuracy"], 1.0)

    def test_tool_eval_runs_default_cases(self):
        result = run_tool_eval()

        self.assertEqual(result["kind"], "tool")
        self.assertEqual(result["metrics"]["total"], 12)
        self.assertEqual(result["metrics"]["accuracy"], 1.0)

    def test_summarize_audit_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_task(
                TaskInput(
                    goal="Answer",
                    user_message="Answer directly.",
                    workspace_root=temp_dir,
                ),
                FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.DIRECT_RESPONSE,
                        reason="Answer directly.",
                        response_text="Done.",
                    )
                ),
                max_steps=1,
            )
            path = Path(temp_dir) / "audit.jsonl"
            append_audit_record(result, "session_test", "Answer directly.", path)

            summary = summarize_audit_jsonl(path)

            self.assertEqual(summary["total_tasks"], 1)
            self.assertEqual(summary["total_steps"], 1)
            self.assertEqual(summary["total_errors"], 0)


if __name__ == "__main__":
    unittest.main()
