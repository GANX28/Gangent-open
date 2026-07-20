import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gangent.audit import append_audit_record
from gangent.llm_client import FakeLLMClient
from gangent.models import ActionDecision, DecisionType, TaskInput
from gangent.output import _console_safe_text, final_answer_from_result, print_quiet_result, result_to_dict
from gangent.runtime import run_task


class OutputAndAuditTests(unittest.TestCase):
    def test_result_to_dict_contains_stats(self):
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

            data = result_to_dict(result)

            self.assertEqual(final_answer_from_result(result), "Done.")
            self.assertIn("stats", data)
            self.assertEqual(data["stats"]["step_count"], 1)

    def test_append_audit_record_writes_jsonl(self):
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

            append_audit_record(
                result,
                session_id="session_test",
                user_message="Answer directly.",
                path=path,
            )

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["session_id"], "session_test")
            self.assertEqual(record["user_message"], "Answer directly.")

    def test_audit_record_redacts_user_message(self):
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

            append_audit_record(
                result,
                session_id="session_test",
                user_message="token=abcdefgh12345678",
                path=path,
            )

            record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("[REDACTED_", record["user_message"])

    def test_console_safe_text_handles_non_gbk_characters(self):
        value = _console_safe_text("\u4e2d\u6587 + emoji \U0001f4cc")

        self.assertIn("emoji", value)
        self.assertTrue("\\U0001f4cc" in value or "\U0001f4cc" in value)


    def test_quiet_output_hides_internal_errors_for_completed_task(self):
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
            result.state.errors.append("Internal guard warning kept for audit.")
            buffer = io.StringIO()

            with patch("sys.stdout", buffer):
                print_quiet_result(result)

            output = buffer.getvalue()
            self.assertIn("Done.", output)
            self.assertNotIn("Errors:", output)
            self.assertNotIn("Internal guard warning", output)

if __name__ == "__main__":
    unittest.main()

