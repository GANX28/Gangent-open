import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from gangent.cli import (
    _is_doctor_command,
    _is_budget_command,
    configure_stdio_utf8,
    enqueue_cli_event,
    enqueue_short_event,
    handle_budget_command,
    print_event_summary,
    print_planner_summary,
    provider_doctor_summary,
)
from gangent.events import AgentEventType, JsonlEventQueue
from gangent.models import AgentState, RuntimeStats, Task, TaskInput, TaskStatus, utc_now
from gangent.planner_eval import append_planner_evaluation
from gangent.budget_stats import PlannerQualityReport
from gangent.runtime import RuntimeResult


class CliCommandHelperTests(unittest.TestCase):
    def test_slash_inspection_aliases_are_registered(self):
        from gangent.cli import _is_context_command, _is_events_command, _is_planner_command

        self.assertTrue(_is_planner_command("/planner"))
        self.assertTrue(_is_context_command("/context"))
        self.assertTrue(_is_events_command("/events"))
        self.assertTrue(_is_doctor_command("/doctor"))
        self.assertTrue(_is_budget_command("/budget show"))

    def test_budget_command_switches_current_profile(self):
        output = io.StringIO()

        with redirect_stdout(output):
            profile = handle_budget_command("/budget heavy", "auto")

        self.assertEqual(profile, "heavy")
        self.assertIn("auto -> heavy", output.getvalue())

    def test_budget_command_show_and_invalid_do_not_change_profile(self):
        output = io.StringIO()

        with redirect_stdout(output):
            shown = handle_budget_command("/budget show", "medium")
            unchanged = handle_budget_command("/budget nonsense", "medium")

        self.assertEqual(shown, "medium")
        self.assertEqual(unchanged, "medium")
        self.assertIn("budget_profile=medium", output.getvalue())
        self.assertIn("usage: /budget", output.getvalue())

    def test_enqueue_cli_event_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.jsonl"

            with redirect_stdout(io.StringIO()):
                enqueue_cli_event("/event user_input 75 update docs", path)
            output = io.StringIO()
            with redirect_stdout(output):
                print_event_summary(path)

            self.assertEqual(len(JsonlEventQueue(path).load()), 1)
            self.assertIn("user_input", output.getvalue())

    def test_enqueue_short_replan_event(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.jsonl"

            with redirect_stdout(io.StringIO()):
                enqueue_short_event("/replan change target", AgentEventType.REPLAN_REQUEST, path)

            event = JsonlEventQueue(path).load()[0].event
            self.assertEqual(event.event_type, AgentEventType.REPLAN_REQUEST)
            self.assertEqual(event.priority, 80)

    def test_print_planner_summary_reads_evaluation_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evaluation.jsonl"
            append_planner_evaluation(
                PlannerQualityReport(
                    task_kind="analysis",
                    outcome="success",
                    granularity="balanced",
                    budget_fit="fit",
                    success=True,
                ),
                path,
            )
            output = io.StringIO()

            with redirect_stdout(output):
                print_planner_summary(path)

            self.assertIn("PLANNER", output.getvalue())
            self.assertIn("success_rate=1.00", output.getvalue())

    def test_print_token_usage_appends_compact_summary(self):
        from gangent.cli import _print_token_usage

        task = Task(
            task_id="task_test",
            goal="test",
            status=TaskStatus.COMPLETED,
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        state = AgentState(task_id=task.task_id, workspace_root=".")
        result = RuntimeResult(
            task=task,
            state=state,
            steps=[],
            stats=RuntimeStats(
                duration_seconds=0,
                step_count=0,
                tool_call_count=0,
                error_count=0,
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "prompt_cache_hit_tokens": 4,
                    "prompt_cache_miss_tokens": 6,
                },
            ),
            task_input=TaskInput("test", "test", "."),
        )
        output = io.StringIO()

        with redirect_stdout(output):
            _print_token_usage(result, "quiet")

        self.assertIn("token_usage: prompt=10; completion=2; total=12", output.getvalue())
        self.assertIn("cache_hit=4", output.getvalue())

    def test_provider_doctor_marks_fake_and_deepseek_without_secret(self):
        fake = provider_doctor_summary("fake", None, False, "light", ".")
        deepseek = provider_doctor_summary("deepseek", "deepseek-v4-flash", False, "light", ".")

        self.assertIn("provider_check=local_fake", fake)
        self.assertIn("provider=deepseek", deepseek)
        self.assertNotIn("sk-", deepseek)

    def test_configure_stdio_utf8_is_safe_to_call(self):
        configure_stdio_utf8()


if __name__ == "__main__":
    unittest.main()
