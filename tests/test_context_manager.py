import tempfile
import unittest
from pathlib import Path
import subprocess

from gangent.context_manager import (
    ContextSegment,
    analyze_context_segments,
    build_context_bundle,
    build_context_segments,
    build_dynamic_context_pack,
)
from gangent.models import ActionDecision, DecisionType
from gangent.models import TaskInput
from gangent.planner import attach_plan, create_initial_plan, start_current_plan_step
from gangent.state import add_error, create_initial_state, create_task


class ContextManagerTests(unittest.TestCase):
    def test_context_bundle_contains_plan_repo_map_and_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("hello", encoding="utf-8")
            (root / ".gangent").mkdir()
            (root / ".gangent" / "secret.txt").write_text("hidden", encoding="utf-8")
            task_input = TaskInput(
                goal="Inspect project",
                user_message="Inspect project.",
                workspace_root=temp_dir,
                constraints=["Session context: previous work"],
            )
            task = create_task(task_input)
            state = create_initial_state(task, task_input)
            attach_plan(state, create_initial_plan(task, task_input))
            start_current_plan_step(state)
            add_error(state, "previous tool failed")

            bundle = build_context_bundle(task, state)

            self.assertIn("Task goal: Inspect project", bundle.text)
            self.assertIn("Planner Control", bundle.text)
            self.assertIn("segment_remaining_steps", bundle.text)
            self.assertIn("Current Plan", bundle.text)
            self.assertIn("[file] README.md", bundle.text)
            self.assertIn("previous tool failed", bundle.text)
            self.assertNotIn(".gangent", bundle.text)

    def test_context_bundle_includes_git_summary_and_focused_file_only_when_relevant(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init"], cwd=temp_dir, capture_output=True, text=True, check=True)
            (root / "note.txt").write_text("hello world", encoding="utf-8")
            task_input = TaskInput(
                goal="Inspect git repository status",
                user_message="Inspect git repository status.",
                workspace_root=temp_dir,
            )
            task = create_task(task_input)
            state = create_initial_state(task, task_input)
            attach_plan(state, create_initial_plan(task, task_input))
            start_current_plan_step(state)
            state.last_decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Read file.",
                tool_name="read_file",
                tool_args={"path": "note.txt"},
            )

            bundle = build_context_bundle(task, state)

            self.assertIn("Git Summary", bundle.text)
            self.assertIn("Focused Files", bundle.text)
            self.assertIn("note.txt", bundle.text)
            self.assertIn("hello world", bundle.text)

    def test_context_bundle_skips_git_summary_when_task_is_not_git_related(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(["git", "init"], cwd=temp_dir, capture_output=True, text=True, check=True)
            (root / "note.txt").write_text("hello world", encoding="utf-8")
            task_input = TaskInput(
                goal="Explain the note file",
                user_message="Explain the note file.",
                workspace_root=temp_dir,
            )
            task = create_task(task_input)
            state = create_initial_state(task, task_input)
            attach_plan(state, create_initial_plan(task, task_input))
            start_current_plan_step(state)

            bundle = build_context_bundle(task, state)

            self.assertNotIn("Git Summary", bundle.text)

    def test_context_segments_include_metadata_and_pollution_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_input = TaskInput(
                goal="Inspect project structure",
                user_message="Inspect project structure.",
                workspace_root=temp_dir,
            )
            task = create_task(task_input)
            state = create_initial_state(task, task_input)
            attach_plan(state, create_initial_plan(task, task_input))

            segments = build_context_segments(task, state)
            report = analyze_context_segments(segments, max_chars=200)
            bundle = build_context_bundle(task, state, max_chars=4000)

            self.assertTrue(any(segment.title == "Task" for segment in segments))
            self.assertGreaterEqual(report.total_segments, 3)
            self.assertIn("source=runtime", bundle.text)
            self.assertIn("Context Pollution Report", bundle.text)

    def test_dynamic_context_pack_keeps_must_include_and_omits_low_priority(self):
        segments = [
            ContextSegment("Task", "task", source="runtime", priority=100),
            ContextSegment("Plan", "plan", source="planner", priority=90),
            ContextSegment("Large Low Priority", "x" * 1000, source="repo", priority=10),
            ContextSegment("Recent Errors", "error", source="runtime_errors", priority=85),
        ]

        pack = build_dynamic_context_pack(segments, max_chars=450)

        self.assertEqual([segment.title for segment in pack.must_include], ["Task", "Plan"])
        self.assertEqual([segment.title for segment in pack.warnings], ["Recent Errors"])
        self.assertIn("Large Low Priority", pack.excluded)


if __name__ == "__main__":
    unittest.main()
