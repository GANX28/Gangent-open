import tempfile
import unittest
from pathlib import Path

from gangent.checkpoint import (
    archive_checkpoint,
    checkpoint_archive_dir_for_active_path,
    checkpoint_from_runtime_result,
    checkpoint_matches_task_input,
    default_checkpoint_archive_dir,
    default_checkpoint_path,
    is_resume_candidate,
    list_resume_candidates,
    load_checkpoint,
    save_checkpoint,
)
from gangent.llm_client import FakeLLMClient
from gangent.models import ActionDecision, DecisionType, TaskInput
from gangent.runtime import run_task


class SequencedClient:
    def __init__(self, decisions):
        self._decisions = iter(decisions)
        self.last_usage = {}

    def decide(self, model_input):
        return next(self._decisions)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_roundtrip_preserves_task_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("hello", encoding="utf-8")
            task_input = TaskInput(
                goal="Inspect workspace",
                user_message="Inspect files.",
                workspace_root=temp_dir,
            )
            client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.DIRECT_RESPONSE,
                    reason="Done.",
                    response_text="Finished.",
                )
            )

            result = run_task(task_input, client, max_steps=1)
            result.state.event_cursor = 3
            result.state.event_summaries.append("event consumed")
            result.state.event_count = 4
            result.state.replan_count = 2
            result.state.interrupt_count = 1
            result.state.pending_event_count = 0
            result.state.stabilization_required = True
            result.state.stale_outputs.append("out/report.md")
            result.state.plan_patch_summaries.append("plan_patch=replace_pending_steps")
            result.state.budget_profile = "light"
            result.state.runtime_step_limit = 10
            result.state.runtime_remaining_steps = 9
            result.state.total_step_budget = 20
            result.state.total_remaining_steps = 19
            checkpoint_path = root / "checkpoint.json"
            save_checkpoint(load_checkpoint_result(result), checkpoint_path)

            restored = load_checkpoint(checkpoint_path)

            self.assertEqual(restored.task_input.goal, task_input.goal)
            self.assertEqual(restored.task_input.user_message, task_input.user_message)
            self.assertEqual(restored.task.task_id, result.task.task_id)
            self.assertEqual(restored.state.task_id, result.state.task_id)
            self.assertEqual(restored.state.step_index, result.state.step_index)
            self.assertEqual(restored.state.event_cursor, 3)
            self.assertEqual(restored.state.event_summaries, ["event consumed"])
            self.assertEqual(restored.state.event_count, 4)
            self.assertEqual(restored.state.replan_count, 2)
            self.assertEqual(restored.state.interrupt_count, 1)
            self.assertEqual(restored.state.pending_event_count, 0)
            self.assertTrue(restored.state.stabilization_required)
            self.assertEqual(restored.state.stale_outputs, ["out/report.md"])
            self.assertEqual(restored.state.plan_patch_summaries, ["plan_patch=replace_pending_steps"])
            self.assertEqual(restored.state.budget_profile, "light")
            self.assertEqual(restored.state.runtime_step_limit, 10)
            self.assertEqual(restored.state.runtime_remaining_steps, 9)
            self.assertEqual(restored.state.total_step_budget, 20)
            self.assertEqual(restored.state.total_remaining_steps, 19)
            self.assertEqual(restored.task.status.value, result.task.status.value)

    def test_resume_from_checkpoint_continues_without_losing_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("hello", encoding="utf-8")
            task_input = TaskInput(
                goal="Inspect workspace",
                user_message="Inspect files.",
                workspace_root=temp_dir,
            )

            first_client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="List files.",
                    tool_name="list_files",
                    tool_args={"path": "."},
                )
            )
            first_result = run_task(
                task_input,
                first_client,
                max_steps=1,
                checkpoint_path=str(root / "checkpoint.json"),
            )
            self.assertEqual(first_result.task.status.value, "failed")
            self.assertEqual(len(first_result.steps), 1)

            restored = load_checkpoint(root / "checkpoint.json")
            second_client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.FINISH,
                    reason="Enough.",
                    response_text="All done.",
                )
            )
            resumed = run_task(
                task_input,
                second_client,
                max_steps=3,
                resume_checkpoint=restored,
            )

            self.assertEqual(resumed.task.status.value, "completed")
            self.assertGreaterEqual(len(resumed.steps), 2)
            self.assertEqual(resumed.steps[0].decision.tool_name, "list_files")
            self.assertEqual(resumed.steps[-1].decision.decision_type, DecisionType.FINISH)
            self.assertTrue(resumed.resume_report.resumed)
            self.assertEqual(resumed.resume_report.restored_step_count, 1)

    def test_resume_reuses_successful_edit_side_effect(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "note.txt").write_text("hello world", encoding="utf-8")
            task_input = TaskInput(
                goal="Edit note.txt then write done.md.",
                user_message="Edit note.txt then write done.md.",
                workspace_root=temp_dir,
            )
            checkpoint_path = root / "checkpoint.json"

            first_result = run_task(
                task_input,
                FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Edit the file.",
                        tool_name="edit_file",
                        tool_args={"path": "note.txt", "old_text": "hello", "new_text": "hi"},
                    )
                ),
                max_steps=1,
                checkpoint_path=str(checkpoint_path),
            )
            self.assertEqual((root / "note.txt").read_text(encoding="utf-8"), "hi world")
            self.assertEqual(first_result.task.status.value, "failed")

            restored = load_checkpoint(checkpoint_path)
            resumed = run_task(
                task_input,
                SequencedClient(
                    [
                        ActionDecision(
                            decision_type=DecisionType.TOOL_CALL,
                            reason="Edit the file.",
                            tool_name="edit_file",
                            tool_args={"path": "note.txt", "old_text": "hello", "new_text": "hi"},
                        ),
                        ActionDecision(
                            decision_type=DecisionType.TOOL_CALL,
                            reason="Write completion marker.",
                            tool_name="write_file",
                            tool_args={"path": "done.md", "content": "Finished."},
                        ),
                    ]
                ),
                max_steps=2,
                resume_checkpoint=restored,
            )

            self.assertEqual(resumed.task.status.value, "completed")
            self.assertEqual((root / "note.txt").read_text(encoding="utf-8"), "hi world")
            self.assertTrue(resumed.steps[1].tool_result.reused)
            self.assertEqual(resumed.resume_report.reused_tool_call_count, 1)

    def test_resume_candidate_detects_max_steps_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_input = TaskInput(
                goal="Inspect workspace",
                user_message="Inspect files.",
                workspace_root=temp_dir,
            )
            result = run_task(
                task_input,
                FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="List files.",
                        tool_name="list_files",
                        tool_args={"path": "."},
                    )
                ),
                max_steps=1,
            )

            checkpoint = load_checkpoint_result(result)

            self.assertTrue(is_resume_candidate(checkpoint))

    def test_checkpoint_match_rejects_unrelated_task_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original = TaskInput(
                goal="Write air conditioner guide",
                user_message="Write the air conditioner guide and save it.",
                workspace_root=temp_dir,
            )
            result = run_task(
                original,
                FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Start writing.",
                        tool_name="write_file",
                        tool_args={"path": "guide.md", "content": "draft"},
                    )
                ),
                max_steps=1,
            )
            checkpoint = load_checkpoint_result(result)
            different = TaskInput(
                goal="Inspect current runtime tools",
                user_message="Tell me what tools you have.",
                workspace_root=temp_dir,
            )

            self.assertFalse(checkpoint_matches_task_input(checkpoint, different))

    def test_checkpoint_match_accepts_same_task_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original = TaskInput(
                goal="Write air conditioner guide",
                user_message="Write the air conditioner guide and save it.",
                workspace_root=temp_dir,
            )
            result = run_task(
                original,
                FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Start writing.",
                        tool_name="write_file",
                        tool_args={"path": "guide.md", "content": "draft"},
                    )
                ),
                max_steps=1,
            )
            checkpoint = load_checkpoint_result(result)
            same = TaskInput(
                goal="Write air conditioner guide",
                user_message="Write the air conditioner guide and save it.",
                workspace_root=temp_dir,
            )

            self.assertTrue(checkpoint_matches_task_input(checkpoint, same))

    def test_custom_checkpoint_path_uses_isolated_archive_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_input = TaskInput(goal="Inspect files", user_message="Inspect files", workspace_root=temp_dir)
            result = run_task(
                task_input,
                FakeLLMClient(ActionDecision(DecisionType.TOOL_CALL, "Inspect.", "list_files", {"path": "."})),
                max_steps=1,
            )
            checkpoint = checkpoint_from_runtime_result(result)
            archive_checkpoint(checkpoint, default_checkpoint_archive_dir(temp_dir))
            custom_active = root / "isolated" / "latest.json"

            self.assertEqual(checkpoint_archive_dir_for_active_path(temp_dir, custom_active), custom_active.parent / "archive")
            self.assertEqual(list_resume_candidates(temp_dir, custom_active), [])

    def test_completed_task_is_archived_and_active_checkpoint_is_cleared(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_input = TaskInput(
                goal="Finish now",
                user_message="Finish now",
                workspace_root=temp_dir,
            )
            checkpoint_path = default_checkpoint_path(temp_dir)

            result = run_task(
                task_input,
                FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.FINISH,
                        reason="Done.",
                        response_text="Finished.",
                    )
                ),
                max_steps=1,
                checkpoint_path=str(checkpoint_path),
            )

            archive_file = default_checkpoint_archive_dir(temp_dir) / f"{result.task.task_id}.json"
            self.assertFalse(checkpoint_path.exists())
            self.assertTrue(archive_file.exists())

    def test_resume_report_contains_plan_step_diff(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("hello", encoding="utf-8")
            task_input = TaskInput(
                goal="Inspect workspace",
                user_message="Inspect files.",
                workspace_root=temp_dir,
            )

            first_result = run_task(
                task_input,
                FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="List files.",
                        tool_name="list_files",
                        tool_args={"path": "."},
                    )
                ),
                max_steps=1,
                checkpoint_path=str(root / "checkpoint.json"),
            )
            restored = load_checkpoint(root / "checkpoint.json")
            resumed = run_task(
                task_input,
                FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.FINISH,
                        reason="Enough.",
                        response_text="All done.",
                    )
                ),
                max_steps=2,
                resume_checkpoint=restored,
            )

            self.assertIn("Understand task and constraints", resumed.resume_report.restored_completed_steps)
            self.assertIn("Gather relevant context", resumed.resume_report.new_completed_steps)


def load_checkpoint_result(result):
    from gangent.checkpoint import checkpoint_from_runtime_result

    return checkpoint_from_runtime_result(result)


if __name__ == "__main__":
    unittest.main()
