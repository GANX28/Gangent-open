from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gangent.audit import append_audit_record
from gangent.checkpoint import checkpoint_from_runtime_result, save_checkpoint
from gangent.handoff import default_handoff_path, export_handoff_file
from gangent.models import RuntimeStats, TaskInput
from gangent.runtime import RuntimeResult
from gangent.session import create_session, update_session_from_result
from gangent.session_store import save_session
from gangent.state import create_initial_state, create_task


class HandoffExportTests(unittest.TestCase):
    def test_default_handoff_path_is_written_to_runtime_handoff_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir) / "Gangent"
            workspace_root.mkdir()

            path = default_handoff_path(str(workspace_root), timestamp="20260627-123456")

            self.assertEqual(path.parent, workspace_root.resolve() / ".gangent" / "handoff")
            self.assertTrue(path.name.startswith("gangent-handoff-20260627-123456"))

    def test_export_handoff_file_includes_timestamp_session_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir) / "Gangent"
            workspace_root.mkdir()

            task_input = TaskInput(
                goal="Inspect workspace and summarize progress",
                user_message="Inspect workspace and summarize progress",
                workspace_root=str(workspace_root),
                constraints=["rule one", "rule two"],
            )
            task = create_task(task_input)
            state = create_initial_state(task, task_input)
            state.context_summary = "Goal: Inspect workspace and summarize progress"
            result = RuntimeResult(
                task=task,
                state=state,
                steps=[],
                stats=RuntimeStats(step_count=0),
                task_input=task_input,
            )

            session = create_session(str(workspace_root))
            update_session_from_result(session, "Inspect workspace and summarize progress", result)
            save_session(session)

            checkpoint_path = save_checkpoint(checkpoint_from_runtime_result(result))
            audit_path = workspace_root / ".gangent" / "audit" / "latest.jsonl"
            append_audit_record(
                result,
                session_id=session.session_id,
                user_message="Inspect workspace and summarize progress",
                path=audit_path,
            )

            output_path = export_handoff_file(str(workspace_root), checkpoint_path=checkpoint_path)
            content = output_path.read_text(encoding="utf-8")

            self.assertTrue(output_path.exists())
            self.assertIn("# Gangent Handoff", content)
            self.assertIn("generated_at:", content)
            self.assertIn(session.session_id, content)
            self.assertIn(task.task_id, content)
            self.assertIn("Current Session", content)
            self.assertIn("Active Checkpoint", content)
            self.assertIn("Recent Audit", content)


if __name__ == "__main__":
    unittest.main()
