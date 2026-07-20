import tempfile
import unittest

from gangent.error_recovery import (
    attach_recovery_hint,
    recovery_hint_for_policy,
    recovery_hint_for_tool_result,
)
from gangent.llm_client import FakeLLMClient
from gangent.models import (
    ActionDecision,
    DecisionType,
    PolicyDecision,
    PolicyMode,
    TaskInput,
    ToolResult,
)
from gangent.runtime import run_task
from gangent.state import create_initial_state, create_task


class ErrorRecoveryTests(unittest.TestCase):
    def test_recovery_hint_for_missing_file(self):
        decision = ActionDecision(
            decision_type=DecisionType.TOOL_CALL,
            reason="Read file.",
            tool_name="read_file",
            tool_args={"path": "missing.txt"},
        )
        result = ToolResult(
            call_id="call_1",
            success=False,
            error="File does not exist: missing.txt",
        )

        hint = recovery_hint_for_tool_result(decision, result)

        self.assertIn("list_files", hint)
        self.assertIn("workspace-relative", hint)

    def test_recovery_hint_for_patch_context_failure(self):
        decision = ActionDecision(
            decision_type=DecisionType.TOOL_CALL,
            reason="Patch file.",
            tool_name="apply_patch",
            tool_args={"patch": "..."},
        )
        result = ToolResult(
            call_id="call_1",
            success=False,
            error="Patch context was not found: note.txt",
        )

        hint = recovery_hint_for_tool_result(decision, result)

        self.assertIn("Read the target file", hint)

    def test_recovery_hint_for_policy_block(self):
        decision = ActionDecision(
            decision_type=DecisionType.TOOL_CALL,
            reason="Run shell.",
            tool_name="run_command",
            tool_args={"args": ["cmd", "/c", "dir"]},
        )
        policy = PolicyDecision(
            mode=PolicyMode.BLOCK,
            allowed=False,
            reason="Executable is blocked: cmd",
        )

        hint = recovery_hint_for_policy(decision, policy)

        self.assertIn("policy prevented", hint)
        self.assertIn("cmd", hint)

    def test_recovery_hint_for_blocked_shell_file_write_points_to_file_tools(self):
        decision = ActionDecision(
            decision_type=DecisionType.TOOL_CALL,
            reason="Write through shell.",
            tool_name="run_command",
            tool_args={"args": ["powershell", "-Command", "Set-Content x.txt hi"]},
        )
        policy = PolicyDecision(
            mode=PolicyMode.BLOCK,
            allowed=False,
            reason="Executable is blocked: powershell",
        )

        hint = recovery_hint_for_policy(decision, policy)

        self.assertIn("write_file", hint)
        self.assertIn("edit_file", hint)
        self.assertIn("powershell", hint)

    def test_attach_recovery_hint_updates_state_context(self):
        task_input = TaskInput(goal="Demo", user_message="Demo", workspace_root=".")
        task = create_task(task_input)
        state = create_initial_state(task, task_input)

        attach_recovery_hint(state, "Recovery hint: retry smaller.")

        self.assertIn("Recovery hints", state.context_summary)
        self.assertEqual(state.messages[-1].content, "Recovery hint: retry smaller.")

    def test_runtime_adds_recovery_hint_after_tool_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Read missing file.",
                    tool_name="read_file",
                    tool_args={"path": "missing.txt"},
                )
            )
            task_input = TaskInput(
                goal="Read missing file",
                user_message="Read missing file.",
                workspace_root=temp_dir,
            )

            result = run_task(task_input, client, max_steps=1)

            self.assertIn("Recovery hints", result.state.context_summary)
            self.assertTrue(any("Recovery hint" in message.content for message in result.state.messages))


if __name__ == "__main__":
    unittest.main()
