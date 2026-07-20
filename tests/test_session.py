import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gangent.checkpoint import (
    checkpoint_from_runtime_result,
    default_ignored_tasks_path,
    list_resume_candidates,
    load_resume_candidate,
    save_checkpoint,
)
from gangent.llm_client import FakeLLMClient
from gangent.models import ActionDecision, DecisionType
from gangent.runtime import run_task, RuntimeResult
from gangent.session import (
    build_task_input_from_session,
    create_session,
    reset_session,
    update_session_from_result,
)
from gangent.cli import format_approval_request, run_interactive_cli
from gangent.session_store import load_or_create_session, load_session, save_session


class SessionStateTests(unittest.TestCase):
    def test_session_context_is_added_to_next_task_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = create_session(temp_dir)
            session.context_summary = "Turn 1: user=Inspect workspace"

            task_input = build_task_input_from_session(session, "Then summarize it")

            self.assertEqual(task_input.workspace_root, temp_dir)
            self.assertIn("Session context:", "\n".join(task_input.constraints))
            self.assertIn("Inspect workspace", "\n".join(task_input.constraints))

    def test_session_redacts_user_message(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = create_session(temp_dir)

            task_input = build_task_input_from_session(session, "token=abcdefgh12345678")

            self.assertIn("[REDACTED_", task_input.user_message)
            self.assertNotIn("abcdefgh12345678", task_input.user_message)

    def test_session_updates_from_runtime_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = create_session(temp_dir)
            client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.DIRECT_RESPONSE,
                    reason="Answer directly.",
                    response_text="The answer is ready.",
                )
            )
            task_input = build_task_input_from_session(session, "Answer this")

            result = run_task(task_input, client, max_steps=1)
            update_session_from_result(session, "Answer this", result)

            self.assertEqual(len(session.turns), 1)
            self.assertIn("Answer this", session.context_summary)
            self.assertIn("The answer is ready.", session.context_summary)

    def test_reset_session_clears_turns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = create_session(temp_dir)
            session.context_summary = "old"

            new_session = reset_session(session)

            self.assertEqual(new_session.workspace_root, temp_dir)
            self.assertEqual(new_session.turns, [])
            self.assertEqual(new_session.context_summary, "")
            self.assertNotEqual(new_session.session_id, session.session_id)

    def test_cli_new_command_starts_new_session(self):
        # 这个测试只验证命令入口不会报错；更细的 session 逻辑由上面的单元测试覆盖。
        with tempfile.TemporaryDirectory() as temp_dir:
            inputs = iter(["/new", "exit"])
            outputs: list[str] = []

            def fake_input(prompt: str) -> str:
                outputs.append(prompt)
                return next(inputs)

            with patch("builtins.input", fake_input):
                run_interactive_cli(
                    provider="fake",
                    model=None,
                    thinking=False,
                    max_steps=1,
                    max_tokens=100,
                    max_seconds=10,
                    workspace_root=temp_dir,
                )

            self.assertTrue(outputs)

    def test_session_can_be_saved_and_loaded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = create_session(temp_dir)
            session.context_summary = "saved context"
            path = save_session(session)

            loaded = load_session(path)

            self.assertEqual(loaded.session_id, session.session_id)
            self.assertEqual(loaded.context_summary, "saved context")
            self.assertEqual(loaded.workspace_root, temp_dir)

    def test_load_or_create_session_resumes_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = create_session(temp_dir)
            path = save_session(session)

            resumed = load_or_create_session(temp_dir, path=path, resume=True)
            fresh = load_or_create_session(temp_dir, path=path, resume=False)

            self.assertEqual(resumed.session_id, session.session_id)
            self.assertNotEqual(fresh.session_id, session.session_id)

    def test_format_approval_request_for_run_command(self):
        decision = ActionDecision(
            decision_type=DecisionType.TOOL_CALL,
            reason="Run tests.",
            tool_name="run_command",
            tool_args={"args": ["python", "-m", "unittest"], "cwd": ".", "timeout_seconds": 30},
        )
        policy = type("Policy", (), {"mode": type("Mode", (), {"value": "escalate"})(), "reason": "Needs approval."})()

        text = format_approval_request(decision, policy)

        self.assertIn("APPROVAL REQUIRED", text)
        self.assertIn("argv=['python', '-m', 'unittest']", text)

    def test_cli_auto_resumes_checkpoint_on_startup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_input = build_task_input_from_session(create_session(temp_dir), "Inspect files")
            partial = run_task(
                task_input,
                FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Inspect workspace.",
                        tool_name="list_files",
                        tool_args={"path": "."},
                    )
                ),
                max_steps=1,
            )
            checkpoint_path = save_checkpoint(checkpoint_from_runtime_result(partial), root / "checkpoint.json")
            inputs = iter(["exit"])
            captured: list[RuntimeResult] = []

            def fake_input(prompt: str) -> str:
                return next(inputs)

            def fake_client_factory(**kwargs):
                return FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.FINISH,
                        reason="Done.",
                        response_text="Finished.",
                    )
                )

            with patch("builtins.input", fake_input), patch("gangent.cli.create_llm_client", fake_client_factory), patch(
                "gangent.cli.print_result",
                lambda result, mode="verbose": captured.append(result),
            ):
                run_interactive_cli(
                    provider="fake",
                    model=None,
                    thinking=False,
                    max_steps=2,
                    max_tokens=100,
                    max_seconds=10,
                    workspace_root=temp_dir,
                    checkpoint_file=str(checkpoint_path),
                    resume=True,
                )

            self.assertTrue(captured)
            self.assertTrue(captured[0].resume_report.resumed)

    def test_cli_prompts_and_resumes_latest_task_when_user_confirms(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_input = build_task_input_from_session(create_session(temp_dir), "Inspect files")
            partial = run_task(
                task_input,
                FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Inspect workspace.",
                        tool_name="list_files",
                        tool_args={"path": "."},
                    )
                ),
                max_steps=1,
            )
            checkpoint_path = save_checkpoint(checkpoint_from_runtime_result(partial), root / "checkpoint.json")
            inputs = iter(["yes", "exit"])
            captured: list[RuntimeResult] = []

            def fake_input(prompt: str) -> str:
                return next(inputs)

            def fake_client_factory(**kwargs):
                return FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.FINISH,
                        reason="Done.",
                        response_text="Finished.",
                    )
                )

            with patch("builtins.input", fake_input), patch("gangent.cli.create_llm_client", fake_client_factory), patch(
                "gangent.cli.print_result",
                lambda result, mode="verbose": captured.append(result),
            ):
                run_interactive_cli(
                    provider="fake",
                    model=None,
                    thinking=False,
                    max_steps=2,
                    max_tokens=100,
                    max_seconds=10,
                    workspace_root=temp_dir,
                    checkpoint_file=str(checkpoint_path),
                )

            self.assertTrue(captured)
            self.assertTrue(captured[0].resume_report.resumed)

    def test_cli_does_not_auto_resume_without_resume_flag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_input = build_task_input_from_session(create_session(temp_dir), "Inspect files")
            partial = run_task(
                task_input,
                FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Inspect workspace.",
                        tool_name="list_files",
                        tool_args={"path": "."},
                    )
                ),
                max_steps=1,
            )
            checkpoint_path = save_checkpoint(checkpoint_from_runtime_result(partial), root / "checkpoint.json")
            inputs = iter(["no", "exit"])
            captured: list[RuntimeResult] = []

            def fake_input(prompt: str) -> str:
                return next(inputs)

            def fake_client_factory(**kwargs):
                return FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.FINISH,
                        reason="Done.",
                        response_text="Finished.",
                    )
                )

            with patch("builtins.input", fake_input), patch("gangent.cli.create_llm_client", fake_client_factory), patch(
                "gangent.cli.print_result",
                lambda result, mode="verbose": captured.append(result),
            ):
                run_interactive_cli(
                    provider="fake",
                    model=None,
                    thinking=False,
                    max_steps=2,
                    max_tokens=100,
                    max_seconds=10,
                    workspace_root=temp_dir,
                    checkpoint_file=str(checkpoint_path),
                )

            self.assertFalse(captured)

    def test_cli_does_not_resume_unrelated_checkpoint_for_new_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_input = build_task_input_from_session(create_session(temp_dir), "Write the air conditioner guide and save it.")
            partial = run_task(
                task_input,
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
            checkpoint_path = save_checkpoint(checkpoint_from_runtime_result(partial), root / "checkpoint.json")
            inputs = iter(["no", "Tell me what tools you have.", "exit"])
            captured: list[RuntimeResult] = []

            def fake_input(prompt: str) -> str:
                return next(inputs)

            def fake_client_factory(**kwargs):
                return FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.DIRECT_RESPONSE,
                        reason="Answer directly.",
                        response_text="These are my tools.",
                    )
                )

            with patch("builtins.input", fake_input), patch("gangent.cli.create_llm_client", fake_client_factory), patch(
                "gangent.cli.print_result",
                lambda result, mode="verbose": captured.append(result),
            ):
                run_interactive_cli(
                    provider="fake",
                    model=None,
                    thinking=False,
                    max_steps=2,
                    max_tokens=100,
                    max_seconds=10,
                    workspace_root=temp_dir,
                    checkpoint_file=str(checkpoint_path),
                )

            self.assertTrue(captured)
            self.assertIsNone(captured[0].resume_report)

    def test_cli_shelves_skipped_active_checkpoint_before_starting_new_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_input = build_task_input_from_session(create_session(temp_dir), "Write the air conditioner guide and save it.")
            partial = run_task(
                task_input,
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
            checkpoint_path = save_checkpoint(checkpoint_from_runtime_result(partial), root / "checkpoint.json")
            inputs = iter(["no", "Tell me what tools you have.", "exit"])
            captured: list[RuntimeResult] = []

            def fake_input(prompt: str) -> str:
                return next(inputs)

            def fake_client_factory(**kwargs):
                return FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.DIRECT_RESPONSE,
                        reason="Answer directly.",
                        response_text="These are my tools.",
                    )
                )

            with patch("builtins.input", fake_input), patch("gangent.cli.create_llm_client", fake_client_factory), patch(
                "gangent.cli.print_result",
                lambda result, mode="verbose": captured.append(result),
            ):
                run_interactive_cli(
                    provider="fake",
                    model=None,
                    thinking=False,
                    max_steps=2,
                    max_tokens=100,
                    max_seconds=10,
                    workspace_root=temp_dir,
                    checkpoint_file=str(checkpoint_path),
                )

            self.assertTrue(captured)
            self.assertIsNone(load_resume_candidate(checkpoint_path))
            resumable = list_resume_candidates(temp_dir, checkpoint_path)
            self.assertEqual(len(resumable), 1)
            self.assertEqual(resumable[0].checkpoint.task.task_id, partial.task.task_id)

    def test_cli_delete_hides_resumable_tasks_from_future_prompts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_input = build_task_input_from_session(create_session(temp_dir), "Inspect files")
            partial = run_task(
                task_input,
                FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Inspect workspace.",
                        tool_name="list_files",
                        tool_args={"path": "."},
                    )
                ),
                max_steps=1,
            )
            checkpoint_path = save_checkpoint(checkpoint_from_runtime_result(partial), root / "checkpoint.json")
            inputs = iter(["delete", "1", "yes", "exit"])

            def fake_input(prompt: str) -> str:
                return next(inputs)

            with patch("builtins.input", fake_input):
                run_interactive_cli(
                    provider="fake",
                    model=None,
                    thinking=False,
                    max_steps=2,
                    max_tokens=100,
                    max_seconds=10,
                    workspace_root=temp_dir,
                    checkpoint_file=str(checkpoint_path),
                )

            self.assertTrue(default_ignored_tasks_path(temp_dir).exists())
            self.assertEqual(list_resume_candidates(temp_dir, checkpoint_path), [])


if __name__ == "__main__":
    unittest.main()
