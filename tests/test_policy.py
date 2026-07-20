import tempfile
import unittest
from pathlib import Path

from gangent.models import ActionDecision, DecisionType, PolicyMode
from gangent.policy import check_policy


class PolicyTests(unittest.TestCase):
    def test_allows_read_only_list_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="List files.",
                tool_name="list_files",
                tool_args={"path": "."},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertTrue(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.ALLOW)

    def test_blocks_path_escape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Unsafe read.",
                tool_name="read_file",
                tool_args={"path": "../outside.txt"},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertFalse(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.BLOCK)
            self.assertIn("escapes workspace root", policy.reason)

    def test_blocks_shell_tool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Run shell.",
                tool_name="shell",
                tool_args={"cmd": "dir"},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertFalse(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.BLOCK)

    def test_blocks_sensitive_read_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Read env.",
                tool_name="read_file",
                tool_args={"path": ".env"},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertFalse(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.BLOCK)

    def test_blocks_secret_write_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Write key.",
                tool_name="write_file",
                tool_args={
                    "path": "config.txt",
                    "content": "api_key=abcdefgh12345678",
                    "overwrite": False,
                },
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertFalse(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.BLOCK)

    def test_allows_memory_add_after_secret_scan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Store memory.",
                tool_name="memory_add",
                tool_args={
                    "node_type": "decision",
                    "content": "Use planner evaluation before changing planner budget.",
                    "summary": "",
                    "project_scope": "gangent",
                    "source": "test",
                    "tags": [],
                    "importance": 0.7,
                    "confidence": 0.9,
                    "layer": "task",
                },
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertTrue(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.ALLOW)

    def test_allows_large_file_read_as_partial_chunk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            large_file = root / "large.txt"
            large_file.write_text("x" * 20_001, encoding="utf-8")
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Read large file.",
                tool_name="read_file",
                tool_args={"path": "large.txt"},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertTrue(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.ALLOW)
            self.assertIn("partial chunk", policy.reason)

    def test_allows_new_workspace_write(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Write new file.",
                tool_name="write_file",
                tool_args={"path": "note.txt", "content": "hello", "overwrite": False},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertTrue(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.ALLOW)

    def test_allows_restricted_apply_patch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            patch = """*** Begin Patch
*** Add File: note.txt
+hello
*** End Patch"""
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Patch file.",
                tool_name="apply_patch",
                tool_args={"patch": patch},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertTrue(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.ALLOW)

    def test_blocks_apply_patch_delete_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            patch = """*** Begin Patch
*** Delete File: note.txt
*** End Patch"""
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Delete file.",
                tool_name="apply_patch",
                tool_args={"patch": patch},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertFalse(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.BLOCK)

    def test_escalates_existing_file_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "note.txt").write_text("old", encoding="utf-8")
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Overwrite file.",
                tool_name="write_file",
                tool_args={"path": "note.txt", "content": "new", "overwrite": False},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertFalse(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.ESCALATE)

    def test_allows_git_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Inspect git.",
                tool_name="git_status",
                tool_args={},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertTrue(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.ALLOW)

    def test_allows_git_log_and_show(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_policy = check_policy(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Inspect history.",
                    tool_name="git_log",
                    tool_args={"limit": 3},
                ),
                workspace_root=temp_dir,
            )
            show_policy = check_policy(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Inspect commit.",
                    tool_name="git_show",
                    tool_args={"revision": "HEAD"},
                ),
                workspace_root=temp_dir,
            )

            self.assertTrue(log_policy.allowed)
            self.assertEqual(log_policy.mode, PolicyMode.ALLOW)
            self.assertTrue(show_policy.allowed)
            self.assertEqual(show_policy.mode, PolicyMode.ALLOW)

    def test_escalates_git_add_and_commit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "note.txt").write_text("hello", encoding="utf-8")
            add_policy = check_policy(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Stage file.",
                    tool_name="git_add",
                    tool_args={"paths": ["note.txt"]},
                ),
                workspace_root=temp_dir,
            )
            commit_policy = check_policy(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Commit changes.",
                    tool_name="git_commit",
                    tool_args={"message": "save work"},
                ),
                workspace_root=temp_dir,
            )

            self.assertFalse(add_policy.allowed)
            self.assertEqual(add_policy.mode, PolicyMode.ESCALATE)
            self.assertFalse(commit_policy.allowed)
            self.assertEqual(commit_policy.mode, PolicyMode.ESCALATE)

    def test_allows_compile_python(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Check syntax.",
                tool_name="compile_python",
                tool_args={},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertTrue(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.ALLOW)

    def test_escalates_run_tests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Run tests.",
                tool_name="run_tests",
                tool_args={},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertFalse(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.ESCALATE)

    def test_allows_low_risk_run_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Check Python.",
                tool_name="run_command",
                tool_args={"args": ["python", "--version"], "cwd": ".", "timeout_seconds": 10},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertTrue(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.ALLOW)

    def test_escalates_unknown_run_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Run formatter.",
                tool_name="run_command",
                tool_args={"args": ["ruff", "check", "."], "cwd": ".", "timeout_seconds": 10},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertFalse(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.ESCALATE)

    def test_blocks_shell_host_run_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Run shell.",
                tool_name="run_command",
                tool_args={"args": ["cmd", "/c", "dir"], "cwd": ".", "timeout_seconds": 10},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertFalse(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.BLOCK)

    def test_allows_search_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Search context.",
                tool_name="search_context",
                tool_args={"query": "runtime policy", "top_k": 3},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertTrue(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.ALLOW)

    def test_blocks_invalid_search_context_top_k(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Search context.",
                tool_name="search_context",
                tool_args={"query": "runtime policy", "top_k": 100},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertFalse(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.BLOCK)

    def test_allows_ensure_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Create workspace.",
                tool_name="ensure_workspace",
                tool_args={"path": "workspace"},
            )

            policy = check_policy(decision, workspace_root=temp_dir)

            self.assertTrue(policy.allowed)
            self.assertEqual(policy.mode, PolicyMode.ALLOW)


if __name__ == "__main__":
    unittest.main()
