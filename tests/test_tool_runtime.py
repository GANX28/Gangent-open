import tempfile
import subprocess
import unittest
from pathlib import Path

from gangent.models import ActionDecision, DecisionType
from gangent.tool_runtime import (
    compile_python,
    edit_file,
    ensure_workspace,
    export_artifact,
    fetch_url,
    file_info,
    grep_files,
    execute_tool_call,
    git_add,
    git_commit,
    git_status,
    git_log,
    git_show,
    list_files,
    memory_add,
    apply_patch,
    read_file,
    read_many_files,
    run_tests,
    run_command,
    write_file,
    scratchpad_note,
)


class ToolRuntimeTests(unittest.TestCase):
    def _init_git_repo(self, path: str) -> None:
        subprocess.run(["git", "init"], cwd=path, capture_output=True, text=True, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, capture_output=True, text=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=path,
            capture_output=True,
            text=True,
            check=True,
        )

    def test_list_files_lists_workspace_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("hello", encoding="utf-8")
            (root / "gangent").mkdir()

            output = list_files(".", str(root))

            self.assertIn("[file] README.md", output)
            self.assertIn("[dir] gangent", output)

    def test_read_file_reads_utf8_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("hello", encoding="utf-8")

            output = read_file("README.md", str(root))

            self.assertEqual(output, "hello")

    def test_read_file_keeps_real_directory_matching_root_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir, "Gangent")
            package_dir = root / "gangent"
            package_dir.mkdir(parents=True)
            (package_dir / "tool_schema.py").write_text("schema code", encoding="utf-8")

            output = read_file("gangent/tool_schema.py", str(root))

            self.assertEqual(output, "schema code")

    def test_read_many_files_reads_multiple_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.md").write_text("alpha", encoding="utf-8")
            (root / "b.md").write_text("beta", encoding="utf-8")

            output = read_many_files(["a.md", "b.md"], temp_dir)

            self.assertIn("--- a.md ---", output)
            self.assertIn("alpha", output)
            self.assertIn("--- b.md ---", output)
            self.assertIn("beta", output)

    def test_read_file_supports_chunked_line_reads_for_large_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lines = [f"line {index}" for index in range(1, 101)]
            (root / "big.txt").write_text("\n".join(lines), encoding="utf-8")

            output = read_file("big.txt", temp_dir, start_line=10, max_lines=5)

            self.assertIn("start_line=10", output)
            self.assertIn("10 | line 10", output)
            self.assertIn("14 | line 14", output)
            self.assertNotIn("15 | line 15", output)

    def test_read_file_large_file_without_chunk_hint_returns_partial_chunk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            content = "\n".join(f"line-{index:05d}" for index in range(3000))
            (root / "big.txt").write_text(content, encoding="utf-8")

            output = read_file("big.txt", temp_dir)

            self.assertIn("partial=true", output)
            self.assertIn("1 | line-00000", output)
            self.assertIn("200 | line-00199", output)
            self.assertNotIn("201 | line-00200", output)

    def test_file_info_distinguishes_file_and_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "docs").mkdir()
            (root / "docs" / "a.md").write_text("one\ntwo\n", encoding="utf-8")

            directory_info = file_info("docs", temp_dir)
            file_metadata = file_info("docs/a.md", temp_dir)

            self.assertIn("type=directory", directory_info)
            self.assertIn("type=file", file_metadata)
            self.assertIn("line_count=2", file_metadata)

    def test_path_escape_is_rejected_as_tool_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Try unsafe path.",
                tool_name="read_file",
                tool_args={"path": "../outside.txt"},
            )

            result = execute_tool_call(decision, workspace_root=temp_dir)

            self.assertFalse(result.success)
            self.assertIn("escapes workspace root", result.error)

    def test_memory_add_writes_memory_graph_node(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = memory_add(
                node_type="decision",
                content="Use planner evaluation for future planning constraints.",
                workspace_root=temp_dir,
                project_scope="gangent",
                layer="task",
            )

            self.assertIn("memory_node_added", output)
            self.assertTrue(Path(temp_dir, ".gangent", "memory", "graph.json").exists())

    def test_memory_add_rejects_secret_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                memory_add("note", "api_key=abcdefgh12345678", temp_dir)

    def test_unknown_tool_is_rejected_as_tool_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Unknown tool.",
                tool_name="shell",
                tool_args={"cmd": "dir"},
            )

            result = execute_tool_call(decision, workspace_root=temp_dir)

            self.assertFalse(result.success)
            self.assertIn("Unknown tool", result.error)

    def test_write_file_writes_workspace_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = write_file("note.txt", "hello", temp_dir)

            self.assertIn("Wrote", output)
            self.assertEqual(Path(temp_dir, "note.txt").read_text(encoding="utf-8"), "hello")

    def test_write_file_rejects_hidden_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Write hidden.",
                tool_name="write_file",
                tool_args={"path": ".env", "content": "SECRET=1", "overwrite": False},
            )

            result = execute_tool_call(decision, workspace_root=temp_dir)

            self.assertFalse(result.success)
            self.assertIn("Hidden", result.error)

    def test_read_file_redacts_secret_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.txt").write_text(
                "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456",
                encoding="utf-8",
            )

            output = read_file("config.txt", temp_dir)

            self.assertIn("[REDACTED_", output)
            self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz123456", output)

    def test_write_file_rejects_secret_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_tool_call(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Write secret.",
                    tool_name="write_file",
                    tool_args={
                        "path": "config.txt",
                        "content": "token=abcdefgh12345678",
                        "overwrite": False,
                    },
                ),
                workspace_root=temp_dir,
            )

            self.assertFalse(result.success)
            self.assertIn("possible secrets", result.error)

    def test_write_file_rejects_encoding_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = execute_tool_call(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Write corrupted generated text.",
                    tool_name="write_file",
                    tool_args={
                        "path": "summary.md",
                        "content": "Planner 鈥? orchestrates runtime.",
                        "overwrite": False,
                    },
                ),
                workspace_root=temp_dir,
            )

            self.assertFalse(result.success)
            self.assertIn("encoding-corrupted", result.error)

    def test_edit_file_requires_unique_old_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "note.txt").write_text("hello\nhello\n", encoding="utf-8")

            with self.assertRaises(Exception):
                edit_file("note.txt", "hello", "hi", temp_dir)

    def test_edit_file_replaces_exact_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "note.txt").write_text("hello world", encoding="utf-8")

            output = edit_file("note.txt", "hello", "hi", temp_dir)

            self.assertIn("Edited", output)
            self.assertEqual((root / "note.txt").read_text(encoding="utf-8"), "hi world")

    def test_edit_file_rejects_encoding_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "note.txt").write_text("hello world", encoding="utf-8")

            with self.assertRaises(Exception) as context:
                edit_file("note.txt", "hello", "璇诲彇 world", temp_dir)

            self.assertIn("encoding-corrupted", str(context.exception))

    def test_apply_patch_adds_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            patch = """*** Begin Patch
*** Add File: note.txt
+hello
*** End Patch"""

            output = apply_patch(patch, temp_dir)

            self.assertIn("Added note.txt", output)
            self.assertEqual(Path(temp_dir, "note.txt").read_text(encoding="utf-8"), "hello\n")

    def test_git_status_runs_read_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._init_git_repo(temp_dir)
            Path(temp_dir, "note.txt").write_text("hello", encoding="utf-8")

            output = git_status(temp_dir)

            self.assertIn("note.txt", output)

    def test_git_log_and_show_read_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._init_git_repo(temp_dir)
            Path(temp_dir, "note.txt").write_text("hello", encoding="utf-8")
            subprocess.run(["git", "add", "--", "note.txt"], cwd=temp_dir, capture_output=True, text=True, check=True)
            subprocess.run(
                ["git", "commit", "-m", "initial commit"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                check=True,
            )

            log_output = git_log(temp_dir, limit=1)
            show_output = git_show("HEAD", temp_dir)

            self.assertIn("initial commit", log_output)
            self.assertIn("initial commit", show_output)

    def test_git_add_and_commit_mutate_repo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._init_git_repo(temp_dir)
            Path(temp_dir, "note.txt").write_text("hello", encoding="utf-8")

            add_output = git_add(["note.txt"], temp_dir)
            commit_output = git_commit("add note", temp_dir)

            self.assertIn("Staged 1 path", add_output)
            self.assertIn("add note", commit_output)

    def test_compile_python_checks_syntax_without_pycache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "demo.py").write_text("x = 1\n", encoding="utf-8")

            output = compile_python(temp_dir)

            self.assertIn("Python syntax ok", output)
            self.assertFalse((root / "__pycache__").exists())

    def test_run_tests_runs_unittest_suite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_ok.py").write_text(
                "import unittest\n\n"
                "class OkTest(unittest.TestCase):\n"
                "    def test_ok(self):\n"
                "        self.assertTrue(True)\n",
                encoding="utf-8",
            )

            output = run_tests(temp_dir)

            self.assertIn("OK", output)

    def test_run_command_runs_low_risk_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = run_command(["python", "--version"], temp_dir, timeout_seconds=10)

            self.assertIn("Python", output)

    def test_run_command_blocks_shell_host(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(Exception):
                run_command(["cmd", "/c", "dir"], temp_dir, timeout_seconds=10)

    def test_execute_search_context_tool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "runtime.py").write_text("runtime policy execution", encoding="utf-8")
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Search context.",
                tool_name="search_context",
                tool_args={"query": "runtime policy", "top_k": 1},
            )

            result = execute_tool_call(decision, workspace_root=temp_dir)

            self.assertTrue(result.success)
            self.assertIn("runtime.py", result.output)

    def test_grep_files_finds_matches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "runtime.py").write_text("runtime policy execution\nother\n", encoding="utf-8")

            output = grep_files("policy", ".", temp_dir, max_results=5)

            self.assertIn("runtime.py:1", output)

    def test_export_artifact_writes_under_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = export_artifact("report.md", "hello", temp_dir)

            self.assertIn("Wrote", output)
            self.assertEqual(Path(temp_dir, "artifacts", "report.md").read_text(encoding="utf-8"), "hello")

    def test_scratchpad_note_appends_internal_note(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = scratchpad_note("next step: inspect tools", temp_dir)

            self.assertIn("Scratchpad", output)
            self.assertIn(
                "inspect tools",
                Path(temp_dir, ".gangent", "scratchpad", "latest.md").read_text(encoding="utf-8"),
            )

    def test_fetch_url_blocks_localhost(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(Exception):
                fetch_url("http://localhost:8000", temp_dir)

    def test_fetch_url_blocks_private_ip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(Exception):
                fetch_url("http://127.0.0.1:8000", temp_dir)

    def test_ensure_workspace_creates_directory_and_readme(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = ensure_workspace("workspace", temp_dir)

            self.assertIn("Workspace ready", output)
            self.assertTrue(Path(temp_dir, "workspace").is_dir())
            self.assertTrue(Path(temp_dir, "workspace", "README.md").is_file())

    def test_redundant_workspace_prefix_is_folded_to_workspace_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir, "Gangent", "workspace")
            root.mkdir(parents=True)

            output = ensure_workspace("workspace", str(root))

            self.assertIn("Workspace ready", output)
            self.assertTrue(Path(root, "README.md").is_file())
            self.assertFalse(Path(root, "workspace").exists())

    def test_redundant_project_workspace_prefix_is_folded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir, "Gangent", "workspace")
            root.mkdir(parents=True)

            write_file("Gangent/workspace/note.txt", "hello", str(root))

            self.assertEqual(Path(root, "note.txt").read_text(encoding="utf-8"), "hello")
            self.assertFalse(Path(root, "Gangent").exists())

    def test_fake_unix_workspace_prefix_is_folded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir, "Gangent", "workspace")
            root.mkdir(parents=True)

            write_file("/workspace/Gangent/workspace/note.txt", "hello", str(root))

            self.assertEqual(Path(root, "note.txt").read_text(encoding="utf-8"), "hello")
            self.assertFalse(Path(root, "workspace").exists())

    def test_ensure_workspace_rejects_hidden_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Create hidden workspace.",
                tool_name="ensure_workspace",
                tool_args={"path": ".workspace"},
            )

            result = execute_tool_call(decision, workspace_root=temp_dir)

            self.assertFalse(result.success)
            self.assertIn("Hidden", result.error)


if __name__ == "__main__":
    unittest.main()
