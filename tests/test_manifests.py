import tempfile
import unittest
from pathlib import Path

from gangent.manifests import (
    build_execution_manifest,
    extract_path_mentions,
    format_blocking_validation_hint,
    format_manifest_prompt,
)
from gangent.models import TaskInput


class ExecutionManifestTests(unittest.TestCase):
    def test_extracts_source_and_output_paths(self):
        mentions = extract_path_mentions(
            "读取 README.md 和 docs/manual.md，总结后写入 workspace/agent_check.md。"
        )

        self.assertEqual(mentions.sources, ["README.md", "docs/manual.md"])
        self.assertEqual(mentions.outputs, ["workspace/agent_check.md"])

    def test_extracts_path_before_write_marker_as_output(self):
        mentions = extract_path_mentions("在 workspace/smoke_test.md 写入一份测试说明。")

        self.assertEqual(mentions.sources, [])
        self.assertEqual(mentions.outputs, ["workspace/smoke_test.md"])

    def test_extracts_real_chinese_source_and_output_paths(self):
        mentions = extract_path_mentions(
            "\u8bfb\u53d6 source.md\uff0c\u603b\u7ed3\u4e3a summary.md\uff0c"
            "\u4e0d\u8981\u4fee\u6539\u5176\u4ed6\u6587\u4ef6\u3002"
        )

        self.assertEqual(mentions.sources, ["source.md"])
        self.assertEqual(mentions.outputs, ["summary.md"])

    def test_missing_output_blocks_finish(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("hello", encoding="utf-8")
            task_input = TaskInput(
                goal="读取 README.md，然后写入 workspace/agent_check.md。",
                user_message="读取 README.md，然后写入 workspace/agent_check.md。",
                workspace_root=str(root),
            )

            manifest = build_execution_manifest(task_input)

            self.assertEqual(manifest.sources[0].status, "exists")
            self.assertEqual(manifest.outputs[0].status, "missing")
            self.assertTrue(manifest.blocking_issues)
            self.assertIn("workspace/agent_check.md", format_blocking_validation_hint(manifest))

    def test_written_text_output_passes_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "workspace").mkdir()
            (root / "workspace" / "agent_check.md").write_text("done", encoding="utf-8")
            task_input = TaskInput(
                goal="写入 workspace/agent_check.md。",
                user_message="写入 workspace/agent_check.md。",
                workspace_root=str(root),
            )

            manifest = build_execution_manifest(task_input)

            self.assertFalse(manifest.blocking_issues)
            self.assertIn("Output Manifest", format_manifest_prompt(manifest))

    def test_missing_output_manifest_tells_model_to_write_next_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "workspace").mkdir()
            task_input = TaskInput(
                goal="在 workspace/stability_retest.md 写一份稳定性复测说明，不要修改其他文件。",
                user_message="在 workspace/stability_retest.md 写一份稳定性复测说明，不要修改其他文件。",
                workspace_root=str(root),
            )

            manifest = build_execution_manifest(task_input)
            prompt = format_manifest_prompt(manifest)

            self.assertIn("Next required output: workspace/stability_retest.md", prompt)
            self.assertIn("write_file", prompt)
            self.assertIn("Do not finish by merely reporting that it is missing.", prompt)

    def test_invalid_json_output_blocks_finish(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "workspace").mkdir()
            (root / "workspace" / "data.json").write_text("{bad", encoding="utf-8")
            task_input = TaskInput(
                goal="生成 workspace/data.json。",
                user_message="生成 workspace/data.json。",
                workspace_root=str(root),
            )

            manifest = build_execution_manifest(task_input)

            self.assertTrue(manifest.blocking_issues)
            self.assertIn("invalid JSON", manifest.blocking_issues[0].message)


if __name__ == "__main__":
    unittest.main()
