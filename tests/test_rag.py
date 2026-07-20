import tempfile
import json
import unittest
from pathlib import Path

from gangent.rag import build_chunks, default_retrieval_log_path, search_chunks, search_context


class RagTests(unittest.TestCase):
    def test_search_context_finds_relevant_chunk_with_citation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "runtime.py").write_text(
                "class RuntimeLoop:\n"
                "    def run_task(self):\n"
                "        return 'runtime loop policy tool execution'\n",
                encoding="utf-8",
            )
            (root / "notes.md").write_text("unrelated content", encoding="utf-8")

            output = search_context("runtime policy", temp_dir, top_k=2)

            self.assertIn("runtime.py:1-3", output)
            self.assertIn("runtime loop policy", output)
            self.assertTrue(default_retrieval_log_path(temp_dir).exists())

    def test_build_chunks_skips_sensitive_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "visible.py").write_text("public runtime text", encoding="utf-8")
            (root / ".env").write_text("token=abcdefgh12345678", encoding="utf-8")

            chunks = build_chunks(temp_dir)
            paths = {chunk.path for chunk in chunks}

            self.assertIn("visible.py", paths)
            self.assertNotIn(".env", paths)
            self.assertEqual(chunks[0].metadata["source_type"], "py")

    def test_search_context_redacts_secrets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.txt").write_text(
                "public token=abcdefgh12345678 setting",
                encoding="utf-8",
            )

            output = search_context("public token", temp_dir, top_k=1)

            self.assertIn("[REDACTED_", output)
            self.assertNotIn("abcdefgh12345678", output)

    def test_search_chunks_returns_empty_for_no_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "a.py").write_text("alpha beta", encoding="utf-8")
            chunks = build_chunks(temp_dir)

            results = search_chunks("missingterm", chunks)

            self.assertEqual(results, [])

    def test_search_context_writes_retrieval_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "runtime.py").write_text("runtime policy execution", encoding="utf-8")

            search_context("runtime", temp_dir, top_k=1)

            log_path = default_retrieval_log_path(temp_dir)
            record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(record["query"], "runtime")
            self.assertEqual(record["top_k"], 1)
            self.assertEqual(record["result_count"], 1)
            self.assertEqual(record["results"][0]["path"], "runtime.py")


if __name__ == "__main__":
    unittest.main()
