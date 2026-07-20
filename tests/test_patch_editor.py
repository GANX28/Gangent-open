import tempfile
import unittest
from pathlib import Path

from gangent.patch_editor import (
    PatchError,
    apply_text_patch,
    inspect_patch_paths,
    parse_patch,
    summarize_patch,
)


class PatchEditorTests(unittest.TestCase):
    def test_add_file_patch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            patch = """*** Begin Patch
*** Add File: note.txt
+hello
+world
*** End Patch"""

            output = apply_text_patch(patch, temp_dir)

            self.assertIn("Added note.txt", output)
            self.assertEqual(Path(temp_dir, "note.txt").read_text(encoding="utf-8"), "hello\nworld\n")

    def test_update_file_patch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "note.txt").write_text(
                "hello\n"
                "old\n"
                "bye\n",
                encoding="utf-8",
            )
            patch = """*** Begin Patch
*** Update File: note.txt
@@
 hello
-old
+new
 bye
*** End Patch"""

            output = apply_text_patch(patch, temp_dir)

            self.assertIn("Updated note.txt", output)
            self.assertEqual(
                Path(temp_dir, "note.txt").read_text(encoding="utf-8"),
                "hello\n"
                "new\n"
                "bye\n",
            )

    def test_update_file_rejects_missing_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "note.txt").write_text("hello\n", encoding="utf-8")
            patch = """*** Begin Patch
*** Update File: note.txt
@@
-missing
+new
*** End Patch"""

            with self.assertRaises(PatchError):
                apply_text_patch(patch, temp_dir)

    def test_delete_file_is_rejected(self):
        patch = """*** Begin Patch
*** Delete File: note.txt
*** End Patch"""

        with self.assertRaises(PatchError):
            parse_patch(patch)

    def test_inspect_patch_paths(self):
        patch = """*** Begin Patch
*** Add File: note.txt
+hello
*** End Patch"""

        self.assertEqual(inspect_patch_paths(patch), ["note.txt"])

    def test_summarize_patch(self):
        patch = """*** Begin Patch
*** Add File: note.txt
+hello
*** Update File: app.py
@@
-old
+new
*** End Patch"""

        summary = summarize_patch(patch)

        self.assertIn("add note.txt", summary)
        self.assertIn("update app.py", summary)


if __name__ == "__main__":
    unittest.main()
