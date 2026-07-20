import sys
import tempfile
import unittest

from gangent.runner import LocalRunner, SandboxCommand


class LocalRunnerTests(unittest.TestCase):
    def test_local_runner_returns_success_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = LocalRunner().run(
                SandboxCommand(
                    name="python",
                    args=[sys.executable, "-c", "print('hello')"],
                    cwd=temp_dir,
                    timeout_seconds=5,
                    max_output_bytes=1000,
                )
            )

            self.assertTrue(result.success)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.output, "hello")

    def test_local_runner_returns_failure_without_raising(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = LocalRunner().run(
                SandboxCommand(
                    name="python",
                    args=[sys.executable, "-c", "import sys; print('bad'); sys.exit(3)"],
                    cwd=temp_dir,
                    timeout_seconds=5,
                    max_output_bytes=1000,
                )
            )

            self.assertFalse(result.success)
            self.assertEqual(result.exit_code, 3)
            self.assertIn("bad", result.output)

    def test_local_runner_reports_timeout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = LocalRunner().run(
                SandboxCommand(
                    name="python",
                    args=[sys.executable, "-c", "import time; time.sleep(2)"],
                    cwd=temp_dir,
                    timeout_seconds=1,
                    max_output_bytes=1000,
                )
            )

            self.assertFalse(result.success)
            self.assertTrue(result.timed_out)
            self.assertIn("timed out", result.error)

    def test_local_runner_truncates_large_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = LocalRunner().run(
                SandboxCommand(
                    name="python",
                    args=[sys.executable, "-c", "print('x' * 2000)"],
                    cwd=temp_dir,
                    timeout_seconds=5,
                    max_output_bytes=100,
                )
            )

            self.assertTrue(result.success)
            self.assertIn("output truncated", result.output)


if __name__ == "__main__":
    unittest.main()
