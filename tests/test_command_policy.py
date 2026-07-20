import unittest

from gangent.command_policy import CommandRisk, assess_run_command_args


class CommandPolicyTests(unittest.TestCase):
    def test_allows_low_risk_version_command(self):
        assessment = assess_run_command_args(
            {"args": ["python", "--version"], "cwd": "."}
        )

        self.assertEqual(assessment.risk, CommandRisk.ALLOW)

    def test_escalates_destructive_command(self):
        assessment = assess_run_command_args({"args": ["rm", "note.txt"], "cwd": "."})

        self.assertEqual(assessment.risk, CommandRisk.ESCALATE)
        self.assertIn("Destructive", assessment.reason)

    def test_blocks_shell_host(self):
        assessment = assess_run_command_args(
            {"args": ["powershell", "-Command", "Get-ChildItem"], "cwd": "."}
        )

        self.assertEqual(assessment.risk, CommandRisk.BLOCK)

    def test_blocks_shell_metacharacters(self):
        assessment = assess_run_command_args(
            {"args": ["python", "-c", "print(1); del important.txt"], "cwd": "."}
        )

        self.assertEqual(assessment.risk, CommandRisk.BLOCK)

    def test_escalates_unknown_command(self):
        assessment = assess_run_command_args({"args": ["ruff", "check", "."], "cwd": "."})

        self.assertEqual(assessment.risk, CommandRisk.ESCALATE)


if __name__ == "__main__":
    unittest.main()
