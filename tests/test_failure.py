import unittest

from gangent.failure import (
    FailureReason,
    failure_message,
    is_recoverable_failure,
    recoverable_failure_reason,
)


class FailureReasonTests(unittest.TestCase):
    def test_structured_max_steps_failure_is_recoverable(self):
        errors = [
            failure_message(
                FailureReason.MAX_STEPS,
                "Runtime stopped after reaching max_steps=4.",
            )
        ]

        self.assertTrue(is_recoverable_failure(errors))
        self.assertEqual(recoverable_failure_reason(errors), FailureReason.MAX_STEPS)

    def test_legacy_deadline_text_is_still_recoverable(self):
        errors = ["Runtime deadline exceeded: max_seconds=1"]

        self.assertTrue(is_recoverable_failure(errors))
        self.assertEqual(recoverable_failure_reason(errors), FailureReason.DEADLINE_EXCEEDED)


if __name__ == "__main__":
    unittest.main()
