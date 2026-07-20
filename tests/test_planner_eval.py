import tempfile
import unittest
from pathlib import Path

from gangent.budget_stats import PlannerQualityReport
from gangent.planner_eval import (
    append_planner_evaluation,
    load_planner_evaluations,
    summarize_planner_evaluations,
)


class PlannerEvaluationTests(unittest.TestCase):
    def test_planner_evaluation_roundtrip_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evaluation.jsonl"
            report = PlannerQualityReport(
                task_kind="build:write",
                outcome="success",
                granularity="balanced",
                budget_fit="fit",
                success=True,
                findings=("ok",),
                recommendations=("keep plan bounded",),
            )

            append_planner_evaluation(report, path)
            loaded = load_planner_evaluations(path)
            summary = summarize_planner_evaluations(path)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].task_kind, "build:write")
            self.assertIn("success_rate=1.00", summary)
            self.assertIn("granularity=balanced:1", summary)
            self.assertIn("token_fit=fit:1", summary)


if __name__ == "__main__":
    unittest.main()
