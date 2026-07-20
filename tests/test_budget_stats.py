import tempfile
import unittest
from pathlib import Path

from gangent.adaptive_runtime import apply_budget_recommendation, resolve_budget
from gangent.budget_stats import (
    BudgetSample,
    append_budget_sample,
    classify_budget_task,
    evaluate_planner_quality,
    planner_feedback_for_task,
    recommend_budget,
    sample_from_result,
)
from gangent.models import RuntimeStats, TaskInput, TaskStatus
from gangent.planner import attach_plan, complete_current_plan_step, create_initial_plan, start_current_plan_step
from gangent.state import create_initial_state, create_task


class BudgetStatsTests(unittest.TestCase):
    def test_classify_write_task(self):
        task_input = TaskInput(
            goal="write a document and save it",
            user_message="write a document and save it",
            workspace_root=".",
        )

        self.assertIn("write", classify_budget_task(task_input))

    def test_recommend_budget_uses_successful_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "history.json"
            task_input = TaskInput(
                goal="write a document",
                user_message="write a document",
                workspace_root=temp_dir,
            )
            for steps in [4, 6, 8, 10]:
                append_budget_sample(
                    BudgetSample(
                        task_kind=classify_budget_task(task_input),
                        success=True,
                        status="completed",
                        step_count=steps,
                        tool_call_count=2,
                        duration_seconds=float(steps),
                        total_tokens=steps * 100,
                    ),
                    path,
                )

            recommendation = recommend_budget(task_input, path, min_samples=3)

            self.assertIsNotNone(recommendation)
            self.assertEqual(recommendation.sample_count, 4)
            self.assertGreaterEqual(recommendation.steps_p80, 8)
            self.assertGreaterEqual(recommendation.tokens_per_step_p80, 0)

    def test_sample_from_result_records_planner_parameters(self):
        task_input = TaskInput(
            goal="write a document",
            user_message="write a document",
            workspace_root=".",
        )
        task = create_task(task_input)
        state = create_initial_state(task, task_input)
        attach_plan(state, create_initial_plan(task, task_input))
        start_current_plan_step(state)
        complete_current_plan_step(state, "done")
        state.budget_profile = "medium"
        state.runtime_step_limit = 12
        state.total_step_budget = 48
        state.total_remaining_steps = 36

        sample = sample_from_result(
            task_input,
            TaskStatus.COMPLETED,
            RuntimeStats(step_count=4, tool_call_count=2, duration_seconds=1.0, usage={"total_tokens": 800}),
            [],
            state,
        )

        self.assertEqual(sample.budget_profile, "medium")
        self.assertEqual(sample.planned_step_count, len(state.plan_steps))
        self.assertEqual(sample.completed_plan_step_count, 1)
        self.assertEqual(sample.runtime_step_limit, 12)
        self.assertEqual(sample.avg_tokens_per_step, 200)

    def test_sample_from_result_records_deepseek_cache_metrics(self):
        task_input = TaskInput(
            goal="inspect files",
            user_message="inspect files",
            workspace_root=".",
        )

        sample = sample_from_result(
            task_input,
            TaskStatus.COMPLETED,
            RuntimeStats(
                step_count=2,
                tool_call_count=1,
                duration_seconds=1.0,
                usage={
                    "total_tokens": 1000,
                    "prompt_cache_hit_tokens": 800,
                    "prompt_cache_miss_tokens": 200,
                },
            ),
            [],
        )

        self.assertEqual(sample.prompt_cache_hit_tokens, 800)
        self.assertEqual(sample.prompt_cache_miss_tokens, 200)
        self.assertEqual(sample.prompt_cache_hit_ratio, 0.8)

    def test_apply_budget_recommendation_raises_budget(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_input = TaskInput(
                goal="write a document",
                user_message="write a document",
                workspace_root=temp_dir,
            )
            budget = resolve_budget(task_input, profile="light")
            recommendation = type(
                "Recommendation",
                (),
                {
                    "steps_p80": 24,
                    "seconds_p80": 100,
                    "tokens_p80": 2500,
                },
            )()

            raised = apply_budget_recommendation(budget, recommendation)

            self.assertGreater(raised.total_steps, budget.total_steps)
            self.assertGreater(raised.max_tokens, budget.max_tokens)

    def test_evaluate_planner_quality_detects_coarse_plan_and_tight_budget(self):
        sample = BudgetSample(
            task_kind="build:write",
            success=False,
            status="failed",
            step_count=10,
            tool_call_count=7,
            duration_seconds=30.0,
            planned_step_count=3,
            completed_plan_step_count=1,
            runtime_step_limit=10,
            total_step_budget=10,
            total_remaining_steps=0,
            avg_tokens_per_step=2500,
        )

        report = evaluate_planner_quality(sample)

        self.assertEqual(report.granularity, "too_coarse")
        self.assertEqual(report.budget_fit, "tight")
        self.assertEqual(report.token_fit, "high_context_cost")
        self.assertIn("high_tokens_per_step", report.findings)
        self.assertTrue(report.recommendations)

    def test_planner_feedback_for_task_summarizes_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "history.json"
            task_input = TaskInput("write a document", "write a document", temp_dir)
            for steps in [4, 6, 8]:
                append_budget_sample(
                    BudgetSample(
                        task_kind=classify_budget_task(task_input),
                        success=True,
                        status="completed",
                        step_count=steps,
                        tool_call_count=2,
                        duration_seconds=float(steps),
                        total_tokens=steps * 100,
                        planned_step_count=5,
                        completed_plan_step_count=5,
                    ),
                    path,
                )

            feedback = planner_feedback_for_task(task_input, path, min_samples=2)

            self.assertIn("Planner History Feedback", feedback)
            self.assertIn("success_rate", feedback)
            self.assertIn("successful_steps_p80", feedback)


if __name__ == "__main__":
    unittest.main()
