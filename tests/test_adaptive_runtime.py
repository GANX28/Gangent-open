import tempfile
import unittest
from pathlib import Path

from gangent.adaptive_runtime import (
    AdaptiveBudget,
    apply_budget_recommendation,
    format_budget_summary,
    infer_budget_profile,
    resolve_budget,
    run_task_adaptive,
)
from gangent.budget_stats import BudgetRecommendation
from gangent.decision import DecisionParseError
from gangent.llm_client import FakeLLMClient
from gangent.models import ActionDecision, DecisionType, TaskInput, TaskStatus


class AdaptiveRuntimeTests(unittest.TestCase):
    def test_infer_light_profile_for_simple_question(self):
        task_input = TaskInput(
            goal="What tools do you have?",
            user_message="What tools do you have?",
            workspace_root=".",
        )

        self.assertEqual(infer_budget_profile(task_input), "light")

    def test_infer_heavy_profile_for_complex_analysis(self):
        task_input = TaskInput(
            goal="Read the agent structure and compare it with commercial systems, then propose optimizations.",
            user_message="Read the agent structure and compare it with commercial systems, then propose optimizations.",
            workspace_root=".",
        )

        self.assertEqual(infer_budget_profile(task_input), "heavy")

    def test_resolve_budget_supports_overrides(self):
        task_input = TaskInput(
            goal="Inspect workspace",
            user_message="Inspect workspace",
            workspace_root=".",
        )

        budget = resolve_budget(task_input, max_steps=10, max_tokens=2000, max_seconds=75)

        self.assertEqual(budget.total_steps, 10)
        self.assertEqual(budget.max_tokens, 2000)
        self.assertEqual(budget.total_seconds, 75)

    def test_resolve_budget_uses_small_budget_for_direct_task(self):
        task_input = TaskInput(
            goal="不要调用工具，直接回答：OK",
            user_message="不要调用工具，直接回答：OK",
            workspace_root=".",
        )

        budget = resolve_budget(task_input, profile="light")

        self.assertEqual(budget.profile, "light:direct")
        self.assertEqual(budget.total_steps, 2)
        self.assertEqual(budget.max_tokens, 160)

    def test_manual_budget_override_wins_over_execution_profile(self):
        task_input = TaskInput(
            goal="不要调用工具，直接回答：OK",
            user_message="不要调用工具，直接回答：OK",
            workspace_root=".",
        )

        budget = resolve_budget(task_input, profile="light", max_tokens=800)

        self.assertEqual(budget.max_tokens, 800)

    def test_partial_manual_step_override_still_keeps_small_token_budget(self):
        task_input = TaskInput(
            goal="不要调用工具，直接回答：OK",
            user_message="不要调用工具，直接回答：OK",
            workspace_root=".",
        )

        budget = resolve_budget(task_input, profile="light", max_steps=5)

        self.assertEqual(budget.total_steps, 5)
        self.assertEqual(budget.max_tokens, 160)

    def test_read_write_task_gets_compact_transformation_budget(self):
        task_input = TaskInput(
            goal="读取 source_notes_cn.md，保存为 summary_cn.md",
            user_message="读取 source_notes_cn.md，保存为 summary_cn.md",
            workspace_root=".",
        )

        budget = resolve_budget(task_input, profile="medium")

        self.assertEqual(budget.profile, "medium:read_write")
        self.assertEqual(budget.total_steps, 6)
        self.assertEqual(budget.max_tokens, 1200)

    def test_single_write_task_gets_enough_budget_to_draft_content(self):
        message = "在 workspace/stability_retest.md 写一份稳定性复测说明，不要修改其他文件。"
        task_input = TaskInput(goal=message, user_message=message, workspace_root=".")

        budget = resolve_budget(task_input, profile="medium")

        self.assertEqual(budget.profile, "medium:single_write")
        self.assertEqual(budget.total_steps, 6)
        self.assertEqual(budget.segment_steps, (6,))
        self.assertEqual(budget.max_tokens, 1400)

    def test_read_analysis_task_gets_summary_budget(self):
        message = (
            "\u8bfb\u53d6 README.md \u548c docs/agent_framework_layering.md\uff0c"
            "\u603b\u7ed3 Gangent \u7684\u5b9a\u4f4d\u548c\u6838\u5fc3\u6a21\u5757\uff0c"
            "\u4e0d\u8981\u4fee\u6539\u6587\u4ef6\u3002"
        )
        task_input = TaskInput(goal=message, user_message=message, workspace_root=".")

        budget = resolve_budget(task_input, profile="medium")

        self.assertEqual(budget.profile, "medium:read_analysis")
        self.assertEqual(budget.total_steps, 12)
        self.assertEqual(budget.segment_steps, (12,))
        self.assertEqual(budget.max_tokens, 2200)

    def test_git_analysis_task_gets_compact_git_budget(self):
        message = "\u67e5\u770b\u5f53\u524d Git \u72b6\u6001\uff0c\u8bf4\u660e\u662f\u5426\u6709\u672a\u63d0\u4ea4\u6539\u52a8\uff0c\u4e0d\u8981\u4fee\u6539\u6587\u4ef6\u3002"
        task_input = TaskInput(goal=message, user_message=message, workspace_root=".")

        budget = resolve_budget(task_input, profile="medium")

        self.assertEqual(budget.profile, "medium:git_analysis")
        self.assertEqual(budget.total_steps, 8)
        self.assertEqual(budget.segment_steps, (8,))
        self.assertEqual(budget.max_tokens, 1800)

    def test_run_task_adaptive_continues_after_segment_max_steps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_input = TaskInput(
                goal="Inspect and then finish",
                user_message="Inspect and then finish",
                workspace_root=temp_dir,
            )
            decisions = iter(
                [
                    ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Inspect workspace.",
                        tool_name="list_files",
                        tool_args={"path": "."},
                    ),
                    ActionDecision(
                        decision_type=DecisionType.FINISH,
                        reason="Enough information.",
                        response_text="Done.",
                    ),
                ]
            )

            def client_factory(token_budget: int):
                return FakeLLMClient(next(decisions))

            budget = AdaptiveBudget(
                profile="test",
                total_steps=2,
                segment_steps=(1, 1),
                total_seconds=120.0,
                segment_seconds=(60.0, 60.0),
                max_tokens=1000,
                retry_max_tokens=2000,
            )
            result = run_task_adaptive(task_input, client_factory, budget)

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertGreaterEqual(result.stats.step_count, 2)
            self.assertEqual(result.state.last_decision.response_text, "Done.")

    def test_run_task_adaptive_retries_json_error_once_with_higher_tokens(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_input = TaskInput(
                goal="Explain tools",
                user_message="Explain tools",
                workspace_root=temp_dir,
            )
            calls: list[int] = []

            class FlakyClient:
                def __init__(self, token_budget: int) -> None:
                    self.token_budget = token_budget
                    self.last_usage = {}

                def decide(self, model_input):
                    calls.append(self.token_budget)
                    if len(calls) == 1:
                        raise RuntimeError("Function arguments are not valid JSON: Unterminated string")
                    return ActionDecision(
                        decision_type=DecisionType.FINISH,
                        reason="Enough information.",
                        response_text="Done.",
                    )

            budget = resolve_budget(task_input, profile="light")
            result = run_task_adaptive(task_input, lambda token_budget: FlakyClient(token_budget), budget)

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertEqual(len(calls), 2)
            self.assertGreater(calls[1], calls[0])

    def test_run_task_adaptive_stops_after_unstable_segment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_input = TaskInput(
                goal="不要调用工具，直接回答：OK",
                user_message="不要调用工具，直接回答：OK",
                workspace_root=temp_dir,
            )
            calls = 0

            class BadLoopClient:
                last_usage = {"total_tokens": 1000}

                def decide(self, model_input):
                    nonlocal calls
                    calls += 1
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Wrong tool.",
                        tool_name="write_file",
                        tool_args={"path": "out.txt", "content": "bad"},
                    )

            budget = AdaptiveBudget(
                profile="test",
                total_steps=6,
                segment_steps=(3, 3),
                total_seconds=120.0,
                segment_seconds=(60.0, 60.0),
                max_tokens=1000,
                retry_max_tokens=2000,
            )

            result = run_task_adaptive(task_input, lambda token_budget: BadLoopClient(), budget)

            self.assertEqual(result.task.status, TaskStatus.FAILED)
            self.assertEqual(calls, 3)
            self.assertTrue(any("Plan guard" in error for error in result.state.errors))

    def test_run_task_adaptive_stops_after_repeated_read_loop(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "README.md").write_text("hello", encoding="utf-8")
            task_input = TaskInput(
                goal="Read README.md and summarize it.",
                user_message="Read README.md and summarize it.",
                workspace_root=temp_dir,
            )
            calls = 0

            class RepeatedReadClient:
                last_usage = {"total_tokens": 1000}

                def decide(self, model_input):
                    nonlocal calls
                    calls += 1
                    return ActionDecision(
                        DecisionType.TOOL_CALL,
                        "Read README again.",
                        "read_file",
                        {"path": "README.md"},
                    )

            budget = AdaptiveBudget(
                profile="test",
                total_steps=6,
                segment_steps=(3, 3),
                total_seconds=120.0,
                segment_seconds=(60.0, 60.0),
                max_tokens=1000,
                retry_max_tokens=2000,
            )

            result = run_task_adaptive(task_input, lambda token_budget: RepeatedReadClient(), budget)

            self.assertEqual(result.task.status, TaskStatus.FAILED)
            self.assertEqual(calls, 3)
            self.assertFalse(any("Repeat guard" in error for error in result.state.errors))
            self.assertTrue(
                any("already been read multiple times" in message.content for message in result.state.messages)
            )

    def test_adaptive_runtime_continues_high_token_segment_with_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_input = TaskInput(
                goal="Create a.txt b.txt c.txt with ordinary file tools.",
                user_message="Create a.txt b.txt c.txt with ordinary file tools.",
                workspace_root=temp_dir,
            )
            calls = 0

            class ProgressThenParseClient:
                last_usage = {"total_tokens": 15000}

                def decide(self, model_input):
                    nonlocal calls
                    calls += 1
                    self.last_usage = {"total_tokens": 15000}
                    if calls == 1:
                        return ActionDecision(DecisionType.TOOL_CALL, "Write a.", "write_file", {"path": "a.txt", "content": "a"})
                    if calls == 2:
                        raise DecisionParseError("plain text tool request")
                    if calls == 3:
                        return ActionDecision(DecisionType.TOOL_CALL, "Write b.", "write_file", {"path": "b.txt", "content": "b"})
                    if calls == 4:
                        return ActionDecision(DecisionType.TOOL_CALL, "Write c.", "write_file", {"path": "c.txt", "content": "c"})
                    return ActionDecision(DecisionType.FINISH, "Done.", response_text="Done.")

            budget = AdaptiveBudget(
                profile="test",
                total_steps=6,
                segment_steps=(3, 3),
                total_seconds=120.0,
                segment_seconds=(60.0, 60.0),
                max_tokens=1000,
                retry_max_tokens=2000,
            )

            result = run_task_adaptive(task_input, lambda token_budget: ProgressThenParseClient(), budget)

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertEqual(calls, 4)
            self.assertTrue((Path(temp_dir) / "c.txt").exists())

    def test_adaptive_budget_control_reaches_model_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_input = TaskInput(
                goal="Inspect workspace",
                user_message="Inspect workspace.",
                workspace_root=temp_dir,
            )
            seen_messages: list[str] = []

            class CapturingClient:
                last_usage = {}

                def decide(self, model_input):
                    seen_messages.append("\n".join(message["content"] for message in model_input.messages))
                    return ActionDecision(
                        decision_type=DecisionType.FINISH,
                        reason="Enough information.",
                        response_text="Done.",
                    )

            budget = resolve_budget(task_input, profile="medium")
            result = run_task_adaptive(task_input, lambda token_budget: CapturingClient(), budget)

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertIn("Planner Budget Control", seen_messages[0])
            self.assertIn("profile: medium", seen_messages[0])
            self.assertIn("total_step_budget: 48", seen_messages[0])

    def test_format_budget_summary_contains_profile(self):
        task_input = TaskInput(
            goal="Inspect workspace",
            user_message="Inspect workspace",
            workspace_root=".",
        )
        budget = resolve_budget(task_input, profile="medium")

        summary = format_budget_summary(budget)

        self.assertIn("profile=medium", summary)

    def test_ultra_profile_budget_is_available(self):
        task_input = TaskInput(
            goal="Inspect workspace",
            user_message="Inspect workspace",
            workspace_root=".",
        )

        budget = resolve_budget(task_input, profile="ultra")

        self.assertEqual(budget.profile, "ultra")
        self.assertEqual(budget.total_steps, 220)
        self.assertEqual(budget.max_tokens, 28000)

    def test_budget_profiles_have_larger_step_headroom(self):
        task_input = TaskInput(
            goal="Inspect workspace",
            user_message="Inspect workspace",
            workspace_root=".",
        )

        self.assertEqual(resolve_budget(task_input, profile="light").total_steps, 20)
        self.assertEqual(resolve_budget(task_input, profile="medium").total_steps, 48)
        self.assertEqual(resolve_budget(task_input, profile="heavy").total_steps, 96)

    def test_history_recommendation_caps_token_budget(self):
        task_input = TaskInput(
            goal="Inspect workspace",
            user_message="Inspect workspace",
            workspace_root=".",
        )
        budget = resolve_budget(task_input, profile="medium")
        recommendation = BudgetRecommendation(
            task_kind="analysis",
            sample_count=3,
            steps_p50=16,
            steps_p80=16,
            steps_p95=16,
            seconds_p50=120.0,
            seconds_p80=220.0,
            seconds_p95=300.0,
            tokens_p50=50_000,
            tokens_p80=100_000,
            tokens_p95=120_000,
        )

        recommended = apply_budget_recommendation(budget, recommendation)

        self.assertEqual(recommended.max_tokens, 8000)
        self.assertLessEqual(recommended.retry_max_tokens, 12000)

    def test_history_recommendation_is_capped_for_read_analysis_profile(self):
        message = (
            "\u8bfb\u53d6 README.md \u548c docs/agent_framework_layering.md\uff0c"
            "\u603b\u7ed3 Gangent \u7684\u5b9a\u4f4d\u548c\u6838\u5fc3\u6a21\u5757\uff0c"
            "\u4e0d\u8981\u4fee\u6539\u6587\u4ef6\u3002"
        )
        task_input = TaskInput(goal=message, user_message=message, workspace_root=".")
        budget = resolve_budget(task_input, profile="medium")
        recommendation = BudgetRecommendation(
            task_kind="read_analysis",
            sample_count=3,
            steps_p50=48,
            steps_p80=80,
            steps_p95=120,
            seconds_p50=120.0,
            seconds_p80=220.0,
            seconds_p95=300.0,
            tokens_p50=50_000,
            tokens_p80=100_000,
            tokens_p95=120_000,
        )

        recommended = apply_budget_recommendation(budget, recommendation)

        self.assertEqual(recommended.total_steps, 12)
        self.assertEqual(recommended.max_tokens, 4000)


if __name__ == "__main__":
    unittest.main()
