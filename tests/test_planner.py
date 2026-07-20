import unittest

from gangent.models import PlanStepStatus, TaskInput
from gangent.planner_contract import (
    PlanPhaseSpec,
    PlanSpec,
    VerificationSpec,
    compile_plan_spec,
    lint_plan_spec,
    plan_spec_from_task_input,
)
from gangent.planner import (
    attach_plan,
    block_current_plan_step,
    complete_current_plan_step,
    create_initial_plan,
    current_plan_step,
    format_plan_for_model,
    infer_task_kind,
    start_current_plan_step,
)
from gangent.state import create_initial_state, create_task
from gangent.task_profile import task_execution_profile


class PlannerTests(unittest.TestCase):
    def _task_and_state(self):
        task_input = TaskInput(
            goal="Create a small Python file",
            user_message="Create a small Python file.",
            workspace_root=".",
        )
        task = create_task(task_input)
        state = create_initial_state(task, task_input)
        return task_input, task, state

    def test_create_initial_plan_has_ordered_steps(self):
        task_input, task, _ = self._task_and_state()

        plan = create_initial_plan(task, task_input)

        self.assertEqual(plan.task_id, task.task_id)
        self.assertGreaterEqual(len(plan.steps), 5)
        self.assertEqual(plan.steps[0].status, PlanStepStatus.TODO)
        self.assertIn("Understand", plan.steps[0].title)
        self.assertIn("allowed_tools", plan.steps[0].description)
        self.assertIn("exit_criteria", plan.steps[0].description)

    def test_plan_spec_linter_rejects_missing_required_verification(self):
        spec = PlanSpec(
            task_kind="build",
            risk_level="medium",
            max_plan_steps=3,
            phases=(PlanPhaseSpec("inspect", "Inspect files.", 2, ("read_file",), "Files inspected."),),
            verification=VerificationSpec(required=True),
        )

        report = lint_plan_spec(spec)

        self.assertFalse(report.valid)
        self.assertTrue(any(finding.code == "missing_required_verification" for finding in report.findings))

    def test_plan_spec_compiler_adds_phase_contract_to_steps(self):
        task_input, task, _ = self._task_and_state()
        spec = plan_spec_from_task_input(task_input, "build")

        plan = compile_plan_spec(task, spec)

        self.assertEqual(plan.task_id, task.task_id)
        self.assertIn("max_steps=", plan.steps[0].description)

    def test_attach_and_start_current_plan_step(self):
        task_input, task, state = self._task_and_state()
        plan = create_initial_plan(task, task_input)

        attach_plan(state, plan)
        step = start_current_plan_step(state)

        self.assertEqual(state.plan_id, plan.plan_id)
        self.assertIsNotNone(step)
        self.assertEqual(step.status, PlanStepStatus.RUNNING)
        self.assertEqual(current_plan_step(state), step)

    def test_complete_current_plan_step_moves_to_next_step(self):
        task_input, task, state = self._task_and_state()
        attach_plan(state, create_initial_plan(task, task_input))
        first = start_current_plan_step(state)

        complete_current_plan_step(state, "Listed files successfully.")
        second = current_plan_step(state)

        self.assertEqual(first.status, PlanStepStatus.DONE)
        self.assertIn("Listed files", first.result_summary)
        self.assertNotEqual(first.step_id, second.step_id)
        self.assertEqual(second.status, PlanStepStatus.TODO)

    def test_block_current_plan_step_records_reason(self):
        task_input, task, state = self._task_and_state()
        attach_plan(state, create_initial_plan(task, task_input))
        step = start_current_plan_step(state)

        block_current_plan_step(state, "User approval unavailable.")

        self.assertEqual(step.status, PlanStepStatus.BLOCKED)
        self.assertIn("approval", step.result_summary)

    def test_format_plan_for_model_marks_current_step(self):
        task_input, task, state = self._task_and_state()
        attach_plan(state, create_initial_plan(task, task_input))
        start_current_plan_step(state)

        text = format_plan_for_model(state)

        self.assertIn("[current]", text)
        self.assertIn("tool_hint", text)

    def test_infer_task_kind_detects_build(self):
        task_input = TaskInput(
            goal="Build a CLI",
            user_message="Implement a CLI command.",
            workspace_root=".",
        )

        self.assertEqual(infer_task_kind(task_input), "build")

    def test_infer_task_kind_detects_debug(self):
        task_input = TaskInput(
            goal="Fix failing test",
            user_message="Debug the runtime error.",
            workspace_root=".",
        )

        self.assertEqual(infer_task_kind(task_input), "debug")

    def test_direct_task_uses_one_step_plan(self):
        task_input = TaskInput(
            goal="不要调用工具，直接回答：OK",
            user_message="不要调用工具，直接回答：OK",
            workspace_root=".",
        )
        task = create_task(task_input)

        plan = create_initial_plan(task, task_input)

        self.assertEqual(infer_task_kind(task_input), "direct")
        self.assertEqual(len(plan.steps), 1)
        self.assertIn("finish_task", plan.steps[0].description)

    def test_single_file_write_uses_short_plan(self):
        task_input = TaskInput(
            goal="Create result.txt with OK",
            user_message="Create result.txt with OK",
            workspace_root=".",
        )
        task = create_task(task_input)

        plan = create_initial_plan(task, task_input)

        self.assertEqual(infer_task_kind(task_input), "single_write")
        self.assertEqual(len(plan.steps), 2)
        self.assertIn("write_file", plan.steps[0].description)

    def test_read_write_task_uses_read_then_write_plan(self):
        task_input = TaskInput(
            goal="读取 source_notes_cn.md，保存为 summary_cn.md",
            user_message="读取 source_notes_cn.md，保存为 summary_cn.md",
            workspace_root=".",
        )
        task = create_task(task_input)

        plan = create_initial_plan(task, task_input)

        self.assertEqual(infer_task_kind(task_input), "read_write")
        self.assertEqual(len(plan.steps), 3)
        self.assertIn("read_file", plan.steps[0].description)
        self.assertNotIn("file_info", plan.steps[0].description)
        self.assertIn("write_file", plan.steps[1].description)

    def test_read_analysis_task_uses_read_only_plan(self):
        message = (
            "\u8bfb\u53d6 README.md \u548c docs/agent_framework_layering.md\uff0c"
            "\u603b\u7ed3 Gangent \u7684\u5b9a\u4f4d\u548c\u6838\u5fc3\u6a21\u5757\uff0c"
            "\u4e0d\u8981\u4fee\u6539\u6587\u4ef6\u3002"
        )
        task_input = TaskInput(goal=message, user_message=message, workspace_root=".")
        task = create_task(task_input)

        profile = task_execution_profile(task_input)
        plan = create_initial_plan(task, task_input)

        self.assertEqual(profile.name, "read_analysis")
        self.assertEqual(infer_task_kind(task_input), "read_analysis")
        self.assertNotIn("write_file", profile.tool_names or ())
        self.assertEqual(len(plan.steps), 2)
        self.assertIn("read_many_files", plan.steps[0].description)
        self.assertIn("file_info", plan.steps[0].description)
        self.assertIn("grep_files", plan.steps[0].description)
        self.assertNotIn("write_file", plan.steps[0].description)
        self.assertIn("read_file", plan.steps[1].description)
        self.assertIn("read_many_files", plan.steps[1].description)
        self.assertIn("finish_task", plan.steps[1].description)

    def test_code_read_analysis_without_explicit_file_path_uses_read_only_plan(self):
        message = "通过真实读取 planner 的代码再给出结论，不要修改文件。"
        task_input = TaskInput(goal=message, user_message=message, workspace_root=".")
        task = create_task(task_input)

        profile = task_execution_profile(task_input)
        plan = create_initial_plan(task, task_input)

        self.assertEqual(profile.name, "read_analysis")
        self.assertEqual(infer_task_kind(task_input), "read_analysis")
        self.assertIn("list_files", profile.tool_names or ())
        self.assertIn("grep_files", profile.tool_names or ())
        self.assertNotIn("write_file", profile.tool_names or ())
        self.assertIn("list_files", plan.steps[0].description)
        self.assertIn("grep_files", plan.steps[0].description)
        self.assertIn("finish_task", plan.steps[1].description)

    def test_chinese_summarize_to_file_is_read_write(self):
        message = "\u8bfb\u53d6 source_notes.md\uff0c\u603b\u7ed3\u4e3a summary_notes.md\u3002"
        task_input = TaskInput(goal=message, user_message=message, workspace_root=".")
        task = create_task(task_input)

        profile = task_execution_profile(task_input)
        plan = create_initial_plan(task, task_input)

        self.assertEqual(profile.name, "read_write")
        self.assertIn("write_file", profile.tool_names or ())
        self.assertEqual(infer_task_kind(task_input), "read_write")
        self.assertIn("read_file", plan.steps[0].description)
        self.assertIn("write_file", plan.steps[1].description)

    def test_write_target_with_other_files_guard_is_read_write(self):
        message = (
            "Read source.md and write summary.md with a concise summary. "
            "Do not modify other files."
        )
        task_input = TaskInput(goal=message, user_message=message, workspace_root=".")
        task = create_task(task_input)

        profile = task_execution_profile(task_input)
        plan = create_initial_plan(task, task_input)

        self.assertEqual(profile.name, "read_write")
        self.assertIn("write_file", profile.tool_names or ())
        self.assertEqual(infer_task_kind(task_input), "read_write")
        self.assertIn("read_file", plan.steps[0].description)
        self.assertIn("write_file", plan.steps[1].description)

    def test_git_analysis_uses_read_only_git_plan(self):
        message = "\u67e5\u770b\u5f53\u524d Git \u72b6\u6001\uff0c\u8bf4\u660e\u662f\u5426\u6709\u672a\u63d0\u4ea4\u6539\u52a8\uff0c\u4e0d\u8981\u4fee\u6539\u6587\u4ef6\u3002"
        task_input = TaskInput(goal=message, user_message=message, workspace_root=".")
        task = create_task(task_input)

        profile = task_execution_profile(task_input)
        plan = create_initial_plan(task, task_input)

        self.assertEqual(profile.name, "git_analysis")
        self.assertEqual(infer_task_kind(task_input), "git_analysis")
        self.assertIn("git_status", profile.tool_names or ())
        self.assertNotIn("write_file", profile.tool_names or ())
        self.assertIn("git_log", plan.steps[0].description)
        self.assertIn("git_diff", plan.steps[1].description)
        self.assertIn("finish_task", plan.steps[1].description)


if __name__ == "__main__":
    unittest.main()
