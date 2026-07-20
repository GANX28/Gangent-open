import tempfile
import unittest
from pathlib import Path

from gangent.events import AgentEventType, JsonlEventQueue
from gangent.models import AgentState, PlanStep, PlanStepStatus, Task, TaskInput
from gangent.replanning import (
    PlanPatchAction,
    apply_plan_patch,
    build_replan_context,
    plan_patch_from_events,
)


class ReplanningTests(unittest.TestCase):
    def test_high_priority_user_input_replaces_only_unfinished_steps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = JsonlEventQueue(Path(temp_dir) / "events.jsonl")
            queue.append(AgentEventType.USER_INPUT, "Now require a table output too.", priority=80)
            task_input = TaskInput(goal="Review project files", user_message="Extract fields.", workspace_root=temp_dir)
            task = Task(task_id="task_1", goal=task_input.goal)
            state = AgentState(
                task_id=task.task_id,
                plan_steps=[
                    PlanStep("step_1", "Read source files", status=PlanStepStatus.DONE, result_summary="read manual"),
                    PlanStep("step_2", "Extract fields", status=PlanStepStatus.RUNNING),
                    PlanStep("step_3", "Write report", status=PlanStepStatus.TODO),
                ],
            )

            events = queue.pending()
            context = build_replan_context(task_input, task, state, events)
            patch = plan_patch_from_events(context, events)
            apply_plan_patch(state, patch)

            self.assertEqual(patch.action, PlanPatchAction.REPLACE_PENDING_STEPS)
            self.assertEqual(state.plan_steps[0].status, PlanStepStatus.DONE)
            self.assertEqual(state.plan_steps[1].status, PlanStepStatus.BLOCKED)
            self.assertEqual(state.plan_steps[2].status, PlanStepStatus.BLOCKED)
            self.assertTrue(any("user input" in step.title.lower() for step in state.plan_steps))
            self.assertTrue(any("allowed_tools=" in step.description for step in state.plan_steps))
            self.assertTrue(any("write_file" in step.description for step in state.plan_steps))

    def test_new_file_event_appends_read_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = JsonlEventQueue(Path(temp_dir) / "events.jsonl")
            queue.append(AgentEventType.NEW_FILE_ADDED, "inputs/manual.pdf arrived", priority=50)
            task_input = TaskInput(goal="Review project files", user_message="Extract fields.", workspace_root=temp_dir)
            task = Task(task_id="task_1", goal=task_input.goal)
            state = AgentState(
                task_id=task.task_id,
                plan_steps=[PlanStep("step_1", "Extract fields", status=PlanStepStatus.TODO)],
            )

            events = queue.pending()
            patch = plan_patch_from_events(build_replan_context(task_input, task, state, events), events)
            apply_plan_patch(state, patch)

            self.assertEqual(patch.action, PlanPatchAction.APPEND_STEPS)
            self.assertTrue(any(step.title == "Read newly arrived source" for step in state.plan_steps))

    def test_event_pressure_enters_stabilization_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = JsonlEventQueue(Path(temp_dir) / "events.jsonl")
            for index in range(9):
                queue.append(AgentEventType.USER_INPUT, f"change {index}", priority=50)
            task_input = TaskInput(goal="Review project files", user_message="Extract fields.", workspace_root=temp_dir)
            task = Task(task_id="task_1", goal=task_input.goal)
            state = AgentState(task_id=task.task_id, plan_steps=[PlanStep("step_1", "Extract fields")])

            events = queue.pending()
            patch = plan_patch_from_events(build_replan_context(task_input, task, state, events), events)
            apply_plan_patch(state, patch)

            self.assertEqual(patch.action, PlanPatchAction.STABILIZE)
            self.assertTrue(state.stabilization_required)
            self.assertTrue(patch.need_user_confirmation)


if __name__ == "__main__":
    unittest.main()
