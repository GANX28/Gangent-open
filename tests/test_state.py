import unittest

from gangent.models import ActionDecision, DecisionType, TaskInput, TaskStatus
from gangent.state import (
    advance_step,
    attach_decision,
    create_initial_state,
    create_task,
    start_task,
    state_summary,
)


class StateLifecycleTests(unittest.TestCase):
    def test_create_task_from_input(self):
        task_input = TaskInput(
            goal="Build skeleton",
            user_message="Build the first runtime skeleton.",
            workspace_root=".",
        )

        task = create_task(task_input)

        self.assertTrue(task.task_id.startswith("task_"))
        self.assertEqual(task.goal, "Build skeleton")
        self.assertEqual(task.status, TaskStatus.PENDING)

    def test_initialize_and_start_state(self):
        task_input = TaskInput(
            goal="Build skeleton",
            user_message="Build the first runtime skeleton.",
            workspace_root=".",
        )
        task = create_task(task_input)
        state = create_initial_state(task, task_input)

        task, state = start_task(task, state)

        self.assertEqual(task.status, TaskStatus.RUNNING)
        self.assertEqual(state.task_id, task.task_id)
        self.assertEqual(state.step_index, 0)
        self.assertEqual(len(state.messages), 1)

    def test_attach_decision_and_advance_step(self):
        task_input = TaskInput(
            goal="Build skeleton",
            user_message="Build the first runtime skeleton.",
            workspace_root=".",
        )
        task = create_task(task_input)
        state = create_initial_state(task, task_input)
        decision = ActionDecision(
            decision_type=DecisionType.DIRECT_RESPONSE,
            reason="Task and state are enough for this step.",
        )

        attach_decision(state, decision)
        advance_step(state)

        self.assertEqual(state.last_decision, decision)
        self.assertEqual(state.step_index, 1)
        self.assertIn("errors=0", state_summary(state))


if __name__ == "__main__":
    unittest.main()
