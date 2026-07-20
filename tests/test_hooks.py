import tempfile
import unittest

from gangent.hooks import HookEvent, HookManager
from gangent.llm_client import FakeLLMClient
from gangent.models import ActionDecision, DecisionType, TaskInput
from gangent.runtime import run_task


class HookTests(unittest.TestCase):
    def test_runtime_emits_core_hook_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            events: list[HookEvent] = []
            hooks = HookManager()
            for event in HookEvent:
                hooks.register(event, lambda context, event=event: events.append(context.event))
            task_input = TaskInput(
                goal="Inspect workspace",
                user_message="Inspect workspace",
                workspace_root=temp_dir,
            )

            run_task(
                task_input,
                FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="List files.",
                        tool_name="list_files",
                        tool_args={"path": "."},
                    )
                ),
                max_steps=1,
                hook_manager=hooks,
            )

            self.assertIn(HookEvent.TASK_START, events)
            self.assertIn(HookEvent.BEFORE_MODEL_CALL, events)
            self.assertIn(HookEvent.AFTER_MODEL_CALL, events)
            self.assertIn(HookEvent.BEFORE_TOOL_CALL, events)
            self.assertIn(HookEvent.AFTER_TOOL_CALL, events)
            self.assertIn(HookEvent.TASK_FINISH, events)


if __name__ == "__main__":
    unittest.main()
