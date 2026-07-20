import tempfile
import unittest
from pathlib import Path

from gangent.events import (
    AgentEventType,
    EventRuntimeState,
    InterruptAction,
    JsonlEventQueue,
    evaluate_interrupts,
    transition_from_interrupt,
)


class EventQueueTests(unittest.TestCase):
    def test_jsonl_event_queue_appends_and_filters_pending_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = JsonlEventQueue(Path(temp_dir) / "events.jsonl")
            first = queue.append(AgentEventType.USER_INPUT, "new requirement", priority=80)
            second = queue.append(AgentEventType.FILE_CHANGE, "README changed", task_id="task_1")

            all_events = queue.load()
            pending_global = queue.pending(cursor=0, task_id="task_2", created_after=first.created_at)
            pending_task = queue.pending(cursor=0, task_id="task_1")

            self.assertEqual(len(all_events), 2)
            self.assertEqual(pending_global[0].event.event_id, first.event_id)
            self.assertEqual(pending_task[-1].event.event_id, second.event_id)

    def test_event_queue_rejects_secret_like_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = JsonlEventQueue(Path(temp_dir) / "events.jsonl")

            with self.assertRaises(ValueError):
                queue.append(AgentEventType.USER_INPUT, "api_key=sk-1234567890abcdef")

    def test_interrupt_policy_replans_for_high_priority_user_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = JsonlEventQueue(Path(temp_dir) / "events.jsonl")
            queue.append(AgentEventType.USER_INPUT, "change goal", priority=80)

            decision = evaluate_interrupts(queue.pending())

            self.assertEqual(decision.action, InterruptAction.REPLAN)
            self.assertIn("plan revision", decision.reason)

    def test_interrupt_policy_pauses_for_high_priority_audit_warning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = JsonlEventQueue(Path(temp_dir) / "events.jsonl")
            queue.append(AgentEventType.AUDIT_WARNING, "possible secret leak", priority=90)

            decision = evaluate_interrupts(queue.pending())

            self.assertEqual(decision.action, InterruptAction.PAUSE)

    def test_replan_request_maps_to_replanning_transition(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = JsonlEventQueue(Path(temp_dir) / "events.jsonl")
            queue.append(AgentEventType.REPLAN_REQUEST, "new requirement", priority=60)

            decision = evaluate_interrupts(queue.pending())
            transition = transition_from_interrupt(EventRuntimeState.EXECUTING, decision)

            self.assertEqual(decision.action, InterruptAction.REPLAN)
            self.assertEqual(transition.to_state, EventRuntimeState.REPLANNING)
            self.assertTrue(transition.reversible)

    def test_rollback_request_requires_user_approval(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = JsonlEventQueue(Path(temp_dir) / "events.jsonl")
            queue.append(AgentEventType.ROLLBACK_REQUEST, "rollback last write", priority=90)

            decision = evaluate_interrupts(queue.pending())

            self.assertEqual(decision.action, InterruptAction.ASK_USER)


if __name__ == "__main__":
    unittest.main()
