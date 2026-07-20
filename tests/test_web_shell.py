import tempfile
import unittest

from gangent.events import AgentEventType
from gangent.web_shell import ShellConfig, WebShellState, enqueue_user_event, snapshot


class WebShellTests(unittest.TestCase):
    def test_snapshot_reports_workspace_and_empty_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = WebShellState(ShellConfig(workspace_root=temp_dir, provider="fake"))

            data = snapshot(state)

            self.assertFalse(data["running"])
            self.assertEqual(data["workspace_root"], temp_dir)
            self.assertEqual(data["events"], [])

    def test_enqueue_user_event_is_visible_in_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = WebShellState(ShellConfig(workspace_root=temp_dir, provider="fake"))

            event = enqueue_user_event(state, "add table output", AgentEventType.USER_INPUT, 80)
            data = snapshot(state)

            self.assertEqual(event["event_type"], "user_input")
            self.assertEqual(data["events"][0]["content"], "add table output")
            self.assertTrue(any(message["role"] == "user" for message in data["messages"]))

    def test_non_user_event_is_visible_as_event_message(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = WebShellState(ShellConfig(workspace_root=temp_dir, provider="fake"))

            enqueue_user_event(state, "pause now", AgentEventType.USER_INTERRUPT, 90)
            data = snapshot(state)

            self.assertTrue(any(message["role"] == "event" for message in data["messages"]))

    def test_snapshot_includes_transient_activity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = WebShellState(ShellConfig(workspace_root=temp_dir, provider="fake"))
            with state.lock:
                state.running = True
                state.activity = "正在思考..."
                state.activity_kind = "thinking"

            data = snapshot(state)

            self.assertEqual(data["activity"], "正在思考...")
            self.assertEqual(data["activity_kind"], "thinking")


if __name__ == "__main__":
    unittest.main()
