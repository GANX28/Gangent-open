import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gangent.decision import DecisionParseError
from gangent.events import AgentEventType, JsonlEventQueue
from gangent.hooks import HookEvent, HookManager
from gangent.llm_client import FakeLLMClient
from gangent.models import ActionDecision, AgentState, DecisionType, PlanStep, PlanStepStatus, TaskInput, TaskStatus
from gangent.runtime import (
    run_task,
    _ensure_output_repair_plan_step,
    _requested_output_paths,
    _requested_source_paths,
    _visible_tool_schemas_for_current_phase,
)
from gangent.checkpoint import checkpoint_from_runtime_result
from gangent.tool_registry import ToolDefinition, ToolRegistry, ToolRisk, ToolSource
from gangent.tool_schema import tool_schema_for_name


class RuntimeLoopTests(unittest.TestCase):
    def test_output_repair_exposes_write_tools_after_finish_guard(self):
        state = AgentState(
            task_id="task_1",
            plan_steps=[
                PlanStep(
                    step_id="step_1",
                    title="Deliver",
                    status=PlanStepStatus.TODO,
                    description="phase=deliver; max_steps=1; allowed_tools=finish_task; exit_criteria=Final answer.",
                )
            ],
        )
        state.errors.append(
            "Finish guard / Validator Layer: the task cannot finish because required outputs are not valid. "
            "outputs/report.md: missing output file."
        )

        schemas = _visible_tool_schemas_for_current_phase(
            ("read_file", "write_file", "edit_file", "finish_task"),
            state,
        )
        names = {schema["name"] for schema in schemas}

        self.assertIn("write_file", names)
        self.assertIn("read_file", names)
        self.assertIn("finish_task", names)

    def test_output_repair_inserts_write_capable_plan_step(self):
        state = AgentState(
            task_id="task_1",
            plan_steps=[
                PlanStep(
                    step_id="step_1",
                    title="Deliver",
                    status=PlanStepStatus.RUNNING,
                    description="phase=deliver; max_steps=1; allowed_tools=finish_task; exit_criteria=Final answer.",
                )
            ],
        )
        state.errors.append(
            "Finish guard / Validator Layer: the task cannot finish because required outputs are not valid. "
            "outputs/report.md: missing output file."
        )

        _ensure_output_repair_plan_step(state)

        self.assertEqual(state.plan_steps[0].status, PlanStepStatus.BLOCKED)
        self.assertEqual(state.plan_steps[1].title, "Repair missing output")
        self.assertIn("allowed_tools=", state.plan_steps[1].description)
        self.assertIn("write_file", state.plan_steps[1].description)

    def test_read_only_task_treats_all_paths_as_sources(self):
        message = (
            "\u8bfb\u53d6 README.md \u548c docs/agent_framework_layering.md\uff0c"
            "\u603b\u7ed3 Gangent \u7684\u5b9a\u4f4d\u548c\u6838\u5fc3\u6a21\u5757\uff0c"
            "\u4e0d\u8981\u4fee\u6539\u6587\u4ef6\u3002"
        )
        task_input = TaskInput(goal=message, user_message=message, workspace_root=".")

        self.assertEqual(
            _requested_source_paths(task_input),
            ["README.md", "docs/agent_framework_layering.md"],
        )
        self.assertEqual(_requested_output_paths(task_input), [])

    def test_finish_text_for_single_missing_output_is_written_as_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            message = (
                "\u5728 workspace/stability_retest.md \u5199\u4e00\u4efd\u7a33\u5b9a\u6027\u590d\u6d4b\u8bf4\u660e\uff0c"
                "\u4e0d\u8981\u4fee\u6539\u5176\u4ed6\u6587\u4ef6\u3002"
            )
            task_input = TaskInput(goal=message, user_message=message, workspace_root=temp_dir)
            result = run_task(
                task_input,
                FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.FINISH,
                        reason="Draft is ready.",
                        response_text="# Stability Retest\n\nThis file records the test purpose and current runtime capability.",
                    )
                ),
                max_steps=2,
            )

            output = Path(temp_dir) / "workspace" / "stability_retest.md"
            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertTrue(output.exists())
            self.assertIn("Stability Retest", output.read_text(encoding="utf-8"))
            self.assertEqual(result.steps[-1].decision.tool_name, "write_file")

    def test_direct_response_text_for_single_missing_output_is_written_as_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            message = (
                "\u5728 workspace/stability_retest.md \u5199\u4e00\u4efd\u7a33\u5b9a\u6027\u590d\u6d4b\u8bf4\u660e\uff0c"
                "\u4e0d\u8981\u4fee\u6539\u5176\u4ed6\u6587\u4ef6\u3002"
            )
            task_input = TaskInput(goal=message, user_message=message, workspace_root=temp_dir)
            result = run_task(
                task_input,
                FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.DIRECT_RESPONSE,
                        reason="Draft is ready.",
                        response_text="# Stability Retest\n\nThis direct response should become the requested file.",
                    )
                ),
                max_steps=2,
            )

            output = Path(temp_dir) / "workspace" / "stability_retest.md"
            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertTrue(output.exists())
            self.assertIn("direct response", output.read_text(encoding="utf-8"))
            self.assertEqual(result.steps[-1].decision.tool_name, "write_file")

    def test_runtime_recovers_from_plain_text_tool_request_parse_error(self):
        class RecoveringClient:
            def __init__(self):
                self.calls = 0
                self.inputs = []

            def decide(self, model_input):
                self.calls += 1
                self.inputs.append(model_input)
                if self.calls == 1:
                    raise DecisionParseError(
                        "DeepSeek response described a tool request in plain text "
                        "instead of returning a structured tool call: read_file."
                    )
                return ActionDecision(
                    decision_type=DecisionType.FINISH,
                    reason="Recovered after parse hint.",
                    response_text="Recovered.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            client = RecoveringClient()
            task_input = TaskInput(
                goal="Inspect workspace",
                user_message="Inspect files.",
                workspace_root=temp_dir,
            )

            result = run_task(task_input, client, max_steps=3)

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertEqual(client.calls, 2)
            self.assertEqual(len(result.steps), 1)
            self.assertTrue(any("Model output parse failed" in error for error in result.state.errors))
            second_messages = " ".join(message["content"] for message in client.inputs[1].messages)
            self.assertIn("Do not answer with plain text", second_messages)

    def test_runtime_executes_one_tool_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("hello", encoding="utf-8")
            client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Inspect workspace.",
                    tool_name="list_files",
                    tool_args={"path": "."},
                )
            )
            task_input = TaskInput(
                goal="Inspect workspace",
                user_message="Inspect files.",
                workspace_root=temp_dir,
            )

            result = run_task(task_input, client, max_steps=1)

            self.assertEqual(len(result.steps), 1)
            self.assertEqual(result.state.step_index, 1)
            self.assertIsNotNone(result.state.last_tool_result)
            self.assertTrue(result.state.last_tool_result.success)
            self.assertIn("README.md", result.state.last_tool_result.output)

    def test_read_analysis_runtime_reads_multiple_files_then_finishes(self):
        class ReadAnalysisClient:
            last_usage = {}

            def __init__(self):
                self.calls = 0
                self.inputs = []

            def decide(self, model_input):
                self.calls += 1
                self.inputs.append(model_input)
                if self.calls == 1:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Read requested source files.",
                        tool_name="read_many_files",
                        tool_args={"paths": ["README.md", "docs/agent_framework_layering.md"]},
                    )
                return ActionDecision(
                    decision_type=DecisionType.FINISH,
                    reason="Source files have been read.",
                    response_text="Gangent is an agent runtime with layered modules.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "docs").mkdir()
            (root / "README.md").write_text("# Gangent\nRuntime overview.", encoding="utf-8")
            (root / "docs" / "agent_framework_layering.md").write_text("Layering notes.", encoding="utf-8")
            message = (
                "\u8bfb\u53d6 README.md \u548c docs/agent_framework_layering.md\uff0c"
                "\u603b\u7ed3 Gangent \u7684\u5b9a\u4f4d\u548c\u6838\u5fc3\u6a21\u5757\uff0c"
                "\u4e0d\u8981\u4fee\u6539\u6587\u4ef6\u3002"
            )
            client = ReadAnalysisClient()
            task_input = TaskInput(goal=message, user_message=message, workspace_root=temp_dir)

            result = run_task(task_input, client, max_steps=3)

            first_tool_names = {tool["name"] for tool in client.inputs[0].tools}
            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertIn("read_many_files", first_tool_names)
            self.assertNotIn("write_file", first_tool_names)
            self.assertEqual([step.decision.tool_name for step in result.steps[:2]], ["read_many_files", None])
            self.assertFalse(any(root.glob("*.tmp")))

    def test_read_analysis_allows_followup_reads_after_initial_evidence(self):
        class FollowupReadClient:
            last_usage = {}

            def __init__(self):
                self.calls = 0

            def decide(self, model_input):
                self.calls += 1
                if self.calls == 1:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Read requested source files.",
                        tool_name="read_many_files",
                        tool_args={"paths": ["README.md", "docs/agent_framework_layering.md"]},
                    )
                if self.calls == 2:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Read the next README chunk because the first read was partial.",
                        tool_name="read_file",
                        tool_args={"path": "README.md", "start_line": 201, "max_lines": 80},
                    )
                if self.calls == 3:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Search supporting context before final synthesis.",
                        tool_name="search_context",
                        tool_args={"query": "Gangent core modules", "top_k": 3},
                    )
                return ActionDecision(
                    decision_type=DecisionType.FINISH,
                    reason="Enough evidence was collected.",
                    response_text="Gangent is a local agent runtime with planning, tool, policy, memory, and audit layers.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "docs").mkdir()
            readme = "\n".join(f"line {index}" for index in range(1, 260))
            (root / "README.md").write_text(readme, encoding="utf-8")
            (root / "docs" / "agent_framework_layering.md").write_text("Layering notes.", encoding="utf-8")
            message = (
                "\u8bfb\u53d6 README.md \u548c docs/agent_framework_layering.md\uff0c"
                "\u603b\u7ed3 Gangent \u7684\u5b9a\u4f4d\u548c\u6838\u5fc3\u6a21\u5757\uff0c"
                "\u4e0d\u8981\u4fee\u6539\u6587\u4ef6\u3002"
            )
            client = FollowupReadClient()
            task_input = TaskInput(goal=message, user_message=message, workspace_root=temp_dir)

            result = run_task(task_input, client, max_steps=5)

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertEqual(
                [step.decision.tool_name for step in result.steps[:4]],
                ["read_many_files", "read_file", "search_context", None],
            )
            self.assertFalse(any("Plan guard" in error for error in result.state.errors))

    def test_read_analysis_keeps_synthesis_phase_open_after_locator_tools(self):
        class LocatorThenReadClient:
            last_usage = {}

            def __init__(self):
                self.calls = 0

            def decide(self, model_input):
                self.calls += 1
                if self.calls == 1:
                    return ActionDecision(
                        DecisionType.TOOL_CALL,
                        "Find planner files.",
                        "search_context",
                        {"query": "planner code", "top_k": 3},
                    )
                if self.calls == 2:
                    return ActionDecision(
                        DecisionType.TOOL_CALL,
                        "List package files.",
                        "list_files",
                        {"path": "gangent"},
                    )
                if self.calls == 3:
                    return ActionDecision(
                        DecisionType.TOOL_CALL,
                        "Read planner implementation.",
                        "read_file",
                        {"path": "gangent/planner.py", "max_lines": 80},
                    )
                return ActionDecision(
                    DecisionType.FINISH,
                    "Enough code evidence.",
                    response_text="Planner evidence summarized.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "gangent").mkdir()
            (root / "gangent" / "planner.py").write_text("def infer_task_kind():\n    return 'analysis'\n", encoding="utf-8")
            message = "通过真实读取 planner 的代码再给出结论，不要修改文件。"
            result = run_task(
                TaskInput(goal=message, user_message=message, workspace_root=temp_dir),
                LocatorThenReadClient(),
                max_steps=4,
            )

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertEqual(
                [step.decision.tool_name for step in result.steps[:4]],
                ["search_context", "list_files", "read_file", None],
            )
            self.assertFalse(any("no active plan phase remains" in error for error in result.state.errors))

    def test_git_analysis_allows_followup_git_evidence_before_finish(self):
        class GitAnalysisClient:
            last_usage = {}

            def __init__(self):
                self.calls = 0

            def decide(self, model_input):
                self.calls += 1
                if self.calls == 1:
                    return ActionDecision(DecisionType.TOOL_CALL, "Check status.", "git_status", {})
                if self.calls == 2:
                    return ActionDecision(DecisionType.TOOL_CALL, "Inspect diff.", "git_diff", {})
                if self.calls == 3:
                    return ActionDecision(DecisionType.TOOL_CALL, "Inspect recent commits.", "git_log", {"limit": 3})
                return ActionDecision(DecisionType.FINISH, "Enough git evidence.", response_text="Git state summarized.")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".git").mkdir()
            registry = ToolRegistry(
                [
                    ToolDefinition("git_status", "status", ToolRisk.READ, ToolSource.LOCAL, lambda args, root: "M file.py", input_schema=tool_schema_for_name("git_status")),
                    ToolDefinition("git_diff", "diff", ToolRisk.READ, ToolSource.LOCAL, lambda args, root: "diff --git", input_schema=tool_schema_for_name("git_diff")),
                    ToolDefinition("git_log", "log", ToolRisk.READ, ToolSource.LOCAL, lambda args, root: "abc123 init", input_schema=tool_schema_for_name("git_log")),
                    ToolDefinition("git_show", "show", ToolRisk.READ, ToolSource.LOCAL, lambda args, root: "commit detail", input_schema=tool_schema_for_name("git_show")),
                ]
            )
            message = "\u67e5\u770b\u5f53\u524d Git \u72b6\u6001\uff0c\u8bf4\u660e\u662f\u5426\u6709\u672a\u63d0\u4ea4\u6539\u52a8\uff0c\u4e0d\u8981\u4fee\u6539\u6587\u4ef6\u3002"
            client = GitAnalysisClient()
            task_input = TaskInput(goal=message, user_message=message, workspace_root=temp_dir)

            result = run_task(task_input, client, max_steps=5, tool_registry=registry)

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertEqual(
                [step.decision.tool_name for step in result.steps[:4]],
                ["git_status", "git_diff", "git_log", None],
            )
            self.assertFalse(any("Plan guard" in error for error in result.state.errors))

    def test_runtime_creates_and_updates_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("hello", encoding="utf-8")
            client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Inspect workspace.",
                    tool_name="list_files",
                    tool_args={"path": "."},
                )
            )
            task_input = TaskInput(
                goal="Inspect workspace",
                user_message="Inspect files.",
                workspace_root=temp_dir,
            )

            result = run_task(task_input, client, max_steps=1)

            self.assertIsNotNone(result.state.plan_id)
            self.assertGreaterEqual(len(result.state.plan_steps), 5)
            self.assertEqual(result.state.plan_steps[0].status, PlanStepStatus.DONE)
            self.assertIn("README.md", result.state.plan_steps[0].result_summary)

    def test_runtime_completes_on_direct_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.DIRECT_RESPONSE,
                    reason="Answer directly.",
                    response_text="Done.",
                )
            )
            task_input = TaskInput(
                goal="Answer",
                user_message="Answer directly.",
                workspace_root=temp_dir,
            )

            result = run_task(task_input, client, max_steps=3)

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertEqual(len(result.steps), 1)

    def test_runtime_completes_on_finish_decision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.FINISH,
                    reason="Enough information.",
                    response_text="Final answer.",
                )
            )
            task_input = TaskInput(
                goal="Finish",
                user_message="Finish now.",
                workspace_root=temp_dir,
            )

            result = run_task(task_input, client, max_steps=3)

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertEqual(result.state.last_decision.response_text, "Final answer.")
            self.assertEqual(len(result.steps), 1)

    def test_runtime_records_policy_block(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Unsafe.",
                    tool_name="read_file",
                    tool_args={"path": "../outside.txt"},
                )
            )
            task_input = TaskInput(
                goal="Unsafe read",
                user_message="Try unsafe read.",
                workspace_root=temp_dir,
            )

            result = run_task(task_input, client, max_steps=1)

            self.assertEqual(len(result.steps), 1)
            self.assertIsNotNone(result.steps[0].policy)
            self.assertFalse(result.steps[0].policy.allowed)
            self.assertTrue(any("escapes workspace root" in error for error in result.state.errors))

    def test_runtime_plan_guard_blocks_tool_outside_current_phase(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Run tests too early.",
                    tool_name="run_tests",
                    tool_args={},
                )
            )
            task_input = TaskInput(
                goal="Build a small feature",
                user_message="Implement a small feature.",
                workspace_root=temp_dir,
            )

            result = run_task(task_input, client, max_steps=1)

            self.assertTrue(any("Plan guard" in error for error in result.state.errors))
            self.assertIsNone(result.steps[0].tool_result)

    def test_runtime_denies_escalated_tool_without_approval(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".hidden").write_text("secret", encoding="utf-8")
            client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Read hidden file.",
                    tool_name="read_file",
                    tool_args={"path": ".hidden"},
                )
            )
            task_input = TaskInput(
                goal="Read hidden",
                user_message="Read hidden.",
                workspace_root=temp_dir,
            )

            result = run_task(task_input, client, max_steps=1)

            self.assertTrue(result.steps[0].approval_required)
            self.assertFalse(result.steps[0].approved)
            self.assertIsNone(result.steps[0].tool_result)
            self.assertTrue(any("approval denied" in error for error in result.state.errors))

    def test_runtime_executes_escalated_tool_after_approval(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".hidden").write_text("secret", encoding="utf-8")
            client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Read hidden file.",
                    tool_name="read_file",
                    tool_args={"path": ".hidden"},
                )
            )
            task_input = TaskInput(
                goal="Read hidden",
                user_message="Read hidden.",
                workspace_root=temp_dir,
            )

            result = run_task(
                task_input,
                client,
                max_steps=1,
                approval_callback=lambda decision, policy: True,
            )

            self.assertTrue(result.steps[0].approval_required)
            self.assertTrue(result.steps[0].approved)
            self.assertIsNotNone(result.steps[0].tool_result)
            self.assertTrue(result.steps[0].tool_result.success)
            self.assertEqual(result.steps[0].tool_result.output, "secret")

    def test_runtime_stops_when_deadline_is_exceeded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Inspect workspace.",
                    tool_name="list_files",
                    tool_args={"path": "."},
                )
            )
            task_input = TaskInput(
                goal="Inspect workspace",
                user_message="Inspect files.",
                workspace_root=temp_dir,
            )

            with patch("gangent.runtime.monotonic", side_effect=[0.0, 2.0, 2.0]):
                result = run_task(task_input, client, max_steps=3, max_seconds=1)

            self.assertEqual(result.task.status, TaskStatus.FAILED)
            self.assertEqual(len(result.steps), 0)
            self.assertIn("deadline exceeded", result.state.errors[0])

    def test_runtime_stats_are_recorded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.DIRECT_RESPONSE,
                    reason="Answer directly.",
                    response_text="Done.",
                )
            )
            task_input = TaskInput(
                goal="Answer",
                user_message="Answer directly.",
                workspace_root=temp_dir,
            )

            result = run_task(task_input, client, max_steps=1)

            self.assertEqual(result.stats.step_count, 1)
            self.assertEqual(result.stats.tool_call_count, 0)
            self.assertEqual(result.stats.error_count, 0)
            self.assertGreaterEqual(result.stats.duration_seconds, 0)

    def test_runtime_marks_failed_when_max_steps_exhausted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Inspect workspace.",
                    tool_name="list_files",
                    tool_args={"path": "."},
                )
            )
            task_input = TaskInput(
                goal="Inspect repeatedly",
                user_message="Inspect repeatedly.",
                workspace_root=temp_dir,
            )

            result = run_task(task_input, client, max_steps=1)

            self.assertEqual(result.task.status, TaskStatus.FAILED)
            self.assertIn("max_steps", result.state.errors[-1])

    def test_runtime_resume_clears_old_recoverable_max_steps_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_input = TaskInput(
                goal="Inspect then finish",
                user_message="Inspect then finish.",
                workspace_root=temp_dir,
            )
            partial = run_task(
                task_input,
                FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Inspect workspace.",
                        tool_name="list_files",
                        tool_args={"path": "."},
                    )
                ),
                max_steps=1,
            )
            checkpoint = checkpoint_from_runtime_result(partial)

            resumed = run_task(
                task_input,
                FakeLLMClient(
                    ActionDecision(
                        decision_type=DecisionType.FINISH,
                        reason="Enough information.",
                        response_text="Done.",
                    )
                ),
                max_steps=1,
                resume_checkpoint=checkpoint,
            )

            self.assertEqual(resumed.task.status, TaskStatus.COMPLETED)
            self.assertFalse(any("max_steps" in error for error in resumed.state.errors))
            self.assertEqual(resumed.resume_report.new_step_count, 1)

    def test_runtime_records_hook_failures_in_state_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            hook_manager = HookManager()

            def broken_hook(context):
                raise RuntimeError("boom")

            hook_manager.register(HookEvent.BEFORE_MODEL_CALL, broken_hook)
            client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.DIRECT_RESPONSE,
                    reason="Answer directly.",
                    response_text="Done.",
                )
            )
            task_input = TaskInput(goal="Answer", user_message="Answer", workspace_root=temp_dir)

            result = run_task(task_input, client, max_steps=1, hook_manager=hook_manager)

            self.assertTrue(any("Hook failed: before_model_call: boom" in error for error in result.state.errors))

    def test_runtime_accepts_injected_tool_registry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ToolRegistry(
                [
                    ToolDefinition(
                        name="list_files",
                        description="Override list_files for testing.",
                        risk=ToolRisk.READ,
                        source=ToolSource.LOCAL,
                        handler=lambda args, root: "custom registry output",
                        input_schema=tool_schema_for_name("list_files"),
                    )
                ]
            )
            client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Use injected tool.",
                    tool_name="list_files",
                    tool_args={"path": "."},
                )
            )
            task_input = TaskInput(goal="Echo", user_message="Echo", workspace_root=temp_dir)

            result = run_task(task_input, client, max_steps=1, tool_registry=registry)

            self.assertTrue(result.state.last_tool_result.success)
            self.assertEqual(result.state.last_tool_result.output, "custom registry output")

    def test_runtime_retries_after_schema_validation_failure(self):
        class RepairingClient:
            last_usage = {}

            def __init__(self):
                self.calls = 0

            def decide(self, model_input):
                self.calls += 1
                if self.calls == 1:
                    return ActionDecision(
                        DecisionType.TOOL_CALL,
                        "Invalid path type.",
                        "list_files",
                        {"path": 3},
                    )
                if self.calls == 2:
                    return ActionDecision(
                        DecisionType.TOOL_CALL,
                        "Corrected path type.",
                        "list_files",
                        {"path": "."},
                    )
                return ActionDecision(
                    DecisionType.FINISH,
                    "Inspection complete.",
                    response_text="Workspace inspected.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "README.md").write_text("hello", encoding="utf-8")
            task_input = TaskInput(
                goal="Inspect the workspace files.",
                user_message="Inspect the workspace files.",
                workspace_root=temp_dir,
            )

            result = run_task(task_input, RepairingClient(), max_steps=3)

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertFalse(result.steps[0].tool_result.success)
            self.assertIn('"error_type":"tool_argument_validation"', result.steps[0].tool_result.error)
            self.assertTrue(result.steps[1].tool_result.success)
            self.assertEqual(result.state.last_decision.response_text, "Workspace inspected.")

    def test_runtime_repeat_guard_blocks_identical_failed_tool_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeLLMClient(
                ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Read missing file.",
                    tool_name="read_file",
                    tool_args={"path": "missing.md"},
                )
            )
            task_input = TaskInput(goal="Read missing", user_message="Read missing", workspace_root=temp_dir)

            result = run_task(task_input, client, max_steps=3)

            self.assertTrue(any("Repeat guard" in error for error in result.state.errors))
            self.assertEqual(sum(1 for step in result.steps if step.policy is not None and not step.policy.allowed), 2)

    def test_runtime_repeated_successful_read_is_hint_not_error(self):
        class RepeatedReadThenFinishClient:
            def __init__(self):
                self.calls = 0

            def decide(self, model_input):
                self.calls += 1
                if self.calls == 1:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Read first source.",
                        tool_name="read_file",
                        tool_args={"path": "README.md"},
                    )
                if self.calls == 2:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Read second source.",
                        tool_name="read_file",
                        tool_args={"path": "docs/agent_framework_layering.md"},
                    )
                if self.calls in {3, 4}:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Read README again.",
                        tool_name="read_file",
                        tool_args={"path": "README.md"},
                    )
                return ActionDecision(
                    decision_type=DecisionType.FINISH,
                    reason="Enough evidence.",
                    response_text="Summary from available evidence.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("Project overview", encoding="utf-8")
            (root / "docs").mkdir()
            (root / "docs" / "agent_framework_layering.md").write_text("Layering notes", encoding="utf-8")
            message = (
                "Read README.md and docs/agent_framework_layering.md, summarize the project, "
                "and do not modify files."
            )
            client = RepeatedReadThenFinishClient()
            task_input = TaskInput(goal=message, user_message=message, workspace_root=temp_dir)

            result = run_task(task_input, client, max_steps=5)

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertFalse(any("Repeat guard" in error for error in result.state.errors))
            self.assertTrue(any("already been read multiple times" in message.content for message in result.state.messages))

    def test_runtime_repairs_module_directory_confusion_to_source_file(self):
        class PlannerReadClient:
            last_usage = {}

            def __init__(self):
                self.calls = 0

            def decide(self, model_input):
                self.calls += 1
                if self.calls == 1:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Inspect planner code.",
                        tool_name="list_files",
                        tool_args={"path": "gangent/planner"},
                    )
                return ActionDecision(
                    decision_type=DecisionType.FINISH,
                    reason="Planner code was read.",
                    response_text="Planner code evidence collected.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "gangent").mkdir()
            (root / "gangent" / "planner.py").write_text("def infer_task_kind():\n    pass\n", encoding="utf-8")
            message = "通过真实读取 planner 的代码再给出结论"
            result = run_task(
                TaskInput(goal=message, user_message=message, workspace_root=temp_dir),
                PlannerReadClient(),
                max_steps=2,
            )

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertEqual(result.steps[0].decision.tool_name, "read_file")
            self.assertEqual(result.steps[0].decision.tool_args["path"], "gangent/planner.py")
            self.assertTrue(result.steps[0].tool_result.success)
            self.assertTrue(any("Runtime repaired an obvious path/tool mismatch" in message.content for message in result.state.messages))

    def test_plan_guard_allows_read_only_recovery_after_missing_file_error(self):
        class MissingThenRecoverClient:
            last_usage = {}

            def __init__(self):
                self.calls = 0

            def decide(self, model_input):
                self.calls += 1
                if self.calls == 1:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Read the requested file.",
                        tool_name="read_file",
                        tool_args={"path": "missing.md"},
                    )
                if self.calls == 2:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Recover by listing the workspace.",
                        tool_name="list_files",
                        tool_args={"path": "."},
                    )
                return ActionDecision(
                    decision_type=DecisionType.FINISH,
                    reason="Reported the missing file with available evidence.",
                    response_text="missing.md does not exist; workspace was inspected.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("hello", encoding="utf-8")
            message = "Read missing.md and answer."
            result = run_task(
                TaskInput(goal=message, user_message=message, workspace_root=temp_dir),
                MissingThenRecoverClient(),
                max_steps=3,
            )

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertIsNotNone(result.steps[1].tool_result)
            self.assertTrue(result.steps[1].tool_result.success)
            self.assertFalse(any("Plan guard" in error and "list_files" in error for error in result.state.errors))

    def test_finish_guard_rejects_unverified_absolute_path_claim(self):
        class BadPathFinishClient:
            last_usage = {}

            def __init__(self):
                self.calls = 0

            def decide(self, model_input):
                self.calls += 1
                if self.calls == 1:
                    return ActionDecision(
                        DecisionType.TOOL_CALL,
                        "Read real planner file.",
                        "read_file",
                        {"path": "gangent/planner.py"},
                    )
                if self.calls == 2:
                    return ActionDecision(
                        DecisionType.FINISH,
                        "Bad path claim.",
                        response_text="I read /workspace/src/agentic/planner_v1.py and confirmed it.",
                    )
                return ActionDecision(
                    DecisionType.FINISH,
                    "Corrected path claim.",
                    response_text="I read gangent/planner.py and confirmed it.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "gangent").mkdir()
            (root / "gangent" / "planner.py").write_text("def create_initial_plan():\n    pass\n", encoding="utf-8")
            message = "通过真实读取 planner 的代码再给出结论，不要修改文件。"
            result = run_task(
                TaskInput(goal=message, user_message=message, workspace_root=temp_dir),
                BadPathFinishClient(),
                max_steps=3,
            )

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertIn("gangent/planner.py", result.state.last_decision.response_text)
            self.assertTrue(any("Final answer guard" in error for error in result.state.errors))

    def test_finish_guard_rejects_missing_relative_path_claim(self):
        class BadRelativePathFinishClient:
            last_usage = {}

            def __init__(self):
                self.calls = 0

            def decide(self, model_input):
                self.calls += 1
                if self.calls == 1:
                    return ActionDecision(
                        DecisionType.TOOL_CALL,
                        "Read planner file.",
                        "read_file",
                        {"path": "gangent/planner.py"},
                    )
                if self.calls == 2:
                    return ActionDecision(
                        DecisionType.FINISH,
                        "Bad relative directory claim.",
                        response_text="The planner depends on `gangent/planner/` for submodules.",
                    )
                return ActionDecision(
                    DecisionType.FINISH,
                    "Corrected relative path claim.",
                    response_text="The planner implementation is in `gangent/planner.py`.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "gangent").mkdir()
            (root / "gangent" / "planner.py").write_text("def create_initial_plan():\n    pass\n", encoding="utf-8")
            message = "Read planner code and explain it."

            result = run_task(
                TaskInput(goal=message, user_message=message, workspace_root=temp_dir),
                BadRelativePathFinishClient(),
                max_steps=3,
            )

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertIn("gangent/planner.py", result.state.last_decision.response_text)
            self.assertTrue(any("gangent/planner/" in error for error in result.state.errors))

    def test_finish_guard_rejects_unverified_python_symbol_claim(self):
        class BadSymbolFinishClient:
            last_usage = {}

            def __init__(self):
                self.calls = 0

            def decide(self, model_input):
                self.calls += 1
                if self.calls == 1:
                    return ActionDecision(
                        DecisionType.TOOL_CALL,
                        "Read planner file.",
                        "read_file",
                        {"path": "gangent/planner.py"},
                    )
                if self.calls == 2:
                    return ActionDecision(
                        DecisionType.FINISH,
                        "Bad symbol claim.",
                        response_text="The planner defines `PlanStage` and `GangentPlanner`.",
                    )
                return ActionDecision(
                    DecisionType.FINISH,
                    "Corrected symbol claim.",
                    response_text="The planner defines `create_initial_plan`.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "gangent").mkdir()
            (root / "gangent" / "planner.py").write_text(
                "def create_initial_plan(task, task_input):\n    return None\n",
                encoding="utf-8",
            )

            result = run_task(
                TaskInput(goal="Read planner code.", user_message="Read planner code.", workspace_root=temp_dir),
                BadSymbolFinishClient(),
                max_steps=3,
            )

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertIn("create_initial_plan", result.state.last_decision.response_text)
            self.assertTrue(any("PlanStage" in error for error in result.state.errors))

    def test_repeated_final_answer_guard_uses_guarded_fallback(self):
        class AlwaysBadSymbolFinishClient:
            last_usage = {}

            def __init__(self):
                self.calls = 0

            def decide(self, model_input):
                self.calls += 1
                if self.calls == 1:
                    return ActionDecision(
                        DecisionType.TOOL_CALL,
                        "Read planner file.",
                        "read_file",
                        {"path": "gangent/planner.py"},
                    )
                return ActionDecision(
                    DecisionType.FINISH,
                    "Still bad.",
                    response_text="The planner defines `PlanStage`, `GangentPlanner`, and `PlannerState`.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "gangent").mkdir()
            (root / "gangent" / "planner.py").write_text(
                "def create_initial_plan(task, task_input):\n    return None\n",
                encoding="utf-8",
            )

            result = run_task(
                TaskInput(goal="Read planner code.", user_message="Read planner code.", workspace_root=temp_dir),
                AlwaysBadSymbolFinishClient(),
                max_steps=6,
            )

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertIn("could not safely accept", result.state.last_decision.response_text)
            self.assertIn("create_initial_plan", result.state.last_decision.response_text)
            self.assertTrue(any("PlanStage" in error for error in result.state.errors))

    def test_finish_guard_rejects_bare_unverified_code_claim_terms(self):
        class BadBareTermClient:
            last_usage = {}

            def __init__(self):
                self.calls = 0

            def decide(self, model_input):
                self.calls += 1
                if self.calls == 1:
                    return ActionDecision(
                        DecisionType.TOOL_CALL,
                        "Read planner file.",
                        "read_file",
                        {"path": "gangent/planner.py"},
                    )
                if self.calls == 2:
                    return ActionDecision(
                        DecisionType.FINISH,
                        "Bad dependency claim.",
                        response_text="The planner sets dependencies and uses pending -> in_progress transitions.",
                    )
                return ActionDecision(
                    DecisionType.FINISH,
                    "Corrected.",
                    response_text="The planner defines `create_initial_plan`.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "gangent").mkdir()
            (root / "gangent" / "planner.py").write_text(
                "def create_initial_plan(task, task_input):\n    return None\n",
                encoding="utf-8",
            )

            result = run_task(
                TaskInput(goal="Read planner code.", user_message="Read planner code.", workspace_root=temp_dir),
                BadBareTermClient(),
                max_steps=4,
            )

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertIn("create_initial_plan", result.state.last_decision.response_text)
            self.assertTrue(any("dependencies" in error for error in result.state.errors))

    def test_model_input_contains_cache_and_budget_diagnostics(self):
        class CapturingClient:
            def __init__(self):
                self.input = None

            def decide(self, model_input):
                self.input = model_input
                return ActionDecision(
                    decision_type=DecisionType.FINISH,
                    reason="Done.",
                    response_text="Done.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            client = CapturingClient()
            task_input = TaskInput(goal="Answer", user_message="Answer", workspace_root=temp_dir)

            run_task(task_input, client, max_steps=1)

            self.assertIsNotNone(client.input)
            self.assertIn("stable_prefix_hash", client.input.diagnostics)
            self.assertIn("estimated_context_tokens", client.input.diagnostics)

    def test_direct_task_exposes_only_finish_tool_and_small_context(self):
        class CapturingClient:
            last_usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}

            def __init__(self):
                self.input = None

            def decide(self, model_input):
                self.input = model_input
                return ActionDecision(
                    decision_type=DecisionType.FINISH,
                    reason="Direct answer.",
                    response_text="OK",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            client = CapturingClient()
            task_input = TaskInput(
                goal="不要调用工具，直接回答：OK",
                user_message="不要调用工具，直接回答：OK",
                workspace_root=temp_dir,
            )

            result = run_task(task_input, client, max_steps=2, total_step_budget=2, total_remaining_steps=2)

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertEqual([tool["name"] for tool in client.input.tools], [])
            self.assertEqual(client.input.diagnostics["execution_profile"], "direct")
            self.assertEqual(client.input.diagnostics["context_tier"], "direct")
            self.assertIn("Direct-answer mode", client.input.messages[0]["content"])
            self.assertNotIn("finish_task immediately", client.input.messages[0]["content"])
            self.assertLessEqual(client.input.diagnostics["estimated_context_tokens"], 550)
            self.assertEqual(result.state.total_remaining_steps, 1)

    def test_single_read_task_uses_narrow_tool_surface(self):
        class CapturingClient:
            last_usage = {}

            def __init__(self):
                self.input = None

            def decide(self, model_input):
                self.input = model_input
                return ActionDecision(
                    decision_type=DecisionType.FINISH,
                    reason="Done.",
                    response_text="Done.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            client = CapturingClient()
            task_input = TaskInput(
                goal="Read note.txt and answer.",
                user_message="Read note.txt and answer.",
                workspace_root=temp_dir,
            )

            run_task(task_input, client, max_steps=2)

            self.assertEqual(
                [tool["name"] for tool in client.input.tools],
                ["read_file", "file_info", "list_files"],
            )

    def test_read_write_task_exposes_read_and_write_tools_in_read_first_order(self):
        class CapturingClient:
            last_usage = {}

            def __init__(self):
                self.input = None

            def decide(self, model_input):
                self.input = model_input
                return ActionDecision(
                    decision_type=DecisionType.FINISH,
                    reason="Done.",
                    response_text="Done.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            client = CapturingClient()
            task_input = TaskInput(
                goal="读取 source_notes_cn.md，保存为 summary_cn.md",
                user_message="读取 source_notes_cn.md，保存为 summary_cn.md",
                workspace_root=temp_dir,
            )

            run_task(task_input, client, max_steps=2)

            self.assertEqual(
                [tool["name"] for tool in client.input.tools],
                ["read_file", "read_many_files"],
            )
            self.assertEqual(client.input.diagnostics["execution_profile"], "read_write")

    def test_read_file_max_lines_is_normalized_before_policy(self):
        class ReadThenFinishClient:
            last_usage = {}

            def __init__(self):
                self.calls = 0
                self.decisions = []

            def decide(self, model_input):
                self.calls += 1
                if self.calls == 1:
                    decision = ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Read too many lines.",
                        tool_name="read_file",
                        tool_args={"path": "note.txt", "max_lines": 1000},
                    )
                    self.decisions.append(decision)
                    return decision
                return ActionDecision(
                    decision_type=DecisionType.FINISH,
                    reason="Done.",
                    response_text="Done.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "note.txt").write_text("hello\nworld\n", encoding="utf-8")
            client = ReadThenFinishClient()
            task_input = TaskInput(
                goal="Read note.txt and answer.",
                user_message="Read note.txt and answer.",
                workspace_root=temp_dir,
            )

            result = run_task(task_input, client, max_steps=2)

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertEqual(client.decisions[0].tool_args["max_lines"], 400)
            self.assertTrue(result.steps[0].tool_result.success)
            self.assertTrue(any("max_lines to 400" in message.content for message in result.state.messages))

    def test_read_write_multi_output_stays_in_write_phase_until_all_outputs_exist(self):
        class MultiOutputClient:
            last_usage = {}

            def __init__(self):
                self.calls = 0

            def decide(self, model_input):
                self.calls += 1
                if self.calls == 1:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Read source.",
                        tool_name="read_file",
                        tool_args={"path": "source.md"},
                    )
                if self.calls == 2:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Write first output.",
                        tool_name="write_file",
                        tool_args={"path": "a.json", "content": "{}"},
                    )
                if self.calls == 3:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Write second output.",
                        tool_name="write_file",
                        tool_args={"path": "b.md", "content": "# OK"},
                    )
                return ActionDecision(
                    decision_type=DecisionType.FINISH,
                    reason="Done.",
                    response_text="Done.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "source.md").write_text("source", encoding="utf-8")
            client = MultiOutputClient()
            task_input = TaskInput(
                goal="读取 source.md，保存两个文件：a.json 和 b.md",
                user_message="读取 source.md，保存两个文件：a.json 和 b.md",
                workspace_root=temp_dir,
            )

            result = run_task(task_input, client, max_steps=4)

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertTrue((root / "a.json").exists())
            self.assertTrue((root / "b.md").exists())
            self.assertEqual([step.decision.tool_name for step in result.steps[:3]], ["read_file", "write_file", "write_file"])
            self.assertTrue(any("additional output files" in message.content for message in result.state.messages))

    def test_multi_output_detection_includes_pdf_and_xlsx(self):
        class MultiInputClient:
            last_usage = {}

            def __init__(self):
                self.calls = 0

            def decide(self, model_input):
                self.calls += 1
                if self.calls == 1:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Write manual.",
                        tool_name="write_file",
                        tool_args={"path": "inputs/manual.pdf", "content": "manual placeholder"},
                    )
                if self.calls == 2:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Write parameters.",
                        tool_name="write_file",
                        tool_args={"path": "inputs/parameters.xlsx", "content": "field,value"},
                    )
                return ActionDecision(
                    decision_type=DecisionType.FINISH,
                    reason="Done.",
                    response_text="Done.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            client = MultiInputClient()
            task_input = TaskInput(
                goal="使用模型逐步生成 inputs/manual.pdf 和 inputs/parameters.xlsx",
                user_message="使用模型逐步生成 inputs/manual.pdf 和 inputs/parameters.xlsx",
                workspace_root=temp_dir,
            )

            result = run_task(task_input, client, max_steps=3)

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertTrue(Path(temp_dir, "inputs", "manual.pdf").exists())
            self.assertTrue(Path(temp_dir, "inputs", "parameters.xlsx").exists())

    def test_read_write_multi_source_stays_in_read_phase_until_all_sources_read(self):
        class MultiSourceClient:
            last_usage = {}

            def __init__(self):
                self.calls = 0

            def decide(self, model_input):
                self.calls += 1
                if self.calls == 1:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Read first source.",
                        tool_name="read_file",
                        tool_args={"path": "source_a.md"},
                    )
                if self.calls == 2:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Read second source.",
                        tool_name="read_file",
                        tool_args={"path": "source_b.json"},
                    )
                if self.calls == 3:
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Write output.",
                        tool_name="write_file",
                        tool_args={"path": "out.json", "content": "{}"},
                    )
                return ActionDecision(
                    decision_type=DecisionType.FINISH,
                    reason="Done.",
                    response_text="Done.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "source_a.md").write_text("source", encoding="utf-8")
            (root / "source_b.json").write_text("{}", encoding="utf-8")
            client = MultiSourceClient()
            task_input = TaskInput(
                goal="读取 source_a.md 和 source_b.json，生成 out.json",
                user_message="读取 source_a.md 和 source_b.json，生成 out.json",
                workspace_root=temp_dir,
            )

            result = run_task(task_input, client, max_steps=4)

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertTrue((root / "out.json").exists())
            self.assertEqual([step.decision.tool_name for step in result.steps[:3]], ["read_file", "read_file", "write_file"])
            self.assertTrue(any("additional source files" in message.content for message in result.state.messages))

    def test_runtime_appends_high_priority_user_event_before_next_model_call(self):
        class EventAppendingClient:
            def __init__(self, queue_path):
                self.calls = 0
                self.inputs = []
                self.queue = JsonlEventQueue(queue_path)

            def decide(self, model_input):
                self.calls += 1
                self.inputs.append(model_input)
                if self.calls == 1:
                    self.queue.append(AgentEventType.USER_INPUT, "Please revise the plan.", priority=80)
                    return ActionDecision(
                        decision_type=DecisionType.TOOL_CALL,
                        reason="Inspect workspace.",
                        tool_name="list_files",
                        tool_args={"path": "."},
                    )
                return ActionDecision(
                    decision_type=DecisionType.FINISH,
                    reason="Finished after event.",
                    response_text="Done.",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = str(Path(temp_dir) / "events.jsonl")
            client = EventAppendingClient(queue_path)
            task_input = TaskInput(goal="Inspect workspace", user_message="Inspect", workspace_root=temp_dir)

            result = run_task(task_input, client, max_steps=2, event_queue_path=queue_path)

            self.assertEqual(result.task.status, TaskStatus.COMPLETED)
            self.assertGreaterEqual(result.state.event_cursor, 1)
            second_input_text = "\n".join(message["content"] for message in client.inputs[1].messages)
            self.assertIn("Runtime Events", second_input_text)
            self.assertIn("Please revise the plan", second_input_text)

    def test_runtime_pauses_for_high_priority_audit_event(self):
        class AuditEventClient:
            def __init__(self, queue_path):
                self.calls = 0
                self.queue = JsonlEventQueue(queue_path)

            def decide(self, model_input):
                self.calls += 1
                self.queue.append(AgentEventType.AUDIT_WARNING, "Possible policy violation.", priority=90)
                return ActionDecision(
                    decision_type=DecisionType.TOOL_CALL,
                    reason="Inspect workspace.",
                    tool_name="list_files",
                    tool_args={"path": "."},
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = str(Path(temp_dir) / "events.jsonl")
            client = AuditEventClient(queue_path)
            task_input = TaskInput(goal="Inspect workspace", user_message="Inspect", workspace_root=temp_dir)

            result = run_task(task_input, client, max_steps=3, event_queue_path=queue_path)

            self.assertEqual(result.task.status, TaskStatus.WAITING_USER)
            self.assertEqual(client.calls, 1)
            self.assertTrue(any("audit warning" in summary for summary in result.state.event_summaries))


if __name__ == "__main__":
    unittest.main()
