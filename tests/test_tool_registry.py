import tempfile
import unittest
from pathlib import Path

from gangent.models import ActionDecision, DecisionType
from gangent.tool_registry import (
    ToolDefinition,
    ToolRegistry,
    ToolRisk,
    ToolSource,
    default_tool_registry,
)
from gangent.tool_schema import available_tool_schemas, tool_names


def _test_tool_schema(name: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": "Test tool.",
        "parameters": {
            "type": "object",
            "properties": properties or {},
            "required": required or [],
            "additionalProperties": False,
        },
        "strict": True,
    }


class ToolRegistryTests(unittest.TestCase):
    def test_register_and_dispatch_tool(self):
        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="Return text.",
                    risk=ToolRisk.READ,
                    source=ToolSource.LOCAL,
                    handler=lambda args, root: str(args["text"]),
                    input_schema=_test_tool_schema(
                        "echo",
                        properties={"text": {"type": "string"}},
                        required=["text"],
                    ),
                )
            ]
        )
        decision = ActionDecision(
            decision_type=DecisionType.TOOL_CALL,
            reason="Echo.",
            tool_name="echo",
            tool_args={"text": "hello"},
        )

        output = registry.dispatch(decision, workspace_root=".")

        self.assertEqual(output, "hello")

    def test_duplicate_tool_registration_is_rejected(self):
        registry = ToolRegistry()
        definition = ToolDefinition(
            name="echo",
            description="Return text.",
            risk=ToolRisk.READ,
            source=ToolSource.LOCAL,
            handler=lambda args, root: "ok",
            input_schema=_test_tool_schema("echo"),
        )

        registry.register(definition)

        with self.assertRaises(ValueError):
            registry.register(definition)

    def test_registration_rejects_missing_schema(self):
        with self.assertRaisesRegex(ValueError, "missing input_schema"):
            ToolRegistry(
                [
                    ToolDefinition(
                        name="echo",
                        description="Return text.",
                        risk=ToolRisk.READ,
                        source=ToolSource.LOCAL,
                        handler=lambda args, root: "ok",
                    )
                ]
            )

    def test_default_registry_contains_local_runtime_tools(self):
        registry = default_tool_registry()

        self.assertIn("list_files", registry.names())
        self.assertIn("file_info", registry.names())
        self.assertIn("read_file", registry.names())
        self.assertIn("write_file", registry.names())
        self.assertIn("run_tests", registry.names())
        self.assertIn("git_log", registry.names())
        self.assertIn("git_commit", registry.names())
        self.assertIn("memory_add", registry.names())
        self.assertEqual(registry.get("read_file").source, ToolSource.LOCAL)
        self.assertTrue(registry.get("read_file").is_read_only())
        self.assertTrue(registry.get("read_file").snip_hint)

    def test_default_registry_dispatches_local_tool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("hello", encoding="utf-8")
            registry = default_tool_registry()
            decision = ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="Read file.",
                tool_name="read_file",
                tool_args={"path": "README.md"},
            )

            output = registry.dispatch(decision, workspace_root=temp_dir)

            self.assertEqual(output, "hello")

    def test_model_visible_tools_are_registered_for_execution(self):
        visible_tools = tool_names(available_tool_schemas()) - {"finish_task"}
        registry = default_tool_registry()

        self.assertTrue(visible_tools.issubset(registry.names()))

    def test_registered_tools_carry_model_schema(self):
        registry = default_tool_registry()

        for definition in registry.definitions():
            self.assertIsNotNone(definition.input_schema)
            self.assertEqual(definition.input_schema["name"], definition.name)


if __name__ == "__main__":
    unittest.main()
