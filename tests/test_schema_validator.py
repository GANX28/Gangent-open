import json
import unittest

from gangent.error_recovery import recovery_hint_for_tool_result
from gangent.models import ActionDecision, DecisionType
from gangent.schema_validator import (
    ToolArgumentsValidationError,
    ToolSchemaDefinitionError,
    validate_tool_arguments,
    validate_tool_schema_definition,
)
from gangent.tool_registry import ToolDefinition, ToolRegistry, ToolRisk, ToolSource
from gangent.tool_runtime import execute_tool_call


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": "Test schema.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["path", "limit"],
            "additionalProperties": False,
        },
        "strict": True,
    }


class SchemaValidatorTests(unittest.TestCase):
    def test_valid_arguments_pass(self):
        validate_tool_arguments("demo", _schema("demo"), {"path": "README.md", "limit": 3})

    def test_invalid_arguments_return_structured_retryable_issues(self):
        with self.assertRaises(ToolArgumentsValidationError) as context:
            validate_tool_arguments(
                "demo",
                _schema("demo"),
                {"path": 123, "limit": 20, "secret_extra": "do-not-log-this-value"},
            )

        payload = json.loads(str(context.exception))
        self.assertEqual(payload["error_type"], "tool_argument_validation")
        self.assertEqual(payload["tool_name"], "demo")
        self.assertTrue(payload["retryable"])
        self.assertEqual({issue["rule"] for issue in payload["issues"]}, {"type", "maximum", "additionalProperties"})
        self.assertNotIn("do-not-log-this-value", str(context.exception))

    def test_missing_required_field_is_reported(self):
        with self.assertRaises(ToolArgumentsValidationError) as context:
            validate_tool_arguments("demo", _schema("demo"), {"path": "README.md"})

        payload = json.loads(str(context.exception))
        self.assertEqual(payload["issues"][0]["rule"], "required")

    def test_invalid_schema_is_rejected_before_registration(self):
        schema = _schema("demo")
        schema["parameters"]["properties"] = []

        with self.assertRaises(ToolSchemaDefinitionError):
            validate_tool_schema_definition("demo", schema)

    def test_schema_name_must_match_registry_name(self):
        with self.assertRaisesRegex(ToolSchemaDefinitionError, "name mismatch"):
            validate_tool_schema_definition("other", _schema("demo"))

    def test_registry_does_not_call_handler_when_arguments_are_invalid(self):
        calls: list[dict] = []
        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="demo",
                    description="Demo.",
                    risk=ToolRisk.READ,
                    source=ToolSource.LOCAL,
                    handler=lambda args, root: calls.append(args) or "ok",
                    input_schema=_schema("demo"),
                )
            ]
        )
        decision = ActionDecision(
            decision_type=DecisionType.TOOL_CALL,
            reason="Invalid call.",
            tool_name="demo",
            tool_args={"path": "README.md", "limit": "three"},
        )

        with self.assertRaises(ToolArgumentsValidationError):
            registry.dispatch(decision, workspace_root=".")

        self.assertEqual(calls, [])

    def test_execute_tool_call_normalizes_schema_failure(self):
        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="demo",
                    description="Demo.",
                    risk=ToolRisk.READ,
                    source=ToolSource.LOCAL,
                    handler=lambda args, root: "must not run",
                    input_schema=_schema("demo"),
                )
            ]
        )
        decision = ActionDecision(
            decision_type=DecisionType.TOOL_CALL,
            reason="Invalid call.",
            tool_name="demo",
            tool_args={"path": "README.md", "limit": "three"},
        )

        result = execute_tool_call(decision, workspace_root=".", tool_registry=registry)

        self.assertFalse(result.success)
        self.assertEqual(json.loads(result.error)["error_type"], "tool_argument_validation")
        self.assertIn("correct only those argument fields", recovery_hint_for_tool_result(decision, result))


if __name__ == "__main__":
    unittest.main()
