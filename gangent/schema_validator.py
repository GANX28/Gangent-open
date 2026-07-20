"""Shared JSON Schema validation for every registered Gangent tool."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


MAX_VALIDATION_ISSUES = 8


class ToolSchemaDefinitionError(ValueError):
    """Raised when a registered tool carries an invalid schema definition."""


@dataclass(frozen=True)
class SchemaValidationIssue:
    """A bounded, non-secret description of one argument validation failure."""

    path: str
    rule: str
    message: str


class ToolArgumentsValidationError(ValueError):
    """Structured and retryable failure for invalid model-generated tool args."""

    def __init__(self, tool_name: str, issues: list[SchemaValidationIssue]) -> None:
        self.tool_name = tool_name
        self.issues = issues[:MAX_VALIDATION_ISSUES]
        super().__init__(self._serialized_payload())

    def as_dict(self) -> dict[str, Any]:
        return {
            "error_type": "tool_argument_validation",
            "tool_name": self.tool_name,
            "retryable": True,
            "issues": [asdict(issue) for issue in self.issues],
        }

    def _serialized_payload(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, separators=(",", ":"))


def validate_tool_schema_definition(
    tool_name: str,
    input_schema: dict[str, Any] | None,
) -> None:
    """Validate one internal function wrapper and its JSON Schema at registration."""

    parameters = _parameters_schema(tool_name, input_schema)
    try:
        Draft202012Validator.check_schema(parameters)
    except SchemaError as exc:
        raise ToolSchemaDefinitionError(
            f"Invalid JSON Schema for tool {tool_name}: {exc.message}"
        ) from exc


def validate_tool_arguments(
    tool_name: str,
    input_schema: dict[str, Any] | None,
    arguments: dict[str, Any],
) -> None:
    """Validate tool arguments without coercing values or exposing actual values."""

    parameters = _parameters_schema(tool_name, input_schema)
    validator = Draft202012Validator(parameters)
    errors = sorted(
        validator.iter_errors(arguments),
        key=lambda error: (_format_path(error), str(error.validator)),
    )
    if not errors:
        return

    issues = [
        SchemaValidationIssue(
            path=_format_path(error),
            rule=str(error.validator),
            message=_safe_validation_message(error),
        )
        for error in errors[:MAX_VALIDATION_ISSUES]
    ]
    raise ToolArgumentsValidationError(tool_name, issues)


def _parameters_schema(
    tool_name: str,
    input_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(input_schema, dict):
        raise ToolSchemaDefinitionError(f"Registered tool {tool_name} is missing input_schema.")
    if input_schema.get("type") != "function":
        raise ToolSchemaDefinitionError(f"Tool {tool_name} schema must have type=function.")
    if input_schema.get("name") != tool_name:
        raise ToolSchemaDefinitionError(
            f"Tool schema name mismatch: registry={tool_name}, schema={input_schema.get('name')!r}."
        )
    parameters = input_schema.get("parameters")
    if not isinstance(parameters, dict):
        raise ToolSchemaDefinitionError(f"Tool {tool_name} schema must contain parameters object.")
    if parameters.get("type") != "object":
        raise ToolSchemaDefinitionError(f"Tool {tool_name} parameters must have type=object.")
    return parameters


def _format_path(error: ValidationError) -> str:
    parts: list[str] = ["$"]
    for part in error.absolute_path:
        if isinstance(part, int):
            parts.append(f"[{part}]")
        else:
            parts.append(f".{part}")
    return "".join(parts)


def _safe_validation_message(error: ValidationError) -> str:
    """Describe the failed rule without copying the model-provided value."""

    rule = str(error.validator)
    expected = error.validator_value
    if rule == "required":
        return error.message
    if rule == "additionalProperties":
        return error.message
    if rule == "type":
        return f"Expected JSON type {expected!r}."
    if rule == "enum":
        return f"Expected one of {expected!r}."
    if rule == "const":
        return "Value does not match the required constant."
    if rule in {"minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"}:
        return f"Numeric value violates {rule}={expected!r}."
    if rule in {"minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"}:
        return f"Value violates {rule}={expected!r}."
    if rule == "pattern":
        return "String does not match the required pattern."
    if rule == "format":
        return f"String does not match format {expected!r}."
    return f"Value violates JSON Schema rule {rule}."
