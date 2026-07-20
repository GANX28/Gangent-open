"""MCP adapter（模型上下文协议适配层）。

MCP（Model Context Protocol，模型上下文协议）里常见角色：

- MCP Host：承载 LLM 应用的宿主，例如一个 coding agent；
- MCP Client：Host 内部连接某个 MCP server 的客户端；
- MCP Server：对外暴露 tools / resources / prompts 的服务；
- Transport：通信方式，例如 stdio 或 HTTP/SSE。

Gangent 现在先实现 adapter 层：把 MCP server 暴露的 tool spec 转换为
内部 ToolDefinition。真正 stdio / HTTP transport 会在下一步接入。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .tool_registry import ToolDefinition, ToolRisk, ToolSource


class MCPAdapterError(ValueError):
    """Raised when an MCP tool cannot be safely adapted."""


@dataclass(frozen=True)
class MCPToolSpec:
    """A normalized MCP tool description."""

    server_name: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]
    risk: ToolRisk = ToolRisk.EXTERNAL


class MCPToolProxy:
    """Placeholder proxy for a future connected MCP client.

    这不是假装已经能调用 MCP server。它只是把接口位置固定下来，避免后续
    接 stdio / HTTP transport 时影响 ToolRegistry 和 policy 层。
    """

    def __init__(self, spec: MCPToolSpec) -> None:
        self.spec = spec

    def __call__(self, args: dict[str, Any], workspace_root: str) -> str:
        raise MCPAdapterError(
            "MCP transport is not connected yet. "
            f"Tool {mcp_tool_id(self.spec.server_name, self.spec.tool_name)} is registered as an adapter stub."
        )


def mcp_tool_id(server_name: str, tool_name: str) -> str:
    """Return a stable internal namespaced MCP tool id."""

    safe_server = _safe_identifier(server_name, "server_name")
    safe_tool = _safe_identifier(tool_name, "tool_name")
    return f"mcp.{safe_server}.{safe_tool}"


def adapt_mcp_tool(spec: MCPToolSpec) -> ToolDefinition:
    """Convert one MCP tool spec into a Gangent ToolDefinition."""

    _validate_input_schema(spec.input_schema)
    internal_name = mcp_tool_id(spec.server_name, spec.tool_name)
    return ToolDefinition(
        name=internal_name,
        description=spec.description or f"MCP tool {internal_name}",
        risk=spec.risk,
        source=ToolSource.MCP,
        handler=MCPToolProxy(spec),
        input_schema={
            "type": "function",
            "name": internal_name,
            "description": spec.description,
            "parameters": spec.input_schema,
            "strict": True,
        },
    )


def adapt_mcp_tools(specs: list[MCPToolSpec]) -> list[ToolDefinition]:
    """Convert multiple MCP tool specs and reject duplicate internal names."""

    definitions = [adapt_mcp_tool(spec) for spec in specs]
    names = [definition.name for definition in definitions]
    if len(names) != len(set(names)):
        raise MCPAdapterError("Duplicate MCP tool names after namespacing.")
    return definitions


def _safe_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MCPAdapterError(f"{field_name} must be a non-empty string.")
    normalized = value.strip().replace("-", "_")
    if not re.fullmatch(r"[A-Za-z0-9_]+", normalized):
        raise MCPAdapterError(f"{field_name} contains unsupported characters: {value}")
    return normalized


def _validate_input_schema(schema: dict[str, Any]) -> None:
    if not isinstance(schema, dict):
        raise MCPAdapterError("MCP input_schema must be an object.")
    if schema.get("type") != "object":
        raise MCPAdapterError("MCP input_schema must have type=object.")
    if "properties" not in schema or not isinstance(schema["properties"], dict):
        raise MCPAdapterError("MCP input_schema must contain object properties.")
    if "additionalProperties" not in schema:
        raise MCPAdapterError("MCP input_schema must explicitly set additionalProperties.")
