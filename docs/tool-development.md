# Gangent Tool Development

This document is the shortest correct checklist for adding a new tool.

## Current Design

One Gangent tool is currently described in at least four places:

1. model-visible schema
2. registry definition
3. policy rule
4. runtime handler

That duplication is manageable for v1, but it is real duplication.

Future direction:

- make `ToolDefinition` closer to a single source of truth
- generate more of the registry or schema wiring automatically

All registered tools now pass through `gangent/schema_validator.py`. Registration
rejects missing or invalid schemas, and dispatch validates model arguments before
Policy or the handler can create side effects.

Important boundary:

- Codex host-side MCP tools are not Gangent runtime tools yet.
- To make an external MCP tool available inside Gangent, first implement live
  MCP transport and remote tool registration in `gangent/mcp_adapter.py`,
  `gangent/tool_registry.py`, and `gangent/tool_runtime.py`.

## Required Steps

### 1. Add schema

File:

- `gangent/tool_schema.py`

What to do:

- add the tool's JSON Schema
- keep it strict and bounded
- reject open-ended free-form inputs unless unavoidable
- mark fields with safe handler defaults as optional rather than falsely required
- use `additionalProperties: false`

The shared validator automatically checks the schema during registry registration
and checks arguments before every dispatch. Do not duplicate basic JSON type checks
inside each handler unless the handler is also a separately callable public API.

### 2. Add runtime handler

File:

- `gangent/tool_runtime.py`

What to do:

- implement the actual local behavior
- keep the output bounded
- respect `workspace_root`
- fail with clear messages

### 3. Register the tool

File:

- `gangent/tool_registry.py`

What to do:

- create a `ToolDefinition`
- wire args into the runtime handler
- choose correct `ToolRisk`

### 4. Add policy

File:

- `gangent/policy.py`

What to do:

- define whether the tool should `allow`, `block`, or `escalate`
- validate arguments if policy depends on them
- keep dangerous actions explicit

### 5. Add tests

Typical files:

- `tests/test_tool_runtime.py`
- `tests/test_tool_registry.py`
- `tests/test_policy.py`

What to cover:

- success path
- invalid args
- boundary conditions
- escalation or block behavior

### 6. Update docs

Update:

- `README.md` if the tool is user-visible
- `docs/extension-guide.md` if it changes extension flow
- `docs/api-contracts.md` if it changes contracts

## External MCP Tool Path

MCP = Model Context Protocol（模型上下文协议），用于让 Agent 通过统一协议连接外部工具、数据源和服务。

When the tool source is an MCP server rather than local Python code, the future
Gangent path should be:

1. discover remote MCP tool schema
2. convert it into a local `ToolDefinition`
3. assign `ToolRisk`
4. require policy approval for risky operations
5. execute through a bounded transport client
6. normalize the response into `ToolResult`
7. write audit records

Do not bypass policy just because a remote MCP server already exposes a tool.
The runtime still decides what actually happens.

## Example: Chunked File Read

Recent example:

- `read_file` was extended with `start_line` and `max_lines`

Why:

- large files should not be read as one giant blob
- chunked reads are cheaper and more practical

Files touched:

- `tool_schema.py`
- `tool_registry.py`
- `tool_runtime.py`
- `policy.py`
- `tests/test_tool_runtime.py`

That is the standard Gangent tool-change pattern.

