# Gangent Extension Guide

This document answers one practical question:

> If I want to add or change a capability, where do I touch the code?

## Extension Map

| Goal | Primary Files | Secondary Files | Tests |
|---|---|---|---|
| Add a model-visible tool | `gangent/tool_schema.py`, `gangent/schema_validator.py`, `gangent/tool_registry.py`, `gangent/tool_runtime.py` | `gangent/policy.py`, `gangent/permissions.py` | `tests/test_schema_validator.py`, `tests/test_tool_runtime.py`, `tests/test_tool_registry.py`, `tests/test_policy.py` |
| Add a command rule | `gangent/command_policy.py` | `gangent/runner.py`, `gangent/policy.py` | `tests/test_command_policy.py`, `tests/test_runner.py` |
| Add an LLM provider | `gangent/providers.py`, `gangent/llm_client.py` | `gangent/decision.py`, `gangent/model_input.py` | `tests/test_providers.py` |
| Add a hook | `gangent/hooks.py`, `gangent/runtime.py` | `gangent/audit.py`, `gangent/session_store.py` | `tests/test_hooks.py`, `tests/test_runtime.py` |
| Add event / interrupt behavior | `gangent/events.py`, `gangent/replanning.py`, `gangent/runtime.py`, `gangent/models.py` | `gangent/checkpoint.py`, `gangent/context_manager.py`, `gangent/cli.py` | `tests/test_events.py`, `tests/test_replanning.py`, `tests/test_runtime.py`, `tests/test_checkpoint.py` |
| Add a skill | `gangent/skills.py`, `skills/<skill-name>/manifest.json`, `skills/<skill-name>/skill.md` | `gangent/model_input.py`, `gangent/context_manager.py` | `tests/test_skills.py` |
| Add a RAG backend | `gangent/rag.py`, `gangent/embeddings.py` | `gangent/context_manager.py`, `gangent/secret_guard.py` | `tests/test_rag.py`, `tests/test_embeddings.py` |
| Add live MCP transport | `gangent/mcp_adapter.py`, `gangent/tool_registry.py`, `gangent/tool_runtime.py` | `gangent/policy.py`, `gangent/audit.py` | future `tests/test_mcp_adapter.py`, `tests/test_tool_registry.py`, `tests/test_policy.py` |
| Add a checkpoint field | `gangent/checkpoint.py`, `gangent/runtime_checkpoint.py` | `gangent/resume.py`, `gangent/idempotency.py` | `tests/test_checkpoint.py` |
| Add failure recovery | `gangent/failure.py`, `gangent/error_recovery.py` | `gangent/adaptive_runtime.py`, `gangent/runtime.py` | `tests/test_failure.py`, `tests/test_error_recovery.py` |
| Add audit metric | `gangent/audit.py`, `gangent/metrics.py` | `gangent/eval.py` | `tests/test_metrics_eval.py`, `tests/test_output_audit.py` |
| Add session or handoff behavior | `gangent/session.py`, `gangent/session_store.py`, `gangent/handoff.py` | `gangent/cli.py`, `gangent/checkpoint.py` | `tests/test_session.py`, `tests/test_handoff.py` |

## Common Change Paths

### Add a new tool

Touch these files:

1. `gangent/tool_schema.py`
2. `gangent/tool_registry.py`
3. `gangent/tool_runtime.py`
4. `gangent/policy.py`
5. related tests

Rule:

- schema defines model-facing shape
- the shared validator rejects invalid schemas at registration and invalid arguments before execution
- registry defines dispatch metadata
- runtime defines execution
- policy defines safety gate

### Add a new provider

Touch these files:

1. `gangent/llm_client.py`
2. `gangent/providers.py`
3. optional parsing changes in `gangent/decision.py`

### Add a new hook consumer

Touch these files:

1. `gangent/hooks.py`
2. caller site in `gangent/runtime.py`
3. optional persistence in `gangent/audit.py` or `gangent/session_store.py`

### Add retrieval capability

Touch these files:

1. `gangent/rag.py`
2. `gangent/embeddings.py`
3. `gangent/context_manager.py`
4. optional log and eval paths

### Add context or interrupt capability

ContextSegment（上下文片段） and AgentEvent（智能体事件） are now separate contracts.

For context selection or pollution diagnostics, touch:

1. `gangent/context_manager.py`
2. `gangent/context_maintenance.py`
3. `gangent/model_input.py`
4. `tests/test_context_manager.py`

For cooperative interrupts, touch:

1. `gangent/events.py`
2. `gangent/runtime.py`
3. `gangent/models.py`
4. `gangent/checkpoint.py`
5. `gangent/cli.py`
6. `tests/test_events.py`
7. `tests/test_runtime.py`
8. `tests/test_checkpoint.py`

Current rule:

- v1 checks events only at safe step boundaries before model calls
- do not force-cancel running tools or model calls until there is a stronger process/sandbox layer
- event cursor must be checkpointed to avoid consuming the same event twice after resume

### Add live MCP capability

MCP = Model Context Protocol（模型上下文协议），用于把外部工具或数据服务接入 Agent。

Touch these files:

1. `gangent/mcp_adapter.py`
2. `gangent/tool_registry.py`
3. `gangent/tool_runtime.py`
4. `gangent/policy.py`
5. `gangent/audit.py`
6. related tests

Current boundary:

- Gangent's own runtime still has only the adapter skeleton.
- Do not describe a Codex host-side MCP tool as a Gangent runtime tool until the runtime transport and registry wiring exist.

Implementation rule:

- discover remote MCP tools
- normalize each remote tool into a `ToolDefinition`
- classify risk before execution
- execute through a bounded transport client
- normalize the remote response into `ToolResult`
- write audit records for every remote call

### Add checkpoint-compatible state

Touch these files:

1. `gangent/checkpoint.py`
2. migration-compatible defaults in deserialization
3. tests that load older and newer shapes

## Stability Notes

More stable extension points:

- `ToolDefinition`
- `ToolRegistry`
- `HookManager`
- `SkillManifest`
- `TaskInput`
- `ActionDecision`
- `PolicyDecision`

Less stable internal areas:

- planner heuristics
- budget heuristics
- context assembly heuristics
- retrieval ranking heuristics

## Practical Advice

If a change affects model-visible behavior, do not patch only one file.

Typical mistake:

- add handler in `tool_runtime.py`
- forget schema
- forget policy
- forget tests

That creates a half-mounted tool. Gangent will then look complete in code, but
the model still cannot use it correctly.
