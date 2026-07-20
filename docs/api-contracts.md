# Gangent API Contracts

This document records the current cross-module contracts.

The goal is not to duplicate every source line. The goal is to make extension
work predictable.

## 1. TaskInput

Source: `gangent/models.py`

Fields:

- `goal: str`
- `user_message: str`
- `workspace_root: str`
- `constraints: list[str]`
- `created_at: str`

Meaning:

- normalized entry object for one task
- should be stable across runtime, planner, session, checkpoint layers

## 2. ActionDecision

Source: `gangent/models.py`

Fields:

- `decision_type: DecisionType`
- `reason: str`
- `tool_name: str | None`
- `tool_args: dict[str, Any] | None`
- `response_text: str | None`

Meaning:

- the single runtime action selected after model output is parsed

Rules:

- `tool_name` is required for tool calls
- `tool_args` must be JSON-object-like for tool calls

## 3. ToolResult

Source: `gangent/models.py`

Fields:

- `call_id: str`
- `success: bool`
- `output: str`
- `error: str | None`
- `reused: bool`
- `finished_at: str`

Meaning:

- normalized result of one tool execution

## 4. PolicyDecision

Source: `gangent/models.py`

Fields:

- `mode: PolicyMode`
- `allowed: bool`
- `reason: str`

Meaning:

- decision returned by the safety layer

Practical interpretation:

- `ALLOW`: execute
- `BLOCK`: reject
- `ESCALATE`: request user approval first

## 5. ToolDefinition

Source: `gangent/tool_registry.py`

Fields:

- `name: str`
- `description: str`
- `risk: ToolRisk`
- `source: ToolSource`
- `handler: ToolHandler`
- `input_schema: dict[str, Any] | None` at construction time; every registered tool must provide a valid schema

`ToolHandler` signature:

```python
Callable[[dict[str, Any], str], str]
```

Meaning:

- first argument: tool args
- second argument: workspace root
- return value: string output consumed by runtime

Current limitation:

- the runtime dispatch contract is registry-driven
- policy still contains static handling for built-in tool names

### 5.1 Tool Schema Validation

Source: `gangent/schema_validator.py`

Registration contract:

- wrapper `type` must be `function`
- schema `name` must match `ToolDefinition.name`
- `parameters` must be a Draft 2020-12 object schema
- invalid or missing schemas are rejected before registration

Execution contract:

- Plan Guard confirms the tool is allowed in the current phase
- the shared validator checks `tool_args` before Policy and handler execution
- values are not silently coerced
- invalid args produce `ToolArgumentsValidationError`
- its serialized payload contains `error_type`, `tool_name`, `retryable`, and bounded `issues`
- issue messages describe the failed rule without copying the actual model-provided value

Important boundary:

- Schema validation checks structure, types, ranges, enums, and unexpected fields
- Policy still owns permissions, workspace boundaries, sensitive data, risk, and approval
- handlers still own real file state and business semantics

## 6. HookContext

Source: `gangent/hooks.py`

Fields:

- `event`
- `task_input`
- `task`
- `state`
- `decision`
- `policy`
- `tool_result`
- `model_input`
- `result`
- `metadata`

Meaning:

- loose payload envelope for hook handlers

Current rule:

- not every field is populated on every event
- hook code must check for `None`

## 7. LLMClient

Source: `gangent/llm_client.py`

Protocol:

```python
def decide(self, model_input: ModelInput) -> ActionDecision
```

Meaning:

- provider clients must reduce model behavior to exactly one runtime decision

Current implementations:

- `FakeLLMClient`
- `DeepSeekChatClient`
- `OpenAIResponsesClient`

## 8. ModelInput

Source: `gangent/model_input.py`

Role:

- the fully assembled provider-facing input bundle
- contains messages, tool schemas, and bounded context

Practical rule:

- this is the last stable object before provider-specific formatting

## 8.1 Context Segment Contract

Source: `gangent/context_manager.py`

ContextSegment（上下文片段） is the unit used before final prompt assembly.

Fields:

- `title: str`
- `content: str`
- `source: str`
- `scope: str`
- `priority: int`
- `confidence: float`
- `sensitivity: str`

Meaning:

- source tells where the context came from
- scope tells where it applies
- priority controls budget pressure
- confidence tells how trustworthy the segment is
- sensitivity marks whether the segment may need stricter handling

ContextPollutionReport fields:

- `total_segments`
- `total_chars`
- `source_counts`
- `warnings`
- `sensitive_segments`
- `low_confidence_segments`

Current limitation:

- diagnostics are deterministic and conservative
- no model-based conflict resolution yet
- no automatic memory invalidation yet

## 9. Checkpoint Contract

Source: `gangent/checkpoint.py`

Top-level shape:

- `version`
- `task_input`
- `task`
- `state`
- `steps`
- `stats`

State additions:

- `event_cursor`: last consumed one-based event log index
- `event_summaries`: compact summaries of runtime events consumed by the task

Compatibility rule:

- deserialization must tolerate missing older fields
- new fields should have safe defaults
- archive and active checkpoint should share the same schema

## 10. Session Contract

Source: `gangent/session.py`

SessionState fields:

- `session_id`
- `workspace_root`
- `context_summary`
- `turns`
- `created_at`
- `updated_at`

SessionTurn fields:

- `user_message`
- `task_id`
- `task_status`
- `final_answer`
- `tool_summaries`
- `state_summary`

Meaning:

- session is short-term conversation memory, not long-term memory

## 11. Skill Manifest Contract

Source: `gangent/skills.py`

Current JSON fields:

- `name`
- `description`
- `when_to_use`
- `recommended_tools`
- `risk_notes`
- `output_contract`

Required runtime files for one skill:

- `skills/<skill-name>/manifest.json`
- `skills/<skill-name>/skill.md`

Current limitation:

- schema is convention-based, not yet JSON-Schema-validated

## 12. MCP Tool Spec Contract

Source: `gangent/mcp_adapter.py`

MCP = Model Context Protocol（模型上下文协议），用于让 Agent 通过统一协议连接外部工具、数据源和服务。

Current normalized spec:

- `server_name`
- `tool_name`
- `description`
- `input_schema`
- `risk`

Current Gangent runtime limitation:

- adapter only
- no real stdio / HTTP / SSE transport connection yet
- no runtime discovery loop that imports live MCP tools into `ToolRegistry`

Next contract if Gangent implements live MCP:

- transport client: process / HTTP / SSE lifecycle
- discovery result: remote tool spec to `ToolDefinition`
- policy mapping: remote tool risk classification
- execution result: remote response normalized to `ToolResult`
- audit record: server name, tool name, args summary, result summary, and failure reason

## 13. Runtime Result

Source: `gangent/runtime.py`

Practical fields used across the project:

- `task_input`
- `task`
- `state`
- `steps`
- `stats`

Meaning:

- one complete runtime execution result
- feeds checkpoint, session, audit, and handoff layers

## 14. Event Queue Contract

Source: `gangent/events.py`

AgentEvent（智能体事件） records an external or internal input that may affect a running task.

Fields:

- `event_id`
- `event_type`
- `content`
- `source`
- `task_id`
- `priority`
- `metadata`
- `created_at`

Supported event types:

- `user_input`
- `tool_result`
- `audit_warning`
- `file_change`
- `system_signal`

InterruptAction（中断动作） values:

- `ignore`
- `append`
- `pause`
- `replan`
- `fork`
- `ask_user`

Current v1 rule:

- events are stored in local JSONL
- runtime checks pending events before model calls
- high-priority user input requests replan
- high-priority audit warning or pause signal moves the task to `waiting_user`
- no forceful interruption of an already running model call or tool call

## 15. Planner Budget Contract

Source: `gangent/planner_budget.py`

PlannerBudgetControl records deterministic budget pressure for one runtime boundary.

Fields:

- `profile`
- `segment_step_limit`
- `segment_remaining_steps`
- `total_step_budget`
- `total_remaining_steps`
- `completed_plan_steps`
- `pending_plan_steps`
- `blocked_plan_steps`
- `pressure`

Meaning:

- one runtime step equals one model decision
- the model must use remaining steps to choose action granularity
- critical pressure tells the model to avoid broad exploration
- low pressure allows focused context gathering for the current plan step

AgentState now persists:

- `budget_profile`
- `runtime_step_limit`
- `runtime_remaining_steps`
- `total_step_budget`
- `total_remaining_steps`

## 16. Planner Contract

Source: `gangent/planner_contract.py`

PlanSpec is the commercial planner boundary: planner output is a candidate, not an executable authority.

Core structures:

- `PlanPhaseSpec`
- `VerificationSpec`
- `PlanSpec`
- `PlanLintFinding`
- `PlanLintReport`

PlanPhaseSpec fields:

- `name`
- `goal`
- `max_steps`
- `allowed_tools`
- `exit_criteria`

PlanSpec fields:

- `task_kind`
- `risk_level`
- `max_plan_steps`
- `phases`
- `verification`

Current v1 compiler:

- validates the PlanSpec with `lint_plan_spec`
- rejects error-level findings
- compiles phases into runtime `PlanStep` objects
- stores phase metadata in `PlanStep.description`

Runtime plan guard:

- reads current phase `allowed_tools`
- blocks tool calls outside the current phase before policy and execution
- leaves path safety, approval, and side-effect safety to policy

## 17. Budget History Contract

Source: `gangent/budget_stats.py`

BudgetSample stores completed and failed task resource usage. Successful samples feed future budget recommendations.

Planner-related fields:

- `budget_profile`
- `planned_step_count`
- `completed_plan_step_count`
- `blocked_plan_step_count`
- `runtime_step_limit`
- `total_step_budget`
- `total_remaining_steps`
- `avg_tokens_per_step`
- `avg_tokens_per_tool_call`
- `prompt_cache_hit_tokens`
- `prompt_cache_miss_tokens`
- `prompt_cache_hit_ratio`

BudgetRecommendation now also exposes:

- `planned_steps_p80`
- `completed_plan_steps_p80`
- `tokens_per_step_p80`
- `cache_hit_ratio_p80`

## 18. DeepSeek Routing and Prefix Cache Diagnostics

Sources: `gangent/providers.py`, `gangent/model_input.py`, `gangent/llm_client.py`

Current v1 rules:

- explicit `--model` wins
- default DeepSeek model is `deepseek-v4-flash`
- `ultra` profile escalates to `deepseek-v4-pro`
- `heavy` profile escalates to Pro for architecture, commercial, security, audit, compliance, or planner tasks
- `thinking=True` escalates to Pro

ModelInput diagnostics include:

- `stable_prefix_hash`
- `prefix_cache_strategy`
- `prefix_cache_note`
- `execution_profile`
- `context_tier`

DeepSeek usage metrics are carried into budget history when returned by the API:

- `prompt_cache_hit_tokens`
- `prompt_cache_miss_tokens`

## 18.1 Lightweight Execution Profile Contract

Source: `gangent/task_profile.py`

Before the first model call, Gangent classifies simple tasks into smaller execution profiles:

- `direct`: answer-only task, no model-visible tools, compact context, low output token budget
- `single_read`: one explicit file read, read-oriented tools only
- `single_write`: one explicit file write, file-write tools only
- `standard`: full runtime plan, context, and tool surface

Context tiers:

- `direct`: smallest task context
- `tool`: single known tool path context
- `repo`: normal repository context
- `long_task`: broader architecture / commercial / analysis context

The profile is used by:

- `planner.py` / `planner_contract.py` for shorter plans
- `runtime.py` / `tool_schema.py` for model-visible tool filtering
- `model_input.py` / `context_manager.py` for smaller context
- `adaptive_runtime.py` for smaller default `max_tokens`
- `cli.py` for visible `token_usage` reporting after each task
- `cli.py` for UTF-8 stdio setup and provider doctor reporting

## 19. Planner Evaluation Contract

Source: `gangent/budget_stats.py`

PlannerQualityReport turns completed runtime data into planner feedback.

Fields:

- `task_kind`
- `outcome`
- `granularity`
- `budget_fit`
- `token_fit`
- `success`
- `findings`
- `recommendations`

Current v1 rules:

- detects `too_coarse` when a small plan required many runtime actions
- detects `too_fine` when a large plan completes only a small fraction of phases
- detects tight budgets from runtime step pressure and exhausted total budget
- records high token-per-step cases as `token_fit=high_context_cost`
- `planner_feedback_for_task` formats similar-task history into model-facing guidance

## 20. Dynamic Context Pack Contract

Source: `gangent/context_manager.py`

DynamicContextPack is the selected context bundle before final prompt formatting.

Fields:

- `must_include`
- `useful_background`
- `warnings`
- `excluded`

Current v1 rules:

- segments with priority >= 90 are kept as must-include task context
- error, event, sensitive, warning, or low-confidence segments are grouped as warnings
- low-priority segments are omitted first under context pressure
- omitted segment titles are written into the Context Pollution Report

## 21. Memory Graph Layer Contract

Source: `gangent/memory_graph.py`

MemoryLayer separates retrieved memory into:

- `data`
- `task`
- `knowledge`

MemoryContextPack groups retrieval results before context assembly.

Fields:

- `data_nodes`
- `task_nodes`
- `knowledge_nodes`
- `conflict_notes`
- `omitted`

Current v1 rules:

- document-like nodes route to data
- task state, decision, issue, and solution nodes route to task memory
- concept, constraint, and preference nodes route to knowledge
- graph expansion keeps relation reasons and depth for debugging

## 22. Event Runtime State Contract

Source: `gangent/events.py`

EventRuntimeState records the cooperative event-driven runtime state.

States:

- `idle`
- `planning`
- `executing`
- `waiting_approval`
- `interrupted`
- `replanning`
- `rolling_back`
- `completed`
- `failed`

EventTransition maps an InterruptDecision into a state transition.

Current v1 rules:

- high-priority user input and explicit replan requests move to `replanning`
- rollback requests move to `waiting_approval`
- pause and high-priority audit warnings move to `interrupted`
- runtime stores `event_runtime_state` in AgentState and checkpoint data
- events are still cooperative: no running tool or model call is force-cancelled

## 22.1 Event-Aware Replanning Contract

Source: `gangent/replanning.py`

Event-aware replanning converts pending runtime events into a bounded plan patch.

Main objects:

- `EventBudget`
- `ReplanContext`
- `PlanPatch`
- `PlanPatchAction`

### EventBudget

Purpose:

- limits automatic replanning pressure
- prevents unlimited user-event loops
- decides whether stabilization mode is required

Fields:

- `event_count`
- `replan_count`
- `interrupt_count`
- `pending_event_count`
- `max_auto_replans`
- `max_interrupts`
- `max_pending_events`

Current v1 rule:

- if replans, interrupts, or pending event batch size exceed limits, runtime enters `stabilize`

### ReplanContext

Purpose:

- packages the original request, new input, current runtime state, and intermediate progress into one auditable structure

Fields:

- `original_user_request`
- `latest_user_events`
- `current_runtime_phase`
- `current_plan_step_title`
- `completed_steps`
- `pending_steps`
- `intermediate_artifacts`
- `current_outputs`
- `constraints`
- `event_budget`

Important design point:

- current state tells the runtime where the first task paused
- original and latest user inputs are both preserved so the next decision can compare semantics
- intermediate artifacts prevent completed evidence from being discarded blindly

### PlanPatch

Purpose:

- describes the smallest deterministic change to the active plan

Fields:

- `action`
- `reason`
- `affected_steps`
- `stale_outputs`
- `new_steps`
- `need_user_confirmation`

Actions:

- `continue`
- `pause`
- `append_steps`
- `replace_pending_steps`
- `mark_outputs_stale`
- `ask_user`
- `stabilize`

Current v1 rules:

- high-priority user input and explicit replan requests replace unfinished plan steps
- new file events append a source-reading step
- rollback requests ask the user instead of mutating files automatically
- high event pressure enters stabilization mode
- completed plan steps are preserved when pending steps are replaced
- patch summaries are stored in `AgentState.plan_patch_summaries`

### AgentState event fields

Source: `gangent/models.py`

Event-aware replanning persists these fields:

- `event_cursor`
- `event_summaries`
- `event_runtime_state`
- `event_count`
- `replan_count`
- `interrupt_count`
- `pending_event_count`
- `stabilization_required`
- `stale_outputs`
- `plan_patch_summaries`

Checkpoint rule:

- all fields above must be serialized and restored by `gangent/checkpoint.py`

## 23. Planner Evaluation Persistence Contract

Source: `gangent/planner_eval.py`

Default path:

- `.gangent/planner/evaluation.jsonl`

Current v1 rules:

- `_finalize_task_result` writes one `PlannerQualityReport` after budget history is recorded
- `planner` CLI command prints recent success rate, granularity distribution, budget fit, top findings, and the last report
- planner evaluation is derived from runtime stats; it is not a learned planner policy yet

## 24. Memory Add Tool Contract

Sources: `gangent/tool_schema.py`, `gangent/tool_runtime.py`, `gangent/policy.py`, `gangent/tool_registry.py`

Tool name:

- `memory_add`

Required args:

- `node_type`
- `content`
- `summary`
- `project_scope`
- `source`
- `tags`
- `importance`
- `confidence`
- `layer`

Current v1 rules:

- writes to `.gangent/memory/graph.json`
- rejects possible secrets before storage
- supports memory layers `data`, `task`, and `knowledge`
- runs through schema, registry, policy, and runtime like other model-visible tools

## 25. CLI Runtime Inspection Contract

Source: `gangent/cli.py`

Current inspection commands:

- `planner`: summarize planner evaluation history
- `context`: build a current dynamic context report from session state
- `events`: list recent runtime events

Current event injection commands:

- `/event <type> <priority> <content>`
- `/replan <content>`
- `/interrupt <content>`

Current v1 rules:

- event commands append to the local JSONL event queue
- `/replan` uses `replan_request` with priority 80
- `/interrupt` uses `user_interrupt` with priority 90
- runtime consumes events only at safe step boundaries
