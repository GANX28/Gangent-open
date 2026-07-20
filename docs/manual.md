# Gangent Manual

This document is the architecture manual for Gangent.

中文说明：这份文档只负责回答三个问题：

1. 现在 Gangent 是什么
2. 现在它做到哪一步了
3. 各模块之间是怎么连起来的

更细的开发扩展说明，请看：

- `docs/extension-guide.md`
- `docs/api-contracts.md`
- `docs/tool-development.md`
- `docs/event-driven-replanning.md`

## 1. Project Positioning

Gangent is a local CLI-first single-agent runtime.

Current positioning:

- local runtime
- real model provider support
- controlled tool calling
- checkpoint / resume
- audit / session persistence
- local retrieval
- extensible hooks / skills / MCP adapter

It is an engineering prototype rather than a hosted production platform.

## 2. High-Level Runtime Flow

```text
User
  -> Event Queue
  -> CLI
  -> SessionState
  -> TaskInput
  -> Adaptive Runtime
  -> Planner
  -> Context Manager
  -> LLM Client
  -> ActionDecision
  -> Policy Check
  -> Tool Registry
  -> Tool Runtime
  -> State Update
  -> Checkpoint / Audit / Session
  -> Final Output
```

Practical meaning of each stage:

- `CLI`: receives user input and handles resume / approval interaction
- `SessionState`: carries cross-task short-term conversation memory
- `TaskInput`: normalizes one task entry
- `Adaptive Runtime`: chooses budget and handles continuation
- `Planner`: gives the task a bounded plan shape
- `Planner Contract`: validates a PlanSpec before it becomes executable runtime steps
- `Planner Budget Control`: turns remaining step budget into model-visible action-granularity rules
- `Context Manager`: assembles prompt context under budget
- `Event Queue`: records external inputs that can be consumed at safe runtime boundaries
- `LLM Client`: talks to the actual provider
- `ActionDecision`: converts free-form model output into runtime actions
- `Policy Check`: decides allow / block / escalate
- `Tool Registry`: maps tool names to executable handlers
- `Tool Runtime`: executes the real local action
- `State Update`: records results into structured state
- `Checkpoint / Audit / Session`: persists recoverable and inspectable state

### Stability-first context controls

- `context_maintenance.py` estimates rough context size without calling another model.
- `context_manager.py` now builds source-aware `ContextSegment` objects with source, scope, priority, confidence, and sensitivity metadata.
- `ContextPollutionReport` records deterministic diagnostics such as source distribution, sensitive segments, low-confidence segments, and budget warnings.
- It records `stable_prefix_hash` so DeepSeek prefix-cache stability can be inspected.
- Large tool results stored back into state are snipped; exact source content can be re-read.
- `read_file` supports line-based chunks and returns numbered lines for safer follow-up edits.
- `file_info` inspects path type, size, binary flag, and line count before choosing a read or directory tool.
- `runtime.py` includes a repeat guard that stops identical failed tool calls after repeated policy/tool failures.
- `planner_contract.py` adds PlanSpec, PlanLinter, and PlanCompiler so planner output can be rejected or compiled before execution.
- `runtime.py` applies a plan guard before tool execution so a phase cannot call tools outside its allowed tool set.
- `events.py` provides a local JSONL event queue and cooperative interrupt policy. The runtime checks events before model calls instead of forcibly killing running tools.
- `replanning.py` turns pending events into `ReplanContext` and `PlanPatch`, so new inputs can revise unfinished work without discarding completed evidence.
- `planner_budget.py` writes the current profile, segment step limit, remaining steps, total step budget, plan progress, and budget pressure into the model context.
- `budget_stats.py` records successful task parameters such as profile, plan step count, completed plan steps, DeepSeek cache hit/miss tokens, and average tokens per step for later planner tuning.
- `providers.py` routes DeepSeek calls through a Flash-first policy and escalates to Pro for ultra or high-risk architecture / audit tasks unless the user explicitly chooses a model.
- `task_profile.py` routes direct answers, single-file reads, and single-file writes through smaller plans, smaller context packs, smaller tool surfaces, and lower default output token budgets.
- `task_profile.py` exposes context tiers: `direct`, `tool`, `repo`, and `long_task`.
- `planner_eval.py` reports step-budget fit and token-budget fit separately through `budget_fit` and `token_fit`.
- `cli.py` prints a final `token_usage` line after each task so real provider cost can be inspected without opening audit logs.
- `cli.py` configures UTF-8 stdio on startup and prints `provider_check`; the `doctor` command shows whether the current provider is fake or a real provider with an API key present.
- `web_shell.py` provides a minimal browser shell so a task can run in the background while new inputs are appended to the event queue.

## 3. Current Module Map

### Runtime control

- `gangent/cli.py`
- `gangent/adaptive_runtime.py`
- `gangent/runtime.py`
- `gangent/planner.py`
- `gangent/planner_contract.py`
- `gangent/planner_budget.py`
- `gangent/task_profile.py`
- `gangent/context_manager.py`
- `gangent/context_maintenance.py`
- `gangent/events.py`
- `gangent/replanning.py`
- `gangent/web_shell.py`

### Model and decision

- `gangent/model_input.py`
- `gangent/llm_client.py`
- `gangent/providers.py`
- `gangent/decision.py`

### Tool execution

- `gangent/tool_schema.py`
- `gangent/schema_validator.py`
- `gangent/tool_registry.py`
- `gangent/tool_runtime.py`
- `gangent/policy.py`
- `gangent/permissions.py`
- `gangent/runner.py`
- `gangent/command_policy.py`
- `gangent/patch_editor.py`

#### Shared JSON Schema Validator

`gangent/schema_validator.py` is the common structural gate for local and
future MCP-backed tools.

Registration path:

```text
ToolDefinition
-> wrapper/name/parameters checks
-> Draft 2020-12 schema check
-> ToolRegistry accepts or rejects the definition
```

Execution path:

```text
ActionDecision
-> Plan Guard
-> shared argument validation
-> Policy / approval
-> handler execution
```

Validation failures are serialized as bounded retryable issues containing the
tool name, argument path, failed rule, and a safe message. The validator never
coerces values and does not copy the actual model-provided value into its own
error message. Runtime keeps the current plan phase available for one corrected
call; existing repeat guards still prevent endless identical retries.

JSON Schema owns structural validity only. Policy still owns permissions,
workspace boundaries, sensitive data, risk, and approval. Handlers still own
real filesystem state and business semantics. The JSON Schema `default` keyword
is descriptive here; safe defaults are applied explicitly by handlers and are
not injected by the validator.

### State and persistence

- `gangent/models.py`
- `gangent/state.py`
- `gangent/checkpoint.py`
- `gangent/runtime_checkpoint.py`
- `gangent/session.py`
- `gangent/session_store.py`
- `gangent/audit.py`
- `gangent/handoff.py`
- `gangent/budget_stats.py`

### Extension layers

- `gangent/hooks.py`
- `gangent/skills.py`
- `gangent/mcp_adapter.py`
- `gangent/rag.py`
- `gangent/embeddings.py`

## 4. Workspace Model

Gangent supports two ways to set `workspace_root`.

### Repo-root mode

Use:

```text
.
```

Use this when Gangent is editing and testing itself.

### Dedicated workspace mode

Use:

```text
.\workspace
```

Use this when Gangent should act on a separate target folder.

Current recommendation:

- for self-hosting development: repo-root mode
- for isolated experiments: dedicated workspace mode

## 5. Safety Boundary

Current safety model is runtime-level control, not OS-level isolation.

Current boundary components:

- path resolution
- allowed read/write roots
- hidden path blocking
- structured command execution
- approval escalation
- timeouts
- output truncation

What this means:

- safer than letting the model write raw shell text directly
- not strong enough to treat as a true hostile sandbox

Future production-grade direction:

- Docker container isolation
- network isolation
- credential isolation
- resource quotas
- stronger policy-to-code

## 6. Retrieval and Memory

Current memory layers:

- session turns
- checkpoint state
- runtime event summaries
- source-aware context segments
- audit logs
- handoff files
- budget history

## 6.1 Event-Aware Replanning

Gangent 当前的 event-driven runtime 是协作式的，不是抢占式的。

它的核心策略是：

```text
do not kill in-flight LLM / tool calls
consume queued events at safe runtime boundaries
compare original request, new input, current progress, and intermediate artifacts
patch only unfinished plan steps when possible
persist the decision into checkpoint state
```

当前事件进入方式：

- `/event <type> <priority> <content>`
- `/replan <content>`
- `/interrupt <content>`
- internal audit / system events

当前事件处理结构：

```text
JsonlEventQueue
  -> evaluate_interrupts
  -> ReplanContext
  -> PlanPatch
  -> AgentState
  -> checkpoint
  -> next model input
```

`ReplanContext` includes:

- original user request
- latest user events
- current runtime phase
- current plan step
- completed steps
- pending steps
- intermediate artifacts
- current outputs
- constraints
- event budget

`PlanPatch` can:

- continue
- pause
- append steps
- replace pending steps
- mark stale outputs
- ask user
- stabilize

Current capability boundary:

- supports event queue, safe-boundary processing, plan patching, checkpoint persistence, and stabilization mode
- supports a minimal browser shell for background task execution plus foreground event input
- does not support true multi-client realtime UI
- does not forcibly cancel in-flight model or tool calls
- does not implement complex disk rollback
- does not implement graph-dependency-level precise replanning

See `docs/event-driven-replanning.md` for the full design note.

### Web Shell

The classic CLI remains synchronous: after a user submits a task, it returns to
the input prompt only after the task ends or pauses.

`gangent.web_shell` adds a small browser shell for validating event-driven
runtime behavior:

```powershell
python -m gangent.web_shell --provider deepseek --workspace-root . --profile auto --port 8765 --open
```

Behavior:

- if no task is running, the composer starts a new background task
- if a task is running, the composer appends a `user_input` event
- quick buttons can append `replan_request`, `user_interrupt`, or `new_file_added`
- the main page behaves like a lightweight ChatGPT / Codex conversation
- transient activity shows model and tool progress such as thinking, reading files, editing files, running commands, and saving checkpoints
- completed transient activity disappears; the conversation keeps user messages and final answers
- the top-right runtime panel can expand to show checkpoint and event queue state
- it shows task status, phase, event runtime state, current step, event counts, stabilization flag, events, and plan patch summaries

This is not a production UI. It is a minimal shell that proves the runtime can
accept new inputs while the task loop continues to run.

Current retrieval:

- chunk local text
- lexical score
- bounded result formatting
- retrieval log output

Not enabled by default:

- model compression
- vector retrieval
- long-term memory
- true multi-client realtime streaming
- forceful interruption of in-flight model/tool calls

## 7. Skills, Hooks, MCP

### Skills

Skills are runtime-side task instruction bundles loaded from `skills/`.

Current limitation:

- static directory loading
- not yet a dynamic plugin platform

### Hooks

Hooks are lifecycle callbacks.

Current events:

- `on_task_start`
- `before_model_call`
- `after_model_call`
- `before_tool_call`
- `after_tool_call`
- `on_checkpoint_save`
- `on_task_finish`

### MCP

MCP = Model Context Protocol（模型上下文协议），也就是让 Agent 通过统一协议连接外部工具、数据源和服务的接口层。

Current Gangent runtime MCP status:

- tool schema adaptation exists
- real server transport is not wired into Gangent's own runtime yet
- no live stdio / HTTP / SSE client loop inside Gangent yet

Important boundary:

- The adapter contract does not mean Gangent's CLI runtime can already call live MCP servers.
- Live MCP support requires connecting `gangent/mcp_adapter.py` to a real stdio / HTTP / SSE transport and registering discovered tools through `tool_registry.py`, Policy, and Audit.

## 8. Known Gaps

Current important gaps relative to mature coding agents:

- no true system sandbox
- no live MCP transport inside Gangent's own runtime
- no dynamic skill / plugin system
- no hybrid retrieval
- no model-based context compression
- no browser toolchain
- tool policy is still mostly static
- command execution is safer, but not fully production-grade

## 9. Recommended Reading Order

If you are extending Gangent, read in this order:

1. `README.md`
2. `docs/manual.md`
3. `docs/extension-guide.md`
4. `docs/api-contracts.md`
5. `docs/event-driven-replanning.md`
6. `docs/tool-development.md`

If you are changing code, inspect the real source before touching behavior.
