# Event-Driven Replanning

本文件记录 Gangent 当前的流式状态控制与事件驱动重规划设计。

中文术语：

- Event-driven runtime：事件驱动运行时
- Cooperative interrupt：协作式打断
- ReplanContext：重规划上下文
- PlanPatch：计划补丁
- EventBudget：事件预算
- Stabilization mode：稳定模式

## 1. 目标

Gangent 需要处理一种真实长任务场景：

```text
任务正在执行
  -> 用户补充新要求
  -> 新文件到达
  -> 审核反馈要求修改
  -> 当前计划不能完全照旧跑
```

普通聊天式 Agent 容易把这些新输入直接塞进历史上下文，导致：

- 模型不知道当前任务跑到哪一步。
- 新要求和旧要求的冲突没有显式判断。
- 已完成结果和未完成步骤混在一起。
- checkpoint 恢复后可能继续执行过期计划。
- 审计时看不清为什么改了计划。

当前实现的目标不是做完整多端实时协作系统，而是先把“运行中接入新输入”变成可记录、可恢复、可测试的 runtime 行为。

## 2. 当前能做到什么程度

当前版本可以做到：

- 通过 JSONL event queue 记录外部输入。
- 通过 `gangent.web_shell` 提供一个轻量浏览器壳子，让任务在后台运行时，前台继续输入事件。
- 在安全边界处理事件，而不是强行中断正在运行的模型或工具调用。
- 把原始用户请求、新事件、当前计划进度、阶段性结果和输出文件状态打包成 `ReplanContext`。
- 生成确定性的 `PlanPatch`，决定是否继续、追加步骤、替换未完成步骤、暂停、询问用户或进入稳定模式。
- 高优先级用户输入和 replan request 会替换未完成步骤，但保留已完成步骤。
- 新文件事件会追加读取步骤。
- rollback request 不自动执行，会要求用户确认。
- 事件过密时进入 stabilization mode，避免无限重规划。
- checkpoint 保存事件处理状态，包括 event cursor、event summaries、event runtime state、event count、replan count、interrupt count、stale outputs 和 plan patch summaries。

当前还不能做到：

- 不支持真正多端 UI 的实时输入流。
- 不强制取消正在运行的 LLM call 或 tool call。
- 原 CLI 仍是同步阻塞式；如果不用 Web Shell 或第二终端写事件，就必须等当前任务输出后才能继续输入。
- 不支持复杂磁盘级 rollback。
- 不支持图依赖级别的精准重规划。
- `PlanPatch` 目前由确定性规则生成，不是完整模型驱动的 plan rewrite。
- 事件处理仍是 safe-boundary polling，不是后台线程抢占式调度。

## 3. 核心流程

```text
User / System / File / Approval Event
  -> JsonlEventQueue
  -> runtime safe boundary
  -> evaluate_interrupts
  -> build ReplanContext
  -> plan_patch_from_events
  -> apply_plan_patch
  -> AgentState / checkpoint
  -> next model call
```

关键点：

- 模型正在调用时不打断。
- 工具正在执行时不打断。
- runtime 在下一次可控边界读取事件。
- 旧输入和新输入都进入 `ReplanContext`，用于语义对比。
- 当前状态用于回答“任务跑到哪一步了”。
- `PlanPatch` 只修改未完成步骤，默认不破坏已完成证据。

## 4. 数据结构

### AgentEvent

来源：`gangent/events.py`

代表一个进入运行时的外部或内部事件。

当前主要事件类型：

- `user_input`
- `new_file_added`
- `file_change`
- `requirement_change`
- `replan_request`
- `user_interrupt`
- `rollback_request`
- `approval`
- `approval_result`
- `audit_warning`
- `tool_result`
- `system_signal`

### ReplanContext

来源：`gangent/replanning.py`

作用是把重规划所需信息集中在一个结构里：

```text
original_user_request
latest_user_events
current_runtime_phase
current_plan_step_title
completed_steps
pending_steps
intermediate_artifacts
current_outputs
constraints
event_budget
```

这里不只包含“当前状态”。当前状态告诉 runtime 第一轮任务跑到哪里了；原始输入和新输入用于比较语义差异；阶段性成果用于避免重复做已完成工作。

### PlanPatch

来源：`gangent/replanning.py`

作用是描述对当前计划的最小修改：

```text
action
reason
affected_steps
stale_outputs
new_steps
need_user_confirmation
```

当前 action：

- `continue`
- `pause`
- `append_steps`
- `replace_pending_steps`
- `mark_outputs_stale`
- `ask_user`
- `stabilize`

### EventBudget

来源：`gangent/replanning.py`

作用是限制事件带来的重规划成本。

当前不是给用户显示“只能输入几次”，而是内部设定软上限：

- `max_auto_replans`
- `max_interrupts`
- `max_pending_events`

超过后进入 stabilization mode，让系统先稳定需求再继续执行。

## 5. 典型场景

### 场景 A：用户补充高优先级要求

```text
原任务：读取资料并生成审核报告
新输入：报告需要同时输出德文版本
```

处理方式：

- runtime 在下一次模型调用前读到事件。
- 构造 `ReplanContext`。
- 生成 `replace_pending_steps`。
- 已完成的读取步骤保留。
- 未完成的报告生成步骤被标记为 blocked。
- 新增“整合高优先级用户输入”的步骤。

### 场景 B：新文件到达

```text
原任务：根据已有说明书生成摘要
新事件：parameters.xlsx arrived
```

处理方式：

- 生成 `append_steps`。
- 追加“读取新到达资料”的步骤。
- 避免模型在未读新文件时直接 finish。

### 场景 C：要求回滚

```text
新事件：rollback last write
```

处理方式：

- 不自动删除或回滚文件。
- 转为 `ask_user`。
- checkpoint 记录原因。

### 场景 D：事件过密

```text
短时间内连续输入多个互相影响的新要求
```

处理方式：

- 进入 `stabilize`。
- 不继续自动重规划。
- 要求先稳定需求，防止无限 loop 和 token 浪费。

## 6. 为什么不用强制中断

强制中断正在运行的 LLM / tool call 会带来几个问题：

- 模型调用可能已经产生费用。
- 工具调用可能已经产生文件副作用。
- 中断点不可控，checkpoint 可能处于半写入状态。
- 审计日志难以解释到底执行到了哪里。

所以当前采用 cooperative interrupt：

```text
不抢占正在执行的动作
只在 runtime 安全边界处理新输入
```

这更适合现在的单机 CLI prototype，也更容易解释为稳定工程设计。

## 7. 和 checkpoint / resume 的关系

事件重规划不是替代 checkpoint，而是补充 checkpoint：

- checkpoint 保存任务状态。
- event cursor 记录哪些事件已经消费。
- event summaries 记录事件如何影响执行。
- plan patch summaries 记录计划为什么改变。
- stale outputs 标记哪些输出可能已经过期。

恢复任务时，runtime 可以知道：

- 之前处理过哪些事件。
- 是否已经进入 replanning。
- 哪些未完成步骤被替换。
- 哪些输出需要重新验证。

## 8. 和 LangGraph 的关系

LangGraph 可以做更完整的状态图、节点、边、interrupt 和 checkpoint。

Gangent 当前实现的是 Harness 层 MVP：

```text
LangGraph 更偏流程编排
Gangent 当前更偏执行可信度、事件接入、计划约束和审计记录
```

未来可以把 Gangent 的 event-aware replanning 接到 LangGraph 后端，让 LangGraph 管节点流转，Gangent 继续管 policy、manifest、validator、audit 和 budget。

## 9. 后续方向

优先级从高到低：

1. 把 `PlanPatch` 的效果写入更明确的 audit log，而不是只写 state summary。
2. 继续完善 Web Shell，把 event health report、audit tail 和 approval flow 放进右侧状态栏。
3. 给 CLI 增加更友好的 multiline / paste 模式，减少长任务输入被拆成多个任务的问题。
4. 增加 event health report，显示哪些事件被消费、哪些步骤被替换、哪些输出过期。
5. 增加轻量模型判断，用于识别新输入和原输入是否冲突。
6. 引入更细的 source / output manifest 绑定，让新文件事件自动影响对应输出。
7. 长期考虑 LangGraph adapter，把 safe boundary 从 loop 扩展成显式 state graph boundary。
