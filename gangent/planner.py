"""Planner v1.

Planner（规划器）负责把用户目标拆成显式步骤。第一版使用确定性计划，
不额外调用大模型，目的是让 runtime 从 reactive loop（反应式循环）
升级为 plan-driven loop（计划驱动循环）。
"""

from __future__ import annotations

from dataclasses import replace

from .models import AgentState, Plan, PlanStep, PlanStepStatus, Task, TaskInput, new_id, utc_now
from .planner_contract import compile_plan_spec, plan_spec_from_task_input
from .task_profile import infer_profile_name


def create_initial_plan(task: Task, task_input: TaskInput) -> Plan:
    """为一次任务创建最小顺序计划。

    第一版不做复杂任务图，只提供稳定的五步骨架：理解、收集、执行、
    验证、总结。每一步都有目的和预期输出，方便模型按步骤行动。
    """

    spec = plan_spec_from_task_input(task_input, infer_task_kind(task_input))
    plan = compile_plan_spec(task, spec)
    if task_input.constraints and plan.steps:
        plan.steps[0].description += " Include the task constraints from TaskInput."
    return plan


def _steps_for_task(task_input: TaskInput) -> list[PlanStep]:
    task_kind = infer_task_kind(task_input)
    templates = _plan_templates()[task_kind]
    return [replace(step, step_id=new_id("step")) for step in templates]


def infer_task_kind(task_input: TaskInput) -> str:
    profile_name = infer_profile_name(task_input)
    if profile_name in {
        "direct",
        "single_read",
        "read_analysis",
        "git_analysis",
        "single_write",
        "read_write",
    }:
        return profile_name
    text = f"{task_input.goal}\n{task_input.user_message}".lower()
    if any(word in text for word in ["fix", "bug", "error", "fail", "debug"]):
        return "debug"
    if any(word in text for word in ["explain", "analyze", "summarize", "review", "compare"]):
        return "analysis"
    if any(word in text for word in ["create", "build", "write", "implement", "add", "make", "edit", "patch", "save", "创建", "写入", "保存", "修改", "实现"]):
        return "build"
    return "general"


def _plan_templates() -> dict[str, list[PlanStep]]:
    return {
        "general": [
            PlanStep(
                step_id="template",
                title="Understand task and constraints",
                description="Clarify the task goal, workspace boundary, and safety constraints.",
                purpose="Avoid acting before the runtime understands the request.",
                expected_output="A clear next action or a decision to inspect the workspace.",
                tool_hint="list_files",
            ),
            PlanStep(
                step_id="template",
                title="Gather relevant context",
                description="Inspect files, repository state, or search relevant project context.",
                purpose="Ground the next action in actual workspace content.",
                expected_output="Relevant files, snippets, or context summary.",
                tool_hint="search_context",
            ),
            PlanStep(
                step_id="template",
                title="Execute the smallest safe action",
                description="Use constrained tools to make the minimal useful progress.",
                purpose="Complete the requested work without unnecessary broad changes.",
                expected_output="A tool result, code change, file update, or direct answer.",
            ),
            PlanStep(
                step_id="template",
                title="Verify the result",
                description="Run safe checks when useful and available.",
                purpose="Detect obvious breakage before reporting completion.",
                expected_output="A verification result or a clear reason verification was skipped.",
                tool_hint="compile_python",
            ),
            PlanStep(
                step_id="template",
                title="Summarize outcome",
                description="Finish with what changed, what was verified, and any remaining limitation.",
                purpose="Return an auditable final answer to the user.",
                expected_output="Final answer through finish_task.",
                tool_hint="finish_task",
            ),
        ],
        "build": [
            PlanStep(
                step_id="template",
                title="Understand the requested feature",
                description="Clarify the requested behavior, boundaries, and workspace constraints.",
                purpose="Prevent implementation from drifting away from the actual task.",
                expected_output="A concrete implementation target and starting file area.",
                tool_hint="list_files",
            ),
            PlanStep(
                step_id="template",
                title="Inspect the existing code path",
                description="Read the relevant files, repo status, or surrounding context before editing.",
                purpose="Align changes with the current project structure.",
                expected_output="Relevant code context and edit target.",
                tool_hint="read_file",
            ),
            PlanStep(
                step_id="template",
                title="Implement the smallest viable change",
                description="Create or edit files with the minimal coherent implementation.",
                purpose="Deliver useful progress while keeping change scope controlled.",
                expected_output="A concrete file change or patch.",
                tool_hint="apply_patch",
            ),
            PlanStep(
                step_id="template",
                title="Verify implementation safety",
                description="Run syntax checks or focused tests that match the changed surface.",
                purpose="Catch obvious regressions before finishing.",
                expected_output="Verification evidence or a stated verification gap.",
                tool_hint="compile_python",
            ),
            PlanStep(
                step_id="template",
                title="Summarize delivered work",
                description="Explain what changed, how it was verified, and what remains open.",
                purpose="Provide a useful engineering handoff.",
                expected_output="Final answer through finish_task.",
                tool_hint="finish_task",
            ),
        ],
        "debug": [
            PlanStep(
                step_id="template",
                title="Frame the failure",
                description="Clarify the failure mode, expected behavior, and current constraints.",
                purpose="Avoid fixing the wrong problem.",
                expected_output="A concrete failure hypothesis and likely surface area.",
                tool_hint="search_context",
            ),
            PlanStep(
                step_id="template",
                title="Inspect evidence and reproduction path",
                description="Read relevant files, logs, diffs, or test surfaces to isolate the issue.",
                purpose="Ground debugging in evidence rather than guessing.",
                expected_output="Relevant code path, failing condition, or narrowed hypothesis.",
                tool_hint="read_file",
            ),
            PlanStep(
                step_id="template",
                title="Apply the fix",
                description="Make the smallest coherent change that addresses the identified cause.",
                purpose="Resolve the failure without broad unrelated edits.",
                expected_output="A focused code change or configuration update.",
                tool_hint="apply_patch",
            ),
            PlanStep(
                step_id="template",
                title="Verify the fix",
                description="Run the most relevant safe validation, such as syntax checks or tests.",
                purpose="Confirm the issue is actually resolved.",
                expected_output="A passing check, narrower failure, or explicit remaining risk.",
                tool_hint="run_tests",
            ),
            PlanStep(
                step_id="template",
                title="Summarize the root cause and fix",
                description="State what broke, why, what changed, and any remaining caveat.",
                purpose="Turn the debugging result into reusable engineering context.",
                expected_output="Final answer through finish_task.",
                tool_hint="finish_task",
            ),
        ],
        "analysis": [
            PlanStep(
                step_id="template",
                title="Clarify the analysis question",
                description="Pin down what must be explained, compared, or reviewed.",
                purpose="Keep the answer scoped to the actual question.",
                expected_output="A clear analysis target and evidence plan.",
                tool_hint="search_context",
            ),
            PlanStep(
                step_id="template",
                title="Collect supporting evidence",
                description="Read relevant files, status, diffs, or search the local context store.",
                purpose="Support the answer with actual repository evidence.",
                expected_output="Relevant snippets, changed files, or repository facts.",
                tool_hint="read_file",
            ),
            PlanStep(
                step_id="template",
                title="Synthesize the findings",
                description="Connect the evidence into a clear technical conclusion.",
                purpose="Convert raw facts into an actionable explanation.",
                expected_output="A concise set of conclusions and tradeoffs.",
            ),
            PlanStep(
                step_id="template",
                title="Check for missing validation",
                description="Verify whether any supporting check or comparison is still needed.",
                purpose="Avoid overclaiming beyond the gathered evidence.",
                expected_output="Explicit confidence level and any remaining gap.",
                tool_hint="git_diff",
            ),
            PlanStep(
                step_id="template",
                title="Deliver the answer",
                description="Return the analysis in a clean, auditable summary.",
                purpose="Produce a useful final explanation rather than raw notes.",
                expected_output="Final answer through finish_task.",
                tool_hint="finish_task",
            ),
        ],
    }


def attach_plan(state: AgentState, plan: Plan) -> AgentState:
    """把计划挂到 AgentState 上。"""

    state.plan_id = plan.plan_id
    state.plan_steps = list(plan.steps)
    state.updated_at = utc_now()
    return state


def current_plan_step(state: AgentState) -> PlanStep | None:
    """返回当前应执行的计划步骤。"""

    for step in state.plan_steps:
        if step.status in {PlanStepStatus.TODO, PlanStepStatus.RUNNING}:
            return step
    return None


def start_current_plan_step(state: AgentState) -> PlanStep | None:
    """把当前 TODO 步骤标记为 RUNNING。"""

    step = current_plan_step(state)
    if step and step.status == PlanStepStatus.TODO:
        step.status = PlanStepStatus.RUNNING
        state.updated_at = utc_now()
    return step


def complete_current_plan_step(state: AgentState, result_summary: str) -> PlanStep | None:
    """把当前 RUNNING 步骤标记为 DONE。"""

    step = current_plan_step(state)
    if step and step.status == PlanStepStatus.RUNNING:
        step.status = PlanStepStatus.DONE
        step.result_summary = _trim_summary(result_summary)
        state.updated_at = utc_now()
    return step


def block_current_plan_step(state: AgentState, reason: str) -> PlanStep | None:
    """把当前 RUNNING/TODO 步骤标记为 BLOCKED。"""

    step = current_plan_step(state)
    if step:
        step.status = PlanStepStatus.BLOCKED
        step.result_summary = _trim_summary(reason)
        state.updated_at = utc_now()
    return step


def format_plan_for_model(state: AgentState) -> str:
    """生成给模型看的计划摘要。"""

    if not state.plan_steps:
        return "(no plan)"

    lines: list[str] = []
    for index, step in enumerate(state.plan_steps, start=1):
        marker = "current" if step == current_plan_step(state) else step.status.value
        line = f"{index}. [{marker}] {step.title}"
        if step.purpose:
            line += f" | purpose: {step.purpose}"
        if step.expected_output:
            line += f" | expected: {step.expected_output}"
        if step.tool_hint:
            line += f" | tool_hint: {step.tool_hint}"
        if step.result_summary:
            line += f" | result: {step.result_summary}"
        lines.append(line)
    return "\n".join(lines)


def _trim_summary(text: str, max_chars: int = 240) -> str:
    compact = text.replace("\n", " ").strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."
