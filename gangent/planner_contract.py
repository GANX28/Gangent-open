"""Commercial-style planner contract.

The planner is treated as a candidate plan producer, not as an unrestricted
executor. A PlanSpec is linted first, then compiled into runtime PlanSteps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Plan, PlanStep, Task, TaskInput, new_id
from .planner_budget import build_planner_budget_control


READ_TOOLS = (
    "list_files",
    "search_context",
    "read_file",
    "read_many_files",
    "file_info",
    "grep_files",
    "git_status",
    "git_diff",
    "git_log",
    "git_show",
)
WRITE_TOOLS = ("write_file", "edit_file", "apply_patch", "export_artifact", "scratchpad_note", "memory_add")
VERIFY_TOOLS = ("compile_python", "run_tests", "git_diff")
FINISH_TOOLS = ("finish_task",)


@dataclass(frozen=True)
class PlanPhaseSpec:
    """One bounded phase in a plan candidate."""

    name: str
    goal: str
    max_steps: int
    allowed_tools: tuple[str, ...] = ()
    exit_criteria: str = ""


@dataclass(frozen=True)
class VerificationSpec:
    """Verification requirements for one plan."""

    required: bool = True
    allowed_tools: tuple[str, ...] = VERIFY_TOOLS
    exit_criteria: str = "Verification evidence is recorded or an explicit verification gap is stated."


@dataclass(frozen=True)
class PlanSpec:
    """A lintable plan candidate before runtime execution."""

    task_kind: str
    risk_level: str
    max_plan_steps: int
    phases: tuple[PlanPhaseSpec, ...]
    verification: VerificationSpec = field(default_factory=VerificationSpec)


@dataclass(frozen=True)
class PlanLintFinding:
    """One deterministic planner validation finding."""

    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class PlanLintReport:
    """Plan linter result."""

    valid: bool
    findings: tuple[PlanLintFinding, ...] = ()


def plan_spec_from_task_input(task_input: TaskInput, task_kind: str) -> PlanSpec:
    """Build the deterministic v1 PlanSpec for a task kind."""

    if task_kind == "direct":
        phases = (
            PlanPhaseSpec("deliver", "Answer directly without workspace inspection.", 1, FINISH_TOOLS, "Final answer is ready."),
        )
        return PlanSpec("direct", "low", 1, phases, VerificationSpec(required=False))
    if task_kind == "single_read":
        phases = (
            PlanPhaseSpec(
                "read",
                "Read the explicitly requested file; prefer read_file over file_info when file content is needed.",
                1,
                ("read_file", "file_info", "list_files"),
                "Requested file content is known.",
            ),
            PlanPhaseSpec("deliver", "Answer from the retrieved file evidence.", 1, FINISH_TOOLS, "Final answer is ready."),
        )
        return PlanSpec("single_read", "low", 2, phases, VerificationSpec(required=False))
    if task_kind == "read_analysis":
        phases = (
            PlanPhaseSpec(
                "read",
                "Read all explicitly requested source files. Prefer read_many_files when several paths are known; do not modify files.",
                4,
                ("list_files", "read_file", "read_many_files", "file_info", "grep_files", "search_context"),
                "Requested source content is available for synthesis.",
            ),
            PlanPhaseSpec(
                "synthesize",
                "Summarize or analyze from gathered evidence only. If evidence is partial, use read tools for follow-up chunks; do not modify files.",
                4,
                ("list_files", "read_file", "read_many_files", "file_info", "grep_files", "search_context", "finish_task"),
                "Final answer is ready.",
            ),
        )
        return PlanSpec("read_analysis", "low", 2, phases, VerificationSpec(required=False))
    if task_kind == "git_analysis":
        phases = (
            PlanPhaseSpec(
                "inspect_git",
                "Inspect read-only Git state with git_status, git_diff, git_log, or git_show; do not modify files.",
                4,
                ("git_status", "git_diff", "git_log", "git_show"),
                "Git evidence relevant to the question is available.",
            ),
            PlanPhaseSpec(
                "synthesize",
                "Summarize Git evidence. If a small follow-up Git lookup is needed, use a read-only Git tool; otherwise finish.",
                3,
                ("git_status", "git_diff", "git_log", "git_show", "finish_task"),
                "Final answer is ready.",
            ),
        )
        return PlanSpec("git_analysis", "low", 2, phases, VerificationSpec(required=False))
    if task_kind == "single_write":
        phases = (
            PlanPhaseSpec(
                "write",
                "Create or update the explicitly requested file with the smallest content needed.",
                2,
                ("read_file", "read_many_files", "write_file", "edit_file", "file_info"),
                "Requested file change is done.",
            ),
            PlanPhaseSpec("deliver", "Summarize the completed file change.", 1, FINISH_TOOLS, "Final answer is ready."),
        )
        return PlanSpec("single_write", "low", 2, phases, VerificationSpec(required=False))
    if task_kind == "read_write":
        phases = (
            PlanPhaseSpec(
                "read",
                "Read the requested source file content. Use read_file or read_many_files, not file_info, when the output depends on content.",
                2,
                ("read_file", "read_many_files"),
                "Source content is available for transformation.",
            ),
            PlanPhaseSpec(
                "write",
                "Write the requested derived output file using the source evidence; use read tools only for small evidence lookups.",
                2,
                ("read_file", "read_many_files", "write_file", "edit_file"),
                "Derived output file is written.",
            ),
            PlanPhaseSpec("deliver", "Summarize the completed transformation.", 1, FINISH_TOOLS, "Final answer is ready."),
        )
        return PlanSpec("read_write", "low", 4, phases, VerificationSpec(required=False))
    if task_kind == "build":
        phases = (
            PlanPhaseSpec("understand", "Understand requested behavior and constraints.", 2, READ_TOOLS + WRITE_TOOLS, "Implementation target is clear."),
            PlanPhaseSpec("inspect", "Inspect the existing code path before editing.", 3, READ_TOOLS, "Relevant files and edit surface are known."),
            PlanPhaseSpec("implement", "Apply the smallest coherent implementation.", 4, WRITE_TOOLS, "A concrete patch or file change exists."),
            PlanPhaseSpec("verify", "Verify implementation safety.", 3, VERIFY_TOOLS, "Focused verification is complete or explicitly skipped."),
            PlanPhaseSpec("summarize", "Summarize delivered work.", 1, FINISH_TOOLS, "Final answer is ready."),
        )
        return PlanSpec("build", "medium", 8, phases, VerificationSpec(required=True))
    if task_kind == "debug":
        phases = (
            PlanPhaseSpec("frame", "Frame the failure and expected behavior.", 2, ("search_context", "git_diff"), "A concrete failure hypothesis exists."),
            PlanPhaseSpec("inspect", "Inspect evidence and reproduction path.", 4, READ_TOOLS + ("run_tests",), "The likely cause is identified."),
            PlanPhaseSpec("fix", "Apply the smallest fix.", 4, WRITE_TOOLS, "A focused fix exists."),
            PlanPhaseSpec("verify", "Verify the fix.", 4, VERIFY_TOOLS, "The fix is validated or a remaining risk is stated."),
            PlanPhaseSpec("summarize", "Summarize root cause and fix.", 1, FINISH_TOOLS, "Final answer is ready."),
        )
        return PlanSpec("debug", "high", 8, phases, VerificationSpec(required=True))
    if task_kind == "analysis":
        phases = (
            PlanPhaseSpec("clarify", "Clarify the analysis question and evidence needs.", 2, ("search_context", "git_status"), "Analysis target is clear."),
            PlanPhaseSpec("collect", "Collect supporting evidence.", 4, READ_TOOLS, "Evidence is gathered."),
            PlanPhaseSpec("synthesize", "Synthesize findings into conclusions.", 2, (), "Conclusions and tradeoffs are clear."),
            PlanPhaseSpec("check", "Check for missing validation.", 2, ("git_diff", "search_context"), "Confidence and gaps are explicit."),
            PlanPhaseSpec("deliver", "Deliver the answer.", 1, FINISH_TOOLS, "Final answer is ready."),
        )
        return PlanSpec("analysis", "low", 7, phases, VerificationSpec(required=False))
    phases = (
        PlanPhaseSpec("understand", "Understand task and constraints.", 2, READ_TOOLS, "Next action is clear."),
        PlanPhaseSpec("gather", "Gather relevant context.", 3, READ_TOOLS, "Relevant context is known."),
        PlanPhaseSpec("execute", "Execute the smallest safe action.", 3, READ_TOOLS + WRITE_TOOLS, "Requested progress is made."),
        PlanPhaseSpec("verify", "Verify the result when useful.", 2, VERIFY_TOOLS, "Result is checked or verification gap is stated."),
        PlanPhaseSpec("summarize", "Summarize outcome.", 1, FINISH_TOOLS, "Final answer is ready."),
    )
    return PlanSpec("general", "low", 7, phases, VerificationSpec(required=False))


def lint_plan_spec(spec: PlanSpec) -> PlanLintReport:
    """Validate plan shape before compiling it into runtime steps."""

    findings: list[PlanLintFinding] = []
    if not spec.phases:
        findings.append(_finding("error", "empty_plan", "Plan must contain at least one phase."))
    if len(spec.phases) > spec.max_plan_steps:
        findings.append(_finding("error", "too_many_phases", "Plan has more phases than max_plan_steps."))
    if spec.max_plan_steps <= 0:
        findings.append(_finding("error", "invalid_max_plan_steps", "max_plan_steps must be positive."))

    names = set()
    for phase in spec.phases:
        if not phase.name.strip():
            findings.append(_finding("error", "missing_phase_name", "Every phase must have a name."))
        if phase.name in names:
            findings.append(_finding("error", "duplicate_phase", f"Duplicate phase name: {phase.name}."))
        names.add(phase.name)
        if phase.max_steps <= 0:
            findings.append(_finding("error", "invalid_phase_budget", f"Phase {phase.name} must have max_steps > 0."))
        if phase.max_steps > 8:
            findings.append(_finding("warning", "oversized_phase", f"Phase {phase.name} may be too broad."))
        if not phase.exit_criteria:
            findings.append(_finding("warning", "missing_exit_criteria", f"Phase {phase.name} has no exit criteria."))
        if _looks_like_mixed_phase(phase.goal):
            findings.append(_finding("warning", "mixed_phase_goal", f"Phase {phase.name} may mix unrelated actions."))

    if spec.verification.required and "verify" not in names:
        findings.append(_finding("error", "missing_required_verification", "Risky plans must include a verify phase."))

    return PlanLintReport(
        valid=not any(finding.severity == "error" for finding in findings),
        findings=tuple(findings),
    )


def compile_plan_spec(task: Task, spec: PlanSpec) -> Plan:
    """Compile a valid PlanSpec into runtime PlanSteps."""

    report = lint_plan_spec(spec)
    if not report.valid:
        messages = "; ".join(f"{item.code}: {item.message}" for item in report.findings)
        raise ValueError(f"Invalid PlanSpec: {messages}")
    return Plan(
        plan_id=new_id("plan"),
        task_id=task.task_id,
        goal=task.goal,
        steps=[
            PlanStep(
                step_id=new_id("step"),
                title=_phase_title(phase),
                description=(
                    f"phase={phase.name}; max_steps={phase.max_steps}; "
                    f"allowed_tools={','.join(phase.allowed_tools) or '(none)'}; "
                    f"exit_criteria={phase.exit_criteria}"
                ),
                purpose=phase.goal,
                expected_output=phase.exit_criteria,
                tool_hint=phase.allowed_tools[0] if phase.allowed_tools else None,
            )
            for phase in spec.phases
        ],
    )


def planner_contract_summary(task_input: TaskInput, state) -> str:
    """Return a compact model-facing contract summary."""

    from .planner import infer_task_kind

    spec = plan_spec_from_task_input(task_input, infer_task_kind(task_input))
    report = lint_plan_spec(spec)
    budget = build_planner_budget_control(state)
    findings = ", ".join(f"{item.severity}:{item.code}" for item in report.findings) or "none"
    return (
        "Planner Contract:\n"
        f"- task_kind: {spec.task_kind}\n"
        f"- risk_level: {spec.risk_level}\n"
        f"- max_plan_steps: {spec.max_plan_steps}\n"
        f"- lint_valid: {str(report.valid).lower()}\n"
        f"- lint_findings: {findings}\n"
        f"- current_budget_pressure: {budget.pressure}\n"
        "- planner is a candidate producer; runtime, policy, budget, and guardrails are authoritative"
    )


def _finding(severity: str, code: str, message: str) -> PlanLintFinding:
    return PlanLintFinding(severity=severity, code=code, message=message)


def _phase_title(phase: PlanPhaseSpec) -> str:
    titles = {
        "understand": "Understand the requested feature"
        if "requested behavior" in phase.goal.lower()
        else "Understand task and constraints",
        "inspect": "Inspect the existing code path",
        "implement": "Implement the smallest viable change",
        "verify": "Verify implementation safety"
        if "implementation" in phase.goal.lower()
        else "Verify the result",
        "summarize": _summarize_title(phase.goal),
        "frame": "Frame the failure",
        "fix": "Apply the fix",
        "clarify": "Clarify the analysis question",
        "collect": "Collect supporting evidence",
        "synthesize": "Synthesize the findings",
        "check": "Check for missing validation",
        "deliver": "Deliver the answer",
        "read": "Read requested file",
        "inspect_git": "Inspect Git state",
        "synthesize": "Synthesize the findings",
        "write": "Write requested file",
        "gather": "Gather relevant context",
        "execute": "Execute the smallest safe action",
    }
    return titles.get(phase.name, phase.name.replace("_", " ").title())


def _summarize_title(goal: str) -> str:
    lowered = goal.lower()
    if "root cause" in lowered:
        return "Summarize the root cause and fix"
    if "delivered" in lowered:
        return "Summarize delivered work"
    return "Summarize outcome"


def _looks_like_mixed_phase(goal: str) -> bool:
    lowered = goal.lower()
    markers = ["inspect", "edit", "verify", "summarize", "test", "implement", "search"]
    return sum(1 for marker in markers if marker in lowered) >= 3
