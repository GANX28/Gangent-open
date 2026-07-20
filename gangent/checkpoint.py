"""TaskCheckpoint（任务检查点）持久化层。

Checkpoint（检查点）把一次任务的关键执行状态存下来，让 runtime
可以在中断后从已完成的阶段继续，而不是从头重跑。
第一版只做本地 JSON 存档，不做数据库、分布式恢复或加密同步。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .failure import is_recoverable_failure
from .models import (
    ActionDecision,
    AgentState,
    AuditEntry,
    AgentPhase,
    TaskInput,
    Message,
    MessageRole,
    Plan,
    PlanStep,
    PlanStepStatus,
    DecisionType,
    PolicyDecision,
    PolicyMode,
    RuntimeStats,
    Task,
    TaskStatus,
    ToolResult,
)


DEFAULT_CHECKPOINT_FILE = Path(".gangent") / "checkpoints" / "latest.json"
DEFAULT_CHECKPOINT_ARCHIVE_DIR = Path(".gangent") / "checkpoints" / "archive"
DEFAULT_IGNORED_TASKS_FILE = Path(".gangent") / "checkpoints" / "ignored_tasks.json"


@dataclass
class StepCheckpoint:
    """一轮 runtime step 的可序列化记录。"""

    step_index: int
    decision: dict[str, Any]
    policy: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    approval_required: bool = False
    approved: bool | None = None


@dataclass
class TaskCheckpoint:
    """一次任务的恢复点。"""

    task_input: TaskInput
    task: Task
    state: AgentState
    steps: list[StepCheckpoint] = field(default_factory=list)
    stats: RuntimeStats = field(default_factory=RuntimeStats)
    version: int = 1


@dataclass(frozen=True)
class CheckpointCandidate:
    """One resumable checkpoint discovered in active or archived storage."""

    path: Path
    checkpoint: TaskCheckpoint
    is_active: bool = False


def default_checkpoint_path(workspace_root: str) -> Path:
    return Path(workspace_root).resolve() / DEFAULT_CHECKPOINT_FILE


def default_checkpoint_archive_dir(workspace_root: str) -> Path:
    return Path(workspace_root).resolve() / DEFAULT_CHECKPOINT_ARCHIVE_DIR


def checkpoint_archive_dir_for_active_path(workspace_root: str, active_path: str | Path | None = None) -> Path:
    """Return the archive directory paired with an active checkpoint path.

    The normal CLI uses `.gangent/checkpoints/latest.json` and therefore shares
    `.gangent/checkpoints/archive`. Tests and external callers may pass a custom
    active checkpoint path; in that case its archive should live beside it so
    isolated runs do not see the main workspace's old checkpoints.
    """

    if active_path is None:
        return default_checkpoint_archive_dir(workspace_root)
    active = Path(active_path)
    resolved_active = active.resolve()
    try:
        if resolved_active == default_checkpoint_path(workspace_root).resolve():
            return default_checkpoint_archive_dir(workspace_root)
    except Exception:
        pass
    return active.parent / "archive"


def default_ignored_tasks_path(workspace_root: str) -> Path:
    return Path(workspace_root).resolve() / DEFAULT_IGNORED_TASKS_FILE


def checkpoint_from_runtime_result(result: Any) -> TaskCheckpoint:
    """从 RuntimeResult 创建检查点。

    这里用 Any 是为了避免 runtime.py 导入 checkpoint.py 时形成循环依赖。
    """

    steps = [_step_checkpoint_from_runtime_step(step) for step in getattr(result, "steps", [])]
    return TaskCheckpoint(
        task_input=task_input_from_dict(task_input_to_dict(getattr(result, "task_input"))),
        task=_copy_task(getattr(result, "task")),
        state=_copy_state(getattr(result, "state")),
        steps=steps,
        stats=_copy_stats(getattr(result, "stats", None)),
    )


def save_checkpoint(checkpoint: TaskCheckpoint, path: str | Path | None = None) -> Path:
    """把检查点保存为 JSON 文件。"""

    if path is None:
        path = default_checkpoint_path(checkpoint.state.workspace_root)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(checkpoint_to_dict(checkpoint), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def archive_checkpoint(checkpoint: TaskCheckpoint, path: str | Path | None = None) -> Path:
    """Archive one completed/non-resumable checkpoint by task id."""

    if path is None:
        archive_dir = default_checkpoint_archive_dir(checkpoint.state.workspace_root)
    else:
        archive_dir = Path(path)
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / f"{checkpoint.task.task_id}.json"
    target.write_text(
        json.dumps(checkpoint_to_dict(checkpoint), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def clear_active_checkpoint(path: str | Path) -> None:
    """Remove the active checkpoint file if it exists."""

    target = Path(path)
    if target.exists():
        target.unlink()


def load_checkpoint(path: str | Path) -> TaskCheckpoint:
    """从 JSON 文件读取检查点。"""

    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    return checkpoint_from_dict(data)


def checkpoint_from_dict(data: dict[str, Any]) -> TaskCheckpoint:
    task_input_data = data.get("task_input")
    if not task_input_data:
        task_data = data.get("task", {})
        state_data = data.get("state", {})
        messages = state_data.get("messages", [])
        user_message = ""
        for message in messages:
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break
        task_input_data = {
            "goal": task_data.get("goal", user_message),
            "user_message": user_message or task_data.get("goal", ""),
            "workspace_root": state_data.get("workspace_root", ""),
            "constraints": [],
            "created_at": task_data.get("created_at", ""),
        }
    return TaskCheckpoint(
        task_input=task_input_from_dict(task_input_data),
        task=task_from_dict(data["task"]),
        state=state_from_dict(data["state"]),
        steps=[step_from_dict(item) for item in data.get("steps", [])],
        stats=stats_from_dict(data.get("stats", {})),
        version=int(data.get("version", 1)),
    )


def checkpoint_to_dict(checkpoint: TaskCheckpoint) -> dict[str, Any]:
    return {
        "version": checkpoint.version,
        "task_input": task_input_to_dict(checkpoint.task_input),
        "task": task_to_dict(checkpoint.task),
        "state": state_to_dict(checkpoint.state),
        "steps": [asdict(step) for step in checkpoint.steps],
        "stats": stats_to_dict(checkpoint.stats),
    }


def _step_checkpoint_from_runtime_step(step: Any) -> StepCheckpoint:
    return StepCheckpoint(
        step_index=int(getattr(step, "step_index", 0)),
        decision=decision_to_dict(getattr(step, "decision", None)),
        policy=policy_to_dict(getattr(step, "policy", None)),
        tool_result=tool_result_to_dict(getattr(step, "tool_result", None)),
        usage=dict(getattr(step, "usage", None) or {}),
        approval_required=bool(getattr(step, "approval_required", False)),
        approved=getattr(step, "approved", None),
    )


def step_from_dict(data: dict[str, Any]) -> StepCheckpoint:
    return StepCheckpoint(
        step_index=int(data.get("step_index", 0)),
        decision=dict(data.get("decision", {})),
        policy=dict(data["policy"]) if data.get("policy") else None,
        tool_result=dict(data["tool_result"]) if data.get("tool_result") else None,
        usage=dict(data.get("usage", {})),
        approval_required=bool(data.get("approval_required", False)),
        approved=data.get("approved"),
    )


def _copy_task(task: Task) -> Task:
    return task_from_dict(task_to_dict(task))


def _copy_state(state: AgentState) -> AgentState:
    return state_from_dict(state_to_dict(state))


def _copy_stats(stats: RuntimeStats | None) -> RuntimeStats:
    return stats_from_dict(stats_to_dict(stats or RuntimeStats()))


def task_to_dict(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "goal": task.goal,
        "status": task.status.value,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


def task_from_dict(data: dict[str, Any]) -> Task:
    return Task(
        task_id=data["task_id"],
        goal=data["goal"],
        status=TaskStatus(data.get("status", TaskStatus.PENDING.value)),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
    )


def task_input_to_dict(task_input: TaskInput) -> dict[str, Any]:
    return {
        "goal": task_input.goal,
        "user_message": task_input.user_message,
        "workspace_root": task_input.workspace_root,
        "constraints": list(task_input.constraints),
        "created_at": task_input.created_at,
    }


def task_input_from_dict(data: dict[str, Any]) -> TaskInput:
    return TaskInput(
        goal=data["goal"],
        user_message=data["user_message"],
        workspace_root=data["workspace_root"],
        constraints=list(data.get("constraints", [])),
        created_at=data.get("created_at", ""),
    )


def state_to_dict(state: AgentState) -> dict[str, Any]:
    return {
        "task_id": state.task_id,
        "workspace_root": state.workspace_root,
        "phase": state.phase.value,
        "step_index": state.step_index,
        "context_summary": state.context_summary,
        "plan_id": state.plan_id,
        "messages": [message_to_dict(message) for message in state.messages],
        "plan_steps": [plan_step_to_dict(step) for step in state.plan_steps],
        "event_cursor": state.event_cursor,
        "event_summaries": list(state.event_summaries),
        "event_runtime_state": state.event_runtime_state,
        "event_count": state.event_count,
        "replan_count": state.replan_count,
        "interrupt_count": state.interrupt_count,
        "pending_event_count": state.pending_event_count,
        "stabilization_required": state.stabilization_required,
        "stale_outputs": list(state.stale_outputs),
        "plan_patch_summaries": list(state.plan_patch_summaries),
        "budget_profile": state.budget_profile,
        "runtime_step_limit": state.runtime_step_limit,
        "runtime_remaining_steps": state.runtime_remaining_steps,
        "total_step_budget": state.total_step_budget,
        "total_remaining_steps": state.total_remaining_steps,
        "last_decision": decision_to_dict(state.last_decision),
        "last_tool_result": tool_result_to_dict(state.last_tool_result),
        "errors": list(state.errors),
        "updated_at": state.updated_at,
    }


def state_from_dict(data: dict[str, Any]) -> AgentState:
    return AgentState(
        task_id=data["task_id"],
        workspace_root=data.get("workspace_root", ""),
        phase=AgentPhase(data.get("phase", AgentPhase.IDLE.value)),
        step_index=int(data.get("step_index", 0)),
        context_summary=data.get("context_summary", ""),
        plan_id=data.get("plan_id"),
        messages=[message_from_dict(item) for item in data.get("messages", [])],
        plan_steps=[plan_step_from_dict(item) for item in data.get("plan_steps", [])],
        event_cursor=int(data.get("event_cursor", 0)),
        event_summaries=list(data.get("event_summaries", [])),
        event_runtime_state=data.get("event_runtime_state", "idle"),
        event_count=int(data.get("event_count", 0)),
        replan_count=int(data.get("replan_count", 0)),
        interrupt_count=int(data.get("interrupt_count", 0)),
        pending_event_count=int(data.get("pending_event_count", 0)),
        stabilization_required=bool(data.get("stabilization_required", False)),
        stale_outputs=list(data.get("stale_outputs", [])),
        plan_patch_summaries=list(data.get("plan_patch_summaries", [])),
        budget_profile=data.get("budget_profile", ""),
        runtime_step_limit=int(data.get("runtime_step_limit", 0)),
        runtime_remaining_steps=int(data.get("runtime_remaining_steps", 0)),
        total_step_budget=int(data.get("total_step_budget", 0)),
        total_remaining_steps=int(data.get("total_remaining_steps", 0)),
        last_decision=decision_from_dict(data.get("last_decision")),
        last_tool_result=tool_result_from_dict(data.get("last_tool_result")),
        errors=list(data.get("errors", [])),
        updated_at=data.get("updated_at", ""),
    )


def plan_step_to_dict(step: PlanStep) -> dict[str, Any]:
    return {
        "step_id": step.step_id,
        "title": step.title,
        "status": step.status.value,
        "description": step.description,
        "purpose": step.purpose,
        "expected_output": step.expected_output,
        "tool_hint": step.tool_hint,
        "result_summary": step.result_summary,
    }


def plan_step_from_dict(data: dict[str, Any]) -> PlanStep:
    return PlanStep(
        step_id=data["step_id"],
        title=data["title"],
        status=PlanStepStatus(data.get("status", PlanStepStatus.TODO.value)),
        description=data.get("description", ""),
        purpose=data.get("purpose", ""),
        expected_output=data.get("expected_output", ""),
        tool_hint=data.get("tool_hint"),
        result_summary=data.get("result_summary", ""),
    )


def message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "role": message.role.value,
        "content": message.content,
        "timestamp": message.timestamp,
    }


def message_from_dict(data: dict[str, Any]) -> Message:
    return Message(
        role=MessageRole(data["role"]),
        content=data["content"],
        timestamp=data.get("timestamp", ""),
    )


def decision_to_dict(decision: ActionDecision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "decision_type": decision.decision_type.value,
        "reason": decision.reason,
        "tool_name": decision.tool_name,
        "tool_args": decision.tool_args,
        "response_text": decision.response_text,
    }


def decision_from_dict(data: dict[str, Any] | None) -> ActionDecision | None:
    if not data:
        return None
    return ActionDecision(
        decision_type=DecisionType(data["decision_type"]),
        reason=data.get("reason", ""),
        tool_name=data.get("tool_name"),
        tool_args=data.get("tool_args"),
        response_text=data.get("response_text"),
    )


def policy_to_dict(policy: PolicyDecision | None) -> dict[str, Any] | None:
    if policy is None:
        return None
    return {
        "mode": policy.mode.value,
        "allowed": policy.allowed,
        "reason": policy.reason,
    }


def policy_from_dict(data: dict[str, Any] | None) -> PolicyDecision | None:
    if not data:
        return None
    return PolicyDecision(
        mode=PolicyMode(data["mode"]),
        allowed=bool(data.get("allowed", False)),
        reason=data.get("reason", ""),
    )


def tool_result_to_dict(tool_result: ToolResult | None) -> dict[str, Any] | None:
    if tool_result is None:
        return None
    return {
        "call_id": tool_result.call_id,
        "success": tool_result.success,
        "output": tool_result.output,
        "error": tool_result.error,
        "reused": tool_result.reused,
        "snipped": tool_result.snipped,
        "finished_at": tool_result.finished_at,
    }


def tool_result_from_dict(data: dict[str, Any] | None) -> ToolResult | None:
    if not data:
        return None
    return ToolResult(
        call_id=data["call_id"],
        success=bool(data.get("success", False)),
        output=data.get("output", ""),
        error=data.get("error"),
        reused=bool(data.get("reused", False)),
        snipped=bool(data.get("snipped", False)),
        finished_at=data.get("finished_at", ""),
    )


def stats_to_dict(stats: RuntimeStats) -> dict[str, Any]:
    return {
        "duration_seconds": stats.duration_seconds,
        "step_count": stats.step_count,
        "tool_call_count": stats.tool_call_count,
        "error_count": stats.error_count,
        "usage": stats.usage,
    }


def stats_from_dict(data: dict[str, Any]) -> RuntimeStats:
    return RuntimeStats(
        duration_seconds=float(data.get("duration_seconds", 0.0)),
        step_count=int(data.get("step_count", 0)),
        tool_call_count=int(data.get("tool_call_count", 0)),
        error_count=int(data.get("error_count", 0)),
        usage=dict(data.get("usage", {})),
    )


def is_resume_candidate(checkpoint: TaskCheckpoint) -> bool:
    """Return whether this checkpoint should auto-resume."""

    if checkpoint.task.status == TaskStatus.RUNNING:
        return True
    if checkpoint.task.status != TaskStatus.FAILED:
        return False
    return is_recoverable_failure(checkpoint.state.errors)


def checkpoint_matches_task_input(checkpoint: TaskCheckpoint, task_input: TaskInput) -> bool:
    """Return whether a resumable checkpoint belongs to the current user request."""

    if not _workspace_roots_match(checkpoint.task_input.workspace_root, task_input.workspace_root):
        return False

    checkpoint_texts = [
        _normalize_resume_text(checkpoint.task_input.goal),
        _normalize_resume_text(checkpoint.task_input.user_message),
    ]
    current_texts = [
        _normalize_resume_text(task_input.goal),
        _normalize_resume_text(task_input.user_message),
    ]

    checkpoint_texts = [text for text in checkpoint_texts if text]
    current_texts = [text for text in current_texts if text]
    if not checkpoint_texts or not current_texts:
        return False

    for checkpoint_text in checkpoint_texts:
        for current_text in current_texts:
            if checkpoint_text == current_text:
                return True
            if min(len(checkpoint_text), len(current_text)) >= 12:
                if checkpoint_text in current_text or current_text in checkpoint_text:
                    return True
            similarity = SequenceMatcher(a=checkpoint_text, b=current_text).ratio()
            if similarity >= 0.72:
                return True
    return False


def load_resume_candidate(path: str | Path) -> TaskCheckpoint | None:
    """Load a resumable checkpoint when one exists and is eligible."""

    source = Path(path)
    if not source.exists():
        return None
    checkpoint = load_checkpoint(source)
    return checkpoint if is_resume_candidate(checkpoint) else None


def load_ignored_task_ids(workspace_root: str) -> set[str]:
    """Load task ids hidden from resume prompts."""

    path = default_ignored_tasks_path(workspace_root)
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    if not isinstance(data, list):
        return set()
    return {str(item) for item in data if isinstance(item, str) and item.strip()}


def save_ignored_task_ids(workspace_root: str, task_ids: set[str]) -> Path:
    """Persist the current ignored task id set."""

    path = default_ignored_tasks_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(task_id for task_id in task_ids if task_id.strip())
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def ignore_task_ids(workspace_root: str, task_ids: list[str]) -> Path:
    """Hide one or more task ids from resume lists without deleting their files."""

    ignored = load_ignored_task_ids(workspace_root)
    ignored.update(task_id for task_id in task_ids if isinstance(task_id, str) and task_id.strip())
    return save_ignored_task_ids(workspace_root, ignored)


def restore_task_ids(workspace_root: str, task_ids: list[str]) -> Path:
    """Remove task ids from the ignored set."""

    ignored = load_ignored_task_ids(workspace_root)
    ignored.difference_update(task_id for task_id in task_ids if isinstance(task_id, str))
    return save_ignored_task_ids(workspace_root, ignored)


def is_task_ignored(workspace_root: str, task_id: str) -> bool:
    """Return whether one task id is hidden from resume surfaces."""

    return task_id in load_ignored_task_ids(workspace_root)


def list_resume_candidates(workspace_root: str, active_path: str | Path | None = None) -> list[CheckpointCandidate]:
    """List resumable checkpoints from active storage and the archive."""

    resolved_active = Path(active_path) if active_path is not None else default_checkpoint_path(workspace_root)
    candidates: list[CheckpointCandidate] = []
    seen_task_ids: set[str] = set()
    ignored_task_ids = load_ignored_task_ids(workspace_root)

    active_checkpoint = load_resume_candidate(resolved_active)
    if active_checkpoint is not None and active_checkpoint.task.task_id not in ignored_task_ids:
        candidates.append(
            CheckpointCandidate(
                path=resolved_active,
                checkpoint=active_checkpoint,
                is_active=True,
            )
        )
        seen_task_ids.add(active_checkpoint.task.task_id)

    archive_dir = checkpoint_archive_dir_for_active_path(workspace_root, resolved_active)
    if archive_dir.exists():
        archived_paths = sorted(
            archive_dir.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for archived_path in archived_paths:
            try:
                archived_checkpoint = load_checkpoint(archived_path)
            except Exception:
                continue
            if not is_resume_candidate(archived_checkpoint):
                continue
            if archived_checkpoint.task.task_id in ignored_task_ids:
                continue
            if archived_checkpoint.task.task_id in seen_task_ids:
                continue
            candidates.append(
                CheckpointCandidate(
                    path=archived_path,
                    checkpoint=archived_checkpoint,
                    is_active=False,
                )
            )
            seen_task_ids.add(archived_checkpoint.task.task_id)
    return candidates


def shelve_active_checkpoint(path: str | Path) -> Path | None:
    """Move the active checkpoint into the archive so a new task can take over."""

    source = Path(path)
    if not source.exists():
        return None
    checkpoint = load_checkpoint(source)
    target = archive_checkpoint(checkpoint, checkpoint_archive_dir_for_active_path(checkpoint.task_input.workspace_root, source))
    clear_active_checkpoint(source)
    return target


def _workspace_roots_match(left: str, right: str) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except Exception:
        return left.strip().lower() == right.strip().lower()


def _normalize_resume_text(text: str) -> str:
    compact = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
    compact = re.sub(r"\s+", " ", compact).strip()
    return compact
