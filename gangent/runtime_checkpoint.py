"""Runtime checkpoint coordination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .checkpoint import (
    TaskCheckpoint,
    archive_checkpoint,
    checkpoint_from_runtime_result,
    clear_active_checkpoint,
    default_checkpoint_archive_dir,
    save_checkpoint,
)
from .failure import is_recoverable_failure
from .models import AgentState, RuntimeStats, Task, TaskInput, TaskStatus


@dataclass
class RuntimeCheckpointSnapshot:
    """Minimal runtime-result shape needed by checkpoint serialization."""

    task: Task
    state: AgentState
    steps: list[Any]
    stats: RuntimeStats
    task_input: TaskInput


def save_runtime_checkpoint(
    checkpoint_path: str | None,
    task_input: TaskInput,
    task: Task,
    state: AgentState,
    steps: list[Any],
    stats: RuntimeStats,
) -> None:
    """Persist, archive, or clear the active checkpoint for the current state."""

    if not checkpoint_path:
        return
    checkpoint = checkpoint_from_runtime_result(
        RuntimeCheckpointSnapshot(
            task=task,
            state=state,
            steps=steps,
            stats=stats,
            task_input=task_input,
        )
    )
    if should_keep_active_checkpoint(checkpoint):
        save_checkpoint(checkpoint, checkpoint_path)
        return
    archive_checkpoint(checkpoint, default_checkpoint_archive_dir(task_input.workspace_root))
    clear_active_checkpoint(checkpoint_path)


def should_keep_active_checkpoint(checkpoint: TaskCheckpoint) -> bool:
    """Return whether a checkpoint should stay as the active resume point."""

    if checkpoint.task.status == TaskStatus.RUNNING:
        return True
    if checkpoint.task.status == TaskStatus.FAILED:
        return is_recoverable_failure(checkpoint.state.errors)
    return False
