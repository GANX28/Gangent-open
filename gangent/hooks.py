"""Runtime hook manager.

Hooks provide lifecycle extension points without hard-coding every side effect
inside runtime.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class HookEvent(str, Enum):
    """Runtime lifecycle events."""

    TASK_START = "on_task_start"
    BEFORE_MODEL_CALL = "before_model_call"
    AFTER_MODEL_CALL = "after_model_call"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    CHECKPOINT_SAVE = "on_checkpoint_save"
    TASK_FINISH = "on_task_finish"


@dataclass
class HookContext:
    """Data passed to hook handlers."""

    event: HookEvent
    task_input: Any = None
    task: Any = None
    state: Any = None
    decision: Any = None
    policy: Any = None
    tool_result: Any = None
    model_input: Any = None
    result: Any = None
    metadata: dict[str, Any] | None = None


HookHandler = Callable[[HookContext], None]


class HookManager:
    """Register and emit runtime hooks."""

    def __init__(self) -> None:
        self._handlers: dict[HookEvent, list[HookHandler]] = {event: [] for event in HookEvent}

    def register(self, event: HookEvent, handler: HookHandler) -> None:
        self._handlers[event].append(handler)

    def emit(self, context: HookContext) -> None:
        for handler in self._handlers.get(context.event, []):
            handler(context)


def default_hook_manager() -> HookManager:
    """Return an empty hook manager."""

    return HookManager()
