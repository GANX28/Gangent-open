"""Event queue and cooperative interrupt primitives.

This module is intentionally local and deterministic. It does not implement
threads, sockets, or a distributed workflow engine. Version one gives Gangent a
structured place where external inputs can be recorded and checked at safe
runtime boundaries.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .models import new_id, utc_now
from .secret_guard import redact_secrets, secret_labels


DEFAULT_EVENT_LOG = Path(".gangent") / "events" / "events.jsonl"


class AgentEventType(str, Enum):
    """Kinds of external or internal events the runtime can react to."""

    USER_INPUT = "user_input"
    TOOL_RESULT = "tool_result"
    AUDIT_WARNING = "audit_warning"
    FILE_CHANGE = "file_change"
    NEW_FILE_ADDED = "new_file_added"
    REQUIREMENT_CHANGE = "requirement_change"
    SYSTEM_SIGNAL = "system_signal"
    USER_INTERRUPT = "user_interrupt"
    REPLAN_REQUEST = "replan_request"
    ROLLBACK_REQUEST = "rollback_request"
    APPROVAL = "approval"
    APPROVAL_RESULT = "approval_result"


class InterruptAction(str, Enum):
    """Cooperative interrupt policy result."""

    IGNORE = "ignore"
    APPEND = "append"
    PAUSE = "pause"
    REPLAN = "replan"
    FORK = "fork"
    ASK_USER = "ask_user"


class EventRuntimeState(str, Enum):
    """Cooperative event-driven runtime state."""

    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_APPROVAL = "waiting_approval"
    INTERRUPTED = "interrupted"
    REPLANNING = "replanning"
    ROLLING_BACK = "rolling_back"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentEvent:
    """One queued runtime event."""

    event_id: str
    event_type: AgentEventType
    content: str
    source: str = "local"
    task_id: str = ""
    priority: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class QueuedEvent:
    """Event plus its append-only log position."""

    index: int
    event: AgentEvent


@dataclass(frozen=True)
class InterruptDecision:
    """Decision made from pending runtime events."""

    action: InterruptAction
    reason: str
    events: tuple[QueuedEvent, ...] = ()
    context_note: str = ""


@dataclass(frozen=True)
class EventTransition:
    """State transition requested by queued events."""

    from_state: EventRuntimeState
    to_state: EventRuntimeState
    action: InterruptAction
    reason: str
    reversible: bool = False


class JsonlEventQueue:
    """Append-only JSONL event queue.

    The cursor is the one-based line count already consumed by AgentState.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(
        self,
        event_type: AgentEventType,
        content: str,
        source: str = "local",
        task_id: str = "",
        priority: int = 50,
        metadata: dict[str, Any] | None = None,
    ) -> AgentEvent:
        if not content.strip():
            raise ValueError("Event content must not be empty.")
        labels = secret_labels(content)
        if labels:
            raise ValueError(f"Refusing to enqueue possible secrets: {', '.join(labels)}")
        event = AgentEvent(
            event_id=new_id("event"),
            event_type=event_type,
            content=redact_secrets(content.strip()),
            source=source.strip() or "local",
            task_id=task_id.strip(),
            priority=max(0, min(100, int(priority))),
            metadata=metadata or {},
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_to_dict(event), ensure_ascii=False, sort_keys=True) + "\n")
        return event

    def load(self) -> list[QueuedEvent]:
        if not self.path.exists():
            return []
        events: list[QueuedEvent] = []
        for index, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                events.append(QueuedEvent(index=index, event=event_from_dict(data)))
            except Exception:
                continue
        return events

    def pending(self, cursor: int = 0, task_id: str = "", created_after: str = "") -> list[QueuedEvent]:
        pending: list[QueuedEvent] = []
        for queued in self.load():
            if queued.index <= cursor:
                continue
            event_task = queued.event.task_id
            if event_task and task_id and event_task != task_id:
                continue
            if not event_task and created_after and queued.event.created_at < created_after:
                continue
            pending.append(queued)
        return pending


def default_event_log_path(workspace_root: str) -> Path:
    return Path(workspace_root).resolve() / DEFAULT_EVENT_LOG


def evaluate_interrupts(events: list[QueuedEvent]) -> InterruptDecision:
    """Choose one cooperative interrupt action for pending events."""

    if not events:
        return InterruptDecision(action=InterruptAction.IGNORE, reason="no pending events")
    ordered = sorted(events, key=lambda item: (-item.event.priority, item.index))
    highest = ordered[0].event
    if highest.event_type == AgentEventType.SYSTEM_SIGNAL and "pause" in highest.content.lower():
        return _decision(InterruptAction.PAUSE, "system signal requested pause", ordered)
    if highest.event_type == AgentEventType.USER_INTERRUPT:
        return _decision(InterruptAction.PAUSE, "user interrupt requested pause", ordered)
    if highest.event_type == AgentEventType.ROLLBACK_REQUEST:
        return _decision(InterruptAction.ASK_USER, "rollback request needs explicit approval", ordered)
    if highest.event_type == AgentEventType.REPLAN_REQUEST:
        return _decision(InterruptAction.REPLAN, "explicit replan request", ordered)
    if highest.event_type in {AgentEventType.APPROVAL, AgentEventType.APPROVAL_RESULT}:
        return _decision(InterruptAction.APPEND, "approval event appended to current context", ordered)
    if highest.event_type == AgentEventType.AUDIT_WARNING and highest.priority >= 80:
        return _decision(InterruptAction.PAUSE, "high-priority audit warning", ordered)
    if highest.event_type in {AgentEventType.USER_INPUT, AgentEventType.REQUIREMENT_CHANGE} and highest.priority >= 70:
        return _decision(InterruptAction.REPLAN, "high-priority user input requires plan revision", ordered)
    if highest.event_type in {
        AgentEventType.USER_INPUT,
        AgentEventType.FILE_CHANGE,
        AgentEventType.NEW_FILE_ADDED,
        AgentEventType.TOOL_RESULT,
        AgentEventType.REQUIREMENT_CHANGE,
    }:
        return _decision(InterruptAction.APPEND, "event appended to current task context", ordered)
    return _decision(InterruptAction.APPEND, "event recorded for current task", ordered)


def transition_from_interrupt(
    current_state: EventRuntimeState,
    decision: InterruptDecision,
) -> EventTransition:
    """Map an interrupt decision to a cooperative runtime state transition."""

    if decision.action == InterruptAction.PAUSE:
        return EventTransition(current_state, EventRuntimeState.INTERRUPTED, decision.action, decision.reason)
    if decision.action == InterruptAction.REPLAN:
        return EventTransition(current_state, EventRuntimeState.REPLANNING, decision.action, decision.reason, reversible=True)
    if decision.action == InterruptAction.ASK_USER:
        return EventTransition(current_state, EventRuntimeState.WAITING_APPROVAL, decision.action, decision.reason)
    if decision.action == InterruptAction.FORK:
        return EventTransition(current_state, EventRuntimeState.INTERRUPTED, decision.action, decision.reason)
    return EventTransition(current_state, current_state, decision.action, decision.reason, reversible=True)


def format_event_note(events: list[QueuedEvent]) -> str:
    lines: list[str] = []
    for queued in events:
        event = queued.event
        content = event.content.replace("\n", " ").strip()
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(
            f"- event#{queued.index} id={event.event_id}; type={event.event_type.value}; "
            f"priority={event.priority}; source={event.source}; content={content}"
        )
    return "\n".join(lines)


def event_to_dict(event: AgentEvent) -> dict[str, Any]:
    data = asdict(event)
    data["event_type"] = event.event_type.value
    return data


def event_from_dict(data: dict[str, Any]) -> AgentEvent:
    return AgentEvent(
        event_id=str(data["event_id"]),
        event_type=AgentEventType(data["event_type"]),
        content=str(data.get("content", "")),
        source=str(data.get("source", "local")),
        task_id=str(data.get("task_id", "")),
        priority=int(data.get("priority", 50)),
        metadata=dict(data.get("metadata", {})),
        created_at=str(data.get("created_at", utc_now())),
    )


def _decision(action: InterruptAction, reason: str, events: list[QueuedEvent]) -> InterruptDecision:
    return InterruptDecision(
        action=action,
        reason=reason,
        events=tuple(events),
        context_note=format_event_note(events),
    )
