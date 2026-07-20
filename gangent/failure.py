"""Structured runtime failure reasons.

FailureReason（失败原因枚举）keeps recoverability decisions out of free-form
error text. Older checkpoints are still understood through a small legacy
fallback so existing saved state does not become unreadable.
"""

from __future__ import annotations

from enum import Enum


class FailureReason(str, Enum):
    """Failure classes that matter to task recovery."""

    MAX_STEPS = "max_steps"
    DEADLINE_EXCEEDED = "deadline_exceeded"


RECOVERABLE_FAILURE_REASONS = {
    FailureReason.MAX_STEPS,
    FailureReason.DEADLINE_EXCEEDED,
}

_PREFIX = "failure_reason="


def failure_message(reason: FailureReason, detail: str) -> str:
    """Create a human-readable error that also carries a structured reason."""

    return f"{_PREFIX}{reason.value}; {detail}"


def failure_reason_from_error(error: str) -> FailureReason | None:
    """Extract a structured failure reason from one error string."""

    text = error.strip()
    if text.startswith(_PREFIX):
        raw_reason = text[len(_PREFIX) :].split(";", 1)[0].strip()
        try:
            return FailureReason(raw_reason)
        except ValueError:
            return None
    return _legacy_failure_reason(text)


def recoverable_failure_reason(errors: list[str]) -> FailureReason | None:
    """Return the first recoverable failure reason in a list of errors."""

    for error in reversed(errors):
        reason = failure_reason_from_error(error)
        if reason in RECOVERABLE_FAILURE_REASONS:
            return reason
    return None


def is_recoverable_failure(errors: list[str]) -> bool:
    """Return whether a failed task should keep an active checkpoint."""

    return recoverable_failure_reason(errors) is not None


def _legacy_failure_reason(error: str) -> FailureReason | None:
    """Read pre-structured checkpoint errors from earlier versions."""

    lowered = error.lower()
    if "max_steps" in lowered:
        return FailureReason.MAX_STEPS
    if "deadline exceeded" in lowered:
        return FailureReason.DEADLINE_EXCEEDED
    return None
