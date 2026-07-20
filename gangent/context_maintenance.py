"""Deterministic context maintenance helpers.

These helpers do not call another model. They keep the runtime cheaper and more
stable by shortening old or large tool outputs before they are fed back into the
next model call.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .models import ToolResult
from .secret_guard import redact_secrets


DEFAULT_SNIP_LIMIT = 1_200
OLD_TOOL_SNIP_LIMIT = 500


@dataclass(frozen=True)
class ContextBudget:
    """A lightweight budget estimate for one model input."""

    char_count: int
    estimated_tokens: int
    level: str


def estimate_tokens(text: str) -> int:
    """Estimate tokens without provider-specific tokenizers.

    This is intentionally rough. For mixed Chinese/English/code, four
    characters per token is a useful conservative local estimate.
    """

    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_context_budget(messages: list[dict[str, str]]) -> ContextBudget:
    text = "\n".join(message.get("content", "") for message in messages)
    tokens = estimate_tokens(text)
    if tokens < 4_000:
        level = "small"
    elif tokens < 16_000:
        level = "medium"
    else:
        level = "large"
    return ContextBudget(char_count=len(text), estimated_tokens=tokens, level=level)


def stable_prefix_hash(system_prompt: str, tools: list[dict[str, Any]]) -> str:
    """Hash the stable prefix shape for DeepSeek prefix-cache diagnostics."""

    payload = {
        "system_prompt": system_prompt,
        "tools": _canonical_tools(tools),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def snip_tool_result_for_state(result: ToolResult, limit: int = DEFAULT_SNIP_LIMIT) -> str:
    """Return a bounded text representation of a tool result for state messages."""

    content = result.output if result.success else result.error or "Tool failed."
    return snip_text(content, limit=limit)


def snip_text(text: str, limit: int = DEFAULT_SNIP_LIMIT) -> str:
    """Shorten text while preserving both the beginning and the end."""

    text = redact_secrets(text or "")
    if len(text) <= limit:
        return text
    head_limit = max(200, int(limit * 0.65))
    tail_limit = max(120, limit - head_limit - 120)
    omitted = len(text) - head_limit - tail_limit
    return (
        text[:head_limit].rstrip()
        + f"\n... [snipped {omitted} chars; re-read source if exact details are needed] ...\n"
        + text[-tail_limit:].lstrip()
    )


def compact_old_tool_message(content: str) -> str:
    """Compact an older tool message for model input."""

    return snip_text(content, limit=OLD_TOOL_SNIP_LIMIT)


def _canonical_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    for tool in tools:
        canonical.append(
            {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            }
        )
    return sorted(canonical, key=lambda item: str(item.get("name", "")))
