"""Lightweight LLM memory extraction.

This module is intentionally outside the runtime loop. It is used after a task
finishes to convert useful task output into semantic memory chunks. If the
provider call fails, callers should fall back to deterministic extraction.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from .secret_guard import redact_secrets


SYSTEM_PROMPT = """You extract durable memory chunks for a local Agent Runtime.
Return JSON only, no markdown.

Goal:
- Extract reusable semantic chunks, not task logs.
- Each chunk must have a short English summary and a more detailed body.
- Include only information that can help future task-specific context loading.
- Skip greetings, one-off trivial answers, and low-value runtime chatter.

JSON shape:
{
  "chunks": [
    {
      "node_type": "concept|decision|issue|solution|fact|constraint|artifact|preference|note",
      "layer": "data|task|knowledge",
      "summary": "English summary under 140 chars",
      "content": "Detailed memory under 900 chars",
      "tags": ["short", "lowercase", "tags"],
      "importance": 0.0,
      "confidence": 0.0
    }
  ]
}
"""


def should_use_llm_memory_extraction(
    *,
    provider: str,
    user_message: str,
    final_answer: str,
    errors: list[str] | None = None,
) -> bool:
    """Return whether a task is worth spending an extra extraction call on."""

    if os.environ.get("GANGENT_LLM_MEMORY", "1").strip().lower() in {"0", "false", "off", "no"}:
        return False
    if provider != "deepseek":
        return False
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return False
    text = f"{user_message}\n{final_answer}".lower()
    markers = [
        "analyze",
        "analysis",
        "summarize",
        "summary",
        "design",
        "architecture",
        "runtime",
        "planner",
        "memory",
        "context",
        "debug",
        "fix",
        "error",
        "issue",
        "read ",
        "docs/",
        "README".lower(),
        "\u5206\u6790",
        "\u603b\u7ed3",
        "\u8bbe\u8ba1",
        "\u67b6\u6784",
        "\u8bb0\u5fc6",
        "\u4fee\u590d",
        "\u9519\u8bef",
    ]
    return bool(errors) or len(final_answer) >= 500 or any(marker in text for marker in markers)


def extract_semantic_memory_chunks(
    *,
    provider: str,
    model: str | None,
    user_message: str,
    final_answer: str,
    errors: list[str] | None = None,
    tool_names: list[str] | None = None,
    timeout_seconds: int = 35,
) -> list[dict[str, Any]]:
    """Call the configured provider for lightweight memory extraction."""

    if provider != "deepseek":
        return []
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return []
    payload = {
        "model": model or "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_payload(
                    user_message=user_message,
                    final_answer=final_answer,
                    errors=errors or [],
                    tool_names=tool_names or [],
                ),
            },
        ],
        "stream": False,
        "max_tokens": 900,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return []
    text = _message_text(data)
    if not text:
        return []
    try:
        parsed = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError:
        return []
    chunks = parsed.get("chunks")
    return chunks if isinstance(chunks, list) else []


def _build_user_payload(
    *,
    user_message: str,
    final_answer: str,
    errors: list[str],
    tool_names: list[str],
) -> str:
    trimmed_answer = final_answer.strip()
    if len(trimmed_answer) > 3_500:
        trimmed_answer = trimmed_answer[:3_500].rstrip() + "\n... truncated"
    return redact_secrets(
        "\n\n".join(
            [
                "User request:\n" + user_message.strip(),
                "Final answer:\n" + trimmed_answer,
                "Errors:\n" + ("\n".join(errors[-5:]) if errors else "(none)"),
                "Tools:\n" + (", ".join(tool_names) if tool_names else "(none)"),
            ]
        )
    )


def _message_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _strip_json_fence(text: str) -> str:
    value = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", value, re.DOTALL)
    return match.group(1).strip() if match else value
