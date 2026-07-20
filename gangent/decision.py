"""把模型输出转换成 ActionDecision。

这一层的职责是隔离模型 API 的返回格式。
外部模型可能返回文本、函数调用、错误对象等；runtime 内部只认 ActionDecision。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .models import ActionDecision, DecisionType


class DecisionParseError(ValueError):
    """模型输出无法转换成内部决策时抛出的错误。"""


def direct_response(text: str, reason: str | None = None) -> ActionDecision:
    """构造直接回复决策。"""

    clean_text = text.strip()
    return ActionDecision(
        decision_type=DecisionType.DIRECT_RESPONSE,
        reason=reason or "Model returned a direct response.",
        response_text=clean_text,
    )


def tool_call_decision(
    tool_name: str,
    tool_args: dict[str, Any],
    reason: str | None = None,
) -> ActionDecision:
    """构造工具调用决策。"""

    return ActionDecision(
        decision_type=DecisionType.TOOL_CALL,
        reason=reason or f"Model requested tool: {tool_name}.",
        tool_name=tool_name,
        tool_args=tool_args,
    )


def finish_decision(answer: str, reason: str | None = None) -> ActionDecision:
    """构造任务完成决策。"""

    clean_answer = answer.strip()
    clean_reason = (reason or "Model decided the task is complete.").strip()
    return ActionDecision(
        decision_type=DecisionType.FINISH,
        reason=clean_reason,
        response_text=clean_answer,
    )


def validate_decision(
    decision: ActionDecision,
    available_tools: set[str] | None = None,
) -> None:
    """校验 ActionDecision 是否满足最小结构要求。

    专业说法：这是 internal contract validation（内部契约校验）。
    通俗说法：模型说要干什么以后，程序先检查这句话是不是格式合格。
    """

    if decision.decision_type == DecisionType.TOOL_CALL:
        if not decision.tool_name:
            raise DecisionParseError("Tool-call decision must include tool_name.")
        if decision.tool_args is None:
            raise DecisionParseError("Tool-call decision must include tool_args.")
        if available_tools is not None and decision.tool_name not in available_tools:
            raise DecisionParseError(f"Unknown tool requested: {decision.tool_name}")

    if decision.decision_type == DecisionType.DIRECT_RESPONSE and not decision.response_text:
        raise DecisionParseError("Direct response decision must include response_text.")

    if decision.decision_type == DecisionType.FINISH and not decision.response_text:
        raise DecisionParseError("Finish decision must include response_text.")


def decision_from_openai_response(
    response: dict[str, Any],
    available_tools: set[str] | None = None,
) -> ActionDecision:
    """从 OpenAI Responses API 返回值中提取内部 ActionDecision。

    第一版只处理两类结果：
    1. function_call：模型想调用某个工具；
    2. output_text 或 message 文本：模型直接回答。

    如果返回格式不符合预期，直接报错，不伪造结果。
    """

    for item in response.get("output", []):
        if item.get("type") == "function_call":
            name = item.get("name")
            raw_args = item.get("arguments", "{}")
            if not isinstance(name, str) or not name:
                raise DecisionParseError("OpenAI function_call missing name.")
            args = _parse_json_object(raw_args, tool_name=name)
            decision = decision_from_function_call(name, args)
            validate_decision(decision, available_tools)
            return decision

    text = response.get("output_text")
    if isinstance(text, str) and text.strip():
        decision = _direct_response_from_text(text, available_tools, provider="OpenAI")
        validate_decision(decision, available_tools)
        return decision

    for item in response.get("output", []):
        if item.get("type") == "message":
            text = _text_from_message_item(item)
            if text:
                decision = _direct_response_from_text(text, available_tools, provider="OpenAI")
                validate_decision(decision, available_tools)
                return decision

    raise DecisionParseError("OpenAI response did not contain a function call or text.")


def decision_from_deepseek_response(
    response: dict[str, Any],
    available_tools: set[str] | None = None,
) -> ActionDecision:
    """从 DeepSeek Chat Completions 返回值中提取内部 ActionDecision。

    DeepSeek 的返回格式和 OpenAI Responses API 不一样：
    - 文本在 choices[0].message.content
    - 工具调用在 choices[0].message.tool_calls

    这里把它统一转换成我们的 ActionDecision，后面的 runtime 就不用关心模型来源。
    """

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise DecisionParseError("DeepSeek response missing choices.")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise DecisionParseError("DeepSeek response missing message.")

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        tool_call = tool_calls[0]
        function = tool_call.get("function")
        if not isinstance(function, dict):
            raise DecisionParseError("DeepSeek tool_call missing function.")
        name = function.get("name")
        raw_args = function.get("arguments", "{}")
        if not isinstance(name, str) or not name:
            raise DecisionParseError("DeepSeek function call missing name.")
        args = _parse_json_object(raw_args, tool_name=name)
        decision = decision_from_function_call(name, args)
        validate_decision(decision, available_tools)
        return decision

    text = message.get("content")
    if isinstance(text, str) and text.strip():
        decision = _direct_response_from_text(text, available_tools, provider="DeepSeek")
        validate_decision(decision, available_tools)
        return decision

    raise DecisionParseError("DeepSeek response did not contain a tool call or text.")


def decision_from_function_call(name: str, args: dict[str, Any]) -> ActionDecision:
    """把函数调用转换成内部 ActionDecision。

    finish_task 是结构化完成信号，不进入 Tool Runtime。
    其他函数调用才是普通工具调用。
    """

    if name == "finish_task":
        answer = args.get("answer")
        reason = args.get("reason")
        if not isinstance(answer, str) or not answer.strip():
            raise DecisionParseError("finish_task requires non-empty answer.")
        if reason is not None and not isinstance(reason, str):
            raise DecisionParseError("finish_task reason must be a string.")
        return finish_decision(answer=answer, reason=reason)

    return tool_call_decision(name, args)


def _parse_json_object(raw: Any, tool_name: str | None = None) -> dict[str, Any]:
    """把模型返回的 arguments 转成 dict。"""

    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise DecisionParseError("Function arguments must be a JSON string or object.")

    normalized = _normalize_tool_arguments(raw)
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError:
        repaired = _repair_json_object_text(normalized)
        try:
            value = json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise DecisionParseError(_invalid_json_message(exc, raw, tool_name)) from exc
    if not isinstance(value, dict):
        raise DecisionParseError("Function arguments must decode to a JSON object.")
    return value


def _direct_response_from_text(
    text: str,
    available_tools: set[str] | None,
    provider: str,
) -> ActionDecision:
    tool_name = _tool_request_placeholder_name(text, available_tools)
    if tool_name:
        raise DecisionParseError(
            f"{provider} response described a tool request in plain text instead of "
            f"returning a structured tool call: {tool_name}."
        )
    return direct_response(text)


def _tool_request_placeholder_name(text: str, available_tools: set[str] | None) -> str | None:
    """Detect model-generated tool-call scaffolding that is not a real answer."""

    matches = re.findall(r"Model requested tool:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\.?", text)
    if not matches:
        return None
    remainder = re.sub(r"Model requested tool:\s*[A-Za-z_][A-Za-z0-9_]*\s*\.?", "", text).strip()
    if remainder:
        return None
    tool_name = matches[0]
    if available_tools is not None and tool_name not in available_tools:
        return None
    return tool_name


def _normalize_tool_arguments(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _repair_json_object_text(text: str) -> str:
    candidate = text.strip()
    first_brace = candidate.find("{")
    last_brace = candidate.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = candidate[first_brace : last_brace + 1]
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
    return candidate


def _invalid_json_message(
    exc: json.JSONDecodeError,
    raw: str,
    tool_name: str | None,
) -> str:
    snippet = raw.strip().replace("\n", " ")
    if len(snippet) > 160:
        snippet = snippet[:160] + "..."
    hint = ""
    lowered = exc.msg.lower()
    if "unterminated string" in lowered or "expecting value" in lowered:
        hint = (
            " Likely cause: the model emitted incomplete tool arguments or the response "
            "was truncated. Retry with a larger --max-tokens value."
        )
    tool_part = f" for tool {tool_name}" if tool_name else ""
    return f"Function arguments{tool_part} are not valid JSON: {exc}. raw={snippet!r}.{hint}"


def _text_from_message_item(item: dict[str, Any]) -> str:
    """从 Responses API 的 message item 中提取文本。"""

    parts: list[str] = []
    for content in item.get("content", []):
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()
