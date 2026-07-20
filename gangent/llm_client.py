"""LLM 客户端。

LLM = Large Language Model，大语言模型。
这一层负责把 ModelInput 发送给真实模型，或在测试中用 FakeLLMClient 模拟模型。
API Key 只从环境变量读取，绝不写入代码或仓库文件。
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from time import sleep
from typing import Any, Protocol

from .decision import decision_from_deepseek_response, decision_from_openai_response
from .model_input import ModelInput
from .models import ActionDecision
from .tool_schema import to_deepseek_tools, tool_names


class LLMClient(Protocol):
    """模型客户端接口。

    只暴露 decide 一个动作，避免业务代码关心具体模型服务商。
    """

    def decide(self, model_input: ModelInput) -> ActionDecision:
        """根据模型输入返回下一步动作决策。"""


@dataclass
class OpenAIResponsesClient:
    """基于 OpenAI Responses API 的最小客户端。

    Responses API（响应 API）是 OpenAI 面向文本、多模态和工具使用的统一接口。
    第一版用标准库 urllib 发送 HTTP 请求，避免引入 SDK 依赖。
    """

    model: str = "gpt-5.5"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1/responses"
    timeout_seconds: int = 60
    last_usage: dict[str, Any] = field(default_factory=dict, init=False)

    def decide(self, model_input: ModelInput) -> ActionDecision:
        """调用真实 OpenAI 模型并解析成 ActionDecision。"""

        response = self.create_response(model_input)
        self.last_usage = _usage_from_response(response)
        return decision_from_openai_response(
            response,
            available_tools=tool_names(model_input.tools),
        )

    def create_response(self, model_input: ModelInput) -> dict[str, Any]:
        """发送 Responses API 请求。

        如果本机没有设置 OPENAI_API_KEY，就明确报错，不伪造模型结果。
        """

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing {self.api_key_env}. Set it before running a real LLM call."
            )

        payload = {
            "model": self.model,
            "input": model_input.messages,
            "tools": model_input.tools,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI API request failed with HTTP {exc.code}: {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI API request failed: {exc.reason}") from exc


@dataclass
class DeepSeekChatClient:
    """基于 DeepSeek Chat Completions API 的最小客户端。

    DeepSeek API 兼容 OpenAI Chat Completions 格式，但不是 OpenAI Responses API。
    所以这里单独实现一个客户端，把同一个 ModelInput 发给 DeepSeek，
    再把 DeepSeek 的返回值转换成内部 ActionDecision。
    """

    model: str = "deepseek-v4-flash"
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str = "https://api.deepseek.com/chat/completions"
    timeout_seconds: int = 180
    request_attempts: int = 2
    reasoning_effort: str = "high"
    thinking_enabled: bool = False
    max_tokens: int | None = None
    last_usage: dict[str, Any] = field(default_factory=dict, init=False)

    def decide(self, model_input: ModelInput) -> ActionDecision:
        """调用真实 DeepSeek 模型并解析成 ActionDecision。"""

        response = self.create_response(model_input)
        self.last_usage = _usage_from_response(response)
        return decision_from_deepseek_response(
            response,
            available_tools=tool_names(model_input.tools),
        )

    def create_response(self, model_input: ModelInput) -> dict[str, Any]:
        """发送 DeepSeek Chat Completions 请求。

        如果本机没有设置 DEEPSEEK_API_KEY，就明确报错，不伪造模型结果。
        """

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing {self.api_key_env}. Set it before running a real DeepSeek call."
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": model_input.messages,
            "tools": to_deepseek_tools(model_input.tools),
            "tool_choice": "auto",
            "stream": False,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.thinking_enabled:
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = self.reasoning_effort
        else:
            payload["thinking"] = {"type": "disabled"}

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        attempts = max(1, int(self.request_attempts))
        last_timeout: RuntimeError | None = None
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"DeepSeek API request failed with HTTP {exc.code}: {error_body}"
                ) from exc
            except (TimeoutError, socket.timeout) as exc:
                last_timeout = RuntimeError(
                    f"DeepSeek API request timed out after {self.timeout_seconds}s "
                    f"(attempt {attempt + 1}/{attempts})."
                )
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                    last_timeout = RuntimeError(
                        f"DeepSeek API request timed out after {self.timeout_seconds}s "
                        f"(attempt {attempt + 1}/{attempts})."
                    )
                else:
                    raise RuntimeError(f"DeepSeek API request failed: {exc.reason}") from exc
            if attempt + 1 < attempts:
                sleep(min(2.0, 0.5 * (attempt + 1)))
        if last_timeout is not None:
            raise last_timeout
        raise RuntimeError("DeepSeek API request failed after retry attempts.")


@dataclass
class FakeLLMClient:
    """测试用假模型客户端。

    它不调用网络，只返回固定决策。这样单元测试不需要 API Key，也不会花钱。
    """

    decision: ActionDecision
    last_usage: dict[str, Any] = field(default_factory=dict, init=False)

    def decide(self, model_input: ModelInput) -> ActionDecision:
        self.last_usage = {}
        return self.decision


def _usage_from_response(response: dict[str, Any]) -> dict[str, Any]:
    """提取模型供应方返回的 usage 字段。"""

    usage = response.get("usage")
    return usage if isinstance(usage, dict) else {}
