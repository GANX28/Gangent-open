"""模型 provider 工厂。

Provider（模型供应方）包括 fake、openai、deepseek。
把创建客户端的逻辑集中到这里，避免 demo.py 和 cli.py 重复维护同一段代码。
"""

from __future__ import annotations

from .llm_client import DeepSeekChatClient, FakeLLMClient, LLMClient, OpenAIResponsesClient
from .models import ActionDecision, DecisionType


def create_llm_client(
    provider: str,
    model: str | None = None,
    thinking: bool = False,
    max_tokens: int | None = None,
    budget_profile: str | None = None,
    task_text: str = "",
) -> LLMClient:
    """根据 provider 创建对应的 LLMClient。

    fake 用于本地测试；deepseek/openai 用于真实模型调用。
    """

    if provider == "fake":
        return FakeLLMClient(
            ActionDecision(
                decision_type=DecisionType.TOOL_CALL,
                reason="The workspace should be inspected before reading files.",
                tool_name="list_files",
                tool_args={"path": "."},
            )
        )

    if provider == "openai":
        return OpenAIResponsesClient(model=model or "gpt-5.5")

    if provider == "deepseek":
        return DeepSeekChatClient(
            model=_resolve_deepseek_model(model, budget_profile=budget_profile, task_text=task_text, thinking=thinking),
            thinking_enabled=thinking,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unknown provider: {provider}")


def _resolve_deepseek_model(
    explicit_model: str | None,
    *,
    budget_profile: str | None,
    task_text: str,
    thinking: bool,
) -> str:
    """Flash-first cost router with Pro escalation for high-risk work."""

    if explicit_model:
        return explicit_model
    profile = (budget_profile or "").split("+", 1)[0].split(":", 1)[0].lower()
    lowered = task_text.lower()
    high_risk_markers = [
        "architecture",
        "commercial",
        "security",
        "audit",
        "compliance",
        "planner",
        "\u5546\u4e1a",
        "\u5ba1\u8ba1",
        "\u5b89\u5168",
        "\u5408\u89c4",
        "\u67b6\u6784",
        "\u89c4\u5212",
    ]
    if thinking or profile == "ultra" or (profile == "heavy" and any(marker in lowered for marker in high_risk_markers)):
        return "deepseek-v4-pro"
    return "deepseek-v4-flash"
