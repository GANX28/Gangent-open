"""Planning Layer 演示。

默认模式使用 FakeLLMClient，不联网也不消耗 API。
传入 --real 时使用 OpenAIResponsesClient 调用真实大模型。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from pprint import pprint

from .models import TaskInput
from .models import DecisionType
from .providers import create_llm_client
from .runtime import run_task
from .state import state_snapshot, state_summary


def run_demo(
    provider: str = "fake",
    model: str | None = None,
    thinking: bool = False,
    max_steps: int = 3,
    max_tokens: int | None = 1000,
    max_seconds: float | None = 60,
) -> None:
    """运行一次 runtime loop。

    这里展示的是：模型可以连续决策多轮，每轮都经过策略检查和工具执行。
    """

    task_input = TaskInput(
        goal="Inspect the workspace and prepare the next runtime step.",
        user_message="Inspect the current runtime workspace.",
        workspace_root=str(Path.cwd()),
        constraints=["Stay within the current workspace boundary."],
    )

    client = create_llm_client(
        provider=provider,
        model=model,
        thinking=thinking,
        max_tokens=max_tokens,
    )
    result = run_task(
        task_input,
        client,
        max_steps=max_steps,
        max_seconds=max_seconds,
    )

    for step in result.steps:
        print(f"\nSTEP {step.step_index}")
        print("DECISION")
        pprint(asdict(step.decision))
        if step.policy:
            print("POLICY")
            pprint(asdict(step.policy))
        if step.tool_result:
            print("TOOL RESULT")
            pprint(asdict(step.tool_result))

    if result.state.last_decision and result.state.last_decision.decision_type in {
        DecisionType.FINISH,
        DecisionType.DIRECT_RESPONSE,
    }:
        print("\nFINAL ANSWER")
        print(result.state.last_decision.response_text)

    print("\nTASK")
    pprint(asdict(result.task))
    print("\nSTATE SUMMARY")
    print(state_summary(result.state))
    print("\nSTATE SNAPSHOT")
    pprint(state_snapshot(result.state))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gangent planning demo.")
    parser.add_argument(
        "--provider",
        choices=["fake", "openai", "deepseek"],
        default="fake",
        help="Choose which model provider to use.",
    )
    parser.add_argument(
        "--model",
        help="Override the provider model name.",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable DeepSeek thinking mode when provider is deepseek.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Backward-compatible alias for --provider openai.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="Maximum runtime loop steps.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum generated tokens per model call. Currently used by DeepSeek.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=60,
        help="Maximum wall-clock seconds for the demo task.",
    )
    args = parser.parse_args()
    provider = "openai" if args.real else args.provider
    run_demo(
        provider=provider,
        model=args.model,
        thinking=args.thinking,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        max_seconds=args.max_seconds,
    )


if __name__ == "__main__":
    main()
