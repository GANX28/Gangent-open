import unittest

from gangent.decision import (
    DecisionParseError,
    decision_from_function_call,
    decision_from_deepseek_response,
    decision_from_openai_response,
    validate_decision,
)
from gangent.llm_client import FakeLLMClient
from gangent.model_input import build_model_input
from gangent.models import ActionDecision, DecisionType, TaskInput
from gangent.state import create_initial_state, create_task, start_task
from gangent.tool_schema import available_tool_schemas, to_deepseek_tools, tool_names


class PlanningLayerTests(unittest.TestCase):
    def _sample_task_and_state(self):
        task_input = TaskInput(
            goal="Inspect project",
            user_message="Look at the repository structure.",
            workspace_root=".",
        )
        task = create_task(task_input)
        state = create_initial_state(task, task_input)
        task, state = start_task(task, state)
        return task, state

    def test_build_model_input_contains_messages_and_tools(self):
        task, state = self._sample_task_and_state()
        tools = available_tool_schemas()

        model_input = build_model_input(task, state, tools)

        self.assertGreaterEqual(len(model_input.messages), 3)
        self.assertEqual(model_input.messages[0]["role"], "system")
        self.assertIn("Task goal: Inspect project", model_input.messages[1]["content"])
        self.assertIn("list_files", tool_names(model_input.tools))
        self.assertIn("prefix_cache_strategy", model_input.diagnostics)

    def test_build_model_input_redacts_secrets(self):
        task_input = TaskInput(
            goal="Use token=abcdefgh12345678",
            user_message="Use token=abcdefgh12345678",
            workspace_root=".",
        )
        task = create_task(task_input)
        state = create_initial_state(task, task_input)
        task, state = start_task(task, state)

        model_input = build_model_input(task, state, available_tool_schemas())
        combined = "\n".join(message["content"] for message in model_input.messages)

        self.assertIn("[REDACTED_", combined)
        self.assertNotIn("abcdefgh12345678", combined)

    def test_parse_openai_function_call_response(self):
        response = {
            "output": [
                {
                    "type": "function_call",
                    "name": "list_files",
                    "arguments": '{"path": "."}',
                }
            ]
        }

        decision = decision_from_openai_response(
            response,
            available_tools={"list_files"},
        )

        self.assertEqual(decision.decision_type, DecisionType.TOOL_CALL)
        self.assertEqual(decision.tool_name, "list_files")
        self.assertEqual(decision.tool_args, {"path": "."})

    def test_parse_deepseek_tool_call_response(self):
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "README.md"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        decision = decision_from_deepseek_response(
            response,
            available_tools={"read_file"},
        )

        self.assertEqual(decision.decision_type, DecisionType.TOOL_CALL)
        self.assertEqual(decision.tool_name, "read_file")
        self.assertEqual(decision.tool_args, {"path": "README.md"})

    def test_parse_deepseek_tool_call_response_with_code_fence_arguments(self):
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '```json\n{"path": "README.md"}\n```',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        decision = decision_from_deepseek_response(
            response,
            available_tools={"read_file"},
        )

        self.assertEqual(decision.tool_name, "read_file")
        self.assertEqual(decision.tool_args, {"path": "README.md"})

    def test_parse_deepseek_tool_call_response_with_trailing_comma(self):
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "README.md",}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        decision = decision_from_deepseek_response(
            response,
            available_tools={"read_file"},
        )

        self.assertEqual(decision.tool_name, "read_file")
        self.assertEqual(decision.tool_args, {"path": "README.md"})

    def test_parse_deepseek_tool_call_response_reports_truncation_hint(self):
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "finish_task",
                                    "arguments": '{"answer": "This is a long unfinished',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        with self.assertRaises(DecisionParseError) as context:
            decision_from_deepseek_response(
                response,
                available_tools={"finish_task"},
            )

        self.assertIn("larger --max-tokens", str(context.exception))

    def test_convert_tools_to_deepseek_format(self):
        deepseek_tools = to_deepseek_tools(available_tool_schemas())

        self.assertEqual(deepseek_tools[0]["type"], "function")
        self.assertIn("function", deepseek_tools[0])
        self.assertEqual(deepseek_tools[0]["function"]["name"], "list_files")
        self.assertNotIn("strict", deepseek_tools[0]["function"])

    def test_convert_tools_to_deepseek_strict_format_when_enabled(self):
        deepseek_tools = to_deepseek_tools(
            available_tool_schemas(),
            include_strict=True,
        )

        self.assertTrue(deepseek_tools[0]["function"]["strict"])

    def test_finish_task_function_call_becomes_finish_decision(self):
        decision = decision_from_function_call(
            "finish_task",
            {"answer": "The project contains a runtime skeleton.", "reason": "Enough context."},
        )

        self.assertEqual(decision.decision_type, DecisionType.FINISH)
        self.assertEqual(decision.response_text, "The project contains a runtime skeleton.")

    def test_parse_deepseek_finish_task_response(self):
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "finish_task",
                                    "arguments": '{"answer": "Done.", "reason": "Enough information."}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        decision = decision_from_deepseek_response(
            response,
            available_tools={"finish_task"},
        )

        self.assertEqual(decision.decision_type, DecisionType.FINISH)
        self.assertEqual(decision.response_text, "Done.")

    def test_deepseek_plain_text_tool_request_is_rejected(self):
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Model requested tool: read_file.",
                    }
                }
            ]
        }

        with self.assertRaises(DecisionParseError) as context:
            decision_from_deepseek_response(
                response,
                available_tools={"read_file"},
            )

        self.assertIn("structured tool call", str(context.exception))

    def test_deepseek_multiline_plain_text_tool_request_is_rejected(self):
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Model requested tool: read_file.\n\nModel requested tool: read_file.",
                    }
                }
            ]
        }

        with self.assertRaises(DecisionParseError) as context:
            decision_from_deepseek_response(
                response,
                available_tools={"read_file"},
            )

        self.assertIn("structured tool call", str(context.exception))

    def test_openai_plain_text_tool_request_is_rejected(self):
        response = {"output_text": "Model requested tool: read_file."}

        with self.assertRaises(DecisionParseError) as context:
            decision_from_openai_response(
                response,
                available_tools={"read_file"},
            )

        self.assertIn("structured tool call", str(context.exception))

    def test_unknown_tool_is_rejected(self):
        decision = ActionDecision(
            decision_type=DecisionType.TOOL_CALL,
            reason="Bad tool.",
            tool_name="delete_everything",
            tool_args={},
        )

        with self.assertRaises(DecisionParseError):
            validate_decision(decision, available_tools={"list_files"})

    def test_fake_llm_client_returns_decision(self):
        task, state = self._sample_task_and_state()
        model_input = build_model_input(task, state, available_tool_schemas())
        expected = ActionDecision(
            decision_type=DecisionType.DIRECT_RESPONSE,
            reason="No tool needed.",
            response_text="Ready.",
        )

        client = FakeLLMClient(expected)

        self.assertEqual(client.decide(model_input), expected)


if __name__ == "__main__":
    unittest.main()
