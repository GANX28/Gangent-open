import unittest

from gangent.llm_client import DeepSeekChatClient, FakeLLMClient
from gangent.providers import create_llm_client


class ProviderFactoryTests(unittest.TestCase):
    def test_create_fake_client(self):
        client = create_llm_client("fake")

        self.assertIsInstance(client, FakeLLMClient)

    def test_create_deepseek_client(self):
        client = create_llm_client(
            "deepseek",
            model="deepseek-v4-pro",
            thinking=True,
            max_tokens=800,
        )

        self.assertIsInstance(client, DeepSeekChatClient)
        self.assertEqual(client.model, "deepseek-v4-pro")
        self.assertTrue(client.thinking_enabled)
        self.assertEqual(client.max_tokens, 800)
        self.assertGreaterEqual(client.timeout_seconds, 180)
        self.assertGreaterEqual(client.request_attempts, 2)

    def test_deepseek_router_defaults_to_flash(self):
        client = create_llm_client("deepseek", budget_profile="medium", task_text="Inspect files")

        self.assertIsInstance(client, DeepSeekChatClient)
        self.assertEqual(client.model, "deepseek-v4-flash")

    def test_deepseek_router_escalates_ultra_or_high_risk(self):
        ultra = create_llm_client("deepseek", budget_profile="ultra", task_text="Inspect files")
        heavy = create_llm_client(
            "deepseek",
            budget_profile="heavy",
            task_text="Analyze commercial architecture and audit risk",
        )

        self.assertEqual(ultra.model, "deepseek-v4-pro")
        self.assertEqual(heavy.model, "deepseek-v4-pro")

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            create_llm_client("unknown")


if __name__ == "__main__":
    unittest.main()
