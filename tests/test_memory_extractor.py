import os
import unittest
from unittest.mock import patch

from gangent.memory_extractor import should_use_llm_memory_extraction


class MemoryExtractorTests(unittest.TestCase):
    def test_skips_fake_provider_and_trivial_chat(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            self.assertFalse(
                should_use_llm_memory_extraction(
                    provider="fake",
                    user_message="Analyze runtime.",
                    final_answer="Runtime analysis result.",
                )
            )
            self.assertFalse(
                should_use_llm_memory_extraction(
                    provider="deepseek",
                    user_message="Say hello.",
                    final_answer="Hello.",
                )
            )

    def test_uses_deepseek_for_durable_memory_tasks(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            self.assertTrue(
                should_use_llm_memory_extraction(
                    provider="deepseek",
                    user_message="Analyze the memory graph architecture.",
                    final_answer="Memory graph uses semantic chunks for dynamic context routing.",
                )
            )

    def test_can_be_disabled_by_env(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key", "GANGENT_LLM_MEMORY": "0"}):
            self.assertFalse(
                should_use_llm_memory_extraction(
                    provider="deepseek",
                    user_message="Analyze the memory graph architecture.",
                    final_answer="Memory graph uses semantic chunks.",
                )
            )


if __name__ == "__main__":
    unittest.main()
