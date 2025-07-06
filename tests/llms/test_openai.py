import unittest

from unittest.mock import MagicMock

from memos.configs.llm import LLMConfigFactory
from memos.llms.factory import LLMFactory


class TestLLMFactoryWithOpenAIBackend(unittest.TestCase):
    def test_llm_factory_with_mocked_openai_backend(self):
        """Test LLMFactory with mocked OpenAI backend."""
        mock_chat_completions_create = MagicMock()
        mock_response = MagicMock()
        mock_response.model_dump_json.return_value = '{"id":"chatcmpl-BWoqIrvOeWdnFVZQUFzCcdVEpJ166","choices":[{"finish_reason":"stop","index":0,"message":{"content":"Hello! I\'m an AI language model created by OpenAI. I\'m here to help answer questions, provide information, and assist with a wide range of topics. How can I assist you today?","role":"assistant"}}],"created":1747161634,"model":"gpt-4o-2024-08-06","object":"chat.completion"}'
        mock_response.choices[0].message.content = "Hello! I'm an AI language model created by OpenAI. I'm here to help answer questions, provide information, and assist with a wide range of topics. How can I assist you today?"  # fmt: skip
        mock_chat_completions_create.return_value = mock_response

        config = LLMConfigFactory.model_validate(
            {
                "backend": "openai",
                "config": {
                    "model_name_or_path": "gpt-4.1-nano",
                    "temperature": 0.8,
                    "max_tokens": 1024,
                    "top_p": 0.9,
                    "top_k": 50,
                    "api_key": "sk-xxxx",
                    "api_base": "https://api.openai.com/v1",
                },
            }
        )
        llm = LLMFactory.from_config(config)
        llm.client.chat.completions.create = mock_chat_completions_create
        messages = [
            {"role": "user", "content": "Hello, who are you"},
        ]
        response = llm.generate(messages)

        self.assertEqual(
            response,
            "Hello! I'm an AI language model created by OpenAI. I'm here to help answer questions, provide information, and assist with a wide range of topics. How can I assist you today?",
        )
