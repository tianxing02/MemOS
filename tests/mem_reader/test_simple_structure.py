import json
import unittest

from unittest.mock import MagicMock, patch

from memos.chunkers import ChunkerFactory
from memos.chunkers.base import Chunk
from memos.configs.mem_reader import SimpleStructMemReaderConfig
from memos.embedders.factory import EmbedderFactory
from memos.llms.factory import LLMFactory
from memos.mem_reader.simple_struct import SimpleStructMemReader
from memos.memories.textual.item import TextualMemoryItem


class TestSimpleStructMemReader(unittest.TestCase):
    def setUp(self):
        # Mock config
        self.config = MagicMock(spec=SimpleStructMemReaderConfig)
        self.config.llm = MagicMock()
        self.config.embedder = MagicMock()
        self.config.chunker = MagicMock()
        self.config.remove_prompt_example = MagicMock()

        # Mock dependencies
        with (
            patch.object(LLMFactory, "from_config", return_value=MagicMock()),
            patch.object(EmbedderFactory, "from_config", return_value=MagicMock()),
            patch.object(ChunkerFactory, "from_config", return_value=MagicMock()),
        ):
            self.reader = SimpleStructMemReader(self.config)

        # Set up mock LLM and embedder
        self.reader.llm = MagicMock()
        self.reader.embedder = MagicMock()
        self.reader.chunker = MagicMock()

    def test_init(self):
        """Test initialization of the reader."""
        self.assertIsNotNone(self.reader.config)
        self.assertIsNotNone(self.reader.llm)
        self.assertIsNotNone(self.reader.embedder)

    def test_process_chat_data(self):
        """Test processing chat data into memory items."""
        scene_data_info = [
            "user: Hello",
            "assistant: Hi there",
            "user: How are you?",
        ]
        info = {"user_id": "user1", "session_id": "session1"}

        # Mock LLM response

        mock_response = (
            '{"memory list": [{"key": "Planned scope adjustment", "memory_type": "UserMemory", '
            '"value": "Tom planned to suggest in a meeting on June 27, 2025 at 9:30 AM", '
            '"tags": ["planning", "deadline change", "feature prioritization"]}], '
            '"summary": "Tom is currently focused on managing a new project with a tight schedule."}'
        )
        self.reader.llm.generate.return_value = mock_response
        self.reader.parse_json_result = lambda x: json.loads(x)

        result = self.reader._process_chat_data(scene_data_info, info)

        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], TextualMemoryItem)
        self.assertEqual(
            result[0].memory, "Tom planned to suggest in a meeting on June 27, 2025 at 9:30 AM"
        )
        self.assertEqual(result[0].metadata.user_id, "user1")

    def test_process_doc_data(self):
        """Test processing document chunks into memory items."""
        scene_data_info = {"file": "tests/mem_reader/test.txt", "text": "Parsed document text"}
        info = {"user_id": "user1", "session_id": "session1"}

        # Mock LLM response
        mock_response = (
            '{"value": "A sample document about testing.", "tags": ["document"], "key": "title"}'
        )
        self.reader.llm.generate.return_value = mock_response
        self.reader.chunker.chunk.return_value = [
            Chunk(text="Parsed document text", token_count=3, sentences=["Parsed document text"])
        ]
        self.reader.parse_json_result = lambda x: json.loads(x)

        result = self.reader._process_doc_data(scene_data_info, info)

        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], TextualMemoryItem)
        self.assertIn("sample document", result[0].memory)

    def test_get_scene_data_info_with_chat(self):
        """Test extracting chat info from scene data."""
        scene_data = [
            [
                {
                    "role": "user",
                    "chat_time": "3 May 2025",
                    "content": "I'm feeling a bit down today.",
                },
                {
                    "role": "assistant",
                    "chat_time": "3 May 2025",
                    "content": "I'm sorry to hear that. Do you want to talk about what's been going on?",
                },
                {
                    "role": "user",
                    "chat_time": "3 May 2025",
                    "content": "It's just been a tough couple of days...",
                },
            ],
        ]
        result = self.reader.get_scene_data_info(scene_data, type="chat")

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "user: [3 May 2025]: I'm feeling a bit down today.")

    @patch("memos.mem_reader.simple_struct.ParserFactory")
    def test_get_scene_data_info_with_doc(self, mock_parser_factory):
        """Test parsing document files."""
        parser_instance = MagicMock()
        parser_instance.parse.return_value = "Parsed document text.\n"
        mock_parser_factory.from_config.return_value = parser_instance

        scene_data = [{"fake_file_like": "should trigger parse"}]
        result = self.reader.get_scene_data_info(scene_data, type="doc")

        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["text"], "Parsed document text.\n")

    def test_parse_json_result_success(self):
        """Test successful JSON parsing."""
        raw_response = '{"summary": "Test summary", "tags": ["test"]}'
        result = self.reader.parse_json_result(raw_response)

        self.assertIsInstance(result, dict)
        self.assertIn("summary", result)

    def test_parse_json_result_failure(self):
        """Test failure in JSON parsing."""
        raw_response = "Invalid JSON string"
        result = self.reader.parse_json_result(raw_response)

        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
