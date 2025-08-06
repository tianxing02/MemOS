import sys
import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

from memos.configs.mem_scheduler import SchedulerConfigFactory
from memos.llms.base import BaseLLM
from memos.mem_cube.general import GeneralMemCube
from memos.mem_scheduler.scheduler_factory import SchedulerFactory
from memos.mem_scheduler.utils.filter_utils import (
    filter_similar_memories,
    filter_too_short_memories,
)
from memos.memories.textual.tree import TreeTextMemory


FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # Enable execution from any working directory


class TestSchedulerRetriever(unittest.TestCase):
    def setUp(self):
        """Initialize test environment with mock objects."""
        example_scheduler_config_path = (
            f"{BASE_DIR}/examples/data/config/mem_scheduler/general_scheduler_config.yaml"
        )
        scheduler_config = SchedulerConfigFactory.from_yaml_file(
            yaml_path=example_scheduler_config_path
        )
        mem_scheduler = SchedulerFactory.from_config(scheduler_config)
        self.scheduler = mem_scheduler
        self.llm = MagicMock(spec=BaseLLM)
        self.mem_cube = MagicMock(spec=GeneralMemCube)
        self.tree_text_memory = MagicMock(spec=TreeTextMemory)
        self.mem_cube.text_mem = self.tree_text_memory
        self.mem_cube.act_mem = MagicMock()

        # Initialize general_modules with mock LLM
        self.scheduler.initialize_modules(chat_llm=self.llm, process_llm=self.llm)
        self.scheduler.mem_cube = self.mem_cube

        self.retriever = self.scheduler.retriever

        # Mock logging to verify messages
        self.logging_warning_patch = patch("logging.warning")
        self.mock_logging_warning = self.logging_warning_patch.start()

        self.logger_info_patch = patch("memos.mem_scheduler.general_modules.retriever.logger.info")
        self.mock_logger_info = self.logger_info_patch.start()

    def tearDown(self):
        """Clean up patches."""
        self.logging_warning_patch.stop()
        self.logger_info_patch.stop()

    def test_filter_similar_memories_empty_input(self):
        """Test filter_similar_memories with empty input list."""
        result = filter_similar_memories([])
        self.assertEqual(result, [])

    def test_filter_similar_memories_no_duplicates(self):
        """Test filter_similar_memories with no duplicate memories."""
        memories = [
            "This is a completely unique first memory",
            "This second memory is also totally unique",
            "And this third one has nothing in common with the others",
        ]

        result = filter_similar_memories(memories)
        self.assertEqual(len(result), 3)
        self.assertEqual(set(result), set(memories))

    def test_filter_similar_memories_with_duplicates(self):
        """Test filter_similar_memories with duplicate memories."""
        memories = [
            "The user is planning to move to Chicago next month, although the exact date of the move is unclear.",
            "The user is planning to move to Chicago next month, which reflects a significant change in their living situation.",
            "The user is planning to move to Chicago in the upcoming month, indicating a significant change in their living situation.",
        ]
        result = filter_similar_memories(memories, similarity_threshold=0.75)
        self.assertLess(len(result), len(memories))

    def test_filter_similar_memories_error_handling(self):
        """Test filter_similar_memories error handling."""
        # Test with non-string input (should return original list due to error)
        memories = ["valid text", 12345, "another valid text"]
        result = filter_similar_memories(memories)
        self.assertEqual(result, memories)

    def test_filter_too_short_memories_empty_input(self):
        """Test filter_too_short_memories with empty input list."""
        result = filter_too_short_memories([])
        self.assertEqual(result, [])

    def test_filter_too_short_memories_all_valid(self):
        """Test filter_too_short_memories with all valid memories."""
        memories = [
            "This memory is definitely long enough to be kept",
            "This one is also sufficiently lengthy to pass the filter",
            "And this third memory meets the minimum length requirements too",
        ]

        result = filter_too_short_memories(memories, min_length_threshold=5)
        self.assertEqual(len(result), 3)
        self.assertEqual(result, memories)

    def test_filter_too_short_memories_with_short_ones(self):
        """Test filter_too_short_memories with some short memories."""
        memories = [
            "This is long enough",  # 5 words
            "Too short",  # 2 words
            "This one passes",  # 3 words (assuming threshold is 3)
            "Nope",  # 1 word
            "This is also acceptable",  # 4 words
        ]

        # Test with word count threshold of 3
        result = filter_too_short_memories(memories, min_length_threshold=3)
        self.assertEqual(len(result), 3)
        self.assertNotIn("Too short", result)
        self.assertNotIn("Nope", result)

    def test_filter_too_short_memories_edge_case(self):
        """Test filter_too_short_memories with edge case length."""
        memories = ["Exactly three words here", "Two words only", "One", "Four words right here"]

        # Test with threshold exactly matching some memories
        # The implementation uses word count, not character count
        result = filter_too_short_memories(memories, min_length_threshold=3)
        self.assertEqual(
            len(result), 3
        )  # "Exactly three words here", "Two words only", "Four words right here"
        self.assertIn("Exactly three words here", result)
        self.assertIn("Four words right here", result)
