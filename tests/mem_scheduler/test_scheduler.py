import sys
import unittest

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from memos.configs.mem_scheduler import SchedulerConfigFactory
from memos.llms.base import BaseLLM
from memos.mem_cube.general import GeneralMemCube
from memos.mem_scheduler.modules.monitor import SchedulerMonitor
from memos.mem_scheduler.modules.retriever import SchedulerRetriever
from memos.mem_scheduler.scheduler_factory import SchedulerFactory
from memos.mem_scheduler.schemas.general_schemas import (
    ANSWER_LABEL,
    QUERY_LABEL,
    TreeTextMemory_SEARCH_METHOD,
)
from memos.mem_scheduler.schemas.message_schemas import (
    ScheduleLogForWebItem,
)
from memos.memories.textual.tree import TreeTextMemory


FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # Enable execution from any working directory


class TestGeneralScheduler(unittest.TestCase):
    def setUp(self):
        """Initialize test environment with mock objects and test scheduler instance."""
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

        # Initialize modules with mock LLM
        self.scheduler.initialize_modules(chat_llm=self.llm, process_llm=self.llm)
        self.scheduler.mem_cube = self.mem_cube

        # Set current user and memory cube ID for testing
        self.scheduler.current_user_id = "test_user"
        self.scheduler.current_mem_cube_id = "test_cube"

    def test_initialization(self):
        """Test that scheduler initializes with correct default values and handlers."""
        # Verify handler registration
        self.assertTrue(QUERY_LABEL in self.scheduler.dispatcher.handlers)
        self.assertTrue(ANSWER_LABEL in self.scheduler.dispatcher.handlers)

    def test_initialize_modules(self):
        """Test module initialization with proper component assignments."""
        self.assertEqual(self.scheduler.chat_llm, self.llm)
        self.assertIsInstance(self.scheduler.monitor, SchedulerMonitor)
        self.assertIsInstance(self.scheduler.retriever, SchedulerRetriever)

    def test_submit_web_logs(self):
        """Test submission of web logs with updated data structure."""
        # Create log message with all required fields
        log_message = ScheduleLogForWebItem(
            user_id="test_user",
            mem_cube_id="test_cube",
            label=QUERY_LABEL,
            from_memory_type="WorkingMemory",  # 新增字段
            to_memory_type="LongTermMemory",  # 新增字段
            log_content="Test Content",
            current_memory_sizes={
                "long_term_memory_size": 0,
                "user_memory_size": 0,
                "working_memory_size": 0,
                "transformed_act_memory_size": 0,
            },
            memory_capacities={
                "long_term_memory_capacity": 1000,
                "user_memory_capacity": 500,
                "working_memory_capacity": 100,
                "transformed_act_memory_capacity": 0,
            },
        )

        # Empty the queue by consuming all elements
        while not self.scheduler._web_log_message_queue.empty():
            self.scheduler._web_log_message_queue.get()

        # Submit the log message
        self.scheduler._submit_web_logs(messages=log_message)

        # Verify the message was added to the queue
        self.assertEqual(self.scheduler._web_log_message_queue.qsize(), 1)

        # Get the actual message from the queue
        actual_message = self.scheduler._web_log_message_queue.get()

        # Verify core fields
        self.assertEqual(actual_message.user_id, "test_user")
        self.assertEqual(actual_message.mem_cube_id, "test_cube")
        self.assertEqual(actual_message.label, QUERY_LABEL)
        self.assertEqual(actual_message.from_memory_type, "WorkingMemory")
        self.assertEqual(actual_message.to_memory_type, "LongTermMemory")
        self.assertEqual(actual_message.log_content, "Test Content")

        # Verify memory sizes
        self.assertEqual(actual_message.current_memory_sizes["long_term_memory_size"], 0)
        self.assertEqual(actual_message.current_memory_sizes["user_memory_size"], 0)
        self.assertEqual(actual_message.current_memory_sizes["working_memory_size"], 0)
        self.assertEqual(actual_message.current_memory_sizes["transformed_act_memory_size"], 0)

        # Verify memory capacities
        self.assertEqual(actual_message.memory_capacities["long_term_memory_capacity"], 1000)
        self.assertEqual(actual_message.memory_capacities["user_memory_capacity"], 500)
        self.assertEqual(actual_message.memory_capacities["working_memory_capacity"], 100)
        self.assertEqual(actual_message.memory_capacities["transformed_act_memory_capacity"], 0)

        # Verify auto-generated fields exist
        self.assertTrue(hasattr(actual_message, "item_id"))
        self.assertTrue(isinstance(actual_message.item_id, str))
        self.assertTrue(hasattr(actual_message, "timestamp"))
        self.assertTrue(isinstance(actual_message.timestamp, datetime))

    def test_search_with_empty_results(self):
        """Test search method with empty results."""
        # Setup mock memory cube and text memory
        mock_mem_cube = MagicMock()
        mock_mem_cube.text_mem = self.tree_text_memory

        # Setup mock search results for both memory types
        self.tree_text_memory.search.side_effect = [
            [],  # results_long_term
            [],  # results_user
        ]

        # Test search
        results = self.scheduler.retriever.search(
            query="Test query", mem_cube=mock_mem_cube, top_k=5, method=TreeTextMemory_SEARCH_METHOD
        )

        # Verify results
        self.assertEqual(results, [])

        # Verify search was called twice (for LongTermMemory and UserMemory)
        self.assertEqual(self.tree_text_memory.search.call_count, 2)
        self.tree_text_memory.search.assert_any_call(
            query="Test query", top_k=5, memory_type="LongTermMemory"
        )
        self.tree_text_memory.search.assert_any_call(
            query="Test query", top_k=5, memory_type="UserMemory"
        )
