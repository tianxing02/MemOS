from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, field_validator

from memos.log import get_logger
from memos.mem_scheduler.modules.misc import AutoDroppingQueue, DictConversionMixin
from memos.mem_scheduler.schemas.general_schemas import (
    DEFAULT_MAX_QUERY_KEY_WORDS,
    DEFAULT_WEIGHT_VECTOR_FOR_RANKING,
    NOT_INITIALIZED,
)
from memos.mem_scheduler.utils.filter_utils import transform_name_to_key
from memos.memories.textual.tree import TextualMemoryItem


logger = get_logger(__name__)

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent.parent.parent


# ============== Queries ==============
class QueryMonitorItem(BaseModel, DictConversionMixin):
    item_id: str = Field(
        description="Unique identifier for the query item", default_factory=lambda: str(uuid4())
    )
    query_text: str = Field(
        ...,
        description="The actual user query text content",
        min_length=1,
    )
    keywords: list[str] | None = Field(
        default=None,
        min_length=1,  # If provided, shouldn't be empty
        description="Semantic keywords extracted from the query text",
    )
    max_keywords: ClassVar[int] = DEFAULT_MAX_QUERY_KEY_WORDS

    timestamp: datetime = Field(
        default_factory=datetime.now, description="Timestamp indicating when query was submitted"
    )

    @field_validator("keywords", mode="before")
    @classmethod
    def validate_keywords(cls, v, values):
        if v is None:
            return None

        if not isinstance(v, list):
            raise ValueError("Keywords must be a list")

        if len(v) > cls.max_keywords:
            logger.warning(
                f"Keywords list truncated from {len(v)} to {cls.max_keywords} items. "
                f"Configure max_keywords class attribute to adjust this limit."
            )
            return v[: cls.max_keywords]
        return v

    @classmethod
    def with_max_keywords(cls, limit: int):
        """Create a new class with custom keywords limit."""
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("Max keywords limit must be positive integer")

        return type(f"{cls.__name__}_MaxKeywords{limit}", (cls,), {"max_keywords": limit})


class QueryMonitorQueue(AutoDroppingQueue[QueryMonitorItem]):
    """
    A thread-safe queue for monitoring queries with timestamp and keyword tracking.
    Each item is expected to be a dictionary containing:
    """

    def put(self, item: QueryMonitorItem, block: bool = True, timeout: float | None = None) -> None:
        """
        Add a query item to the queue. Ensures the item is of correct type.

        Args:
            item: A QueryMonitorItem instance
        """
        if not isinstance(item, QueryMonitorItem):
            raise ValueError("Item must be an instance of QueryMonitorItem")
        super().put(item, block, timeout)

    def get_queries_by_timestamp(
        self, start_time: datetime, end_time: datetime
    ) -> list[QueryMonitorItem]:
        """
        Retrieve queries added between the specified time range.
        """
        with self.mutex:
            return [item for item in self.queue if start_time <= item.timestamp <= end_time]

    def get_keywords_collections(self) -> Counter:
        """
        Generate a Counter containing keyword frequencies across all queries.

        Returns:
            Counter object with keyword counts
        """
        with self.mutex:
            all_keywords = [kw for item in self.queue for kw in item.keywords]
            return Counter(all_keywords)

    def get_queries_with_timesort(self, reverse: bool = True) -> list[str]:
        """
        Retrieve all queries sorted by timestamp.

        Args:
            reverse: If True, sort in descending order (newest first),
                     otherwise sort in ascending order (oldest first)

        Returns:
            List of query items sorted by timestamp
        """
        with self.mutex:
            return [
                monitor.query_text
                for monitor in sorted(self.queue, key=lambda x: x.timestamp, reverse=reverse)
            ]


# ============== Memories ==============
class MemoryMonitorItem(BaseModel, DictConversionMixin):
    item_id: str = Field(
        description="Unique identifier for the memory item", default_factory=lambda: str(uuid4())
    )
    memory_text: str = Field(
        ...,
        description="The actual content of the memory",
        min_length=1,
    )
    tree_memory_item: TextualMemoryItem | None = Field(
        default=None, description="Optional textual memory item"
    )
    tree_memory_item_mapping_key: str = Field(
        description="Key generated from memory_text using transform_name_to_key",
    )
    keywords_score: float = Field(
        default=NOT_INITIALIZED,
        description="The score generate by counting keywords in queries",
        ge=NOT_INITIALIZED,  # Minimum value of 0
    )
    sorting_score: float = Field(
        default=NOT_INITIALIZED,
        description="The score generate from rerank process",
        ge=NOT_INITIALIZED,  # Minimum value of 0
    )
    importance_score: float = Field(
        default=NOT_INITIALIZED,
        description="Numerical score representing the memory's importance",
        ge=NOT_INITIALIZED,  # Minimum value of 0
    )
    recording_count: int = Field(
        default=1,
        description="How many times this memory has been recorded",
        ge=1,  # Greater than or equal to 1
    )

    @field_validator("tree_memory_item_mapping_key", mode="before")
    def generate_mapping_key(cls, v, values):  # noqa: N805
        if v is None and "memory_text" in values:
            return transform_name_to_key(values["memory_text"])
        return v

    def get_importance_score(self, weight_vector: list[float] | None = None) -> float:
        """
        Calculate the effective score for the memory item.

        Returns:
            float: The importance_score if it has been initialized (>=0),
                   otherwise the recording_count converted to float.

        Note:
            This method provides a unified way to retrieve a comparable score
            for memory items, regardless of whether their importance has been explicitly set.
        """
        if weight_vector is None:
            logger.warning("weight_vector of get_importance_score is None.")
            weight_vector = DEFAULT_WEIGHT_VECTOR_FOR_RANKING
        assert sum(weight_vector) == 1
        normalized_keywords_score = min(self.keywords_score * weight_vector[1], 5)
        normalized_recording_count_score = min(self.recording_count * weight_vector[2], 2)
        self.importance_score = (
            self.sorting_score * weight_vector[0]
            + normalized_keywords_score
            + normalized_recording_count_score
        )
        return self.importance_score


class MemoryMonitorManager(BaseModel, DictConversionMixin):
    user_id: str = Field(..., description="Required user identifier", min_length=1)
    mem_cube_id: str = Field(..., description="Required memory cube identifier", min_length=1)
    memories: list[MemoryMonitorItem] = Field(
        default_factory=list, description="Collection of memory items"
    )
    max_capacity: int | None = Field(
        default=None, description="Maximum number of memories allowed (None for unlimited)", ge=1
    )

    @computed_field
    @property
    def memory_size(self) -> int:
        """Automatically calculated count of memory items."""
        return len(self.memories)

    @property
    def memories_mapping_dict(self) -> dict[str, MemoryMonitorItem]:
        """
        Generate a mapping dictionary for the memories in MemoryMonitorManager,
        using tree_memory_item_mapping_key as the key and MemoryMonitorItem as the value.

        Returns:
            Dict[str, MemoryMonitorItem]: A dictionary where keys are
            tree_memory_item_mapping_key values from MemoryMonitorItem,
            and values are the corresponding MemoryMonitorItem objects.
        """
        mapping_dict = {
            mem_item.tree_memory_item_mapping_key: mem_item for mem_item in self.memories
        }

        logger.debug(
            f"Generated memories mapping dict for user_id={self.user_id}, "
            f"mem_cube_id={self.mem_cube_id}, "
            f"total_items={len(mapping_dict)}, "
            f"source_memory_count={len(self.memories)}"
        )
        return mapping_dict

    def get_sorted_mem_monitors(self, reverse=True) -> list[MemoryMonitorItem]:
        """
        Retrieve memory monitors sorted by their ranking score in descending order.

        Returns:
            list[MemoryMonitorItem]: Sorted list of memory monitor items.
        """
        return sorted(
            self.memories,
            key=lambda item: item.get_importance_score(
                weight_vector=DEFAULT_WEIGHT_VECTOR_FOR_RANKING
            ),
            reverse=reverse,
        )

    def update_memories(
        self, new_memory_monitors: list[MemoryMonitorItem], partial_retention_number: int
    ) -> MemoryMonitorItem:
        """
        Update memories based on monitor_working_memories.
        """

        # Validate partial_retention_number
        if partial_retention_number < 0:
            raise ValueError("partial_retention_number must be non-negative")

        # Step 1: Update existing memories or add new ones
        added_count = 0
        memories_mapping_dict = self.memories_mapping_dict
        new_mem_set = set()
        for memory_monitor in new_memory_monitors:
            if memory_monitor.tree_memory_item_mapping_key in memories_mapping_dict:
                # Update existing memory
                item: MemoryMonitorItem = memories_mapping_dict[
                    memory_monitor.tree_memory_item_mapping_key
                ]
                item.recording_count += 1
                item.keywords_score = memory_monitor.keywords_score
                item.sorting_score = memory_monitor.sorting_score
            else:
                # Add new memory
                self.memories.append(memory_monitor)
                added_count += 1

            new_mem_set.add(memory_monitor.tree_memory_item_mapping_key)

        # Step 2: Identify memories to remove
        old_mem_monitor_list = []
        for mem_monitor in self.memories:
            if mem_monitor.tree_memory_item_mapping_key not in new_mem_set:
                old_mem_monitor_list.append(mem_monitor)

        # Sort memories by recording_count in descending order
        sorted_old_mem_monitors = sorted(
            old_mem_monitor_list,
            key=lambda item: item.get_importance_score(
                weight_vector=DEFAULT_WEIGHT_VECTOR_FOR_RANKING
            ),
            reverse=True,
        )

        # Keep the top N old memories
        memories_to_remove = sorted_old_mem_monitors[partial_retention_number:]
        memories_to_change_score = sorted_old_mem_monitors[:partial_retention_number]

        # Step 3: Remove identified memories and change the scores of left old memories
        for memory in memories_to_remove:
            self.memories.remove(memory)

        for memory in memories_to_change_score:
            memory.sorting_score = 0
            memory.recording_count = 0
            memory.keywords_score = 0

        # Step 4: Enforce max_capacity if set
        sorted_memories = sorted(
            self.memories,
            key=lambda item: item.get_importance_score(
                weight_vector=DEFAULT_WEIGHT_VECTOR_FOR_RANKING
            ),
            reverse=True,
        )
        # Keep only the top max_capacity memories
        self.memories = sorted_memories[: self.max_capacity]

        # Log the update result
        logger.info(
            f"Updated monitor manager for user {self.user_id}, mem_cube {self.mem_cube_id}: "
            f"Total memories: {len(self.memories)}, "
            f"Added/Updated: {added_count}, "
            f"Removed: {len(memories_to_remove)} (excluding top {partial_retention_number} by recording_count)"
        )

        return self.memories
