from memos import log
from memos.configs.mem_reader import SimpleStructMemReaderConfig
from memos.configs.memory import TreeTextMemoryConfig
from memos.mem_reader.simple_struct import SimpleStructMemReader
from memos.memories.textual.tree import TreeTextMemory


logger = log.get_logger(__name__)
tree_config = TreeTextMemoryConfig.from_json_file(
    "examples/data/config/tree_config_shared_database.json"
)
my_tree_textual_memory = TreeTextMemory(tree_config)
my_tree_textual_memory.delete_all()

# Create a memory reader instance
reader_config = SimpleStructMemReaderConfig.from_json_file(
    "examples/data/config/simple_struct_reader_config.json"
)
reader = SimpleStructMemReader(reader_config)

# Processing Documents
doc_paths = [
    "evaluation/data/mmlongbench/documents/0b85477387a9d0cc33fca0f4becaa0e5.pdf",
    "evaluation/data/mmlongbench/documents/0e94b4197b10096b1f4c699701570fbf.pdf",
]
# Acquiring memories from documents
doc_memory = reader.get_memory(doc_paths, "doc", info={"user_id": "1111", "session_id": "2222"})

for m_list in doc_memory:
    added_ids = my_tree_textual_memory.add(m_list)

# my_tree_textual_memory.dump
my_tree_textual_memory.dump("tmp/my_tree_textual_memory")
