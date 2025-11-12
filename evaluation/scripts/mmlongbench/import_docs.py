import asyncio
import os
import traceback
import uuid

from memos import log
from memos.configs.mem_reader import SimpleStructMemReaderConfig
from memos.configs.memory import TreeTextMemoryConfig
from memos.mem_reader.simple_struct import SimpleStructMemReader
from memos.memories.textual.tree import TreeTextMemory


logger = log.get_logger(__name__)

# Create a memory reader instance
reader_config = SimpleStructMemReaderConfig.from_json_file(
    "examples/data/config/simple_struct_reader_config.json"
)
reader = SimpleStructMemReader(reader_config)

# Processing Documents
doc_paths = [
    "evaluation/data/mmlongbench/documents_xcy/" + f
    for f in os.listdir("evaluation/data/mmlongbench/documents_xcy")
    if os.path.isfile(os.path.join("evaluation/data/mmlongbench/documents_xcy", f))
]


async def process_doc(doc_path):
    print(f"🔄 Processing document: {doc_path}")
    session_id = str(uuid.uuid4())
    tree_config = TreeTextMemoryConfig.from_json_file(
        "examples/data/config/tree_config_shared_database.json"
    )
    doc_file = doc_path.split("/")[-1]
    if doc_file != "38331-h80.pdf":
        return doc_path

    user_id = "user_" + doc_file
    tree_config.graph_db.config.user_name = user_id
    temp_dir = "tmp/" + doc_file

    my_tree_textual_memory = TreeTextMemory(tree_config)
    doc_memory = await reader.get_memory(
        [doc_path], "doc", info={"user_id": user_id, "session_id": session_id}
    )

    count = 0
    for m_list in doc_memory:
        count += len(m_list)
        my_tree_textual_memory.add(m_list)
    print("total memories: ", count)

    my_tree_textual_memory.dump(temp_dir)
    return doc_path


async def main():
    batch_size = 4

    for i in range(0, len(doc_paths), batch_size):
        batch = doc_paths[i : i + batch_size]
        print(f"🚀 Starting batch {i // batch_size + 1} with {len(batch)} docs")

        tasks = [process_doc(p) for p in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for p, result in zip(batch, results, strict=False):
            if isinstance(result, Exception):
                print(f"❌ Error processing {p}: {result}")
                tb_text = "".join(traceback.TracebackException.from_exception(result).format())
                print(tb_text)
            else:
                print(f"✅ Finished {result}")


if __name__ == "__main__":
    asyncio.run(main())
