import json
import sys

from pathlib import Path

from memos.configs.mem_chat import MemChatConfigFactory
from memos.configs.mem_reader import NaiveMemReaderConfig
from memos.configs.mem_scheduler import SchedulerConfigFactory
from memos.configs.memory import TreeTextMemoryConfig
from memos.llms.factory import LLMFactory
from memos.mem_cube.general import GeneralMemCube
from memos.mem_reader.naive import NaiveMemReader
from memos.mem_scheduler.general_scheduler import GeneralScheduler
from memos.mem_scheduler.scheduler_factory import SchedulerFactory
from memos.memories.textual.tree import TreeTextMemory


FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # Enable execution from any working directory


def run_mem_scheduler(mem_scheduler):
    turns = [
        {"question": "What is quantum entanglement?"},
        {"question": "How is it different from classical physics?"},
        {"question": "So, what is its relationship with quantum computing?"},
    ]
    for turn in turns:
        print(f"Processing turn: {turn['question']}")
        print(
            f"Working memory: {[m.memory for m in mem_scheduler.mem_cube.text_mem.get_working_memory()]}"
        )
        session_result = mem_scheduler.process_session_turn(turn)
        print(
            f"Working memory after process:{[m.memory for m in mem_scheduler.mem_cube.text_mem.get_working_memory()]}"
        )
    print(session_result)


if __name__ == "__main__":
    print("Initializing MemChatConfig...")
    config = MemChatConfigFactory.from_yaml_file(
        f"{BASE_DIR}/examples/data/config/mem_scheduler/mem_chat_config.yaml"
    )
    chat_llm = LLMFactory.from_config(config.config.chat_llm)

    # initialize mem cube
    init_mem_cube = f"{BASE_DIR}/examples/data/mem_cube_2"
    print("Loading MemChatCube...")
    mem_cube = GeneralMemCube.init_from_dir(init_mem_cube)

    # initialize mem scheduler
    example_scheduler_config_path = (
        f"{BASE_DIR}/examples/data/config/mem_scheduler/general_scheduler_config.yaml"
    )
    scheduler_config = SchedulerConfigFactory.from_yaml_file(
        yaml_path=example_scheduler_config_path
    )
    mem_scheduler: GeneralScheduler = SchedulerFactory.from_config(scheduler_config)
    mem_scheduler.initialize_modules(chat_llm=chat_llm)
    mem_scheduler.mem_cube = mem_cube

    tree_config = TreeTextMemoryConfig.from_json_file(
        f"{BASE_DIR}/examples/data/config/tree_config.json"
    )
    tree_config.graph_db.config.uri = "bolt://123.57.48.226:7687"
    text_mem = TreeTextMemory(tree_config)
    mem_scheduler.mem_cube.text_mem = text_mem

    # Create a memory reader instance
    reader_config = NaiveMemReaderConfig.from_json_file(
        f"{BASE_DIR}/examples/data/config/naive_reader_config.json"
    )

    reader = NaiveMemReader(reader_config)
    scene_data_file = Path(f"{BASE_DIR}/examples/data/mem_scheduler/scene_data.json")
    scene_data = json.load(scene_data_file.open("r", encoding="utf-8"))
    # Acquiring memories
    memory = reader.get_memory(
        scene_data, type="chat", info={"user_id": "1234", "session_id": "2222"}
    )

    print("==== Add memories ====")
    for m_list in memory:
        text_mem.add(m_list)
    run_mem_scheduler(mem_scheduler)
