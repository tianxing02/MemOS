import json
import shutil
import sys

from pathlib import Path

from memos.configs.mem_cube import GeneralMemCubeConfig
from memos.configs.mem_os import MOSConfig
from memos.configs.mem_scheduler import AuthConfig
from memos.log import get_logger
from memos.mem_cube.general import GeneralMemCube
from memos.mem_scheduler.mos_for_test_scheduler import MOSForTestScheduler


FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # Enable execution from any working directory

logger = get_logger(__name__)


def init_task():
    conversations = [
        {
            "role": "user",
            "content": "I have two dogs - Max (golden retriever) and Bella (pug). We live in Seattle.",
        },
        {"role": "assistant", "content": "Great! Any special care for them?"},
        {
            "role": "user",
            "content": "Max needs joint supplements. Actually, we're moving to Chicago next month.",
        },
        {
            "role": "user",
            "content": "Correction: Bella is 6, not 5. And she's allergic to chicken.",
        },
        {
            "role": "user",
            "content": "My partner's cat Whiskers visits weekends. Bella chases her sometimes.",
        },
    ]

    questions = [
        # 1. Basic factual recall (simple)
        {
            "question": "What breed is Max?",
            "category": "Pet",
            "expected": "golden retriever",
            "difficulty": "easy",
        },
        # 2. Temporal context (medium)
        {
            "question": "Where will I live next month?",
            "category": "Location",
            "expected": "Chicago",
            "difficulty": "medium",
        },
        # 3. Information correction (hard)
        {
            "question": "How old is Bella really?",
            "category": "Pet",
            "expected": "6",
            "difficulty": "hard",
            "hint": "User corrected the age later",
        },
        # 4. Relationship inference (harder)
        {
            "question": "Why might Whiskers be nervous around my pets?",
            "category": "Behavior",
            "expected": "Bella chases her sometimes",
            "difficulty": "harder",
        },
        # 5. Combined medical info (hardest)
        {
            "question": "Which pets have health considerations?",
            "category": "Health",
            "expected": "Max needs joint supplements, Bella is allergic to chicken",
            "difficulty": "hardest",
            "requires": ["combining multiple facts", "ignoring outdated info"],
        },
    ]
    return conversations, questions


if __name__ == "__main__":
    # set up data
    conversations, questions = init_task()

    # set configs
    mos_config = MOSConfig.from_yaml_file(
        f"{BASE_DIR}/examples/data/config/mem_scheduler/memos_config_w_scheduler_and_openai.yaml"
    )

    mem_cube_config = GeneralMemCubeConfig.from_yaml_file(
        f"{BASE_DIR}/examples/data/config/mem_scheduler/mem_cube_config.yaml"
    )

    # default local graphdb uri
    if AuthConfig.default_config_exists():
        auth_config = AuthConfig.from_local_config()

        mos_config.mem_reader.config.llm.config.api_key = auth_config.openai.api_key
        mos_config.mem_reader.config.llm.config.api_base = auth_config.openai.base_url

        mem_cube_config.text_mem.config.graph_db.config.uri = auth_config.graph_db.uri
        mem_cube_config.text_mem.config.graph_db.config.user = auth_config.graph_db.user
        mem_cube_config.text_mem.config.graph_db.config.password = auth_config.graph_db.password
        mem_cube_config.text_mem.config.graph_db.config.db_name = auth_config.graph_db.db_name
        mem_cube_config.text_mem.config.graph_db.config.auto_create = (
            auth_config.graph_db.auto_create
        )

    # Initialization
    mos = MOSForTestScheduler(mos_config)

    user_id = "user_1"
    mos.create_user(user_id)

    mem_cube_id = "mem_cube_5"
    mem_cube_name_or_path = f"{BASE_DIR}/outputs/mem_scheduler/{user_id}/{mem_cube_id}"

    if Path(mem_cube_name_or_path).exists():
        shutil.rmtree(mem_cube_name_or_path)
        print(f"{mem_cube_name_or_path} is not empty, and has been removed.")

    mem_cube = GeneralMemCube(mem_cube_config)
    mem_cube.dump(mem_cube_name_or_path)
    mos.register_mem_cube(
        mem_cube_name_or_path=mem_cube_name_or_path, mem_cube_id=mem_cube_id, user_id=user_id
    )

    mos.add(conversations, user_id=user_id, mem_cube_id=mem_cube_id)

    # Add interfering conversations
    file_path = Path(f"{BASE_DIR}/examples/data/mem_scheduler/scene_data.json")
    scene_data = json.load(file_path.open("r", encoding="utf-8"))
    mos.add(scene_data[0], user_id=user_id, mem_cube_id=mem_cube_id)
    mos.add(scene_data[1], user_id=user_id, mem_cube_id=mem_cube_id)

    for item in questions:
        print("===== Chat Start =====")
        query = item["question"]
        print(f"Query:\n {query}\n")
        response = mos.chat(query=query, user_id=user_id)
        print(f"Answer:\n {response}\n")

    mos.mem_scheduler.stop()
