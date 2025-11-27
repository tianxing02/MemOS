import json
import os

from concurrent.futures import ThreadPoolExecutor, as_completed

import datasets as ds_mod

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from memos.configs.mem_cube import GeneralMemCubeConfig
from memos.configs.mem_os import MOSConfig
from memos.mem_cube.general import GeneralMemCube
from memos.mem_os.main import MOS


load_dotenv()

db_name = "stx-hotpot-001"

openapi_config = {
    "model_name_or_path": "gpt-4o",
    "top_k": 50,
    "remove_think_prefix": True,
    "api_key": os.getenv("OPENAI_API_KEY", "sk-xxxxx"),
    "api_base": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
}
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")

base_config = {
    "chat_model": {
        "backend": "openai",
        "config": openapi_config,
    },
    "mem_reader": {
        "backend": "simple_struct",
        "config": {
            "llm": {"backend": "openai", "config": openapi_config},
            "embedder": {
                "backend": "universal_api",
                "config": {
                    "provider": "openai",
                    "api_key": os.getenv("OPENAI_API_KEY", "sk-xxxxx"),
                    "model_name_or_path": "text-embedding-3-large",
                    "base_url": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
                },
            },
            "chunker": {
                "backend": "sentence",
                "config": {
                    "tokenizer_or_token_counter": "gpt2",
                    "chunk_size": 512,
                    "chunk_overlap": 128,
                    "min_sentences_per_chunk": 1,
                },
            },
        },
    },
    "max_turns_window": 20,
    "top_k": 5,
    "enable_textual_memory": True,
    "enable_activation_memory": False,
    "enable_parametric_memory": False,
}


def filter_memory_data(memories_data):
    filtered_data = {}
    for key, value in memories_data.items():
        if key == "text_mem":
            filtered_data[key] = []
            for mem_group in value:
                if "memories" in mem_group and isinstance(mem_group["memories"], list):
                    # New data structure: directly a list of TextualMemoryItem objects
                    filtered_memories = []
                    for memory_item in mem_group["memories"]:
                        # Create filtered dictionary
                        filtered_item = {
                            "id": memory_item.id,
                            "memory": memory_item.memory,
                            "metadata": {},
                        }
                        # Filter metadata, excluding embedding
                        if hasattr(memory_item, "metadata") and memory_item.metadata:
                            for attr_name in dir(memory_item.metadata):
                                if not attr_name.startswith("_") and attr_name != "embedding":
                                    attr_value = getattr(memory_item.metadata, attr_name)
                                    if not callable(attr_value):
                                        filtered_item["metadata"][attr_name] = attr_value
                        filtered_memories.append(filtered_item)

                    filtered_group = {
                        "cube_id": mem_group.get("cube_id", ""),
                        "memories": filtered_memories,
                    }
                    filtered_data[key].append(filtered_group)
                else:
                    # Old data structure: dictionary with nodes and edges
                    filtered_group = {
                        "memories": {"nodes": [], "edges": mem_group["memories"].get("edges", [])}
                    }
                    for node in mem_group["memories"].get("nodes", []):
                        filtered_node = {
                            "id": node.get("id"),
                            "memory": node.get("memory"),
                            "metadata": {
                                k: v
                                for k, v in node.get("metadata", {}).items()
                                if k != "embedding"
                            },
                        }
                        filtered_group["memories"]["nodes"].append(filtered_node)
                    filtered_data[key].append(filtered_group)
        else:
            filtered_data[key] = value
    return filtered_data


def init_mos_and_cube(user_name: str) -> MOS:
    cfg = dict(base_config)
    cfg["user_id"] = user_name
    mos_config = MOSConfig(**cfg)
    mos = MOS(mos_config)
    cube_conf = GeneralMemCubeConfig.model_validate(
        {
            "user_id": user_name,
            "cube_id": f"{user_name}",
            "text_mem": {
                "backend": "tree_text",
                "config": {
                    "extractor_llm": {"backend": "openai", "config": openapi_config},
                    "dispatcher_llm": {"backend": "openai", "config": openapi_config},
                    "graph_db": {
                        "backend": "neo4j",
                        "config": {
                            "uri": neo4j_uri,
                            "user": "neo4j",
                            "password": "iaarlichunyu",
                            "db_name": db_name,
                            "auto_create": True,
                        },
                    },
                    "embedder": {
                        "backend": "universal_api",
                        "config": {
                            "provider": "openai",
                            "api_key": os.getenv("OPENAI_API_KEY", "sk-xxxxx"),
                            "model_name_or_path": "text-embedding-3-large",
                            "base_url": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
                        },
                    },
                    "reorganize": True,
                },
            },
            "act_mem": {},
            "para_mem": {},
        }
    )
    mem_cube = GeneralMemCube(cube_conf)
    temp_dir = "tmp/" + user_name
    if not os.path.exists(temp_dir) or not os.listdir(temp_dir):
        mem_cube.dump(temp_dir)
    mos.register_mem_cube(temp_dir, mem_cube_id=user_name)
    return mos


data = load_dataset("hotpotqa/hotpot_qa", "distractor")


def get_dataset_items(ds):
    dataset_cls = getattr(ds_mod, "Dataset", None) if ds_mod else None
    dataset_dict_cls = getattr(ds_mod, "DatasetDict", None) if ds_mod else None
    if dataset_dict_cls and isinstance(ds, dataset_dict_cls):
        split = ds.get("validation") or ds.get("train") or next(iter(ds.values()))
        return [split[i] for i in range(len(split))]
    if dataset_cls and isinstance(ds, dataset_cls):
        return [ds[i] for i in range(len(ds))]
    if isinstance(ds, list):
        return ds
    return []


items = get_dataset_items(data)


def build_context_text(context_list):
    parts = []
    for title, sentences in context_list:
        text = " ".join(s.strip() for s in sentences if s.strip())
        parts.append(f"{title}: {text}")
    return "\n".join(parts)


def build_and_ask(item):
    qid = item.get("_id") or item.get("id")
    question = item["question"]
    mos = init_mos_and_cube(qid)
    for entry in item["context"]:
        if isinstance(entry, list) and len(entry) >= 2:
            title, sentences = entry[0], entry[1]
        elif isinstance(entry, dict):
            title = entry.get("title", "")
            sentences = entry.get("sentences", [])
        else:
            continue
        text = " ".join(s.strip() for s in sentences if isinstance(s, str) and s.strip())
        if title or text:
            mos.add(memory_content=f"{title}: {text}")

    answer = mos.chat(question, qid).strip()
    print("question:", question)
    print("answer:", answer)
    return qid, answer


pred_answers = {}
output_dir = "evaluation/data/hotpot/output"
os.makedirs(output_dir, exist_ok=True)
pred_path = os.path.join(output_dir, "dev_distractor_pred.json")
gold_path = os.path.join(output_dir, "dev_distractor_gold.json")


def write_gold(items_list):
    try:
        with open(gold_path, "w", encoding="utf-8") as f:
            json.dump(items_list, f, ensure_ascii=False)
    except Exception:
        pass


def run_eval():
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "hotpot_eval_v1", "evaluation/scripts/hotpot/hotpot_evaluate_v1.py"
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        print("评估分数:")
        m.eval(pred_path, gold_path)
    except Exception as e:
        print("评估失败:", e)


write_gold(items)
processed = 0
interval = 200
print("开始评估，总样本:", len(items))
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(build_and_ask, item): idx for idx, item in enumerate(items)}
    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            qid, answer = future.result()
            pred_answers[qid] = answer
            processed += 1
            if processed % 50 == 0:
                print("已完成:", processed, "剩余:", len(items) - processed)
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump({"answer": pred_answers, "sp": {}}, f, ensure_ascii=False)
            if processed % interval == 0:
                print("阶段评估，当前进度:", processed)
                run_eval()
        except Exception as e:
            print("Error:", e)
print("最终评估:")
run_eval()
