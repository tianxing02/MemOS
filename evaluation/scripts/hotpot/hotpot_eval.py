import importlib.util
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from memos.configs.mem_cube import GeneralMemCubeConfig
from memos.configs.mem_os import MOSConfig
from memos.mem_cube.general import GeneralMemCube
from memos.mem_os.main import MOS


load_dotenv()

DB_NAME = "stx-hotpot-001"
OPENAPI_CONFIG = {
    "model_name_or_path": "gpt-4o",
    "top_k": 50,
    "remove_think_prefix": True,
    "api_key": os.getenv("OPENAI_API_KEY", "sk-xxxxx"),
    "api_base": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
}

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
DATASET = load_dataset("hotpotqa/hotpot_qa", "distractor")

BASE_CONFIG = {
    "chat_model": {"backend": "openai", "config": OPENAPI_CONFIG},
    "mem_reader": {
        "backend": "simple_struct",
        "config": {
            "llm": {"backend": "openai", "config": OPENAPI_CONFIG},
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


def init_mos_and_cube(user_name: str) -> MOS:
    cfg = dict(BASE_CONFIG)
    cfg["user_id"] = user_name

    mos_config = MOSConfig(**cfg)
    mos = MOS(mos_config)

    cube_conf = GeneralMemCubeConfig.model_validate(
        {
            "user_id": user_name,
            "cube_id": user_name,
            "text_mem": {
                "backend": "tree_text",
                "config": {
                    "extractor_llm": {"backend": "openai", "config": OPENAPI_CONFIG},
                    "dispatcher_llm": {"backend": "openai", "config": OPENAPI_CONFIG},
                    "graph_db": {
                        "backend": "neo4j",
                        "config": {
                            "uri": NEO4J_URI,
                            "user": "neo4j",
                            "password": "iaarlichunyu",
                            "db_name": DB_NAME,
                            "user_name": user_name,
                            "use_multi_db": False,
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
                    "reorganize": False,
                },
            },
            "act_mem": {},
            "para_mem": {},
        }
    )

    mem_cube = GeneralMemCube(cube_conf)

    temp_dir = Path("tmp") / user_name
    temp_dir.mkdir(parents=True, exist_ok=True)

    if not list(temp_dir.iterdir()):
        mem_cube.dump(str(temp_dir))

    mos.register_mem_cube(str(temp_dir), mem_cube_id=user_name)
    return mos


def build_context_text(context_list: List[Tuple[str, List[str]]]) -> str:
    parts = []
    for title, sentences in context_list:
        clean_text = " ".join(s.strip() for s in sentences if s.strip())
        parts.append(f"{title}: {clean_text}")
    return "\n".join(parts)


def build_and_ask(item: Dict[str, Any]) -> Tuple[str, str]:
    qid = item.get("_id") or item.get("id")
    question = item["question"]

    mos = init_mos_and_cube(qid)

    ctx = item.get("context")
    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sentences_list = ctx.get("sentences", [])
        for title, sentences in zip(titles, sentences_list, strict=False):
            text = " ".join(
                s.strip() for s in sentences if isinstance(s, str) and s.strip()
            )
            if title or text:
                mos.add(memory_content=f"{title}: {text}")
    else:
        for entry in ctx or []:
            if isinstance(entry, list) and len(entry) >= 2:
                title, sentences = entry[0], entry[1]
            elif isinstance(entry, dict):
                title = entry.get("title", "")
                sentences = entry.get("sentences", [])
            else:
                continue

            text = " ".join(
                s.strip() for s in sentences if isinstance(s, str) and s.strip()
            )
            if title or text:
                mos.add(memory_content=f"{title}: {text}")

    answer = mos.chat(question, qid).strip()
    print("question:", question)
    print("answer:", answer)

    return qid, answer


PRED_ANSWERS: Dict[str, str] = {}

OUTPUT_DIR = Path("evaluation/data/hotpot/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_PATH = OUTPUT_DIR / "dev_distractor_pred.json"
GOLD_PATH = OUTPUT_DIR / "dev_distractor_gold.json"


def write_gold(dataset):
    split = dataset.get("validation")

    items_list = [split[i] for i in range(10)]
    out = []

    for item in items_list:
        qid = item.get("_id") or item.get("id")

        sp = item.get("supporting_facts")
        if isinstance(sp, dict):
            titles = sp.get("title", [])
            sent_ids = sp.get("sent_id", [])
            sp_list = [[t, s] for t, s in zip(titles, sent_ids, strict=False)]
        else:
            sp_list = sp or []

        ctx = item.get("context")
        if isinstance(ctx, dict):
            titles = ctx.get("title", [])
            sentences = ctx.get("sentences", [])
            ctx_list = [[t, s] for t, s in zip(titles, sentences, strict=False)]
        else:
            ctx_list = ctx or []

        out.append(
            {
                "_id": qid,
                "question": item.get("question"),
                "answer": item.get("answer"),
                "supporting_facts": sp_list,
                "context": ctx_list,
            }
        )

    with open(GOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def run_eval():
    spec = importlib.util.spec_from_file_location(
        "hotpot_eval_v1", "evaluation/scripts/hotpot/hotpot_evaluate_v1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    print("评估分数:")
    module.eval(str(PRED_PATH), str(GOLD_PATH))


def save_pred():
    tmp_path = PRED_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"answer": PRED_ANSWERS, "sp": {}}, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, PRED_PATH)


def main():
    write_gold(DATASET)

    interval = 50
    split = DATASET.get("validation")
    items_list = list(split)

    if PRED_PATH.exists():
        try:
            with open(PRED_PATH, encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict) and isinstance(prev.get("answer"), dict):
                PRED_ANSWERS.update(prev["answer"])
        except json.JSONDecodeError:
            pass

    processed = len(PRED_ANSWERS)

    print("开始评估，总样本:", len(items_list))
    print("已存在预测:", processed)

    pending_items = [
        item for item in items_list
        if (item.get("_id") or item.get("id")) not in PRED_ANSWERS
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(build_and_ask, item): idx
            for idx, item in enumerate(pending_items)
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            qid, answer = future.result()
            PRED_ANSWERS[qid] = answer
            processed += 1

            if processed % 10 == 0:
                print("已完成:", processed, "剩余:", len(items_list) - processed)

            save_pred()

            if processed % interval == 0:
                print("阶段评估，当前进度:", processed)
                run_eval()

    print("最终评估:")
    run_eval()


if __name__ == "__main__":
    main()
