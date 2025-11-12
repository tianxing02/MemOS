import json
import os
import re
import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from eval.eval_score2 import eval_acc_and_f1, eval_score, show_results
from eval.extract_answer import extract_answer

from memos.configs.mem_cube import GeneralMemCubeConfig
from memos.configs.mem_os import MOSConfig
from memos.mem_cube.general import GeneralMemCube
from memos.mem_os.main import MOS


load_dotenv()
openapi_config = {
    "model_name_or_path": "gpt-4o",
    "top_k": 50,
    "remove_think_prefix": True,
    "api_key": os.getenv("OPENAI_API_KEY", "sk-xxxxx"),
    "api_base": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
}
neo4j_uri = os.getenv("NEO4J_URI", "bolt://47.117.41.207:7687")
space_name = "nebula-longbench-stx-new"

doc_paths = [
    f
    for f in os.listdir("evaluation/data/mmlongbench/documents_xcy")
    if os.path.isfile(os.path.join("evaluation/data/mmlongbench/documents_xcy", f))
]

with open("evaluation/data/mmlongbench/all_samples_with_gt.json") as f:
    samples = json.load(f)


def process_doc(doc_file):
    user_name = "user_" + doc_file
    config = {
        "user_id": user_name,
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
    mos_config = MOSConfig(**config)
    mos = MOS(mos_config)

    mem_cube_config = GeneralMemCubeConfig.model_validate(
        {
            "user_id": user_name,
            "cube_id": user_name,
            "text_mem": {
                "backend": "tree_text",
                "config": {
                    "extractor_llm": {"backend": "openai", "config": openapi_config},
                    "dispatcher_llm": {"backend": "openai", "config": openapi_config},
                    "graph_db": {
                        "backend": "neo4j",
                        "config": {
                            "uri": "bolt://47.117.45.189:7687",
                            "user": "neo4j",
                            "password": "iaarlichunyu",
                            "db_name": "stx-mmlongbench-001",
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
    mem_cube = GeneralMemCube(mem_cube_config)

    temp_dir = os.path.join("tmp", doc_file)

    # Only dump when the directory does not exist or is empty
    # Avoid calling dump on a non-empty directory to prevent MemCubeError
    if (not os.path.exists(temp_dir)) or (not os.listdir(temp_dir)):
        mem_cube.dump(temp_dir)

    mos.register_mem_cube(temp_dir, mem_cube_id=user_name)

    with open("evaluation/scripts/mmlongbench/eval/prompt_for_answer_extraction.md") as f:
        prompt = f.read()

    samples_res = []
    doc_samples = [s for s in samples if s.get("doc_id") == doc_file]

    if len(doc_samples) == 0:
        return []

    sample = doc_samples[0]
    question_list = sample["question"]
    answer_list = sample["answer"]

    for idx, question in enumerate(question_list):
        gt = answer_list.get(str(idx))

        try_cnt, is_success = 0, False
        while True:
            try:
                mos.clear_messages()
                response = mos.chat(question, user_name)
                is_success = True
            except Exception as e:
                print(f"[{doc_file}] Error:", e)
                traceback.print_exc()
                try_cnt += 1
                response = "Failed"
            if is_success or try_cnt > 5:
                break

        sample_item = dict(sample)
        sample_item["question"] = question
        sample_item["answer"] = gt
        sample_item["response"] = response

        extracted_res = extract_answer(sample_item["question"], response, prompt)
        sample_item["extracted_res"] = extracted_res

        print("--------------------------------------")
        pred_ans = extracted_res.split("Answer format:")[0].split("Extracted answer:")[1].strip()
        score = eval_score(question, gt, response)

        sample_item["pred"] = pred_ans
        sample_item["score"] = score
        samples_res.append(sample_item)

        print(f"Question: {question}")
        print(f"Response: {sample_item['response']}")
        print(f"Ground true: {gt}\tPred: {sample_item['pred']}\tScore: {sample_item['score']}")

    print("samples_res length: ", len(samples_res))
    return samples_res


if __name__ == "__main__":
    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_doc = {executor.submit(process_doc, doc_file): doc_file for doc_file in doc_paths}

        for future in as_completed(future_to_doc):
            doc_file = future_to_doc[future]
            try:
                res = future.result()
                results.extend(res)

                if len(res) > 0:
                    acc, f1 = eval_acc_and_f1(results)
                    print()
                    print(f"Avg acc: {acc}")
                    print(f"Avg f1: {f1}")

                with open(
                    "evaluation/data/mmlongbench/test_results2.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"[{doc_file}] failed with {e}")
                traceback.print_exc()

    acc, f1 = eval_acc_and_f1(results)
    print("--------------------------------------")
    print(f"Final avg acc: {acc}")
    print(f"Final avg f1: {f1}")

    show_results(
        results,
        show_path=re.sub(
            r"\.json$", ".txt", "evaluation/data/mmlongbench/test_results_report2.txt"
        ),
    )
