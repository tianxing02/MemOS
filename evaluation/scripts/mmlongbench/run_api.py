import argparse
import json
import os
import re
import uuid

from dotenv import load_dotenv
from eval.eval_score import eval_acc_and_f1, eval_score, show_results
from eval.extract_answer import extract_answer
from tqdm import tqdm

from memos.configs.mem_cube import GeneralMemCubeConfig
from memos.configs.mem_os import MOSConfig
from memos.mem_cube.general import GeneralMemCube
from memos.mem_os.main import MOS


load_dotenv()
openapi_config = {
    "model_name_or_path": "gpt-4o-mini",
    "temperature": 0.8,
    "max_tokens": 1024,
    "top_p": 0.9,
    "top_k": 50,
    "remove_think_prefix": True,
    "api_key": os.getenv("OPENAI_API_KEY", "sk-xxxxx"),
    "api_base": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
}
neo4j_uri = os.getenv("NEO4J_URI", "bolt://47.117.41.207:7687")


def process_document(doc_path):
    """Process a single document by creating MOS and MemCube with unique user ID"""
    # Create MOS Config with unique user_id
    user_name = str(uuid.uuid4())
    config = {
        "user_id": user_name,
        "chat_model": {
            "backend": "openai",
            "config": openapi_config,
        },
        "mem_reader": {
            "backend": "simple_struct",
            "config": {
                "llm": {
                    "backend": "openai",
                    "config": openapi_config,
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

    # Create MemCube configuration with fixed database name
    mem_cube_config = GeneralMemCubeConfig.model_validate(
        {
            "user_id": user_name,
            "cube_id": user_name,
            "text_mem": {
                "backend": "tree_text",
                "config": {
                    "extractor_llm": {
                        "backend": "openai",
                        "config": openapi_config,
                    },
                    "dispatcher_llm": {
                        "backend": "openai",
                        "config": openapi_config,
                    },
                    "graph_db": {
                        "backend": "neo4j",
                        "config": {
                            "uri": neo4j_uri,
                            "user": "neo4j",
                            "password": "iaarlichunyu",
                            "db_name": "mm-long-bench-shared",
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
        },
    )
    mem_cube = GeneralMemCube(mem_cube_config)

    # Save and register MemCube
    temp_dir = f"/tmp/{user_name}"
    os.makedirs(temp_dir, exist_ok=True)
    mem_cube.dump(temp_dir)
    mos.register_mem_cube(temp_dir, mem_cube_id=user_name)

    # Add the current document
    mos.add(doc_path=[doc_path])
    return mos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, default="evaluation/data/mmlongbench/samples.json"
    )
    parser.add_argument(
        "--document_path", type=str, default="evaluation/data/mmlongbench/documents"
    )
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--max_pages", type=int, default=120)
    parser.add_argument("--resolution", type=int, default=144)
    parser.add_argument("--max_try", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--extractor_prompt_path",
        type=str,
        default="evaluation/scripts/mmlongbench/eval/prompt_for_answer_extraction.md",
    )
    args = parser.parse_args()
    args.output_path = "evaluation/data/mmlongbench/res.json"

    with open(args.extractor_prompt_path) as f:
        prompt = f.read()

    # Load input samples
    with open(args.input_path) as f:
        samples = json.load(f)

    # Get list of document files to process
    doc_files = [
        f
        for f in os.listdir(args.document_path)
        if os.path.isfile(os.path.join(args.document_path, f))
    ]

    # Process each document separately
    res_samples = []
    for doc_file in tqdm(doc_files, desc="Processing documents"):
        print(f"Processing {doc_file}")
        doc_path = os.path.join(args.document_path, doc_file)

        # Process current document and create MOS
        mos = process_document(doc_path)

        # Filter samples for this document
        doc_id = os.path.basename(doc_path)
        doc_samples = [s for s in samples if s["doc_id"] == doc_id]

        # Process samples for this document
        for sample in doc_samples:
            if sample["evidence_sources"] != "['Pure-text (Plain-text)']":
                continue

            try_cnt = 0
            while try_cnt <= args.max_try:
                try:
                    mos.clear_messages()
                    response = mos.chat(sample["question"])
                    break
                except Exception as e:
                    try_cnt += 1
                    response = f"Error: {e!s}"
                    if try_cnt > args.max_try:
                        break

            sample["response"] = response
            # Extract and evaluate response
            if "Error:" not in response:
                extracted_res = extract_answer(sample["question"], response, prompt)
                sample["extracted_res"] = extracted_res
                try:
                    pred_ans = (
                        extracted_res.split("Answer format:")[0]
                        .split("Extracted answer:")[1]
                        .strip()
                    )
                    score = eval_score(sample["answer"], pred_ans, sample["answer_format"])
                except Exception:
                    pred_ans = ""
                    score = 0
                sample["pred"] = pred_ans
                sample["score"] = score
            else:
                sample["pred"] = ""
                sample["score"] = 0

            res_samples.append(sample)
            print("--------------------------------------")
            print("Question: {}".format(sample["question"]))
            print("Response: {}".format(sample["response"]))
            print(
                "Gt: {}\tPred: {}\tScore: {}".format(
                    sample["answer"], sample["pred"], sample["score"]
                )
            )

        acc, f1 = eval_acc_and_f1(res_samples)
        print(f"\nMetrics - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

        # Save progress after each sample
        with open(args.output_path, "w") as f:
            json.dump(res_samples, f)

    # Calculate and display final results
    acc, f1 = eval_acc_and_f1(res_samples)
    print(f"\nFinal Metrics - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    show_results(res_samples, show_path=re.sub("\.json$", ".txt", args.output_path))
