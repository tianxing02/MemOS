import json
import os
import sys
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.append(str(SCRIPTS_ROOT))
sys.path.append(str(SRC_ROOT))
load_dotenv()


def _get_clients():
    from utils.client import MemosApiClient

    memos_client = MemosApiClient()
    openai_client = OpenAI(
        api_key=os.getenv("CHAT_MODEL_API_KEY"), base_url=os.getenv("CHAT_MODEL_BASE_URL")
    )
    return memos_client, openai_client


def _load_dataset(split: str):
    from datasets import load_dataset

    return load_dataset("zai-org/LongBench-v2", split=split)


def _dump_dataset_to_local(dataset, split: str):
    out_dir = Path("evaluation/data/longbenchV2")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"longbenchv2_{split}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            s = dataset[i]
            row = {
                "_id": s.get("_id") or s.get("id") or str(i),
                "domain": s.get("domain"),
                "sub_domain": s.get("sub_domain"),
                "difficulty": s.get("difficulty"),
                "length": s.get("length"),
                "question": s.get("question"),
                "choice_A": s.get("choice_A"),
                "choice_B": s.get("choice_B"),
                "choice_C": s.get("choice_C"),
                "choice_D": s.get("choice_D"),
                "answer": s.get("answer"),
                "context": s.get("context") or s.get("document") or s.get("documents"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved dataset to {out_path}")


def _ingest_context(client, user_id: str, context: str) -> None:
    iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    messages = [{"role": "user", "content": context or "", "created_at": iso}]
    client.add(messages=messages, user_id=user_id, conv_id=user_id)


def memos_search(client, user_id: str, query: str, top_k: int = 10) -> list[str]:
    results = client.search(query=query, user_id=user_id, top_k=top_k)
    memories = results["text_mem"][0]["memories"]
    mem_texts = [m["memory"] for m in memories]
    print(f"[Search] user={user_id} top_k={top_k} memories={len(memories)}")
    return mem_texts


def llm_answer(oai_client, memories: list[str], question: str, choices: dict) -> str:
    from memos.mem_os.core import MOSCore

    system_prompt = MOSCore._build_system_prompt(MOSCore.__new__(MOSCore), memories)
    choice_text = f"A. {choices.get('A', '')}\nB. {choices.get('B', '')}\nC. {choices.get('C', '')}\nD. {choices.get('D', '')}\n"
    user_content = (
        "You are answering a multiple-choice question. Select the best option (A/B/C/D).\n\n"
        f"Question: {question}\n\nChoices:\n{choice_text}\nAnswer with only the option letter."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    resp = oai_client.chat.completions.create(
        model=os.getenv("CHAT_MODEL"), messages=messages, temperature=0
    )
    return resp.choices[0].message.content or ""


def process_sample(client, oai_client, sample, top_k: int) -> dict:
    sample_id = str(
        sample.get("_id")
        or sample.get("id")
        or sample.get("qid")
        or sample.get("name")
        or sample.get("task")
        or "sample"
    )
    user_id = sample_id
    context = sample.get("context") or ""
    _ingest_context(client, user_id, str(context))
    question = sample.get("question") or ""
    choices = {
        "A": sample.get("choice_A") or "",
        "B": sample.get("choice_B") or "",
        "C": sample.get("choice_C") or "",
        "D": sample.get("choice_D") or "",
    }
    memories = memos_search(client, user_id, str(question), top_k=top_k)
    response = llm_answer(oai_client, memories, str(question), choices)
    out = {
        "_id": sample_id,
        "domain": sample.get("domain"),
        "sub_domain": sample.get("sub_domain"),
        "difficulty": sample.get("difficulty"),
        "length": sample.get("length"),
        "question": question,
        "choice_A": choices["A"],
        "choice_B": choices["B"],
        "choice_C": choices["C"],
        "choice_D": choices["D"],
        "answer": sample.get("answer"),
        "context": context,
        "memories_used": memories,
        "response": response,
    }
    return out


def main():
    client, oai_client = _get_clients()
    dataset = _load_dataset("train")
    _dump_dataset_to_local(dataset, "train")
    results: list[dict] = []
    os.makedirs("evaluation/data/longbenchV2", exist_ok=True)
    out_json = Path("evaluation/data/longbenchV2/memos_results.json")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_sample, client, oai_client, dataset[i], 15)
            for i in range(len(dataset))
        ]
        for f in as_completed(futures):
            try:
                res = f.result()
                results.append(res)
                out_json.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as e:
                print("Error: ", e)

    print(f"Saved {len(results)} results to {out_json}")


if __name__ == "__main__":
    main()
