import argparse
import json
import os
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prompts import ANSWER_PROMPT


def lme_response(llm_client, context, question, question_date):
    prompt = ANSWER_PROMPT.format(
        question=question,
        question_date=question_date,
        context=context,
    )

    response = llm_client.chat.completions.create(
        model=os.getenv("CHAT_MODEL"),
        messages=[
            {"role": "system", "content": prompt},
        ],
        temperature=0,
    )
    result = response.choices[0].message.content or ""

    return result


def process_qa(user_id, search_result, llm_client):
    start = time()
    search_result = search_result[0]
    question = search_result.get("question")
    question_date = search_result.get("date")
    context = search_result.get("search_context", "")
    anwer = lme_response(llm_client, context, question, question_date)

    response_duration_ms = (time() - start) * 1000

    print("\n" + "-" * 80)
    print(f"🤖 Processed User: \033[1m{user_id}\033[0m")
    print(f"⏱️  Duration: \033[92m{response_duration_ms:.2f} ms\033[0m")
    print(f"❓ Question: \033[93m{question}\033[0m")
    print(
        f"💬 Answer: \033[96m{anwer[:150]}...\033[0m"
        if len(anwer) > 150
        else f"💬 Answer: \033[96m{anwer}\033[0m"
    )
    print("-" * 80)

    return {
        "user_id": user_id,
        "category": search_result.get("category"),
        "question": question,
        "answer": anwer,
        "question_date": question_date,
        "golden_answer": search_result.get("golden_answer"),
        "response_duration_ms": response_duration_ms,
        "search_context": context,
        "search_duration_ms": search_result.get("search_duration_ms"),
        "answer_evidences": search_result.get("answer_evidences", []),
    }


def main(frame, version, num_workers=4):
    print("\n" + "=" * 80)
    print(
        f"🚀 \033[1;36mLONGMEMEVAL RESPONSE GENERATION - {frame.upper()} v{version}\033[0m".center(
            80
        )
    )
    print("=" * 80)

    load_dotenv()

    oai_client = OpenAI(
        api_key=os.getenv("CHAT_MODEL_API_KEY"), base_url=os.getenv("CHAT_MODEL_BASE_URL")
    )

    print(
        f"🔌 \033[1mUsing OpenAI client with model:\033[0m \033[94m{os.getenv('CHAT_MODEL')}\033[0m"
    )

    search_path = f"results/lme/{frame}-{version}/{frame}_lme_search_results.json"
    response_path = f"results/lme/{frame}-{version}/{frame}_lme_responses.json"

    print(f"📂 \033[1mLoading search results from:\033[0m \033[94m{search_path}\033[0m")
    with open(search_path) as file:
        lme_search_results = json.load(file)
    print(f"📊 \033[1mFound\033[0m \033[93m{len(lme_search_results)}\033[0m users to process")
    print(f"⚙️  \033[1mUsing\033[0m \033[93m{num_workers}\033[0m worker threads")
    print("-" * 80)

    lme_responses = {}
    start_time = time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_user_id = {}

        for user_id, search_results in lme_search_results.items():
            future = executor.submit(process_qa, user_id, search_results, oai_client)
            future_to_user_id[future] = user_id

        for future in tqdm(
            as_completed(future_to_user_id),
            total=len(future_to_user_id),
            desc="📝 Generating responses",
        ):
            user_id = future_to_user_id[future]
            try:
                result = future.result()
                lme_responses[user_id] = result
            except Exception as exc:
                print(f"\033[91m❌ Error processing user {user_id}: {exc}\033[0m")

    end_time = time()
    elapsed_time = end_time - start_time
    elapsed_sec = int(elapsed_time)

    print("\n" + "=" * 80)
    print("✅ \033[1;32mRESPONSE GENERATION COMPLETE\033[0m".center(80))
    print("=" * 80)
    print(f"⏱️  \033[1mTotal time:\033[0m \033[92m{elapsed_sec // 60}m {elapsed_sec % 60}s\033[0m")
    print(f"📊 \033[1mProcessed:\033[0m \033[93m{len(lme_responses)}\033[0m users")
    print(
        f"🔄 \033[1mFramework:\033[0m \033[94m{frame}\033[0m | \033[1mVersion:\033[0m \033[94m{version}\033[0m"
    )

    with open(response_path, "w") as f:
        json.dump(lme_responses, f, indent=4)

    print(f"📁 \033[1mResponses saved to:\033[0m \033[1;94m{response_path}\033[0m")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongMemeval Response Generation Script")
    parser.add_argument(
        "--lib",
        type=str,
        choices=["mem0-local", "mem0-api", "memos-local", "memos-api", "zep", "memobase"],
    )
    parser.add_argument(
        "--version", type=str, default="v1", help="Version of the evaluation framework."
    )
    parser.add_argument(
        "--workers", type=int, default=3, help="Number of runs for LLM-as-a-Judge evaluation."
    )

    args = parser.parse_args()
    main(frame=args.lib, version=args.version, num_workers=args.workers)
