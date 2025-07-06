import concurrent.futures
import json
import os
import shutil
import sys

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import time
from uuid import uuid4

import pandas as pd

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from memos.configs.mem_cube import GeneralMemCubeConfig
from memos.configs.mem_os import MOSConfig
from memos.configs.mem_scheduler import SchedulerConfigFactory
from memos.configs.memory import MemoryConfigFactory
from memos.log import get_logger
from memos.mem_cube.general import GeneralMemCube
from memos.mem_os.main import MOS
from memos.mem_scheduler.scheduler_factory import SchedulerFactory
from memos.mem_scheduler.utils import parse_yaml
from memos.memories.factory import MemoryFactory


logger = get_logger(__name__)

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # Enable execution from any working directory


custom_instructions = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""

ANSWER_PROMPT_MEMOS = """
    You are a knowledgeable and helpful AI assistant.
    You have access to conversation memories that help you provide more personalized responses.
    Use the memories to understand the user's context, preferences, and past interactions.
    If memories are provided, reference them naturally when relevant, but don't explicitly mention having memories.


    ## Memories:

    {context}

    Question: {question}
    Answer:
    """


def get_client(frame: str, user_id: str | None = None, version: str = "default"):
    config = MemoryConfigFactory(
        backend="general_text",
        config={
            "extractor_llm": {
                "backend": "openai",
                "config": {
                    "model_name_or_path": os.getenv("MODEL"),
                    "temperature": 0,
                    "max_tokens": 8192,
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "api_base": os.getenv("OPENAI_BASE_URL"),
                },
            },
            "vector_db": {
                "backend": "qdrant",
                "config": {
                    "path": f"results/locomo/memos-{version}/storages/{user_id}/qdrant",
                    "collection_name": "test_textual_memory",
                    "distance_metric": "cosine",
                    "vector_dimension": 768,  # nomic-embed-text model's embedding dimension is 768
                },
            },
            "embedder": {
                "backend": "ollama",
                "config": {
                    "model_name_or_path": os.getenv("EMBEDDING_MODEL"),
                },
            },
        },
    )
    m = MemoryFactory.from_config(config)
    return m


def get_mem_cube(user_id: str | None = None, model_name: str | None = None):
    config = MemoryConfigFactory(
        backend="general_text",
        config={
            "extractor_llm": {
                "backend": "openai",
                "config": {
                    "model_name_or_path": os.getenv("MODEL"),
                    "temperature": 0,
                    "max_tokens": 8192,
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "api_base": os.getenv("OPENAI_BASE_URL"),
                },
            },
            "vector_db": {
                "backend": "qdrant",
                "config": {
                    "path": f"{BASE_DIR}/outputs/evaluation/locomo/{model_name}/memos/storages/{user_id}/qdrant",
                    "collection_name": "test_textual_memory",
                    "distance_metric": "cosine",
                    "vector_dimension": 768,  # nomic-embed-text model's embedding dimension is 768
                },
            },
            "embedder": {
                "backend": "ollama",
                "config": {
                    "model_name_or_path": os.getenv("EMBEDDING_MODEL"),
                },
            },
        },
    )
    m = MemoryFactory.from_config(config)
    return m


def ingest_session(client, session, metadata):
    session_date = metadata["session_date"]
    date_format = "%I:%M %p on %d %B, %Y UTC"
    date_string = datetime.strptime(session_date, date_format).replace(tzinfo=timezone.utc)
    iso_date = date_string.isoformat()
    conv_idx = metadata["conv_idx"]
    conv_id = "locomo_exp_user_" + str(conv_idx)

    for chat in session:
        blip_caption = chat.get("blip_captions")
        img_description = (
            f"(description of attached image: {blip_caption})" if blip_caption is not None else ""
        )
        data = chat.get("speaker") + ": " + chat.get("text") + img_description
        logger.info({"context": data, "conv_id": conv_id, "created_at": iso_date})
        msg = [{"role": "user", "content": data}]

        try:
            memories = client.extract(msg)
        except Exception as ex:
            logger.error(f"Error extracting message {msg}: {ex}")
            memories = []

        client.add(memories)


def search_query(client, query, metadata):
    start = time()
    search_results = client.search(query, top_k=20)
    context = ""
    for item in search_results:
        item = item.to_dict()
        context += f"{item['memory']}\n"
    print(query, context)
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def process_qa(qa, search_result, llm_client):
    start = time()
    query = qa.get("question")
    gold_answer = qa.get("answer")
    qa_category = qa.get("category")

    prompt = ANSWER_PROMPT_MEMOS.format(
        context=search_result.get("context"),
        question=query,
    )
    response = llm_client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=[
            {"role": "system", "content": prompt},
        ],
        temperature=0,
    )
    answer = response.choices[0].message.content or ""

    response_duration_ms = (time() - start) * 1000

    print(f"Processed question: {query}")
    print(f"Answer: {answer}")
    return {
        "question": query,
        "answer": answer,
        "category": qa_category,
        "golden_answer": gold_answer,
        "search_context": search_result.get("context", ""),
        "response_duration_ms": response_duration_ms,
        "search_duration_ms": search_result.get("duration_ms", 0),
    }


def calculate_f1_score(gold_tokens, response_tokens):
    try:
        gold_set = set(gold_tokens)
        response_set = set(response_tokens)

        if len(gold_set) == 0 or len(response_set) == 0:
            return 0.0

        precision = len(gold_set.intersection(response_set)) / len(response_set)
        recall = len(gold_set.intersection(response_set)) / len(gold_set)

        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0
    except Exception as e:
        print(f"Failed to calculate F1 score: {e}")
        return 0.0


class LLMGrade(BaseModel):
    llm_judgment: str = Field(description="CORRECT or WRONG")
    llm_reasoning: str = Field(description="Explain why the answer is correct or incorrect.")


def locomo_grader(llm_client, question: str, gold_answer: str, response: str) -> bool:
    system_prompt = """
        You are an expert grader that determines if answers to questions match a gold standard answer
        """

    accuracy_prompt = f"""
    Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You williolw23 be given the following data:
        (1) a question (posed by one user to another user),
        (2) a ’gold’ (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it’s time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """

    response = llm_client.beta.chat.completions.parse(
        model=os.getenv("MODEL"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": accuracy_prompt},
        ],
        response_format=LLMGrade,
        temperature=0,
    )
    result = response.choices[0].message.parsed

    return result.llm_judgment.strip().lower() == "correct"


# the entry function
def process_user(conv_idx, locomo_df, model, model_name):
    try:
        # ============= 1. generate memories =================
        logger.info("============= 1. generate memories =================")
        conversation = locomo_df["conversation"].iloc[conv_idx]
        max_session_count = 35

        conv_id = "locomo_exp_user_" + str(conv_idx)
        client = get_client("memos")

        for session_idx in range(max_session_count):
            session_key = f"session_{session_idx}"
            session = conversation.get(session_key)
            if session is None:
                continue
            print(f"User {conv_idx}, {session_key}")
            metadata = {
                "session_date": conversation.get(f"session_{session_idx}_date_time") + " UTC",
                "speaker_a": conversation.get("speaker_a"),
                "speaker_b": conversation.get("speaker_b"),
                "speaker_a_user_id": f"{conversation.get('speaker_a')}_{conv_idx}",
                "speaker_b_user_id": f"{conversation.get('speaker_b')}_{conv_idx}",
                "conv_idx": conv_idx,
                "session_key": session_key,
            }
            ingest_session(client, session, metadata)

        conv_id = "locomo_exp_user_" + str(conv_idx)
        client.dump(f"{BASE_DIR}/outputs/evaluation/locomo/{model_name}/memos/storages/{conv_id}")

        logger.info(f"Completed processing user {conv_idx}")

        # ============= 2. search memories =================
        logger.info("============= 2. search memories =================")
        search_results = defaultdict(list)
        qa_set = locomo_df["qa"].iloc[conv_idx]

        metadata = {
            "speaker_a": conversation.get("speaker_a"),
            "speaker_b": conversation.get("speaker_b"),
            "speaker_a_user_id": f"{conversation.get('speaker_a')}_{conv_idx}",
            "speaker_b_user_id": f"{conversation.get('speaker_b')}_{conv_idx}",
            "conv_idx": conv_idx,
            "conv_id": conv_id,
        }
        qa_filtered_set = []
        for qa in qa_set:
            query = qa.get("question")
            if qa.get("category") == 5:
                continue
            qa_filtered_set.append(qa)
            context, duration_ms = search_query(client, query, metadata)
            search_results[conv_id].append({"context": context, "duration_ms": duration_ms})
            logger.info({"context": context[:20] + "...", "duration_ms": duration_ms})

        search_path = Path(
            f"{BASE_DIR}/outputs/evaluation/locomo/{model_name}/locomo_search_results.json"
        )
        search_path.parent.mkdir(exist_ok=True, parents=True)
        with search_path.open("w", encoding="utf-8") as fw:
            json.dump(dict(search_results), fw, indent=2)
            logger.info(f"Save search results {conv_idx}")

    except Exception as e:
        return f"Error processing user {conv_idx}: {e!s}"


def main():
    # Load environment variables
    load_dotenv()

    # Load JSON data
    locomo_df = pd.read_json(f"{BASE_DIR}/evaluation/data/locomo/locomo10.json")

    # Process each user in parallel
    num_users = 10

    max_workers = min(num_users, os.cpu_count() * 2)

    # 1. Create Mos Config
    config = parse_yaml(f"{BASE_DIR}/examples/data/config/mem_scheduler/memos_config.yaml")

    mos_config = MOSConfig(**config)
    mos = MOS(mos_config)

    # 2. Initialization
    user_id = f"user_{uuid4!s}"
    mos.create_user(user_id)

    config = GeneralMemCubeConfig.from_yaml_file(
        f"{BASE_DIR}/examples/data/config/mem_scheduler/mem_cube_config.yaml"
    )
    mem_cube_id = "mem_cube_5"
    mem_cube_name_or_path = f"{BASE_DIR}/outputs/mem_scheduler/{user_id}/{mem_cube_id}"
    if Path(mem_cube_name_or_path).exists():
        shutil.rmtree(mem_cube_name_or_path)
        print(f"{mem_cube_name_or_path} is not empty, and has been removed.")
    mem_cube = GeneralMemCube(config)
    mem_cube.dump(mem_cube_name_or_path)
    mos.register_mem_cube(
        mem_cube_name_or_path=mem_cube_name_or_path, mem_cube_id=mem_cube_id, user_id=user_id
    )

    # 3. set mem_scheduler
    example_scheduler_config_path = (
        f"{BASE_DIR}/examples/data/config/mem_scheduler/general_scheduler_config.yaml"
    )
    scheduler_config = SchedulerConfigFactory.from_yaml_file(
        yaml_path=example_scheduler_config_path
    )
    mem_scheduler = SchedulerFactory.from_config(scheduler_config)
    mem_scheduler.initialize_modules(chat_llm=mos.chat_llm)
    mos.mem_scheduler = mem_scheduler

    mos.mem_scheduler.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_user, i, locomo_df, mos, "memos"): i for i in range(num_users)
        }
        for future in concurrent.futures.as_completed(futures):
            user_id = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"Error processing user {user_id}: {e!s}")


if __name__ == "__main__":
    os.environ["MODEL"] = "gpt-4o-mini"
    # TODO: This code is not finished yet.
    main()
