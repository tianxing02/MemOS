import base64
import json
import mimetypes
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import openai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from evaluation.scripts.utils.eval_score import (
    eval_acc_and_f1,
    eval_score,
    show_results,
)
from evaluation.scripts.utils.extract_answer import extract_answer
from evaluation.scripts.utils.prompts import MMLONGBENCH_ANSWER_PROMPT


load_dotenv()


def create_openai_client() -> openai.Client:
    return openai.Client(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )


def _encode_image_to_data_url(image_path: str) -> str | None:
    try:
        mime, _ = mimetypes.guess_type(image_path)
        mime = mime or "image/jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception as exc:
        print("Failed to encode image '%s' to data URL: %s", image_path, exc)
        return None


def build_images_index(base_dir: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    if not base_dir.exists():
        return index

    for images_dir in base_dir.rglob("auto/images"):
        if images_dir.is_dir():
            for img_file in images_dir.iterdir():
                if img_file.is_file():
                    index[img_file.name] = str(img_file.resolve())
    return index


def get_images(
    sources: list[Any],
    image_index: dict[str, str],
) -> list[str]:
    if not sources:
        return []

    found: list[str] = []

    md_img_pattern = re.compile(r"\[Image: images/\s*(.+?)\s*-")
    images_substr_pattern = re.compile(
        r"images/[^\s)]+\.(?:png|jpg|jpeg|webp)",
        re.IGNORECASE,
    )

    for src in sources:
        if not isinstance(src, str):
            continue

        for candidate in md_img_pattern.findall(src) + images_substr_pattern.findall(src):
            basename = os.path.basename(candidate)
            if basename in image_index:
                found.append(image_index[basename])

    # deduplicate
    seen: set[str] = set()
    return [p for p in found if not (p in seen or seen.add(p))]


def add_images_context(
    messages: list[dict[str, Any]],
    image_paths: list[str],
) -> list[dict[str, Any]]:
    if not image_paths:
        return messages

    user_idx = next(
        i for i in range(len(messages) - 1, -1, -1)
        if messages[i].get("role") == "user"
    )
    user_msg = messages[user_idx]
    content = user_msg.get("content", "")

    parts: list[dict[str, Any]]
    if isinstance(content, list):
        parts = content
    else:
        parts = [{"type": "text", "text": str(content)}]

    for img_path in image_paths[:6]:
        data_url = _encode_image_to_data_url(img_path)
        if data_url:
            parts.append(
                {"type": "image_url", "image_url": {"url": data_url}},
            )

    user_msg["content"] = parts
    messages[user_idx] = user_msg
    return messages


def multimodal_answer(
    client: openai.Client,
    chat_model: str,
    memories: list[str],
    question: str,
    sources: list[Any],
    image_index: dict[str, str],
) -> tuple[str, int | None]:
    image_paths = get_images(sources, image_index)

    system_prompt = MMLONGBENCH_ANSWER_PROMPT.format(
        memories="\n\n".join(memories),
        question=question,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    messages = add_images_context(messages, image_paths)

    resp = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        temperature=0,
    )
    return resp.choices[0].message.content or "", resp.usage.prompt_tokens


def process_single_item(
    client: openai.Client,
    chat_model: str,
    image_index: dict[str, str],
    item: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    try:
        response, prompt_tokens = multimodal_answer(
            client,
            chat_model,
            item["memories"],
            item["question"],
            item.get("sources", []),
            image_index,
        )

        extracted = extract_answer(item["question"], response)
        pred = extracted or response.strip()
        score = eval_score(item["answer"], pred, item.get("answer_format", "Str"))

        return {
            "index": index,
            "result": {
                "response": response,
                "pred": pred,
                "score": score,
                "prompt_tokens": prompt_tokens,
                "eval_success": True,
                "eval_error": None,
            },
        }

    except Exception as exc:
        traceback.print_exc()
        return {
            "index": index,
            "result": {
                "response": None,
                "pred": None,
                "score": 0,
                "eval_success": False,
                "eval_error": str(exc),
            },
        }


def run_eval(
    questions_file: Path,
    output_file: Path,
    version_dir: Path,
    chat_model: str,
    max_workers: int,
    limit: int | None,
) -> None:
    client = create_openai_client()
    image_index = build_images_index(
        Path("/Users/tianxingshi/Desktop/lcy/ppt_test_result"),
    )

    data = json.loads(questions_file.read_text(encoding="utf-8"))
    items = data["results"][:limit] if limit else data["results"]

    results: dict[int, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_item,
                client,
                chat_model,
                image_index,
                item,
                i,
            ): i
            for i, item in enumerate(items)
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            results[res["index"]] = res["result"]

    for i, item in enumerate(items):
        if i in results:
            item.update(results[i])

    acc, f1 = eval_acc_and_f1(items)
    data["eval_summary"] = {
        "accuracy": acc,
        "f1_score": f1,
        "eval_timestamp": datetime.now().isoformat(),
    }

    output_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_path = version_dir / "eval_results.txt"
    show_results(items, show_path=str(report_path))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser("MMLongBench Eval")
    parser.add_argument("--lib", required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--version-dir", required=True)
    parser.add_argument("--chat-model", required=True)
    parser.add_argument("--limit", type=int)

    args = parser.parse_args()

    base_dir = Path("evaluation/data/mmlongbench")
    version_dir = base_dir / args.version_dir
    input_path = version_dir / f"{args.lib}_search_results.json"

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)

    run_eval(
        questions_file=input_path,
        output_file=input_path,
        version_dir=version_dir,
        chat_model=args.chat_model,
        max_workers=args.workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
