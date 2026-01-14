#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from evaluation.scripts.utils.metrics import Metrics

load_dotenv()

FASTGPT_DATASET_ID = os.getenv("FASTGPT_DATASET_ID_MM_LONGBENCH")
MEMOS_KNOWLEDGEBASE_ID = os.getenv("MEMOS_KNOWLEDGEBASE_ID_MM_LONGBENCH")


def retry_operation(
    func,
    *args,
    retries: int = 5,
    delay: int = 2,
    **kwargs,
):
    """Retry wrapper with exponential backoff."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if attempt >= retries - 1:
                raise
            traceback.print_exc()
            func_name = getattr(func, "__name__", "operation")
            print(f"[Retry] {func_name} failed: {exc}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2
    return None


def read_filenames(filepath: str | Path) -> list[str]:
    """Read filenames from text file, one per line."""
    path = Path(filepath)
    filenames: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                filenames.append(name)
    return filenames


def _get_lib_client(lib: str):
    if lib == "memos":
        from evaluation.scripts.utils.client import MemosApiClient

        return MemosApiClient()
    if lib == "mem0":
        from evaluation.scripts.utils.client import Mem0Client

        return Mem0Client(enable_graph=False)
    if lib == "supermemory":
        from evaluation.scripts.utils.client import SupermemoryClient

        return SupermemoryClient()
    if lib == "fastgpt":
        from evaluation.scripts.utils.client import FastGPTClient

        return FastGPTClient()
    if lib == "memos-online":
        from evaluation.scripts.utils.client import MemosApiOnlineClient

        return MemosApiOnlineClient()

    msg = f"Unknown lib type: {lib}"
    raise ValueError(msg)


def run_concurrent_add(
    *,
    lib: str,
    filenames: list[str],
    url_prefix: str,
    user_prefix: str,
    workers: int,
    mode: str = "fine",
    async_mode: str = "sync",
) -> dict[str, Any]:
    """Run concurrent ingestion."""

    client = _get_lib_client(lib)
    metrics = Metrics()

    total_files = len(filenames)
    completed = 0
    completed_lock = threading.Lock()

    added_ids: dict[str, str] = {}

    base_dir = Path("ppt_test_result")
    all_md_files = list(base_dir.rglob("*.md"))

    def add_single_file(filename: str, doc_id: str) -> tuple[bool, Any]:
        nonlocal completed

        file_url = f"{url_prefix.rstrip('/')}/{filename}"
        stem = Path(filename).stem.lower()
        name = filename.lower()

        md_path: Path | None = None
        for md_file in all_md_files:
            pstr = str(md_file).lower()
            if (stem and stem in pstr) or (name and name in pstr):
                md_path = md_file
                break

        if md_path is None:
            raise FileNotFoundError(f"No markdown found for {filename}")

        text = md_path.read_text(encoding="utf-8", errors="ignore")
        start_time = time.perf_counter()
        user_id = f"{user_prefix}_{doc_id}"

        try:
            result = None

            if lib == "memos-online":
                result = retry_operation(
                    client.upload_file,
                    MEMOS_KNOWLEDGEBASE_ID,
                    file_url,
                )
                file_id = None
                if isinstance(result, dict):
                    data = result.get("data") or []
                    if isinstance(data, list) and data:
                        file_id = data[0].get("id")
                if file_id:
                    added_ids[filename] = str(file_id)

            elif lib == "fastgpt":
                result = retry_operation(
                    client.upload_file,
                    datasetId=FASTGPT_DATASET_ID,
                    file_url=file_url,
                )

            elif lib == "supermemory":
                result = client.add(content=text, user_id=user_id)

            elif lib == "mem0":
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.PYTHON,
                    chunk_size=5120,
                    chunk_overlap=128,
                )
                chunks = [c for c in splitter.split_text(text) if c.strip()]
                messages = [{"role": "user", "content": c} for c in chunks]
                result = client.add(
                    messages=messages,
                    user_id=doc_id,
                    timestamp=int(time.time()),
                    batch_size=10,
                )

            duration = time.perf_counter() - start_time
            metrics.record(duration, True)

            with completed_lock:
                completed += 1
                print(
                    f"[{completed}/{total_files}] ✓ Success: {filename} "
                    f"({duration * 1000:.0f}ms)"
                )

            return True, result

        except Exception as exc:
            duration = time.perf_counter() - start_time
            metrics.record(duration, False, str(exc))

            with completed_lock:
                completed += 1
                print(
                    f"[{completed}/{total_files}] ✗ Failed: {filename} - {exc!s:.100}"
                )

            return False, str(exc)

    print(f"\nStarting concurrent add for {total_files} files...")
    print(f"Concurrency: {workers}")
    print(f"Version Dir: {user_prefix}")
    print(f"URL prefix: {url_prefix}")
    print("-" * 60)

    start_time = time.time()

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(add_single_file, fn, fn[:-3] + ".pdf")
            for fn in filenames
        ]

        for filename, future in zip(filenames, futures, strict=True):
            try:
                success, result = future.result()
                results.append(
                    {"filename": filename, "success": success, "result": result}
                )
            except Exception as exc:
                results.append(
                    {"filename": filename, "success": False, "result": str(exc)}
                )

    total_duration = time.time() - start_time
    summary = metrics.summary()

    print("\n" + "=" * 60)
    print("Ingestion finished! Statistics:")
    print("=" * 60)
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Success: {summary['counts']['success']} / Failed: {summary['counts']['failed']}")

    return {
        "summary": summary,
        "total_duration": total_duration,
        "results": results,
        "added": added_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MMLongbench-doc Product Add Concurrent Script",
    )
    parser.add_argument("--lib", "-b", required=True)
    parser.add_argument(
        "--filenames-file",
        "-f",
        default="evaluation/data/mmlongbench/pdf_file_list.txt",
    )
    parser.add_argument(
        "--url-prefix",
        "-u",
        default="https://memos-knowledge-base-file-pre.oss-cn-shanghai.aliyuncs.com/mmlongbench_pdf_files/",
    )
    parser.add_argument("--workers", "-w", type=int, default=5)
    parser.add_argument("--limit", "-l", type=int)
    parser.add_argument("--version-dir", "-v", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    filenames = read_filenames(args.filenames_file)
    if args.limit:
        filenames = filenames[: args.limit]

    version_dir = Path("evaluation/data/mmlongbench") / args.version_dir
    version_dir.mkdir(parents=True, exist_ok=True)

    output_path = version_dir / f"{args.lib}_add_results.json"

    existing_added: dict[str, str] = {}
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            obj = json.load(f)
            existing_added = obj.get("added", {}) if isinstance(obj, dict) else {}

    filenames = [f for f in filenames if f not in existing_added]

    if not filenames:
        print("[Add] no pending files.")
        return

    result = run_concurrent_add(
        lib=args.lib,
        filenames=filenames,
        url_prefix=args.url_prefix,
        user_prefix=args.version_dir,
        workers=args.workers,
    )

    output = {
        "summary": result["summary"],
        "total_duration": result["total_duration"],
        "added": {**existing_added, **result.get("added", {})},
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
