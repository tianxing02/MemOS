#!/usr/bin/env python3

import argparse
import json
import os
import threading
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from evaluation.scripts.utils.metrics import Metrics


load_dotenv()

fastgpt_dataset_id = os.getenv("FASTGPT_DATASET_ID_MM_LONGBENCH")
memos_knowledgebase_id = os.getenv("MEMOS_KNOWLEDGEBASE_ID_MM_LONGBENCH")


def retry_operation(func, *args, retries=5, delay=2, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                func_name = getattr(func, "__name__", "Operation")
                print(f"[Retry] {func_name} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise e


def load_samples(filepath: str) -> list[dict]:
    """
    Read sample list from JSON file
    """
    with open(filepath, encoding="utf-8") as f:
        samples = json.load(f)
    return samples


def memos_search(client, user_id: str, query: str, top_k: int, mode: str) -> list[str]:
    results = retry_operation(
        client.search,
        query=query,
        user_id=user_id,
        top_k=top_k,
        mode=mode,
        knowledgebase_ids=[memos_knowledgebase_id],
    )
    if "memory_detail_list" in results["data"] and results["data"]["memory_detail_list"]:
        memories = results["data"]["memory_detail_list"]
        return [m.get("memory_value", "") for m in memories]
    return []


def mem0_search(client, user_id: str, query: str, top_k: int) -> tuple[list[str], list[str]]:
    res = retry_operation(client.search, query, user_id, top_k)
    results = res.get("results", [])
    mem_texts = [m.get("memory", "") for m in results if m.get("memory")]
    return mem_texts, mem_texts


def supermemory_search(client, user_id: str, query: str, top_k: int) -> tuple[list[str], list[str]]:
    chunk_list = retry_operation(client.search, query, user_id, top_k)
    return chunk_list, chunk_list


def fastgpt_search(client, query: str, top_k: int) -> list[str]:
    result = retry_operation(client.search, datasetId=fastgpt_dataset_id, query=query, top_k=top_k)
    return [item["q"] for item in result[:top_k]]


def _load_existing_results(path: str | os.PathLike[str]) -> tuple[list[dict], set[str]]:
    p = Path(path)
    if not p.exists():
        return [], set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        rows: list[dict] = []
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            rows = data.get("results") or []
        elif isinstance(data, list):
            rows = data
        success_rows = [r for r in rows if r.get("success") is True]
        ids = {str(r.get("doc_id")) for r in success_rows if r.get("doc_id")}
        return success_rows, ids
    except Exception:
        return [], set()


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


def run_concurrent_search(
    lib: str, samples: list[dict], user_prefix: str, concurrency: int, top_k: int, mode: str
) -> dict:
    """
    Execute concurrent search operations

    Args:
        lib: Client name
        samples: Sample list, each containing doc_id and question
        user_prefix: User ID prefix
        concurrency: Concurrency
        top_k: Number of results to return
        mode: Query mode ['fast', 'fine']

    Returns:
        Search results
    """

    client = _get_lib_client(lib)
    metrics = Metrics()
    total_samples = len(samples)
    completed = 0
    completed_lock = threading.Lock()

    # 用于存储所有搜索结果
    all_results = []
    results_lock = threading.Lock()

    user_id = user_prefix

    def search_single(sample: dict, index: int):
        nonlocal completed

        doc_id = sample.get("doc_id", "")
        question = sample.get("question", "")

        # user_id = user_prefix + "_" + doc_id
        user_id = doc_id[:20]
        start_time = time.perf_counter()
        try:
            memories, sources = [], []
            if lib == "memos" or lib == "memos-online":
                memories = memos_search(
                    client=client,
                    query=question,
                    user_id=user_id,
                    top_k=top_k,
                    mode=mode,
                )
            elif lib == "mem0":
                memories, sources = mem0_search(client, user_id, question, top_k=top_k)
            elif lib == "supermemory":
                memories, sources = supermemory_search(client, user_id, question, top_k=top_k)
            elif lib == "fastgpt":
                memories = fastgpt_search(client, question, top_k=top_k)

            duration = time.perf_counter() - start_time
            metrics.record(duration, True)

            result = {
                "index": index,
                "doc_id": doc_id,
                "question": question,
                "answer": sample.get("answer", ""),
                "evidence_pages": sample.get("evidence_pages", ""),
                "evidence_sources": sample.get("evidence_sources", ""),
                "answer_format": sample.get("answer_format", ""),
                "doc_type": sample.get("doc_type", ""),
                "memories": memories,
                "memory_count": len(memories),
                "success": True,
                "duration_ms": duration * 1000,
                "mode": mode,
            }

            with results_lock:
                all_results.append(result)

            with completed_lock:
                completed += 1
                print(
                    f"[{completed}/{total_samples}] ✓ Success: {doc_id[:30]}... ({duration * 1000:.0f}ms, {len(memories)} memories)"
                )

            return True, result

        except Exception as e:
            duration = time.perf_counter() - start_time
            error_msg = str(e)
            metrics.record(duration, False, error_msg)

            result = {
                "index": index,
                "doc_id": doc_id,
                "question": question,
                "answer": sample.get("answer", ""),
                "evidence_pages": sample.get("evidence_pages", ""),
                "evidence_sources": sample.get("evidence_sources", ""),
                "answer_format": sample.get("answer_format", ""),
                "doc_type": sample.get("doc_type", ""),
                "memories": [],
                "memory_count": 0,
                "success": False,
                "error": error_msg,
                "duration_ms": duration * 1000,
            }

            with results_lock:
                all_results.append(result)

            with completed_lock:
                completed += 1
                print(
                    f"[{completed}/{total_samples}] ✗ Failed: {doc_id[:30]}... - {error_msg[:80]}"
                )

            return False, result

    print(f"\nStarting concurrent search for {total_samples} questions...")
    print(f"Concurrency: {concurrency}")
    print(f"User ID: {user_id}")
    print(f"Top-K: {top_k}")
    print("-" * 60)

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i, sample in enumerate(samples):
            future = executor.submit(search_single, sample, i)
            futures.append(future)

        # Wait for all tasks to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Task execution exception: {e}")

    end_time = time.time()
    total_duration = end_time - start_time

    # Sort results by original index
    all_results.sort(key=lambda x: x["index"])

    # Print statistics
    summary = metrics.summary()

    print("\n" + "=" * 60)
    print("Search finished! Statistics:")
    print("=" * 60)
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Success: {summary['counts']['success']} / Failed: {summary['counts']['failed']}")

    if summary["stats"]:
        stats = summary["stats"]
        qps = stats["count"] / total_duration if total_duration > 0 else 0
        print(f"QPS: {qps:.2f}")
        print("Latency stats (ms):")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Min: {stats['min']:.2f}")
        print(f"  Max: {stats['max']:.2f}")
        print(f"  P95: {stats['p95']:.2f}")
        print(f"  P99: {stats['p99']:.2f}")

    if summary["errors"]:
        print("\nError statistics:")
        for error, count in sorted(summary["errors"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  [{count} times] {error[:100]}...")

    return {"summary": summary, "total_duration": total_duration, "results": all_results}


def parse_args():
    parser = argparse.ArgumentParser(
        description="MMLongbench-doc Product Search Concurrent Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--lib", "-b", required=True, help="Product name to evaluate")

    parser.add_argument(
        "--samples-file",
        "-s",
        default="evaluation/data/mmlongbench/samples.json",
        help="Path to JSON file containing samples",
    )

    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8001",
        help="API service address (default: http://127.0.0.1:8001)",
    )

    parser.add_argument("--api-key", default="", help="API key (optional)")

    parser.add_argument("--workers", "-c", type=int, default=5, help="Concurrency (default: 5)")

    parser.add_argument(
        "--timeout", type=float, default=120.0, help="Request timeout in seconds (default: 120)"
    )

    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=15,
        help="Number of results to return per search (default: 20)",
    )

    parser.add_argument("--version-dir", "-v", default=None, help="Version directory name")

    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing, default all)",
    )

    parser.add_argument(
        "--mode", "-m", type=str, default="fast", help="Search mode (default: fast)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("MMLongbench-doc Product Search Concurrent Tool")
    print("=" * 60)

    # Read sample data
    samples_path = "evaluation/data/mmlongbench/samples.json"
    print(f"\nReading sample file: {samples_path}")
    try:
        samples = load_samples(samples_path)
        print(f"Total {len(samples)} samples read")

        # Limit number of samples
        if args.limit and args.limit > 0:
            samples = samples[: args.limit]
            print(f"Limiting to first {len(samples)} samples")

        if len(samples) == 0:
            print("Error: Sample list is empty!")
            return

    except FileNotFoundError:
        print(f"Error: File not found {args.samples_file}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: JSON parse failed - {e}")
        return
    except Exception as e:
        print(f"Error: Failed to read file - {e}")
        return

    # Determine output file path
    import os

    output_dir = os.path.join("evaluation/data/mmlongbench", args.version_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{args.lib}_search_results.json"
    output_path = os.path.join(output_dir, output_filename)

    existing_results, processed_ids = _load_existing_results(output_path)
    if processed_ids:
        before = len(samples)
        samples = [s for s in samples if str(s.get("doc_id", "")) not in processed_ids]
        print(
            f"[Resume] found {len(processed_ids)} successful samples in checkpoint, "
            f"skip {before - len(samples)} samples, pending={len(samples)}"
        )

    if not samples:
        print("[Search] no pending samples, nothing to do.")
        return

    # Execute concurrent search only on pending samples
    result = run_concurrent_search(
        lib=args.lib,
        samples=samples,
        user_prefix=args.version_dir,
        concurrency=args.workers,
        top_k=args.top_k,
        mode=args.mode,
    )

    new_results = [r for r in result["results"] if r.get("success")]
    all_results = existing_results + new_results

    # Save results
    output_data = {
        "summary": result["summary"],
        "total_duration": result["total_duration"],
        "config": {
            "samples_file": args.samples_file,
            "api_url": args.api_url,
            "workers": args.workers,
            "top_k": args.top_k,
        },
        "results": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Calculate valid results
    success_results = all_results
    total_memories = sum(r["memory_count"] for r in success_results)
    avg_memories = total_memories / len(success_results) if success_results else 0
    print(f"Average {avg_memories:.1f} memories returned per question")


if __name__ == "__main__":
    main()
