import argparse
import json
import os
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from evaluation.scripts.utils.metrics import Metrics
from memos.reranker.strategies.dialogue_common import extract_texts_and_sp_from_sources


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


def _get_lib_client(lib: str):
    if lib == "mem0":
        from utils.client import Mem0Client  # type: ignore

        return Mem0Client(enable_graph=False)
    if lib == "supermemory":
        from utils.client import SupermemoryClient  # type: ignore

        return SupermemoryClient()
    from utils.client import MemosApiClient  # type: ignore

    return MemosApiClient()


def _load_existing_results(output_path: Path) -> tuple[list[dict], set[str]]:
    if not output_path.exists():
        return [], set()
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            ids = {str(r.get("_id")) for r in data if r.get("_id")}
            return data, ids
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            rows = data.get("results") or []
            ids = {str(r.get("_id")) for r in rows if r.get("_id")}
            return rows, ids
    except Exception:
        return [], set()
    return [], set()


def _save_json_list(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def get_dedup_sp(sources):
    sp_list: list[list[str | int]] = []
    _, sps = extract_texts_and_sp_from_sources(sources)
    for t, s in sps:
        sp_list.append([t, s])

    seen = set()
    dedup_sp = []
    for t, s in sp_list:
        key = (t, s)
        if key not in seen:
            seen.add(key)
            dedup_sp.append([t, s])
    return dedup_sp


def memos_search(client, user_id: str, query: str, top_k: int) -> tuple[str, list[list[str | int]]]:
    results = retry_operation(client.search, query=query, user_id=user_id, top_k=top_k)
    memories = results["text_mem"][0]["memories"]
    mem_texts = [i["memory"] for i in memories]

    sources = []
    for m in memories:
        source = (m.get("metadata", {}) or {}).get("sources") or []
        sources.extend(source)

    dedup_sp = get_dedup_sp(sources)
    return mem_texts, dedup_sp


def mem0_search(client, user_id: str, query: str, top_k: int) -> tuple[str, list[list[str | int]]]:
    res = retry_operation(client.search, query, user_id, top_k)
    results = res.get("results", [])
    mem_texts = [m.get("memory", "") for m in results if m.get("memory")]
    dedup_sp = get_dedup_sp(mem_texts)
    return mem_texts, dedup_sp


def supermemory_search(
    client, user_id: str, query: str, top_k: int
) -> tuple[str, list[list[str | int]]]:
    chunk_list = retry_operation(client.search, query, user_id, top_k)
    dedup_sp = get_dedup_sp(chunk_list)
    return chunk_list, dedup_sp


def search_one(client, lib: str, item: dict, top_k: int) -> dict:
    qid = item.get("_id") or item.get("id")
    question = item.get("question") or ""

    if lib == "memos":
        memories, sp_list = memos_search(client, str(qid), str(question), top_k)
    elif lib == "mem0":
        memories, sp_list = mem0_search(client, str(qid), str(question), top_k)
    elif lib == "supermemory":
        memories, sp_list = supermemory_search(client, str(qid), str(question), top_k)
    else:
        memories, sp_list = [], []

    return {
        "_id": str(qid),
        "question": question,
        "answer": item.get("answer"),
        "memories": memories,
        "sp": sp_list,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="HotpotQA search (search only).")
    parser.add_argument(
        "--lib",
        type=str,
        default="memos",
        choices=["memos", "mem0", "supermemory"],
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples (was max_samples)"
    )
    parser.add_argument("--version-dir", "-v", default=None, help="Version directory name")
    parser.add_argument("--mode", default="fine", help="Search mode")

    args = parser.parse_args(argv)

    # Handle limit/max_samples compatibility
    limit = args.limit if args.limit is not None else args.max_samples

    data = load_dataset("hotpotqa/hotpot_qa", "distractor")
    split = data.get("validation")
    items_list = [split[i] for i in range(len(split))]
    if limit is not None:
        items_list = items_list[:limit]

    output_dir = Path(f"evaluation/data/hotpot/{args.version_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.lib}_search_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results, processed_ids = _load_existing_results(output_path)
    pending_items = []
    for it in items_list:
        qid = it.get("_id") or it.get("id")
        if str(qid) not in processed_ids:
            pending_items.append(it)

    print(
        f"[Search] lib={args.lib} total={len(items_list)} pending={len(pending_items)} top_k={args.top_k}"
    )
    if not pending_items:
        return

    client = _get_lib_client(args.lib)
    metrics = Metrics()
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:

        def do_search(item):
            st = time.perf_counter()
            try:
                r = search_one(client, args.lib, item, args.top_k)
                dur = time.perf_counter() - st
                metrics.record(dur, True)
                return r
            except Exception as e:
                dur = time.perf_counter() - st
                metrics.record(dur, False, str(e))
                raise e

        futures = [executor.submit(do_search, it) for it in pending_items]
        for idx, f in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Searching"), 1
        ):
            try:
                r = f.result()
                results.append(r)
                if idx % 20 == 0:
                    _save_json_list(output_path, results)
            except Exception as e:
                print(f"[Search] Error: {e}")

    _save_json_list(output_path, results)
    print(f"[Search] saved {len(results)} rows to {output_path}")

    # Save performance metrics
    total_duration = time.time() - start_time
    summary = metrics.summary()
    # Merge perf into results json file
    combined_obj = {
        "results": results,
        "perf": {
            "summary": summary,
            "total_duration": total_duration,
            "config": {
                "workers": args.workers,
                "top_k": args.top_k,
                "limit": limit,
                "mode": args.mode,
                "lib": args.lib,
            },
        },
    }
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(combined_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, output_path)

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
        print("\nError stats:")
        for error, count in sorted(summary["errors"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  [{count} times] {error[:100]}...")


if __name__ == "__main__":
    main()
