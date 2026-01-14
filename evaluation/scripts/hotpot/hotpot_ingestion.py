import argparse
import json
import os
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from evaluation.scripts.hotpot.data_loader import load_hotpot_data
from evaluation.scripts.utils.metrics import Metrics


load_dotenv()
memos_knowledgebase_id = os.getenv("MEMOS_KNOWLEDGEBASE_ID_HOTPOT")


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
        from evaluation.scripts.utils.client import Mem0Client

        return Mem0Client(enable_graph=False)
    if lib == "supermemory":
        from evaluation.scripts.utils.client import SupermemoryClient

        return SupermemoryClient()
    if lib == "memos-online":
        from evaluation.scripts.utils.client import MemosApiOnlineClient

        return MemosApiOnlineClient()
    from evaluation.scripts.utils.client import MemosApiClient

    return MemosApiClient()


def _load_added_ids(records_path: Path) -> dict[str, str | None]:
    if not records_path.exists():
        return {}

    try:
        obj = json.loads(records_path.read_text(encoding="utf-8"))
        added = obj.get("added") if isinstance(obj, dict) else None
        if isinstance(added, dict):
            return {str(k): (str(v) if v is not None else None) for k, v in added.items()}
    except Exception:
        return {}

    return {}


def _save_added_ids(
    records_path: Path,
    added: dict[str, str | None],
    perf: dict | None = None,
) -> None:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = records_path.with_suffix(records_path.suffix + ".tmp")

    obj = {"added": dict(sorted(added.items()))}
    if perf is not None:
        obj["perf"] = perf

    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, records_path)


def _build_tasks(ctx: dict | list | None) -> list[str]:
    tasks: list[str] = []
    for item in ctx:
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue
        title, sentences = item
        if not isinstance(sentences, list):
            continue
        for idx, sentence in enumerate(sentences):
            tasks.append(
                json.dumps({"idx": idx, "title": title, "sentence": sentence}, ensure_ascii=False)
            )
    return tasks


def _build_memory_texts(ctx: dict | list | None) -> list[str]:
    texts: list[str] = []

    if not ctx:
        return texts

    for item in ctx:
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue

        title, sentences = item
        if not isinstance(sentences, list):
            continue

        for sentence in sentences:
            texts.append(f"{title}: {sentence}")

    return texts


def add_context_memories(
    client,
    lib: str,
    user_id: str,
    ctx: dict | list | None,
    url_prefix: str,
    mode: str = "fine",
    async_mode: str = "sync",
) -> str | None:
    tasks = _build_tasks(ctx)
    if not tasks:
        return None

    file_id = None

    if lib == "memos-online":
        file_url = f"{url_prefix.rstrip('/')}/{user_id}_context.txt"
        result = retry_operation(
            client.upload_file,
            memos_knowledgebase_id,
            file_url,
        )
        file_id = result["data"][0]["id"]

    if lib == "memos":
        messages = [{"type": "text", "text": content} for content in tasks]
        writable_cube_ids = [user_id]
        retry_operation(
            client.add,
            messages=messages,
            user_id=user_id,
            writable_cube_ids=writable_cube_ids,
            source_type="batch_import",
            mode=mode,
            async_mode=async_mode,
        )

    if lib == "mem0":
        ts = int(time.time())
        messages = [{"role": "user", "content": content} for content in tasks]
        retry_operation(client.add, messages=messages, user_id=user_id, timestamp=ts, batch_size=10)

    if lib == "supermemory":
        for content in tasks:
            retry_operation(client.add, content=content, user_id=user_id)

    return file_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="HotpotQA ingestion (add only).")
    parser.add_argument(
        "--lib",
        type=str,
        default="memos",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--version-dir", "-v", default=None, help="Version directory name")
    parser.add_argument(
        "--mode", default="fine", choices=["fine", "fast"], help="Processing mode (default: fine)"
    )
    parser.add_argument(
        "--async-mode", default="sync", choices=["sync", "async"], help="Async mode (default: sync)"
    )
    parser.add_argument(
        "--url-prefix",
        "-u",
        default="https://memos-knowledge-base-file-pre.oss-cn-shanghai.aliyuncs.com/hotpot_text_files/",
        help="URL prefix to be prepended to filenames",
    )

    args = parser.parse_args(argv)

    print("=" * 60)
    print("hotpotQA Product Add Concurrent Tool")
    print("=" * 60)

    output_dir = Path("evaluation/data/hotpot")
    if args.version_dir:
        output_dir = output_dir / args.version_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    items_list = load_hotpot_data("evaluation/data/hotpot")
    if args.limit is not None:
        items_list = items_list[: args.limit]

    records_path = output_dir / f"{args.lib}_added_records.json"
    added_ids: dict[str, str | None] = _load_added_ids(records_path)

    pending_items = []
    for it in items_list:
        qid = it.get("_id") or it.get("id")
        if str(qid) not in added_ids:
            pending_items.append(it)

    print(f"[Add] lib={args.lib} total={len(items_list)} pending={len(pending_items)}")
    if not pending_items:
        return

    client = _get_lib_client(args.lib)
    metrics = Metrics()

    def do_ingest(item):
        start_time = time.perf_counter()
        try:
            qid = item.get("_id") or item.get("id")
            ctx = item.get("context")
            user_id = str(qid)
            file_id = add_context_memories(client, args.lib, user_id, ctx, args.url_prefix)

            duration = time.perf_counter() - start_time
            metrics.record(duration, True)
            return str(qid), file_id
        except Exception as e:
            duration = time.perf_counter() - start_time
            metrics.record(duration, False, str(e))
            raise e

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(do_ingest, it) for it in pending_items]
        for _idx, f in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Adding"), 1):
            try:
                sid, fid = f.result()
                if sid:
                    if args.lib == "memos-online":
                        added_ids[sid] = str(fid) if fid else None
                    else:
                        added_ids.setdefault(sid, None)

                    if len(added_ids) % 20 == 0:
                        _save_added_ids(records_path, added_ids)

            except Exception as e:
                print(f"[Add] Error: {e}")

    _save_added_ids(records_path, added_ids)
    print(f"[Add] saved records to {records_path}")

    total_duration = time.time() - start_time
    summary = metrics.summary()

    _save_added_ids(
        records_path,
        added_ids,
        perf={
            "summary": summary,
            "total_duration": total_duration,
            "config": {
                "workers": args.workers,
                "mode": args.mode,
                "async_mode": args.async_mode,
                "lib": args.lib,
            },
        },
    )

    print("\n" + "=" * 60)
    print("Ingestion finished! Statistics:")
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


if __name__ == "__main__":
    main()
