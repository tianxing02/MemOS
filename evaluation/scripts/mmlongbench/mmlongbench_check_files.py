import argparse
import json
import os

from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from evaluation.scripts.utils.client import MemosApiOnlineClient


load_dotenv()
memos_knowledgebase_id = os.getenv("MEMOS_KNOWLEDGEBASE_ID_MM_LONGBENCH")


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


def _check_file_status(
    client: MemosApiOnlineClient, file_ids: list[str], batch_size: int
) -> dict[str, dict[str, str | None]]:
    file_status: dict[str, dict[str, str | None]] = {}
    for i in tqdm(range(0, len(file_ids), batch_size), desc="Checking files"):
        batch = file_ids[i : i + batch_size]
        try:
            resp = client.check_file(batch)
        except Exception as e:
            print(f"[Check] error for batch starting at {i}: {e}")
            continue
        if not isinstance(resp, dict):
            continue
        data = resp.get("data") or {}
        details = data.get("file_detail_list") or []
        for item in details:
            if not isinstance(item, dict):
                continue
            fid = item.get("id")
            if not fid:
                continue
            file_status[str(fid)] = {
                "name": item.get("name"),
                "size": item.get("size"),
                "status": item.get("status"),
            }
    return file_status


def _reupload_failed_files(
    client: MemosApiOnlineClient,
    file_status: dict[str, dict[str, str | None]],
    added_ids: dict[str, str | None],
    url_prefix: str,
) -> list[dict[str, str | None]]:
    fid_to_filename: dict[str, str] = {}
    for filename, fid in added_ids.items():
        if fid:
            fid_to_filename[str(fid)] = str(filename)

    reupload_results: list[dict[str, str | None]] = []
    failed_ids = [
        fid for fid, info in file_status.items() if (info.get("status") == "PROCESSING_FAILED")
    ]

    for fid in tqdm(failed_ids, desc="Reuploading failed files"):
        filename = fid_to_filename.get(fid)
        if not filename:
            reupload_results.append(
                {
                    "old_file_id": fid,
                    "filename": None,
                    "new_file_id": None,
                    "ok": "false",
                    "error": "filename_not_found",
                }
            )
            continue

        file_url = f"{url_prefix.rstrip('/')}/{filename}"
        try:
            resp = client.upload_file(memos_knowledgebase_id or "", file_url)
            new_id = None
            if isinstance(resp, dict):
                data = resp.get("data") or {}
                if isinstance(data, list) and data:
                    first = data[0] if isinstance(data[0], dict) else {}
                    new_id = str(first.get("id")) if first.get("id") else None
            reupload_results.append(
                {
                    "old_file_id": fid,
                    "filename": filename,
                    "new_file_id": new_id,
                    "ok": "true",
                    "error": None,
                }
            )
        except Exception as e:
            reupload_results.append(
                {
                    "old_file_id": fid,
                    "filename": filename,
                    "new_file_id": None,
                    "ok": "false",
                    "error": str(e),
                }
            )

    return reupload_results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Check MMLongbench memos-online file status and reupload failed files."
    )
    parser.add_argument("--lib", type=str, default="memos-online")
    parser.add_argument("--version-dir", "-v", default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--url-prefix",
        "-u",
        default="https://memos-knowledge-base-file-pre.oss-cn-shanghai.aliyuncs.com/mmlongbench_pdf_files/",
    )
    args = parser.parse_args(argv)

    if args.lib != "memos-online":
        print(f"Only memos-online is supported, got lib={args.lib}")
        return

    output_dir = Path("evaluation/data/mmlongbench")
    if args.version_dir:
        output_dir = output_dir / args.version_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records_path = output_dir / f"{args.lib}_add_results.json"
    print(f"[Check] loading records from {records_path}")

    added_ids = _load_added_ids(records_path)
    file_ids = sorted({fid for fid in added_ids.values() if fid})
    print(f"[Check] total file ids: {len(file_ids)}")
    if not file_ids:
        return

    client = MemosApiOnlineClient()
    batch_size = max(1, args.batch_size)

    file_status = _check_file_status(client, file_ids, batch_size)
    reupload_results = _reupload_failed_files(client, file_status, added_ids, args.url_prefix)

    if reupload_results:
        try:
            obj: dict = {}
            if records_path.exists():
                txt = records_path.read_text(encoding="utf-8")
                if txt:
                    parsed = json.loads(txt)
                    if isinstance(parsed, dict):
                        obj = parsed
            added_obj: dict[str, str | None] = {}
            if isinstance(obj.get("added"), dict):
                added_obj = {
                    str(k): (str(v) if v is not None else None) for k, v in obj["added"].items()
                }
            else:
                added_obj = dict(added_ids)
            for item in reupload_results:
                if item.get("ok") == "true" and item.get("filename") and item.get("new_file_id"):
                    added_obj[str(item["filename"])] = str(item["new_file_id"])
            obj["added"] = dict(sorted(added_obj.items()))
            tmp_r = records_path.with_suffix(records_path.suffix + ".tmp")
            tmp_r.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_r, records_path)
            print(f"[Update] updated add_results with new file ids -> {records_path}")
        except Exception as e:
            print(f"[Update] failed to update add_results: {e}")

    output_path = output_dir / f"{args.lib}_file_status.json"
    result_obj = {
        "lib": args.lib,
        "version_dir": args.version_dir,
        "total": len(file_ids),
        "file_detail_list": [{"id": fid, **(file_status.get(fid) or {})} for fid in file_ids],
        "reupload_results": reupload_results,
    }
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, output_path)
    print(f"[Check] saved file status for {len(file_status)} files to {output_path}")


if __name__ == "__main__":
    main()

