import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from evaluation.scripts.utils.client import MemosApiOnlineClient

load_dotenv()

MEMOS_KNOWLEDGEBASE_ID = os.getenv("MEMOS_KNOWLEDGEBASE_ID_MM_LONGBENCH")


def _load_added_ids(records_path: Path) -> dict[str, str | None]:
    if not records_path.exists():
        return {}

    try:
        obj = json.loads(records_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(obj, dict):
        return {}

    added = obj.get("added")
    if not isinstance(added, dict):
        return {}

    return {
        str(key): str(value) if value is not None else None
        for key, value in added.items()
    }


def _check_file_status(
    client: MemosApiOnlineClient,
    file_ids: list[str],
    batch_size: int,
) -> dict[str, dict[str, str | None]]:
    file_status: dict[str, dict[str, str | None]] = {}

    for start in tqdm(
        range(0, len(file_ids), batch_size),
        desc="Checking files",
    ):
        batch = file_ids[start : start + batch_size]
        try:
            resp = client.check_file(batch)
        except Exception as exc:
            print(f"[Check] error for batch starting at {start}: {exc}")
            continue

        if not isinstance(resp, dict):
            continue

        data = resp.get("data")
        if not isinstance(data, dict):
            continue

        details = data.get("file_detail_list")
        if not isinstance(details, list):
            continue

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
    fid_to_filename: dict[str, str] = {
        str(fid): str(filename)
        for filename, fid in added_ids.items()
        if fid
    }

    failed_ids = [
        fid
        for fid, info in file_status.items()
        if info.get("status") == "PROCESSING_FAILED"
    ]

    reupload_results: list[dict[str, str | None]] = []

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
            resp = client.upload_file(
                MEMOS_KNOWLEDGEBASE_ID or "",
                file_url,
            )
            new_id: str | None = None

            if isinstance(resp, dict):
                data = resp.get("data")
                if isinstance(data, list) and data:
                    first = data[0]
                    if isinstance(first, dict) and first.get("id"):
                        new_id = str(first["id"])

            reupload_results.append(
                {
                    "old_file_id": fid,
                    "filename": filename,
                    "new_file_id": new_id,
                    "ok": "true",
                    "error": None,
                }
            )
        except Exception as exc:
            reupload_results.append(
                {
                    "old_file_id": fid,
                    "filename": filename,
                    "new_file_id": None,
                    "ok": "false",
                    "error": str(exc),
                }
            )

    return reupload_results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Check MMLongBench memos-online file status and reupload failed files.",
    )
    parser.add_argument("--lib", default="memos-online")
    parser.add_argument("--version-dir", "-v")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--url-prefix",
        "-u",
        default=(
            "https://memos-knowledge-base-file-pre.oss-cn-shanghai.aliyuncs.com/"
            "mmlongbench_pdf_files/"
        ),
    )
    args = parser.parse_args(argv)

    if args.lib != "memos-online":
        print(f"Only memos-online is supported, got lib={args.lib}")
        return

    output_dir = Path("evaluation/data/mmlongbench")
    if args.version_dir:
        output_dir /= args.version_dir
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
    reupload_results = _reupload_failed_files(
        client,
        file_status,
        added_ids,
        args.url_prefix,
    )

    if reupload_results:
        try:
            obj: dict[str, Any] = {}
            if records_path.exists():
                txt = records_path.read_text(encoding="utf-8")
                if txt:
                    parsed = json.loads(txt)
                    if isinstance(parsed, dict):
                        obj = parsed

            added_obj: dict[str, str | None]
            if isinstance(obj.get("added"), dict):
                added_obj = {
                    str(k): str(v) if v is not None else None
                    for k, v in obj["added"].items()
                }
            else:
                added_obj = dict(added_ids)

            for item in reupload_results:
                if (
                    item.get("ok") == "true"
                    and item.get("filename")
                    and item.get("new_file_id")
                ):
                    added_obj[str(item["filename"])] = str(item["new_file_id"])

            obj["added"] = dict(sorted(added_obj.items()))

            tmp_path = records_path.with_suffix(records_path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(obj, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, records_path)

            print(f"[Update] updated add_results -> {records_path}")
        except Exception as exc:
            print(f"[Update] failed to update add_results: {exc}")

    output_path = output_dir / f"{args.lib}_file_status.json"
    result_obj = {
        "lib": args.lib,
        "version_dir": args.version_dir,
        "total": len(file_ids),
        "file_detail_list": [
            {"id": fid, **(file_status.get(fid) or {})}
            for fid in file_ids
        ],
        "reupload_results": reupload_results,
    }

    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_output.write_text(
        json.dumps(result_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_output, output_path)

    print(
        f"[Check] saved file status for {len(file_status)} files to {output_path}",
    )


if __name__ == "__main__":
    main()
