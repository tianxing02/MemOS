import argparse
import json
import os

from pathlib import Path


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        txt = path.read_text(encoding="utf-8")
        if not txt:
            return {}
        obj = json.loads(txt)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Update added_records.json from status JSON reupload results."
    )
    parser.add_argument(
        "--status-json",
        type=str,
        default="evaluation/data/hotpot/test_0113_memos/memos-online_file_status.json",
    )
    parser.add_argument(
        "--records-json",
        type=str,
        default="evaluation/data/hotpot/test_0113_memos/memos-online_added_records.json",
    )
    args = parser.parse_args(argv)

    status_path = Path(args.status_json)
    records_path = Path(args.records_json)

    status_obj = _read_json(status_path)
    records_obj = _read_json(records_path)

    added = {}
    if isinstance(records_obj.get("added"), dict):
        added = {
            str(k): (str(v) if v is not None else None) for k, v in records_obj["added"].items()
        }

    reupload_results = status_obj.get("reupload_results") or []
    updated_count = 0
    for item in reupload_results:
        if not isinstance(item, dict):
            continue
        if item.get("ok") != "true":
            continue
        uid = item.get("user_id")
        new_id = item.get("new_file_id")
        if not uid or not new_id:
            continue
        uid = str(uid)
        new_id = str(new_id)
        if added.get(uid) != new_id:
            added[uid] = new_id
            updated_count += 1

    records_obj["added"] = dict(sorted(added.items()))
    _write_json_atomic(records_path, records_obj)

    print(f"Updated {updated_count} entries in {records_path}")


if __name__ == "__main__":
    main()
