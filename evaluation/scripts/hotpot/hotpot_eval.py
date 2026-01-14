import argparse
import importlib.util
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from evaluation.scripts.hotpot.data_loader import load_hotpot_data
from evaluation.scripts.utils.extract_answer import (
    extract_answer,
    parse_extracted_answer,
)
from evaluation.scripts.utils.metrics import Metrics
from evaluation.scripts.utils.prompts import HOTPOT_ANSWER_PROMPT

load_dotenv()

HOT_POT_DIR = Path("evaluation/data/hotpot")


def llm_response(
    oai_client: OpenAI,
    chat_model: str,
    context: str,
    question: str,
) -> str:
    prompt = HOTPOT_ANSWER_PROMPT.format(
        question=question,
        context=context,
    )
    resp = oai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "system", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    raise ValueError(f"Invalid json format: {path}")


def _save_pred(
    pred_path: Path,
    pred_answers: dict[str, str],
    pred_sp: dict[str, list[Any]],
    perf: dict[str, Any] | None = None,
) -> None:
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = pred_path.with_suffix(pred_path.suffix + ".tmp")

    safe_pred_answers = {
        k: v if isinstance(v, str) else "" if v is None else str(v) for k, v in pred_answers.items()
    }

    obj: dict[str, Any] = {
        "answer": safe_pred_answers,
        "sp": pred_sp,
    }
    if perf is not None:
        obj["perf"] = perf

    tmp_path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, pred_path)


def run_eval(pred_path: Path, gold_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "hotpot_eval_v1",
        "evaluation/scripts/hotpot/hotpot_evaluate_v1.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load hotpot_evaluate_v1")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    metrics: dict[str, Any] = module.eval(
        str(pred_path),
        str(gold_path),
    )

    # Save metrics back into pred json
    try:
        if pred_path.exists():
            current_data = json.loads(
                pred_path.read_text(encoding="utf-8"),
            )
        else:
            current_data = {}

        if isinstance(current_data, list):
            new_data: Any = [metrics, *current_data]
        elif isinstance(current_data, dict):
            new_data = metrics.copy()
            for key, value in current_data.items():
                if key not in new_data:
                    new_data[key] = value
        else:
            new_data = metrics

        pred_path.write_text(
            json.dumps(new_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[Eval] Failed to save metrics to {pred_path}: {exc}")

    # Save metrics to xlsx
    try:
        xlsx_path = pred_path.with_name(
            f"{pred_path.stem}_metrics.xlsx",
        )

        rows: list[dict[str, Any]] = []
        row = {
            "category": "overall",
            "question_number": metrics.get("count"),
            "em": metrics.get("em"),
            "f1": metrics.get("f1"),
            "sp_em": metrics.get("sp_em"),
            "sp_f1": metrics.get("sp_f1"),
            "joint_em": metrics.get("joint_em"),
            "joint_f1": metrics.get("joint_f1"),
        }

        for key, value in metrics.items():
            if key not in row and key != "count":
                row[key] = value

        rows.append(row)

        df = pd.DataFrame(rows)
        preferred_cols = [
            "category",
            "question_number",
            "em",
            "f1",
            "sp_em",
            "sp_f1",
            "joint_em",
            "joint_f1",
        ]
        remaining_cols = [c for c in df.columns if c not in preferred_cols]
        df = df[preferred_cols + remaining_cols]

        df.to_excel(xlsx_path, index=False)
        print(f"[Eval] Metrics xlsx saved to: {xlsx_path}")
    except Exception as exc:
        print(f"[Eval] Failed to save metrics xlsx: {exc}")


def evaluate_one(
    oai_client: OpenAI,
    row: dict[str, Any],
    chat_model: str,
) -> tuple[str, str, list[Any]]:
    qid = str(row.get("_id"))
    question = row.get("question") or ""
    context = row.get("context") or ""
    sp_list = row.get("sp") or []

    raw_answer = llm_response(
        oai_client,
        chat_model,
        context=context,
        question=question,
    )
    extracted = extract_answer(question, raw_answer)
    answer = parse_extracted_answer(extracted, raw_answer)

    return qid, answer, sp_list


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="HotpotQA evaluation (OpenAI only).",
    )
    parser.add_argument(
        "--lib",
        default="memos",
        choices=["memos", "mem0", "supermemory"],
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--version-dir", "-v")
    parser.add_argument("--chat-model", required=True)
    parser.add_argument("--search-mode", default="fine")

    args = parser.parse_args(argv)

    output_dir = HOT_POT_DIR / str(args.version_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.lib == "memos":
        search_path = output_dir / f"{args.lib}_{args.search_mode}_search_results.json"
        pred_path = output_dir / f"{args.lib}_{args.search_mode}_search_eval_results.json"
    else:
        search_path = output_dir / f"{args.lib}_search_results.json"
        pred_path = output_dir / f"{args.lib}_eval_results.json"

    gold_path = HOT_POT_DIR / "dev_distractor_gold.json"

    if not search_path.exists():
        raise FileNotFoundError(f"Search results not found: {search_path}")

    if not gold_path.exists():
        load_hotpot_data(str(HOT_POT_DIR))

    pred_answers: dict[str, str] = {}
    pred_sp: dict[str, list[Any]] = {}

    if pred_path.exists():
        try:
            prev = json.loads(pred_path.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                pred_answers.update(prev.get("answer", {}))
                pred_sp.update(prev.get("sp", {}))
        except Exception as exc:
            print(f"[Eval] Failed to load existing pred: {exc}")

    rows = _load_json_list(search_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    pending = [row for row in rows if str(row.get("_id")) not in pred_answers]

    print(
        f"[Eval] lib={args.lib} total={len(rows)} pending={len(pending)} workers={args.workers}",
    )

    if not pending:
        run_eval(pred_path, gold_path)
        return

    oai_client = OpenAI(
        api_key=os.getenv("CHAT_MODEL_API_KEY"),
        base_url=os.getenv("CHAT_MODEL_BASE_URL"),
    )

    metrics = Metrics()
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:

        def do_eval(row: dict[str, Any]) -> tuple[str, str, list[Any]]:
            start = time.perf_counter()
            try:
                result = evaluate_one(
                    oai_client,
                    row,
                    args.chat_model,
                )
                metrics.record(time.perf_counter() - start, True)
                return result
            except Exception as exc:
                metrics.record(
                    time.perf_counter() - start,
                    False,
                    str(exc),
                )
                raise

        futures = [executor.submit(do_eval, row) for row in pending]

        for idx, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Evaluating"),
            start=1,
        ):
            try:
                qid, answer, sp_list = future.result()
                pred_answers[qid] = answer
                pred_sp[qid] = sp_list
                if idx % 20 == 0:
                    _save_pred(pred_path, pred_answers, pred_sp)
            except Exception as exc:
                print(f"[Eval] Error: {exc}")

    total_duration = time.time() - start_time
    summary = metrics.summary()

    perf_obj = {
        "summary": summary,
        "total_duration": total_duration,
        "config": {
            "workers": args.workers,
            "chat_model": args.chat_model,
            "lib": args.lib,
        },
    }

    _save_pred(pred_path, pred_answers, pred_sp, perf=perf_obj)
    run_eval(pred_path, gold_path)

    print("\n" + "=" * 60)
    print("Evaluation finished!")
    print("=" * 60)
    print(f"Total duration: {total_duration:.2f}s")
    print(
        f"Success: {summary['counts']['success']} / Failed: {summary['counts']['failed']}",
    )

    if summary["errors"]:
        print("\nTop errors:")
        for error, count in sorted(
            summary["errors"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]:
            print(f"  [{count} times] {error[:100]}...")


if __name__ == "__main__":
    main()
