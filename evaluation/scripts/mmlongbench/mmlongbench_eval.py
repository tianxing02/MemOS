import json
import os
import sys
import time

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from eval.eval_score import eval_acc_and_f1, eval_score, show_results  # type: ignore
from eval.extract_answer import extract_answer  # type: ignore
from openai import OpenAI


# Ensure project paths for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.append(str(SCRIPTS_ROOT))
sys.path.append(str(SRC_ROOT))
load_dotenv()
max_retries = 5
RESULTS_PATH = "evaluation/data/mmlongbench/test_results.json"


def iter_markdown_files(base_dir: str | Path) -> Iterator[Path]:
    base = Path(base_dir)
    if not base.exists():
        return
    # glob all 'auto/*.md'
    for md in base.rglob("auto/*.md"):
        if md.is_file():
            yield md


def _get_clients():
    from utils.client import MemosApiClient  # type: ignore
    from utils.prompts import MEMOS_CONTEXT_TEMPLATE  # type: ignore

    from memos.mem_os.core import add_images_context, get_images  # type: ignore

    memos_client = MemosApiClient()
    openai_client = OpenAI(
        api_key=os.getenv("CHAT_MODEL_API_KEY"), base_url=os.getenv("CHAT_MODEL_BASE_URL")
    )
    return memos_client, openai_client, MEMOS_CONTEXT_TEMPLATE, add_images_context, get_images


def _load_existing_results():
    completed_pairs: set[tuple[str, str]] = set()
    completed_docs: set[str] = set()
    existing: list[dict] = []
    p = Path(RESULTS_PATH)
    if p.exists():
        try:
            existing = json.loads(p.read_text(encoding="utf-8"))
            for r in existing:
                did = str(r.get("doc_id") or "").strip()
                q = str(r.get("question") or "").strip()
                # normalize whitespace for robust resume
                did_norm = did
                q_norm = " ".join(q.split())
                if did:
                    completed_docs.add(did)
                if did_norm and q_norm:
                    completed_pairs.add((did_norm, q_norm))
        except Exception:
            existing = []
    return existing, completed_pairs, completed_docs


def register_and_add_markdown(client, user_id: str, md_path: Path) -> None:
    client.register_user(user_id=user_id, user_name=user_id, mem_cube_id=user_id)
    # Read markdown and add per paragraph
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    # Simple paragraph split: double newline blocks
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    print(f"[Ingest] doc_id={user_id} paragraphs={len(paragraphs)} path={md_path}")

    def _add_one(content: str) -> None:
        for attempt in range(max_retries):
            try:
                client.add(memory_content=content, user_id=user_id, conv_id=user_id)
                return
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_add_one, p): i for i, p in enumerate(paragraphs)}
        success_count = 0
        failure_count = 0
        for f in as_completed(futures):
            try:
                f.result()
                success_count += 1
            except Exception:
                failure_count += 1
    print(
        f"[Add] user={user_id} success={success_count} fail={failure_count} total={len(paragraphs)}"
    )


def memos_search(
    client, memos_context_template, get_images, user_id: str, query: str, top_k: int = 15
) -> tuple[str, list[str]]:
    results = None
    for attempt in range(max_retries):
        try:
            results = client.search(query=query, user_id=user_id, top_k=top_k)
            break
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                raise
    memories = results["text_mem"][0]["memories"]
    mem_texts = [m["memory"] for m in memories]
    for memory in mem_texts[:5]:
        print(memory)
    context = "\n".join(mem_texts) + f"\n{results.get('pref_string', '')}"
    context = memos_context_template.format(user_id=user_id, memories=context)

    # Collect possible image paths from memory texts (and any source content if present)
    sources_texts: list[str] = []
    for m in memories:
        srcs = (m.get("metadata", {}) or {}).get("sources") or []
        for s in srcs:
            content = s.get("content") if isinstance(s, dict) else str(s)
            if content:
                sources_texts.append(content)
    candidates = mem_texts + sources_texts
    image_paths = get_images(candidates)
    print(
        f"[Search] user={user_id} top_k={top_k} memories={len(memories)} images={len(image_paths)}"
    )
    return context, image_paths


def multimodal_answer(
    add_images_context, oai_client, context: str, question: str, image_paths: list[str]
) -> str:
    # Build messages with multimodal user content (align with eval_docs: use question only, no extra system prompt)
    messages = [{"role": "user", "content": question + "\n\n" + context}]
    add_images_context(messages, image_paths)
    resp = oai_client.chat.completions.create(
        model=os.getenv("CHAT_MODEL"), messages=messages, temperature=0
    )
    return resp.choices[0].message.content or ""


def run_ingest_and_eval(
    ppt_root: str | Path = "ppt_test_result",
    questions_file: str | Path | None = None,
    top_k: int = 15,
) -> None:
    client, oai_client, memos_context_template, add_images_context, get_images = _get_clients()
    if questions_file and Path(questions_file).exists():
        data = json.loads(Path(questions_file).read_text(encoding="utf-8"))
        print(f"[Load] samples={len(data)} file={questions_file}")

        # Build allowed doc list from documents directory (align with eval_docs grouping by doc_id)
        docs_dir = Path("evaluation/data/mmlongbench/documents")
        allowed_docs: set[str] = set()
        if docs_dir.exists():
            for f in docs_dir.iterdir():
                if f.is_file():
                    allowed_docs.add(f.name)
        if allowed_docs:
            print(f"[Filter] allowed_docs={len(allowed_docs)} from {docs_dir}")

        # Determine doc_ids present in samples and apply allowed filter
        doc_ids_in_samples = {str(s.get("doc_id") or "").strip() for s in data if s.get("doc_id")}
        doc_list = [d for d in doc_ids_in_samples if (not allowed_docs or d in allowed_docs)]
        doc_list.sort()

        # Resume state
        existing, completed_pairs, completed_docs = _load_existing_results()
        print(f"[Resume] loaded_results={len(existing)} completed_docs={len(completed_docs)}")
        results: list[dict] = list(existing)
        ingested_doc_ids: set[str] = set(completed_docs)

        for doc_id in doc_list:
            if not doc_id:
                continue
            print(f"\n===== [Doc] {doc_id} =====")
            user_id = doc_id

            # Locate markdown under ppt_test_result for this doc_id
            md_path = None
            base = Path(ppt_root)
            if base.exists():
                stem = Path(doc_id).stem.lower()
                name = doc_id.lower()
                for md in base.rglob("*.md"):
                    pstr = str(md).lower()
                    if (stem and stem in pstr) or (name and name in pstr):
                        md_path = md
                        break

            # Ingest markdown once per doc
            if doc_id not in ingested_doc_ids:
                if md_path is not None:
                    register_and_add_markdown(client, user_id, md_path)
                else:
                    print(f"[Skip] markdown not found for doc_id={doc_id}")
                ingested_doc_ids.add(doc_id)

            # Evaluate all samples for this doc_id
            doc_samples = [s for s in data if str(s.get("doc_id") or "").strip() == doc_id]
            for item in doc_samples:
                question = item["question"]
                q_norm = " ".join(str(question).split())
                if (doc_id, q_norm) in completed_pairs:
                    print(f"[Skip] already done doc_id={doc_id} question={question[:60]}...")
                    continue
                context, images = memos_search(
                    client, memos_context_template, get_images, user_id, question, top_k=top_k
                )
                response = multimodal_answer(
                    add_images_context, oai_client, context, question, images
                )
                with open(
                    Path("evaluation/scripts/mmlongbench/eval/prompt_for_answer_extraction.md"),
                    encoding="utf-8",
                ) as f:
                    prompt = f.read()
                extracted_res = extract_answer(question, response, prompt)
                try:
                    pred_ans = (
                        extracted_res.split("Answer format:")[0]
                        .split("Extracted answer:")[1]
                        .strip()
                    )
                except Exception:
                    pred_ans = response.strip()
                score = eval_score(item.get("answer"), pred_ans, item.get("answer_format", "Str"))
                print(f"[QA] doc_id={doc_id} images={len(images)} score={score}")

                sample_res = dict(item)
                sample_res["response"] = response
                sample_res["extracted_res"] = extracted_res
                sample_res["pred"] = pred_ans
                sample_res["score"] = score
                results.append(sample_res)
                completed_pairs.add((doc_id, q_norm))

                print("[Question]:", question)
                print("[Answer]:", pred_ans)
                print("[Ground truth]:", item.get("answer"))
                print("[Score]:", score)
                out_json = Path("evaluation/data/mmlongbench/test_results.json")
                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_json.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                acc, f1 = eval_acc_and_f1(results)
                total_target = sum(1 for s in data if str(s.get("doc_id") or "") in doc_list)
                print(f"[Metric] acc={acc} f1={f1} processed={len(results)}/{total_target}")

        report_path = Path("evaluation/data/mmlongbench/test_results_report.txt")
        show_results(results, show_path=str(report_path))
        print(f"[Report] saved to {report_path}")


if __name__ == "__main__":
    run_ingest_and_eval(questions_file=Path("evaluation/data/mmlongbench/samples.json"))
