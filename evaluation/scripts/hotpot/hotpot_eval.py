import importlib.util
import json
import os
import sys
import time
import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.client import MemosApiClient
from utils.prompts import LME_ANSWER_PROMPT, MEMOS_CONTEXT_TEMPLATE


load_dotenv()
os.environ["SEARCH_MODE"] = os.environ.get("SEARCH_MODE", "fine")
data = load_dataset("hotpotqa/hotpot_qa", "distractor")
client = MemosApiClient()
oai_client = OpenAI(
    api_key=os.getenv("CHAT_MODEL_API_KEY"), base_url=os.getenv("CHAT_MODEL_BASE_URL")
)
max_retries = 5


def add_context_memories(user_id: str, ctx: dict | list | None):
    titles = ctx.get("title") or []
    sentences_list = ctx.get("sentences") or []

    client.register_user(user_id=user_id, user_name=user_id, mem_cube_id=user_id)

    tasks = []
    for title, sentences in zip(titles, sentences_list, strict=False):
        for sentence in sentences:
            memory_content = f"{title}: {sentence}"
            tasks.append(memory_content)

    def _add_one_memory(content: str):
        for attempt in range(max_retries):
            try:
                client.add(memory_content=content, user_id=user_id, conv_id=user_id)
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    print(f"[FAILED] {content}, error={e}")
                    raise e

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_add_one_memory, m): m for m in tasks}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print("线程执行失败:", e)


def memos_search(user_id: str, query: str, top_k: int = 30) -> tuple[str, float]:
    start_time = datetime.now()
    results = None
    for attempt in range(max_retries):
        try:
            results = client.search(query=query, user_id=user_id, top_k=top_k)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                raise e
    print("Search memories:", len(results["text_mem"][0]["memories"]))
    context = (
        "\n".join([i["memory"] for i in results["text_mem"][0]["memories"]])
        + f"\n{results.get('pref_string', '')}"
    )
    context = MEMOS_CONTEXT_TEMPLATE.format(user_id=user_id, memories=context)
    elapsed = (datetime.now() - start_time).total_seconds() * 1000
    return context, elapsed


def lme_response(context: str, question: str, question_date: str | None = None) -> str:
    prompt = LME_ANSWER_PROMPT.format(
        question=question, question_date=question_date or "", context=context
    )
    resp = oai_client.chat.completions.create(
        model=os.getenv("CHAT_MODEL"),
        messages=[{"role": "system", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def extract_answer(question: str, output: str, model_name: str | None = None) -> str:
    try:
        response = oai_client.chat.completions.create(
            model=model_name or os.getenv("CHAT_MODEL"),
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are an answer extractor. Given a question and a verbose response, "
                        "return ONLY the concise final answer suitable for HotpotQA exact match.\n\n"
                        "Rules:\n"
                        "- If the question asks yes/no, answer strictly 'yes' or 'no'.\n"
                        "- Otherwise, output the shortest noun phrase/entity/date/number that answers the question.\n"
                        "- No explanations, no punctuation beyond what's necessary for the answer.\n\n"
                        f"Question: {question}\nVerbose response: {output}\nFinal answer:"
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=64,
            top_p=1,
        )
        ans = (response.choices[0].message.content or "").strip()
        return ans
    except Exception:
        text = (output or "").lower()
        if " yes" in text or text.startswith("yes"):
            return "yes"
        if " no" in text or text.startswith("no"):
            return "no"
        for sep in ["\n", ". ", ".", "?", "!"]:
            if sep in output:
                cand = output.split(sep)[0].strip()
                if cand:
                    return cand
        return (output or "").strip()


def build_context_text(context_list):
    parts = []
    for title, sentences in context_list:
        text = " ".join(s.strip() for s in sentences if s.strip())
        parts.append(f"{title}: {text}")
    return "\n".join(parts)


def build_and_ask(item):
    qid = item.get("_id") or item.get("id")
    question = item["question"]
    ctx = item.get("context")
    add_context_memories(qid, ctx)

    try:
        context, _ = memos_search(qid, question, top_k=30)
        raw_answer = lme_response(context=context, question=question, question_date="")
        answer = extract_answer(question, raw_answer)
        print("Question:", question)
        print("Answer (raw):", raw_answer)
        print("Answer (final):", answer)
        return qid, answer
    except Exception as e:
        print(f"[Question {qid}] Error:", e)
        traceback.print_exc()
        try:
            if isinstance(ctx, dict):
                titles = ctx.get("title") or []
                sentences = ctx.get("sentences") or []
                ctx_list = [[t, s] for t, s in zip(titles, sentences, strict=False)]
            else:
                ctx_list = ctx or []
            fallback_context = MEMOS_CONTEXT_TEMPLATE.format(
                user_id=qid, memories=build_context_text(ctx_list)
            )
            raw_answer = lme_response(context=fallback_context, question=question, question_date="")
            answer = extract_answer(question, raw_answer)
            print("Question:", question)
            print("Fallback Answer (raw):", raw_answer)
            print("Fallback Answer (final):", answer)
            return qid, answer
        except Exception as e2:
            print(f"[Question {qid}] Fallback failed:", e2)
            traceback.print_exc()
            return qid, None


pred_answers = {}
output_dir = "evaluation/data/hotpot/output"
os.makedirs(output_dir, exist_ok=True)
pred_path = os.path.join(output_dir, "dev_distractor_pred.json")
gold_path = os.path.join(output_dir, "dev_distractor_gold.json")


def write_gold(data, out_path: str | None = None):
    split = data.get("validation")
    items_list = [split[i] for i in range(len(split))]
    out = []
    for it in items_list:
        qid = it.get("_id") or it.get("id")
        sp = it.get("supporting_facts")
        if isinstance(sp, dict):
            titles = sp.get("title") or []
            sent_ids = sp.get("sent_id") or []
            sp_list = [[t, s] for t, s in zip(titles, sent_ids, strict=False)]
        else:
            sp_list = sp or []
        ctx = it.get("context")
        if isinstance(ctx, dict):
            titles = ctx.get("title") or []
            sentences = ctx.get("sentences") or []
            ctx_list = [[t, s] for t, s in zip(titles, sentences, strict=False)]
        else:
            ctx_list = ctx or []
        out.append(
            {
                "_id": qid,
                "question": it.get("question"),
                "answer": it.get("answer"),
                "supporting_facts": sp_list,
                "context": ctx_list,
            }
        )
    target_path = out_path or gold_path
    tmp_path = target_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)
    except Exception as e:
        print("保存gold失败:", e)


def run_eval(pred_file: str | None = None, gold_file: str | None = None):
    spec = importlib.util.spec_from_file_location(
        "hotpot_eval_v1", "evaluation/scripts/hotpot/hotpot_evaluate_v1.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.eval(pred_file or pred_path, gold_file or gold_path)


def save_pred():
    tmp_path = pred_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"answer": pred_answers, "sp": {}}, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, pred_path)
    except Exception as e:
        print("保存失败:", e)


def main():
    interval = 20
    split = data.get("validation")
    items_list = [split[i] for i in range(10)]
    write_gold(data)

    if os.path.exists(pred_path):
        try:
            with open(pred_path, encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict) and isinstance(prev.get("answer"), dict):
                pred_answers.update(prev["answer"])
        except Exception as e:
            print("读取历史预测失败:", e)

    processed = len(pred_answers)
    print("开始评估，总样本:", len(items_list))
    print("已存在预测:", processed)

    pending_items = []
    for it in items_list:
        qid = it.get("_id") or it.get("id")
        if qid not in pred_answers:
            pending_items.append(it)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(build_and_ask, item): idx for idx, item in enumerate(pending_items)
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                qid, answer = future.result()
            except Exception as e:
                print("线程执行失败:", e)
                traceback.print_exc()
                continue
            pred_answers[qid] = answer
            processed += 1
            if processed % 10 == 0:
                print("已完成:", processed, "剩余:", len(items_list) - processed)
            save_pred()
            if processed % interval == 0:
                print("阶段评估，当前进度:", processed)
                run_eval()

    run_eval()


if __name__ == "__main__":
    main()
