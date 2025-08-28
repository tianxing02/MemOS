import argparse
import json
import os
import re
import signal
import sys
import time
import uuid

from dotenv import load_dotenv
from eval.eval_score import eval_acc_and_f1, eval_score, show_results
from eval.extract_answer import extract_answer
from tqdm import tqdm

from memos.configs.mem_cube import GeneralMemCubeConfig
from memos.configs.mem_os import MOSConfig
from memos.mem_cube.general import GeneralMemCube
from memos.mem_os.main import MOS


load_dotenv()
openapi_config = {
    "model_name_or_path": "gpt-4o-mini",
    "temperature": 0.8,
    "max_tokens": 1024,
    "top_p": 0.9,
    "top_k": 50,
    "remove_think_prefix": True,
    "api_key": os.getenv("OPENAI_API_KEY", "sk-xxxxx"),
    "api_base": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
}
neo4j_uri = os.getenv("NEO4J_URI", "bolt://47.117.41.207:7687")
db_name = "mm-long-bench-single-"

# Global state tracking
completed_docs = set()
completed_samples = set()
skipped = 0
doc_user_map = {}  # Map document name to user_name
doc_temp_dir = {}  # Map document name to temp directory


# Graceful shutdown handler
def graceful_shutdown(signum, frame):
    """Handle interrupt signals and exit gracefully"""
    print("\n🛑 Received interrupt signal. Saving progress and exiting...")
    save_test_results(args.test_results_path, test_results)
    save_doc_processing_state(args.doc_state_path, get_state_dict())
    sys.exit(1)


# Register signal handlers
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)


def get_state_dict():
    """Convert state to dictionary for JSON serialization"""
    return {
        "completed_docs": list(completed_docs),
        "completed_samples": [{"doc": d, "sample": s} for d, s in completed_samples],
        "skipped": skipped,
        "doc_user_map": doc_user_map,
        "doc_temp_dir": doc_temp_dir,
    }


def load_state_dict(data):
    """Load state from dictionary"""
    global completed_docs, completed_samples, skipped, doc_user_map, doc_temp_dir
    completed_docs = set(data.get("completed_docs", []))
    completed_samples = set()
    for item in data.get("completed_samples", []):
        completed_samples.add((item["doc"], item["sample"]))
    skipped = data.get("skipped", 0)
    doc_user_map = data.get("doc_user_map", {})
    doc_temp_dir = data.get("doc_temp_dir", {})


def import_document(doc_path):
    """Import a single document and return user_name and temp_dir"""
    try:
        user_name = str(uuid.uuid4())
        config = {
            "user_id": user_name,
            "chat_model": {
                "backend": "openai",
                "config": openapi_config,
            },
            "mem_reader": {
                "backend": "simple_struct",
                "config": {
                    "llm": {"backend": "openai", "config": openapi_config},
                    "embedder": {
                        "backend": "universal_api",
                        "config": {
                            "provider": "openai",
                            "api_key": os.getenv("OPENAI_API_KEY", "sk-xxxxx"),
                            "model_name_or_path": "text-embedding-3-large",
                            "base_url": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
                        },
                    },
                    "chunker": {
                        "backend": "sentence",
                        "config": {
                            "tokenizer_or_token_counter": "gpt2",
                            "chunk_size": 512,
                            "chunk_overlap": 128,
                            "min_sentences_per_chunk": 1,
                        },
                    },
                },
            },
            "max_turns_window": 20,
            "top_k": 5,
            "enable_textual_memory": True,
            "enable_activation_memory": False,
            "enable_parametric_memory": False,
        }
        mos_config = MOSConfig(**config)
        mos = MOS(mos_config)

        mem_cube_config = GeneralMemCubeConfig.model_validate(
            {
                "user_id": user_name,
                "cube_id": user_name,
                "text_mem": {
                    "backend": "tree_text",
                    "config": {
                        "extractor_llm": {"backend": "openai", "config": openapi_config},
                        "dispatcher_llm": {"backend": "openai", "config": openapi_config},
                        "graph_db": {
                            "backend": "neo4j",
                            "config": {
                                "uri": neo4j_uri,
                                "user": "neo4j",
                                "password": "iaarlichunyu",
                                "db_name": db_name + user_name[:8],
                                "auto_create": True,
                            },
                        },
                        "embedder": {
                            "backend": "universal_api",
                            "config": {
                                "provider": "openai",
                                "api_key": os.getenv("OPENAI_API_KEY", "sk-xxxxx"),
                                "model_name_or_path": "text-embedding-3-large",
                                "base_url": os.getenv(
                                    "OPENAI_API_BASE", "https://api.openai.com/v1"
                                ),
                            },
                        },
                        "reorganize": True,
                    },
                },
                "act_mem": {},
                "para_mem": {},
            }
        )
        mem_cube = GeneralMemCube(mem_cube_config)
        temp_dir = f"/tmp/{user_name}"
        os.makedirs(temp_dir, exist_ok=True)
        mem_cube.dump(temp_dir)
        mos.register_mem_cube(temp_dir, mem_cube_id=user_name)
        mos.add(doc_path=[doc_path])

        # Save mapping
        doc_name = os.path.basename(doc_path)
        doc_user_map[doc_name] = user_name
        doc_temp_dir[doc_name] = temp_dir

        return user_name, temp_dir
    except Exception as e:
        print(f"❌ Failed to import document {os.path.basename(doc_path)}: {e!s}")
        return None, None


def create_mos_for_doc(user_name, temp_dir):
    """Create MOS instance for a document using saved user_name and temp_dir"""
    try:
        config = {
            "user_id": user_name,
            "chat_model": {
                "backend": "openai",
                "config": openapi_config,
            },
            "mem_reader": {
                "backend": "simple_struct",
                "config": {
                    "llm": {"backend": "openai", "config": openapi_config},
                    "embedder": {
                        "backend": "universal_api",
                        "config": {
                            "provider": "openai",
                            "api_key": os.getenv("OPENAI_API_KEY", "sk-xxxxx"),
                            "model_name_or_path": "text-embedding-3-large",
                            "base_url": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
                        },
                    },
                    "chunker": {
                        "backend": "sentence",
                        "config": {
                            "tokenizer_or_token_counter": "gpt2",
                            "chunk_size": 512,
                            "chunk_overlap": 128,
                            "min_sentences_per_chunk": 1,
                        },
                    },
                },
            },
            "max_turns_window": 20,
            "top_k": 5,
            "enable_textual_memory": True,
            "enable_activation_memory": False,
            "enable_parametric_memory": False,
        }
        mos_config = MOSConfig(**config)
        mos = MOS(mos_config)
        mos.register_mem_cube(temp_dir, mem_cube_id=user_name)
        return mos
    except Exception as e:
        print(f"❌ Failed to create MOS for user {user_name}: {e!s}")
        return None


# Improved answer extraction with better error handling
def extract_answer_safe(question, response, prompt):
    """Robust answer extraction with fallback handling"""
    try:
        return extract_answer(question, response, prompt)
    except Exception as e:
        print(f"⚠️ Answer extraction failed: {e!s}")
        # Fallback: return the entire response if extraction fails
        return response


def process_sample(sample, mos, prompt, max_try, doc_file):
    """Process a single sample with retry and evaluation"""
    global skipped
    if sample["evidence_sources"] != "['Pure-text (Plain-text)']":
        skipped += 1
        return None

    doc_id = os.path.basename(doc_file)
    sample_id = sample.get("question_id", hash(sample["question"]))

    # Skip already completed samples
    if (doc_id, sample_id) in completed_samples:
        skipped += 1
        return None

    try_cnt = 0
    response = ""
    while try_cnt <= max_try:
        try:
            mos.clear_messages()
            response = mos.chat(sample["question"])
            break
        except Exception as e:
            try_cnt += 1
            response = f"Error: {e!s}"
            if try_cnt > max_try:
                break
            # Exponential backoff
            sleep_time = min(2**try_cnt, 30)  # Cap at 30 seconds
            print(f"⚠️ Retrying sample {sample_id} (attempt {try_cnt}/{max_try}) in {sleep_time}s")
            time.sleep(sleep_time)

    sample["response"] = response
    sample["doc_file"] = doc_file  # Track which doc it came from

    if "Error:" not in response:
        extracted_res = extract_answer_safe(sample["question"], response, prompt)
        sample["extracted_res"] = extracted_res
        try:
            # Handle different extraction formats
            if "Extracted answer:" in extracted_res:
                pred_ans = (
                    extracted_res.split("Answer format:")[0].split("Extracted answer:")[1].strip()
                )
            else:
                pred_ans = extracted_res.strip()

            score = eval_score(sample["answer"], pred_ans, sample["answer_format"])
            sample["pred"] = pred_ans
            sample["score"] = score
        except Exception as e:
            print(f"⚠️ Evaluation failed for sample {sample_id}: {e!s}")
            sample["pred"] = ""
            sample["score"] = 0
    else:
        sample["pred"] = ""
        sample["score"] = 0

    # Print detailed results
    print("\n" + "-" * 80)
    print(f"📄 Document: {doc_file}")
    print(f"📌 Sample ID: {sample_id}")
    print(f"❓ Question: {sample['question']}")
    print(f"💬 Response: {response[:500]}{'...' if len(response) > 500 else ''}")
    print(f"🎯 Ground Truth: {sample['answer']}")
    print(f"📦 Extracted: {sample.get('pred', '')}")
    print(f"⭐ Score: {sample.get('score', 0)}")
    print("-" * 80 + "\n")

    # Mark sample as completed
    completed_samples.add((doc_id, sample_id))
    return sample


def import_all_documents(args):
    """Import all documents sequentially"""
    print("\n" + "=" * 80)
    print("🚀 Starting document import phase")
    print("=" * 80)

    # Find all document files
    doc_files = [
        f
        for f in os.listdir(args.document_path)
        if os.path.isfile(os.path.join(args.document_path, f))
    ]

    # Load existing mappings if available
    if os.path.exists(args.doc_state_path):
        state_data = load_doc_processing_state(args.doc_state_path)
        load_state_dict(state_data)
        print(f"🔍 Loaded existing document processing state for {len(doc_user_map)} documents")

    # Import new documents
    new_docs = [doc for doc in doc_files if doc not in doc_user_map]

    if not new_docs:
        print("✅ All documents already imported")
        return

    print(f"📂 Importing {len(new_docs)} new documents")

    for doc_file in tqdm(new_docs, desc="Importing documents"):
        print(f"📂 Importing document：{doc_file}")
        doc_path = os.path.join(args.document_path, doc_file)
        user_name, temp_dir = import_document(doc_path)
        if user_name and temp_dir:
            print(f"✅ Imported {doc_file} with user {user_name}")
            # Save state after each document
            save_doc_processing_state(args.doc_state_path, get_state_dict())
        else:
            print(f"❌ Failed to import {doc_file}")

    print("✅ Document import completed")


def process_document_with_samples(doc_file, args, samples, prompt):
    """Process one document and all its samples sequentially"""
    global skipped
    # Skip already completed documents
    if doc_file in completed_docs:
        print(f"⏭️ Skipping already processed document: {doc_file}")
        return []

    print(f"🔄 Processing document: {doc_file}")

    # Get user_name and temp_dir for this document
    user_name = doc_user_map.get(doc_file)
    temp_dir = doc_temp_dir.get(doc_file)
    if not user_name or not temp_dir:
        print(f"❌ No user mapping found for {doc_file}. Skipping.")
        return []

    # Create MOS instance for this document
    mos = create_mos_for_doc(user_name, temp_dir)
    if not mos:
        print(f"❌ Failed to create MOS for {doc_file}. Skipping.")
        return []

    doc_id = os.path.basename(doc_file)
    doc_samples = [s for s in samples if s.get("doc_id") == doc_id]

    results = []
    for sample in tqdm(doc_samples, desc=f"Processing samples in {doc_file}"):
        sample_id = sample.get("question_id", hash(sample["question"]))
        if (doc_file, sample_id) in completed_samples:
            skipped += 1
            continue

        try:
            result = process_sample(sample, mos, prompt, args.max_try, doc_file)
            if result:
                results.append(result)
        except Exception as e:
            print(f"⚠️ Error processing sample: {e!s}")

    # Mark document as completed
    completed_docs.add(doc_file)
    return results


def load_test_results(output_path):
    """Load existing test results from output file"""
    test_results = []
    if os.path.exists(output_path):
        print(f"🔍 Found existing test results file: {output_path}")
        try:
            with open(output_path) as f:
                test_results = json.load(f)

            # Rebuild progress state from loaded samples
            for sample in test_results:
                if "doc_file" in sample:
                    doc_file = sample["doc_file"]
                    sample_id = sample.get("question_id", hash(sample["question"]))
                    completed_docs.add(doc_file)
                    completed_samples.add((doc_file, sample_id))

            processed_docs = len({s["doc_file"] for s in test_results if "doc_file" in s})
            print(f"📊 Loaded existing test results: {len(test_results)} samples")
            print(f"📂 {processed_docs} documents already processed")
        except Exception as e:
            print(f"⚠️ Failed to load test results file: {e!s}")
            # Clear the broken file
            os.remove(output_path)
            print("🚮 Removed corrupted test results file")
    return test_results


def save_test_results(output_path, test_results):
    """Save current test results to output file"""
    print(f"💾 Auto-saving test results to {output_path}")
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save with pretty formatting
        with open(output_path, "w") as f:
            json.dump(test_results, f, indent=2)
        return True
    except Exception as e:
        print(f"❌ Failed to save test results: {e!s}")
        return False


def save_doc_processing_state(map_path, state_data):
    """Save document processing state to JSON file"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(map_path), exist_ok=True)

        # Save as JSON
        with open(map_path, "w") as f:
            json.dump(state_data, f, indent=2)
        return True
    except Exception as e:
        print(f"❌ Failed to save document processing state: {e!s}")
        return False


def load_doc_processing_state(map_path):
    """Load document processing state from JSON file"""
    if os.path.exists(map_path):
        try:
            with open(map_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load document processing state: {e!s}")
            return {}
    return {}


def main(args):
    global test_results

    # Load document processing state
    if os.path.exists(args.doc_state_path):
        state_data = load_doc_processing_state(args.doc_state_path)
        load_state_dict(state_data)
        print(f"🔍 Loaded document processing state for {len(doc_user_map)} documents")

    # Import all documents first
    import_all_documents(args)

    with open(args.extractor_prompt_path) as f:
        prompt = f.read()

    # Load all input samples (original data)
    with open(args.input_path) as f:
        original_samples = json.load(f)

    # Find all document files
    doc_files = [
        f
        for f in os.listdir(args.document_path)
        if os.path.isfile(os.path.join(args.document_path, f))
    ]

    print("\n" + "=" * 80)
    print(f"🚀 Starting sample processing with {len(doc_files)} documents")
    print(f"📂 {len(original_samples)} total samples in dataset")
    print(f"⏭️ {skipped} samples skipped due to filtering")
    print("=" * 80 + "\n")

    # Load existing test results
    test_results = load_test_results(args.test_results_path)

    # Process documents sequentially
    completed = 0
    for doc_file in tqdm(doc_files, desc="Processing documents"):
        try:
            doc_results = process_document_with_samples(doc_file, args, original_samples, prompt)
            if doc_results:
                test_results.extend(doc_results)
                completed += 1

                # Save test results after each document
                save_test_results(args.test_results_path, test_results)

                # Save document processing state
                save_doc_processing_state(args.doc_state_path, get_state_dict())

                # Calculate metrics
                processed_samples = [s for s in test_results if "score" in s]
                if processed_samples:
                    acc, f1 = eval_acc_and_f1(processed_samples)
                    print("\n" + "=" * 80)
                    print(f"📈 Cumulative Metrics After Document: {doc_file}")
                    print(f"✅ Processed: {completed}/{len(doc_files)} documents")
                    print(f"🧪 Processed: {len(processed_samples)}/{len(original_samples)} samples")
                    print(f"🎯 Accuracy: {acc:.4f}")
                    print(f"📊 F1 Score: {f1:.4f}")
                    print("=" * 80 + "\n")
        except Exception as e:
            print(f"⛔ Unhandled error processing {doc_file}: {e!s}")
            # Try to save current progress on error
            save_test_results(args.test_results_path, test_results)
            save_doc_processing_state(args.doc_state_path, get_state_dict())

    # Final evaluation
    if test_results:
        print("\n" + "=" * 80)
        print("🏆 FINAL RESULTS")
        print("=" * 80)

        processed_samples = [s for s in test_results if "score" in s]
        acc, f1 = eval_acc_and_f1(processed_samples)

        print(f"✅ Completed: {completed}/{len(doc_files)} documents")
        print(f"🧪 Completed: {len(processed_samples)}/{len(original_samples)} samples")
        print(f"⏭️ Skipped: {skipped} samples (filtered)")
        print(f"🎯 Final Accuracy: {acc:.4f}")
        print(f"📊 Final F1 Score: {f1:.4f}")
        print("=" * 80)

        # Generate detailed report
        report_path = re.sub("\.json$", "_report.txt", args.test_results_path)
        show_results(test_results, show_path=report_path)
        print(f"📝 Detailed report saved to: {report_path}")
    else:
        print("⚠️ No results to report. Processing may have failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, default="evaluation/data/mmlongbench/samples.json"
    )
    parser.add_argument(
        "--document_path", type=str, default="evaluation/data/mmlongbench/documents"
    )
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--max_pages", type=int, default=120)
    parser.add_argument("--resolution", type=int, default=144)
    parser.add_argument("--max_try", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--extractor_prompt_path",
        type=str,
        default="evaluation/scripts/mmlongbench/eval/prompt_for_answer_extraction.md",
    )
    parser.add_argument(
        "--doc_state_path",
        type=str,
        default="evaluation/data/mmlongbench/doc_processing_state.json",
    )
    parser.add_argument(
        "--test_results_path",
        type=str,
        default="evaluation/data/mmlongbench/test_results.json",
    )
    args = parser.parse_args()

    # Initialize test results list
    test_results = []

    try:
        main(args)
    except Exception as e:
        print(f"⛔ Critical error: {e!s}")
        print("💾 Attempting to save progress...")
        save_test_results(args.test_results_path, test_results)
        save_doc_processing_state(args.doc_state_path, get_state_dict())
        sys.exit(1)
