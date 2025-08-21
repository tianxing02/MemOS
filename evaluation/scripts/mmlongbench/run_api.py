import argparse
import json
import os
import re
import signal
import sys
import threading
import time
import uuid

from concurrent.futures import ThreadPoolExecutor, as_completed

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
class ProcessingState:
    def __init__(self):
        self.lock = threading.Lock()
        self.completed_docs = set()
        self.completed_samples = set()
        self.skipped = 0
        self.doc_user_map = {}  # Map document name to user_name
        self.doc_temp_dir = {}  # Map document name to temp directory

    def mark_doc_completed(self, doc_id):
        with self.lock:
            self.completed_docs.add(doc_id)

    def is_doc_completed(self, doc_id):
        with self.lock:
            return doc_id in self.completed_docs

    def mark_sample_completed(self, doc_id, sample_id):
        with self.lock:
            self.completed_samples.add((doc_id, sample_id))

    def is_sample_completed(self, doc_id, sample_id):
        with self.lock:
            return (doc_id, sample_id) in self.completed_samples

    def increment_skipped(self):
        with self.lock:
            self.skipped += 1

    def get_skipped(self):
        with self.lock:
            return self.skipped

    def add_doc_user_mapping(self, doc_name, user_name, temp_dir):
        with self.lock:
            self.doc_user_map[doc_name] = user_name
            self.doc_temp_dir[doc_name] = temp_dir

    def get_user_name_for_doc(self, doc_name):
        with self.lock:
            return self.doc_user_map.get(doc_name)

    def get_temp_dir_for_doc(self, doc_name):
        with self.lock:
            return self.doc_temp_dir.get(doc_name)

    def to_dict(self):
        """Convert state to dictionary for JSON serialization"""
        with self.lock:
            return {
                "completed_docs": list(self.completed_docs),
                "completed_samples": [{"doc": d, "sample": s} for d, s in self.completed_samples],
                "skipped": self.skipped,
                "doc_user_map": self.doc_user_map,
                "doc_temp_dir": self.doc_temp_dir,
            }

    def from_dict(self, data):
        """Load state from dictionary"""
        with self.lock:
            self.completed_docs = set(data.get("completed_docs", []))
            self.completed_samples = set()
            for item in data.get("completed_samples", []):
                self.completed_samples.add((item["doc"], item["sample"]))
            self.skipped = data.get("skipped", 0)
            self.doc_user_map = data.get("doc_user_map", {})
            self.doc_temp_dir = data.get("doc_temp_dir", {})


# Initialize global state
state = ProcessingState()


# Graceful shutdown handler
def graceful_shutdown(signum, frame):
    """Handle interrupt signals and exit gracefully"""
    print("\n🛑 Received interrupt signal. Saving progress and exiting...")
    save_test_results(args.test_results_path, test_results)
    save_doc_processing_state(args.doc_state_path, state.to_dict())
    sys.exit(1)


# Register signal handlers
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)


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
        state.add_doc_user_mapping(doc_name, user_name, temp_dir)

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
    if sample["evidence_sources"] != "['Pure-text (Plain-text)']":
        state.increment_skipped()
        return None

    doc_id = os.path.basename(doc_file)
    sample_id = sample.get("question_id", hash(sample["question"]))

    # Skip already completed samples
    if state.is_sample_completed(doc_id, sample_id):
        state.increment_skipped()
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
    state.mark_sample_completed(doc_id, sample_id)
    return sample


def import_document_task(doc_file, args):
    """Task for importing a single document"""
    doc_path = os.path.join(args.document_path, doc_file)
    user_name, temp_dir = import_document(doc_path)
    return doc_file, user_name, temp_dir


def import_all_documents(args):
    """Import all documents concurrently and save user_name mappings"""
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
        state.from_dict(state_data)
        print(
            f"🔍 Loaded existing document processing state for {len(state.doc_user_map)} documents"
        )

    # Import new documents
    new_docs = [doc for doc in doc_files[:20] if doc not in state.doc_user_map]

    if not new_docs:
        print("✅ All documents already imported")
        return

    print(f"📂 Importing {len(new_docs)} new documents")

    # Use ThreadPoolExecutor for concurrent imports
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(import_document_task, doc_file, args): doc_file for doc_file in new_docs
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Importing documents"):
            doc_file = futures[future]
            print(f"📂 Importing document：{doc_file}")
            try:
                doc_file, user_name, temp_dir = future.result()
                if user_name and temp_dir:
                    print(f"✅ Imported {doc_file} with user {user_name}")
                    # Save state after each document
                    save_doc_processing_state(args.doc_state_path, state.to_dict())
                else:
                    print(f"❌ Failed to import {doc_file}")
            except Exception as e:
                print(f"❌ Error importing {doc_file}: {e!s}")

    print("✅ Document import completed")


def process_document_with_samples(doc_file, args, samples, prompt):
    """Process one document and all its samples concurrently"""
    # Skip already completed documents
    if state.is_doc_completed(doc_file):
        print(f"⏭️ Skipping already processed document: {doc_file}")
        return []

    print(f"🔄 Processing document: {doc_file}")

    # Get user_name and temp_dir for this document
    user_name = state.get_user_name_for_doc(doc_file)
    temp_dir = state.get_temp_dir_for_doc(doc_file)
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
    # Process samples concurrently
    with ThreadPoolExecutor(max_workers=args.max_concurrent_samples) as executor:
        futures = [
            executor.submit(process_sample, sample, mos, prompt, args.max_try, doc_file)
            for sample in doc_samples
            if not state.is_sample_completed(
                doc_file, sample.get("question_id", hash(sample["question"]))
            )
        ]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"Processing samples in {doc_file}"
        ):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"⚠️ Error processing sample: {e!s}")

    # Mark document as completed
    state.mark_doc_completed(doc_file)
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
                    state.mark_doc_completed(doc_file)
                    state.mark_sample_completed(doc_file, sample_id)

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
    # Load existing test results
    test_results = load_test_results(args.test_results_path)

    # Load document processing state
    if os.path.exists(args.doc_state_path):
        state_data = load_doc_processing_state(args.doc_state_path)
        state.from_dict(state_data)
        print(f"🔍 Loaded document processing state for {len(state.doc_user_map)} documents")

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
    print(f"⏭️ {state.get_skipped()} samples skipped due to filtering")
    print("=" * 80 + "\n")

    # Process documents concurrently
    completed = 0
    with ThreadPoolExecutor(max_workers=min(args.max_concurrent_docs, len(doc_files))) as executor:
        future_to_doc = {
            executor.submit(
                process_document_with_samples, doc_file, args, original_samples, prompt
            ): doc_file
            for doc_file in doc_files
        }

        for future in tqdm(
            as_completed(future_to_doc), total=len(future_to_doc), desc="Processing documents"
        ):
            doc_file = future_to_doc[future]
            try:
                doc_results = future.result()
                if doc_results:
                    test_results.extend(doc_results)
                    completed += 1

                    # Save test results after each document
                    save_test_results(args.test_results_path, test_results)

                    # Save document processing state
                    save_doc_processing_state(args.doc_state_path, state.to_dict())

                    # Calculate metrics
                    processed_samples = [s for s in test_results if "score" in s]
                    if processed_samples:
                        acc, f1 = eval_acc_and_f1(processed_samples)
                        print("\n" + "=" * 80)
                        print(f"📈 Cumulative Metrics After Document: {doc_file}")
                        print(f"✅ Processed: {completed}/{len(doc_files)} documents")
                        print(
                            f"🧪 Processed: {len(processed_samples)}/{len(original_samples)} samples"
                        )
                        print(f"🎯 Accuracy: {acc:.4f}")
                        print(f"📊 F1 Score: {f1:.4f}")
                        print("=" * 80 + "\n")
            except Exception as e:
                print(f"⛔ Unhandled error processing {doc_file}: {e!s}")
                # Try to save current progress on error
                save_test_results(args.test_results_path, test_results)
                save_doc_processing_state(args.doc_state_path, state.to_dict())

    # Final evaluation
    if test_results:
        print("\n" + "=" * 80)
        print("🏆 FINAL RESULTS")
        print("=" * 80)

        processed_samples = [s for s in test_results if "score" in s]
        acc, f1 = eval_acc_and_f1(processed_samples)

        print(f"✅ Completed: {completed}/{len(doc_files)} documents")
        print(f"🧪 Completed: {len(processed_samples)}/{len(original_samples)} samples")
        print(f"⏭️ Skipped: {state.get_skipped()} samples (filtered)")
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
    parser.add_argument(
        "--max_concurrent_docs",
        type=int,
        default=5,
        help="Maximum number of documents to process concurrently",
    )
    parser.add_argument(
        "--max_concurrent_samples",
        type=int,
        default=4,
        help="Maximum number of samples to process concurrently per document",
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
        save_doc_processing_state(args.doc_state_path, state.to_dict())
        sys.exit(1)
