import urllib.error
import urllib.request

from pathlib import Path


BASE_URL = "https://memos-knowledge-base-file-pre.oss-cn-shanghai.aliyuncs.com/evaluation_datasets/"
TARGET_DIR = Path("evaluation/data")

DATASETS = {
    "hotpot": ["dev_distractor_gold.json"],
    "longmemeval": ["longmemeval_s.json"],
    "locomo": ["locomo10.json", "locomo10_rag.json"],
    "personamem": ["questions_32k.csv", "shared_contexts_32k.jsonl"],
    "mmlongbench": ["pdf_file_list.txt", "md_file_list.txt", "samples.json"],
    "longbench_v2": ["longbenchv2_train.json"],
    "prefeval": ["train.jsonl", "pref_processed.jsonl"],
}


def check_and_download(url, local_path):
    if local_path.exists():
        print(f"✅ Exists: {local_path}")
        return

    print(f"Checking {url}...")
    try:
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                print(f"Downloading to {local_path}...")
                with open(local_path, "wb") as f:
                    f.write(response.read())
                print("✅ Success")
            else:
                print(f"⚠️  Skipping (Status {response.status})")
    except urllib.error.HTTPError as e:
        print(f"❌ Failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Target Directory: {TARGET_DIR.absolute()}")
    print(f"Base URL: {BASE_URL}")
    print("=" * 60)

    for folder, files in DATASETS.items():
        folder_path = TARGET_DIR / folder
        folder_path.mkdir(exist_ok=True)

        for filename in files:
            url = f"{BASE_URL}{folder}/{filename}"
            local_path = folder_path / filename
            check_and_download(url, local_path)


if __name__ == "__main__":
    main()
