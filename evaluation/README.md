# Evaluation Memory Framework

This repository provides tools and scripts for evaluating the LoCoMo and LongMemEval dataset using various models and APIs.

## Installation

1. Set the `PYTHONPATH` environment variable:
   ```bash
   export PYTHONPATH=../src
   cd evaluation
   ```

2. Install the required dependencies:
   ```bash
   poetry install --with eval
   ```

## Configuration

Create an `.env` file in the `evaluation/` directory and include the following environment variables:

```plaintext
OPENAI_API_KEY="sk-xxx"
OPENAI_BASE_URL="your_base_url"

MEM0_API_KEY="your_mem0_api_key"
MEM0_PROJECT_ID="your_mem0_proj_id"
MEM0_ORGANIZATION_ID="your_mem0_org_id"

MODEL="gpt-4o-mini"  # or your preferred model
EMBEDDING_MODEL="text-embedding-3-small"  # or your preferred embedding model
ZEP_API_KEY="your_zep_api_key"
```

## Dataset
The smaller dataset "LoCoMo" has already been included in the repo to facilitate reproducing.

To download the "LongMemEval" dataset, run the following command:
```bash
huggingface-cli download --repo-type dataset --resume-download xiaowu0162/longmemeval --local-dir data/longmemeval
```

After downloading, rename the files as follows:
- `longmemeval_m.json`
- `longmemeval_s.json`
- `longmemeval_oracle.json`

## Evaluation Scripts

To evaluate the `locomo` dataset, execute the following scripts in order:

1. **Ingest locomo history into MemOS:**
   ```bash
   python scripts/locomo/locomo_ingestion.py --lib memos
   ```

2. **Search Memory for each QA pair in locomo:**
   ```bash
   python scripts/locomo/locomo_search.py --lib memos
   ```

3. **Generate responses from OpenAI with provided context:**
   ```bash
   python scripts/locomo/locomo_responses.py --lib memos
   ```

4. **Evaluate the generated answers:**
   ```bash
   python scripts/locomo/locomo_eval.py --lib memos
   ```

5. **Calculate fine-grained scores for each category:**
   ```bash
   python scripts/locomo/locomo_metric.py --lib memos
   ```

## Contributing Guidelines

1. **Add New Metrics**
When incorporating the evaluation of reflection duration, ensure to record related data in `{lib}_locomo_judged.json`. For additional NLP metrics like BLEU and ROUGE-L score, make adjustments to the `locomo_grader` function in `scripts/locomo/locomo_eval.py`.

2. **Intermediate Results**
While I have provided intermediate results like `{lib}_locomo_search_results.json`, `{lib}_locomo_responses.json`, and `{lib}_locomo_judged.json` for reproducibility, contributors are encouraged to report final results in the PR description rather than editing these files directly. Any valuable modifications will be combined into an updated version of the evaluation code containing revised intermediate results (at specified intervals).
