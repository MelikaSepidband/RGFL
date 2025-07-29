# RGFL: Reasoning-Guided Fault Localization

## Setup

First, create the environment:
```bash
git clone https://github.com/MelikaSepidband/RGFL.git
cd RGFL

conda create -n rgfl python=3.11
conda activate rgfl
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Then export your API credentials:

**OpenAI**:
```bash
export OPENAI_API_KEY=your_openai_key_here
```

**Anthropic**:
```bash
export ANTHROPIC_API_KEY=your_anthropic_key_here
```

**Google Gemini (Vertex AI)**:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path_to_your_service_account_key.json"
export VERTEXAI_PROJECT="your_project_id"
export VERTEXAI_LOCATION="your_project_region"  # e.g. us-central1
```

Create a folder to save results:
```bash
mkdir results
```

---

### Tips

- Supported datasets: `SWE-bench_Lite`, `SWE-bench_Verified`
  - Use `--dataset=princeton-nlp/SWE-bench_Lite`
- To target a specific bug: `--target_id=django__django-10914`
- Use `--num_threads` to parallelize operations for speedup

---

## Localization Pipeline

### 1. Localize Suspicious Files (Agentless(baseline))

#### ✅ LLM-based File Localization
```bash
python rgfl/fl/localize.py \
  --file_level \
  --output_folder results/swe-bench-lite/file_level \
  --num_threads 10 \
  --skip_existing
```
Saves to:
```
results/swe-bench-lite/file_level/loc_outputs.jsonl
results/swe-bench-lite/file_level/localization_logs
```

#### ✅ Filter Irrelevant Files with LLM
```bash
python rgfl/fl/localize.py \
  --file_level \
  --irrelevant \
  --output_folder results/swe-bench-lite/file_level_irrelevant \
  --num_threads 10 \
  --skip_existing
```
Saves to:
```
results/swe-bench-lite/file_level_irrelevant/loc_outputs.jsonl
results/swe-bench-lite/file_level_irrelevant/localization_logs
```

#### ✅ Embedding-based Retrieval
```bash
python rgfl/fl/retrieve.py \
  --index_type simple \
  --filter_type given_files \
  --filter_file results/swe-bench-lite/file_level_irrelevant/loc_outputs.jsonl \
  --output_folder results/swe-bench-lite/retrievel_embedding \
  --persist_dir embedding/swe-bench_simple \
  --num_threads 10
```
Saves to:
```
results/swe-bench-lite/retrievel_embedding/retrieve_locs.jsonl
results/swe-bench-lite/retrievel_embedding/retrieval_logs
```

#### ✅ Merge Suspicious Files
```bash
python rgfl/fl/combine.py \
  --retrieval_loc_file results/swe-bench-lite/retrievel_embedding/retrieve_locs.jsonl \
  --model_loc_file results/swe-bench-lite/file_level/loc_outputs.jsonl \
  --top_n 3 \
  --output_folder results/swe-bench-lite/file_level_combined
```
Saves final file list to:
```
results/swe-bench-lite/file_level_combined/combined_locs.jsonl
```

---

### 2. Rerank Files by Reasoning

#### ✅ Generate Reasoning
```bash
python rgfl/fl/file_reasoning.py \
  --model gpt-4o \
  --backend openai \
  --dataset princeton-nlp/SWE-bench_Lite \
  --combined_locs results/swe-bench-lite/file_level_combined/combined_locs.jsonl \
  --output results/swe-bench-lite/file_reasoning_results.json
```

#### ✅ Rank Files
```bash
python rgfl/fl/file_ranking.py \
  --model gpt-4o \
  --backend openai \
  --dataset princeton-nlp/SWE-bench_Lite \
  --reasoning_file results/swe-bench-lite/file_reasoning_results.json \
  --output results/swe-bench-lite/file_ranking_results.json
```

#### ✅ Evaluate File Ranking
```bash
python rgfl/fl/file_evaluation.py \
  --baseline_file results/swe-bench-lite/file_level/loc_outputs.jsonl \
  --ours_file results/swe-bench-lite/file_ranking_results.json \
  --dataset princeton-nlp/SWE-bench_Lite
```

---

### 3. Localize to Related Elements (Agentless(baseline))
```bash
python rgfl/fl/localize.py \
  --related_level \
  --output_folder results/swe-bench-lite/related_elements \
  --top_n 3 \
  --compress_assign \
  --compress \
  --start_file results/swe-bench-lite/file_level_combined/combined_locs.jsonl \
  --num_threads 10 \
  --skip_existing
```
Saves to:
```
results/swe-bench-lite/related_elements/loc_outputs.jsonl
results/swe-bench-lite/related_elements/localization_logs
```

---

### 4. Rerank Elements by Reasoning

#### ✅ Generate Element-Level Reasoning
```bash
python rgfl/fl/element_reasoning.py \
  --model gpt-4o \
  --backend openai \
  --dataset princeton-nlp/SWE-bench_Lite \
  --input results/swe-bench-lite/file_ranking_results.json \
  --output results/swe-bench-lite/element_reasoning_results.json
```

#### ✅ Rank Elements
```bash
python rgfl/fl/element_ranking.py \
  --model gpt-4o \
  --backend openai \
  --dataset princeton-nlp/SWE-bench_Lite \
  --input results/swe-bench-lite/element_reasoning_results.json \
  --output results/swe-bench-lite/element_ranking_results.json
```

#### ✅ Evaluate Element Ranking
```bash
python rgfl/fl/element_evaluation.py \
  --dataset princeton-nlp/SWE-bench_Lite \
  --data_baseline results/swe-bench-lite/related_elements/loc_outputs.jsonl \
  --data_ranking_ours results/swe-bench-lite/element_ranking_results.json \
  --output results/swe-bench-lite/element_eval_summary.json
```
