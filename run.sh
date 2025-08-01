#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
# Load variables from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

python rgfl/fl/localize.py \
  --file_level \
  --output_folder results/swe-bench-lite/file_level_irrelevant \
  --num_threads 10 \
  --skip_existing  --model=$MODEL --dataset=princeton-nlp/SWE-bench_Lite --backend=$BACKEND

python rgfl/fl/localize.py \
  --file_level \
  --irrelevant \
  --output_folder results/swe-bench-lite/file_level_irrelevant \
  --num_threads 10 \
  --skip_existing  --model=$MODEL --dataset=princeton-nlp/SWE-bench_Lite --backend=$BACKEND


python rgfl/fl/retrieve.py \
  --index_type simple \
  --filter_type given_files \
  --filter_file results/swe-bench-lite/file_level_irrelevant/loc_outputs.jsonl \
  --output_folder results/swe-bench-lite/retrievel_embedding \
  --persist_dir embedding/swe-bench_simple \
  --num_threads 10 --dataset=princeton-nlp/SWE-bench_Lite

python rgfl/fl/combine.py \
  --retrieval_loc_file results/swe-bench-lite/retrievel_embedding/retrieve_locs.jsonl \
  --model_loc_file results/swe-bench-lite/file_level/loc_outputs.jsonl \
  --top_n 3 \
  --output_folder results/swe-bench-lite/file_level_combined

