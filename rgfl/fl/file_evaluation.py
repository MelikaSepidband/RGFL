import json
import argparse
from collections import defaultdict
from datasets import load_dataset

# ----------- Modified File Extractor (Git Patch Parser) ----------- #
def extract_modified_files(patch_text):
    files = []
    for line in patch_text.splitlines():
        if line.startswith('diff --git'):
            path = line.split(' b/')[-1]
            files.append(path)
    return files

# ---------------------- Evaluation Function ------------------------ #
def evaluate(swebench, data, key='found_files', ks=[1, 2, 3, 5, 10], max_instances=None):
    hit_at_k = defaultdict(int)
    mrr_list = []
    recall_at_k = defaultdict(list)
    avg_rank_list = []

    count = min(len(swebench), max_instances or len(swebench))

    for i in range(count):
        example = swebench[i]
        instance_id = example['instance_id']
        patch_text = example['patch']
        oracle_files = extract_modified_files(patch_text)

        try:
            this1 = next(j for j in range(len(data)) if data[j]['instance_id'] == instance_id)
        except StopIteration:
            continue

        found_files = data[this1].get(key, [])
        positions = [found_files.index(f) if f in found_files else -1 for f in oracle_files]

        # MRR (first hit)
        if any(p >= 0 for p in positions):
            first_hit = min(p for p in positions if p >= 0)
            mrr_list.append(1.0 / (first_hit + 1))
        else:
            mrr_list.append(0.0)

        # Hit@K and Recall@K
        for k in ks:
            hits = [1 for p in positions if 0 <= p < k]
            if hits:
                hit_at_k[k] += 1
            recall_at_k[k].append(len(hits) / len(oracle_files))

        # Avg rank
        ranks = [p for p in positions if p >= 0]
        if ranks:
            avg_rank = sum(ranks) / len(ranks)
            avg_rank_list.append(avg_rank)

    # Summary
    results = {
        'Hit@K': {f'Hit@{k}': hit_at_k[k] / count for k in ks},
        'Recall@K': {f'Recall@{k}': sum(recall_at_k[k]) / count for k in ks},
        'MRR': sum(mrr_list) / count,
        'AverageRank': sum(avg_rank_list) / len(avg_rank_list) if avg_rank_list else None,
        'Total': count
    }

    return results

# -------------------------- CLI Setup ---------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fault Localization Results")
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--baseline_file", type=str, help="Path to JSON file for baseline (found_files)")
    parser.add_argument("--ours_file", type=str, help="Path to JSON file for our method (retrieved_files)")
    parser.add_argument("--max_instances", type=int, default=500, help="Max number of examples to evaluate")
    parser.add_argument("--target_id", type=str, help="Evaluate only a specific instance ID")

    args = parser.parse_args()

    swebench = load_dataset(args.dataset, split=args.split)
    if args.target_id:
        target_instance = next((ex for ex in swebench if ex["instance_id"] == args.target_id), None)
        if not target_instance:
            raise ValueError(f"Target instance ID {args.target_id} not found.")
        swebench = [target_instance]
    elif args.max_instances:
        swebench = swebench


    if args.baseline_file:
        with open(args.baseline_file, 'r', encoding='utf-8') as f:
            data_baseline = json.load(f)
        print("\n=== Baseline Evaluation (found_files) ===")
        baseline_results = evaluate(swebench, data_baseline, key='found_files', max_instances=args.max_instances)
        for k, v in baseline_results['Hit@K'].items():
            print(f"{k}: {v:.3f}")
        for k, v in baseline_results['Recall@K'].items():
            print(f"{k}: {v:.3f}")
        print(f"MRR: {baseline_results['MRR']:.3f}")
        if baseline_results['AverageRank'] is not None:
            print(f"Average Rank: {baseline_results['AverageRank']:.2f}")
        else:
            print("Average Rank: N/A")


    if args.ours_file:
        with open(args.ours_file, 'r', encoding='utf-8') as f:
            data_ours = json.load(f)
        print("\n=== Our Method Evaluation (retrieved_files) ===")
        ours_results = evaluate(swebench, data_ours, key='retrieved_files', max_instances=args.max_instances)
        for k, v in ours_results['Hit@K'].items():
            print(f"{k}: {v:.3f}")
        for k, v in ours_results['Recall@K'].items():
            print(f"{k}: {v:.3f}")
        print(f"MRR: {ours_results['MRR']:.3f}")
        if ours_results['AverageRank'] is not None:
            print(f"Average Rank: {ours_results['AverageRank']:.2f}")
        else:
            print("Average Rank: N/A")

