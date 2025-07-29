import os
import re
import ast
import json
import argparse
import requests
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
from unidiff import PatchSet

# ----------------- Element Normalization ----------------- #

def normalize_element(elem):
    if not isinstance(elem, str) or ':' not in elem:
        return []
    kind, name = elem.split(':', 1)
    kind = kind.strip()
    name = name.strip()
    if '.' in name and kind == 'function':
        class_name, func_name = name.split('.', 1)
        return [f'function: {func_name}', f'class: {class_name}']
    return [f'{kind}: {name}']

# ----------------- Patch + AST Extraction ----------------- #

def get_repo_file(repo: str, commit: str, filepath: str) -> str | None:
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{filepath}"
    try:
        response = requests.get(url)
        return response.text if response.status_code == 200 else None
    except:
        return None

def extract_code_elements_from_source(source_code: str):
    tree = ast.parse(source_code)
    code_elements = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            code_elements.append(('function', node.name, node.lineno, getattr(node, 'end_lineno', node.lineno)))
            self.generic_visit(node)
        def visit_ClassDef(self, node):
            code_elements.append(('class', node.name, node.lineno, getattr(node, 'end_lineno', node.lineno)))
            self.generic_visit(node)
        def visit_Assign(self, node):
            if isinstance(getattr(node, 'parent', None), ast.Module):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        code_elements.append(('global', target.id, node.lineno, getattr(node, 'end_lineno', node.lineno)))
            self.generic_visit(node)

    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    Visitor().visit(tree)
    return code_elements

def extract_ground_truth_elements(patch_text: str, repo: str, commit: str) -> list[str]:
    patch = PatchSet(patch_text)
    results = []

    def get_changed_lines(patch_file) -> set[int]:
        lines = set()
        for hunk in patch_file:
            lineno = hunk.target_start
            for line in hunk:
                if line.is_added or line.is_removed:
                    lines.add(lineno)
                if not line.is_removed:
                    lineno += 1
        return lines

    for file in patch:
        filepath = file.path[2:] if file.path.startswith("b/") else file.path
        changed_lines = get_changed_lines(file)
        source_code = get_repo_file(repo, commit, filepath)
        if not source_code:
            continue
        for kind, name, start, end in extract_code_elements_from_source(source_code):
            if any(start <= line <= end for line in changed_lines):
                results.append(f"{kind}: {name}")
    return sorted(set(results))

# ----------------- Evaluation ----------------- #

def evaluate_all_methods(samples, k_values=[1, 3, 5, 10, 20, 50, 100]):
    results = {'baseline': defaultdict(list), 'new': defaultdict(list)}
    for sample in samples:
        gt = set(e for raw in sample['ground_truth'] for e in normalize_element(raw))
        pred_baseline = [e for raw in sample['baseline'] for e in normalize_element(raw)]
        pred_new = [e for raw in sample['new'] for e in normalize_element(raw)]

        for method_name, prediction in [('baseline', pred_baseline), ('new', pred_new)]:
            pred_set = set(prediction)

            results[method_name]['jaccard'].append(len(gt & pred_set) / len(gt | pred_set) if gt | pred_set else 1.0)
            results[method_name]['exact_match'].append(int(gt <= pred_set))
            results[method_name]['partial_match'].append(int(bool(gt & pred_set)))

            for k in k_values:
                topk = set(prediction[:k])
                recall = len(gt & topk) / len(gt) if gt else 0
                hit = int(bool(gt & topk))
                results[method_name][f'recall@{k}'].append(recall)
                results[method_name][f'hit@{k}'].append(hit)

    summary = {
        method: {
            metric: round(sum(vals) / len(vals), 4) for metric, vals in result.items()
        }
        for method, result in results.items()
    }
    return summary

# ----------------- Main Evaluation Loop ----------------- #

def build_evaluation_samples(swebench, data_baseline, data_ranking_ours):
    samples, all_related_elements = [], []

    for i, example in enumerate(tqdm(swebench, desc="Building samples")):
        instance_id = example['instance_id']
        patch, repo, commit = example['patch'], example['repo'], example['base_commit']

        try:
            this1 = next(j for j in range(len(data_baseline)) if data_baseline[j]['instance_id'] == instance_id)
            elem_list1 = list(data_baseline[this1]['found_related_locs'].values())[0][0].split('\n') if 'found_related_locs' in data_baseline[this1] else []
            elem_list2 = list(data_baseline[this1]['found_related_locs'].values())[1][0].split('\n') if len(data_baseline[this1]['found_related_locs']) > 1 else []
            elem_list3 = list(data_baseline[this1]['found_related_locs'].values())[2][0].split('\n') if len(data_baseline[this1]['found_related_locs']) > 2 else []
        except:
            elem_list1 = elem_list2 = elem_list3 = []

        baseline = elem_list1 + elem_list2 + elem_list3
        all_related_elements.append(baseline)

        try:
            gt_list = extract_ground_truth_elements(patch, repo, commit)
        except Exception as e:
            print(f"[{instance_id}] GT extraction failed: {e}")
            continue

        try:
            this2 = next(j for j in range(len(data_ranking_ours)) if data_ranking_ours[j]['instance_id'] == instance_id)
            new_method = data_ranking_ours[this2].get('similar_elements_file1', []) + \
                         data_ranking_ours[this2].get('similar_elements_file2', []) + \
                         data_ranking_ours[this2].get('similar_elements_file3', [])
        except:
            new_method = []

        samples.append({
            'ground_truth': gt_list,
            'baseline': baseline,
            'new': new_method,
        })

    return samples

# ----------------- CLI Entry Point ----------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate element-level fault localization")
    parser.add_argument("--dataset", type=str, required=True, help="SWE-bench dataset (JSON or HF dataset name)")
    parser.add_argument("--data_baseline", type=str, required=True, help="Baseline file with found_related_locs")
    parser.add_argument("--data_ranking_ours", type=str, required=True, help="File with similar_elements_fileX keys")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--target_id", type=str, help="Optional: evaluate a single instance")

    args = parser.parse_args()

    print(f"Loading SWE-bench: {args.dataset}")
    if args.dataset.endswith(".json"):
        with open(args.dataset, "r", encoding="utf-8") as f:
            swebench = json.load(f)
    else:
        dataset_full = load_dataset(args.dataset, split="test")

        if args.target_id:
            swebench = [ex for ex in dataset_full if ex["instance_id"] == args.target_id]
            if not swebench:
                raise ValueError(f"Target ID {args.target_id} not found in dataset.")
        else:
            swebench = dataset_full


    print(f"Loading baseline: {args.data_baseline}")
    with open(args.data_baseline, "r", encoding="utf-8") as f:
        data_baseline = json.load(f)

    print(f"Loading new method: {args.data_ranking_ours}")
    with open(args.data_ranking_ours, "r", encoding="utf-8") as f:
        data_ranking_ours = json.load(f)

    samples = build_evaluation_samples(swebench, data_baseline, data_ranking_ours)
    results = evaluate_all_methods(samples)

    from pprint import pprint
    pprint(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {args.output}")
