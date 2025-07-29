import os
import json
import ast
import re
import argparse
import tempfile
import subprocess
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------- Utility Functions ----------------------- #

def clone_and_checkout(repo_name, commit):
    url = f"https://github.com/{repo_name}.git"
    temp_dir = tempfile.mkdtemp()
    subprocess.run(["git", "clone", "--quiet", url, temp_dir], check=True)
    subprocess.run(["git", "checkout", commit], cwd=temp_dir, check=True)
    return temp_dir

def extract_code_elements_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    code_elements = []

    class CodeVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            code_elements.append(('function', node.name, node.lineno, node.end_lineno))
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            code_elements.append(('class', node.name, node.lineno, node.end_lineno))
            self.generic_visit(node)

        def visit_Assign(self, node):
            if isinstance(node.parent, ast.Module):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        code_elements.append(('global', target.id, node.lineno, node.end_lineno))
            self.generic_visit(node)

    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    CodeVisitor().visit(tree)
    return code_elements, source

def get_source_code(element, file_source):
    source_lines = file_source.splitlines()
    return "\n".join(source_lines[element[2] - 1:element[3]])

# ------------------------- Reasoning LLM Call ---------------------- #

def get_reasoning(element_code, bug_report):
    prompt = f"""A user is trying to fix a bug described in the following report:\n\n{bug_report}\n\nBelow is a code element (a function, a class, or a global variable) in a file in a repository:\n\n{element_code}\n\nExplain the purpose and functionality of this code element in the context of the bug report. Focus on what this element does and whether it may be related to the bug."""

    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    elif backend == "gemini":
        from google import genai
        PROJECT_ID = os.environ.get("VERTEXAI_PROJECT", "")
        LOCATION = os.environ.get("VERTEXAI_LOCATION", "us-central1")
        client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        return response.text

    elif backend == "openai":
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return response.choices[0].message.content

# ------------------------- Element Reasoning ------------------------ #

def process_file(file_path, bug_report, source_code, elements):
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(get_reasoning, get_source_code(el, source_code), bug_report): el
            for el in elements
        }
        for future in as_completed(futures):
            el = futures[future]
            try:
                reasoning = future.result()
                results[f"{el[0]}: {el[1]}"] = reasoning
            except Exception as e:
                print(f"Error processing element {el[1]}: {e}")
                results[f"{el[0]}: {el[1]}"] = f"ERROR: {str(e)}"
    return results

# ------------------------- Argument Parsing ------------------------- #

parser = argparse.ArgumentParser(description="Element-Level Reasoning Generator")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--backend", type=str, choices=["anthropic", "gemini", "openai"], required=True)
parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Verified")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--input", type=str, required=True, help="JSON file with file-level retrieved files")
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--top_k", type=int, default=3, help="Top-K files per instance to analyze")
parser.add_argument("--target_id", type=str, help="Optional: instance_id to run only one task")

args = parser.parse_args()

MODEL_ID = args.model
backend = args.backend

# ---------------------------- Load Data ----------------------------- #

print(f"Loading dataset: {args.dataset}")
dataset_full = load_dataset(args.dataset, split=args.split)

if args.target_id:
    swebench = [ex for ex in dataset_full if ex["instance_id"] == args.target_id]
    if not swebench:
        raise ValueError(f"Target ID {args.target_id} not found.")
else:
    swebench = dataset_full


print(f"Loading file: {args.input}")
with open(args.input, "r", encoding="utf-8") as f:
    data4 = json.load(f)

# ----------------------------- Main Loop ---------------------------- #

for i, example in enumerate(swebench):
    instance_id = example['instance_id']
    bug_report = example['problem_statement']
    print(f"Processing {i}: {instance_id}")

    try:
        this1 = next(j for j in range(len(data4)) if data4[j]['instance_id'] == instance_id)
    except StopIteration:
        continue

    list_of_files = data4[this1].get("retrieved_files", [])[:args.top_k]
    try:
        repo_root = clone_and_checkout(example['repo'], example['base_commit'])
    except Exception as e:
        print(f"Error cloning repo: {e}")
        continue

    for idx, file in enumerate(list_of_files):
        abs_path = os.path.join(repo_root, file.strip("'\""))
        try:
            elements, source_code = extract_code_elements_from_file(abs_path)
            reasoning_dict = process_file(abs_path, bug_report, source_code, elements)
            data4[this1][f'file{idx+1}_elements_reasoning'] = reasoning_dict
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data4, f, indent=2, ensure_ascii=False)

print(f"\n✅ Element-level reasoning saved to {args.output}")
