import os
import json
import tempfile
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
import argparse

# --------------- Argument Parser ---------------- #
parser = argparse.ArgumentParser(description="Reasoning Generator Script")
parser.add_argument("--model", type=str, required=True, help="Model ID to use for reasoning")
parser.add_argument("--backend", type=str, choices=["anthropic", "gemini", "openai"], required=True, help="LLM backend")
parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Verified", help="Dataset to use")
parser.add_argument("--split", type=str, default="test", help="Dataset split")
parser.add_argument("--combined_locs", type=str, required=True, help="Path to combined_locs.jsonl file")
parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
parser.add_argument("--target_id", type=str, help="Optional: instance_id to process a single instance")
args = parser.parse_args()

MODEL_ID = args.model
backend = args.backend

# ----------------- Load Dataset ------------------ #
print(f"Loading dataset: {args.dataset} [{args.split}]")
swebench = load_dataset(args.dataset, split=args.split)
if args.target_id:
    target_instance = next((ex for ex in swebench if ex["instance_id"] == args.target_id), None)
    if target_instance is None:
        raise ValueError(f"Instance ID {args.target_id} not found in dataset.")
    swebench = [target_instance]  # process only this instance


# ---------------- Load combined_locs ------------- #
print(f"Loading file: {args.combined_locs}")
with open(args.combined_locs, 'r', encoding='utf-8') as f:
    data4 = [json.loads(line) for line in f]

# ---------------- Clone Function ----------------- #
def clone_and_checkout(repo_name, commit):
    url = f"https://github.com/{repo_name}.git"
    temp_dir = tempfile.mkdtemp()
    subprocess.run(["git", "clone", "--quiet", url, temp_dir], check=True)
    subprocess.run(["git", "checkout", commit], cwd=temp_dir, check=True)
    return temp_dir

# ----------------- Reasoning Call ---------------- #
def get_reasoning(file_content, bug_report):
    prompt = f"""A user is trying to fix a bug described in the following report:\n\n{bug_report}\n\nBelow is a code file from a repository:\n\n{file_content}\n\nExplain the purpose and functionality of this code in the context of the bug report. Focus on what this file does and whether it may be related to the bug."""

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
        import os

        PROJECT_ID = os.environ.get("VERTEXAI_PROJECT", "")
        LOCATION = os.environ.get("VERTEXAI_LOCATION", "us-central1")

        client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        return response.text

    elif backend == "openai":
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

# -------------------- Main Loop ------------------ #
for i, example in enumerate(tqdm(swebench, desc="Processing instances", total=len(swebench))):
    bug_report = example['problem_statement']
    repo = example['repo']
    base_commit = example['base_commit']
    instance_id = example['instance_id']

    try:
        repo_path = clone_and_checkout(repo, base_commit)
    except Exception as e:
        print(f"[{instance_id}] Repo checkout failed: {e}")
        continue

    file_texts, file_paths = [], []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                        file_texts.append(content)
                        file_paths.append(os.path.relpath(path, repo_path))
                except Exception:
                    continue

    try:
        this1 = next(j for j in range(len(data4)) if data4[j]['instance_id'] == instance_id)
    except StopIteration:
        print(f"[{instance_id}] No match in combined_locs")
        continue

    def reasoning_task(file_name):
        if file_name in file_paths:
            file_content = file_texts[file_paths.index(file_name)]
            try:
                reasoning = get_reasoning(file_content, bug_report)
            except Exception as e:
                reasoning = f"Error during reasoning: {e}"
            return file_name, reasoning
        return file_name, "File not found in repo"

    data4[this1]['file_reasoning'] = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(reasoning_task, fn): fn for fn in data4[this1]['found_files']}
        for future in as_completed(futures):
            file_name, reasoning = future.result()
            data4[this1]['file_reasoning'][file_name] = reasoning

# ------------------- Save Results ---------------- #
with open(args.output, "w", encoding="utf-8") as f:
    json.dump(data4, f, indent=2, ensure_ascii=False)

print(f"\n✅ Reasoning results saved to {args.output}")
