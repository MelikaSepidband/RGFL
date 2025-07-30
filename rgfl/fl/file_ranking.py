import os
import json
import argparse
import re
from datasets import load_dataset
from tqdm import tqdm

# ------------------------- Argument Parser ---------------------------- #
parser = argparse.ArgumentParser(description="Rank files based on reasoning similarity to bug report")
parser.add_argument("--model", type=str, required=True, help="Model ID to use")
parser.add_argument("--backend", type=str, choices=["anthropic", "gemini", "openai"], required=True)
parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Verified")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--reasoning_file", type=str, required=True, help="Path to file_reasoning JSONL")
parser.add_argument("--output", type=str, required=True, help="Output path for ranked file list")
parser.add_argument("--target_id", type=str, help="Optional: instance_id to process a single instance")

args = parser.parse_args()

MODEL_ID = args.model
backend = args.backend

# ------------------------- LLM Wrapper ---------------------------- #
def get_ranking(file_reasoning, bug_report):
    prompt = f"""Below is a list of files from a repository and the reasonings behind the codes of these files:\n\n{file_reasoning}\n\nCan you rank the files based on the similarity of their reasoning to the bug report:\n\n{bug_report}\n\nPlease just return the list of ranked files."""

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

# ------------------------- Output Parser ---------------------------- #
def extract_file_list(llm_output):
    if llm_output.strip().startswith("[") and llm_output.strip().endswith("]"):
        try:
            return eval(llm_output)
        except:
            pass
    return re.findall(r"[\w\-/]+\.py", llm_output)

# ------------------------ Load Data ---------------------------------- #
print(f"Loading SWE-bench dataset: {args.dataset}")
swebench = load_dataset(args.dataset, split=args.split)

if args.target_id:
    target_instance = next((ex for ex in swebench if ex["instance_id"] == args.target_id), None)
    if target_instance is None:
        raise ValueError(f"Instance ID {args.target_id} not found in dataset.")
    swebench = [target_instance]  # process only this instance

print(f"Loading reasoning file: {args.reasoning_file}")
with open(args.reasoning_file, "r", encoding="utf-8") as f:
    data4 = json.load(f)  # load full JSON array

# ------------------------ Main Loop ---------------------------------- #
for i, example in enumerate(tqdm(swebench, desc="Ranking files", total=len(swebench))):
    instance_id = example['instance_id']
    bug_report = example['problem_statement']

    try:
        this1 = next(j for j in range(len(data4)) if data4[j]['instance_id'] == instance_id)
    except StopIteration:
        print(f"[{instance_id}] No match found in reasoning file.")
        continue

    file_reasoning = data4[this1].get("file_reasoning", {})
    if not file_reasoning:
        print(f"[{instance_id}] Missing `file_reasoning` data.")
        continue

    try:
        reasoning_text = json.dumps(file_reasoning, indent=2)
        ranking = get_ranking(reasoning_text, bug_report)
        list_of_files = extract_file_list(ranking)
        data4[this1]["retrieved_files"] = list_of_files
    except Exception as e:
        print(f"[{instance_id}] Ranking failed: {e}")
        data4[this1]["retrieved_files"] = []

# ------------------------ Save Output -------------------------------- #
with open(args.output, "w", encoding="utf-8") as f:
    json.dump(data4, f, indent=2, ensure_ascii=False)

print(f"\n✅ File rankings saved to {args.output}")
