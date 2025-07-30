import os
import re
import ast
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm

def normalize_to_list(entry):
    def clean_item(item):
        item = re.sub(r'^\d+\.\s*', '', item)
        return item.strip("`'\" \t\n,")
    
    bad_tokens = {'[', ']', 'json', '```', '```json', ''}

    if isinstance(entry, list):
        return [clean_item(item) for item in entry if clean_item(item) not in bad_tokens]

    elif isinstance(entry, str):
        entry = entry.strip()
        entry = re.sub(r'^```.*$', '', entry, flags=re.MULTILINE)
        try:
            parsed = ast.literal_eval(entry)
            if isinstance(parsed, list):
                return [clean_item(item) for item in parsed if clean_item(item) not in bad_tokens]
        except:
            pass
        lines = re.split(r'[\n,]+', entry)
        return [clean_item(line) for line in lines if clean_item(line) not in bad_tokens]

    return []

def get_similar_elements(file_elements_reasoning, bug_report):
    prompt = f"""You are provided with a list of code elements (functions, classes, and global variables) from a repository, along with an explanation of what each element does:\n\n{file_elements_reasoning}\n\nAlso you are given the following bug report:\n\n{bug_report}\n\nBased on the reasoning for each code element and the bug report, which of these elements are most likely related to the bug? Please just return a ranked list of the potentially buggy elements (keys in the file_elements_reasoning dictionary) without any further explanation."""

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
        )
        return response.choices[0].message.content

def rank_all_elements(data2, swebench, top_k=3):
    for i, example in enumerate(tqdm(swebench, desc="Ranking elements")):
        instance_id = example['instance_id']
        bug_report = example['problem_statement']

        try:
            this1 = next(j for j in range(len(data2)) if data2[j]['instance_id'] == instance_id)
        except StopIteration:
            continue

        for file_idx in range(1, top_k + 1):
            key = f'file{file_idx}_elements_reasoning'
            output_key = f'similar_elements_file{file_idx}'

            elem_reasoning = data2[this1].get(key, {})
            if not elem_reasoning:
                data2[this1][output_key] = []
                continue

            try:
                reasoning_json = json.dumps(elem_reasoning, indent=2)
                sim_elems = get_similar_elements(reasoning_json, bug_report)
                data2[this1][output_key] = normalize_to_list(sim_elems)
            except Exception as e:
                print(f"[{instance_id}] Error for {key}: {e}")
                data2[this1][output_key] = []
    return data2

def postprocess_keys(data2, swebench, top_k=3):
    for i, example in enumerate(tqdm(swebench, desc="Postprocessing element names")):
        instance_id = example['instance_id']
        try:
            this1 = next(j for j in range(len(data2)) if data2[j]['instance_id'] == instance_id)
        except StopIteration:
            continue

        for file_idx in range(1, top_k + 1):
            key = f'file{file_idx}_elements_reasoning'
            output_key = f'similar_elements_file{file_idx}'

            elem_reasoning = data2[this1].get(key, {})
            sim_elems = data2[this1].get(output_key, [])

            corrected = []
            for s in sim_elems:
                if ':' in s:
                    corrected.append(s)
                else:
                    for full_key in elem_reasoning.keys():
                        if full_key.endswith(f": {s}"):
                            corrected.append(full_key)
                            break
            if corrected:
                data2[this1][output_key] = corrected
    return data2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank code elements based on reasoning and bug report.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--backend", type=str, choices=["gemini", "anthropic", "openai"], required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--target_id", type=str, help="Optional: instance_id to process a single instance")
    args = parser.parse_args()

    MODEL_ID = args.model
    backend = args.backend

    print(f"Loading dataset: {args.dataset}")
    dataset_full = load_dataset(args.dataset, split=args.split)
    if args.target_id:
        swebench = [ex for ex in dataset_full if ex["instance_id"] == args.target_id]
        if not swebench:
            raise ValueError(f"Instance ID {args.target_id} not found in dataset.")
    else:
        swebench = dataset_full


    print(f"Loading reasoning file: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data2 = json.load(f)

    data2 = rank_all_elements(data2, swebench)
    data2 = postprocess_keys(data2, swebench)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data2, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Ranked and normalized elements saved to {args.output}")
