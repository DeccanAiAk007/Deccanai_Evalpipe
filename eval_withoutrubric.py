import pandas as pd
import requests
import json
from tqdm import tqdm
import logging
import re
from sklearn.metrics import mean_absolute_error

# Setup logging
logging.basicConfig(level=logging.INFO)

TOGETHER_API_KEY = "your_api_key"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

generator_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
verifier_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

def build_generation_prompt(question):
    return f"""You are a reliable and concise medical assistant.

Answer the following medical question in exactly 3 to 4 medically accurate words. Use only verified clinical facts. Avoid any elaboration, guessing, or vague language. Avoid extra explanations.

If the question cannot be answered confidently with medical evidence, respond with: "Not enough information".
Question: {question}

Answer:"""

def build_verification_prompt(question, human_answer, model_answer):
    return f"""
You are an expert medical reviewer. Strictly output only JSON format.

Evaluate these answers to the medical question:
Question: {question}

Human-written answer:
"{human_answer}"

Model-generated answer:
"{model_answer}"

Evaluation criteria:
1. Clarity (0-1): Understandability
2. Medical Accuracy (0-1): Factual correctness
3. Helpfulness (0-1): Usefulness/relevance

Output ONLY this JSON format with scores as floats:
{{
  "human_score": 0.0,
  "model_score": 0.0
}}

Do NOT include any other text, explanations, or formatting.
"""

def extract_json(text):
    logging.debug(f"Raw model response: {text}")
    try:
        # First, try to parse the entire string as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            # If that fails, look for JSON objects within the text
            matches = re.findall(r"\{[\s\S]*?\}", text)
            if matches:
                # Try each potential JSON match
                for match in matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
    # If all parsing attempts fail
    logging.warning(f"Unexpected parsing failure\nRaw text: {text}")
    return {}

def query_together(prompt, model, expect_json=False, max_tokens=128):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0 if expect_json else 0.3,
        "top_p": 0.8,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=data)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return extract_json(content) if expect_json else content
    except Exception as e:
        logging.error(f"API call failed: {e}")
        return {} if expect_json else ""

def rerank_best_answer(question, human_answer, candidates):
    best = {"answer": "", "score": -1}
    for ans in candidates:
        verifier_prompt = build_verification_prompt(question, human_answer, ans)
        scores = query_together(verifier_prompt, model=verifier_model, expect_json=True)
        if scores.get("model_score", -1) > best["score"]:
            best = {"answer": ans, "score": scores["model_score"]}
    return best["answer"], best["score"]

def evaluate_pipeline(input_csv, output_csv, num_candidates=3):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        human_answer = row["answer"]
        candidates = set()

        for _ in range(num_candidates * 2):  # Overgenerate to avoid duplicates
            if len(candidates) >= num_candidates:
                break
            prompt = build_generation_prompt(question)
            answer = query_together(prompt, model=generator_model).strip()
            if answer:
                candidates.add(answer)

        if not candidates:
            logging.warning(f"No model answer generated for: {question}")
            continue

        best_model_answer, _ = rerank_best_answer(question, human_answer, list(candidates))
        final_prompt = build_verification_prompt(question, human_answer, best_model_answer)
        scores = query_together(final_prompt, model=verifier_model, expect_json=True)

        row_data = row.to_dict()
        row_data["model_answer"] = best_model_answer
        row_data["human_score"] = float(scores.get("human_score", -1))
        row_data["model_score"] = float(scores.get("model_score", -1))
        row_data["score_diff"] = abs(row_data["human_score"] - row_data["model_score"]) if row_data["human_score"] >= 0 and row_data["model_score"] >= 0 else -1

        results.append(row_data)

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")

def compute_mean_score(df):
    df["human_score"] = pd.to_numeric(df["human_score"], errors="coerce")
    df["model_score"] = pd.to_numeric(df["model_score"], errors="coerce")
    valid = df[(df["human_score"] >= 0) & (df["model_score"] >= 0)]

    if valid.empty:
        print("No valid scores found")
        return

    mae = mean_absolute_error(valid["human_score"], valid["model_score"])
    print("\n--- Score Summary ---")
    print(f"Human Score Avg: {valid['human_score'].mean():.3f}")
    print(f"Model Score Avg: {valid['model_score'].mean():.3f}")
    print(f"MAE: {mae:.3f}")

if __name__ == "__main__":
    input_csv = "que_ans.csv"
    output_csv = "eval_wrubric.csv"
    evaluate_pipeline(input_csv, output_csv, num_candidates=3)
    df = pd.read_csv(output_csv)
    compute_mean_score(df)
