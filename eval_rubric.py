# Imports
import pandas as pd
import requests
import re
import json
from tqdm import tqdm
import logging
from sklearn.metrics import mean_absolute_error

# Set logging level to INFO
logging.basicConfig(level=logging.INFO)

# API key and endpoint for Together AI
TOGETHER_API_KEY = "your_api_key"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# Model to be used for both generation and verification
generator_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
verifier_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Prompt to generate concise and medically accurate answers
def build_generation_prompt(question):
    return f"""You are a reliable and concise medical assistant.

Answer the following medical question in exactly 3 to 4 medically accurate words. Use only verified clinical facts. Avoid any elaboration, guessing, or vague language.

If the question cannot be answered confidently with medical evidence, respond with: "Not enough information".
Question: {question}

Answer:"""


def build_verification_prompt(question, human_answer, model_answer):
    return f"""
You are a highly competent and unbiased medical evaluator. Your task is to assess the quality of two responses—one written by a human, and one generated by a model—to a given medical question.

You must evaluate both responses based on **five well-defined criteria**, using the definitions below. Your ratings must be fair, consistent, and grounded in clinical knowledge and reasoning.

Assess each on a scale from 0.0 (poor) to 1.0 (excellent) based on:

1. **Relevance**: Does it address the question directly and fully?
2. **Conciseness**: Is the answer efficient? Short correct answers should not be penalized.
3. **Medical Accuracy**: Is it factually correct, up-to-date, and consistent with clinical knowledge?
4. **Hallucination**: Are any medical claims unsupported or fabricated?  
   → Full score (1.0) if no hallucination.
5. **Clarity**: Is the answer clear and easy to understand?

Scoring Guide(between 0 to 1):

- 0.0 — Poor:
Entirely incorrect, irrelevant, or misleading.
Includes hallucinated facts or unrelated medical information.
Fails to address the question or shows no valid reasoning.

- Between 0.0 and 0.5 — Weak to Acceptable:
Partially correct, but vague, incomplete, or slightly inaccurate.
May mix some correct and incorrect info.
Shows some medical reasoning but lacks clarity and precision.

- Exactly 0.5 — Minimally Acceptable:
Somewhat helpful but basic or underdeveloped.
Lacks detail or specificity.
No serious errors, but not clinically sufficient.

- Between 0.5 and 1.0 — Good to Very Good:
Mostly correct and clearly presented.
May have small phrasing issues or minor omissions.
Factually grounded, useful, and mostly accurate.

- 1.0 — Excellent:
Fully accurate, concise, and medically sound.
Clearly addresses the question with verified clinical facts.
Free from hallucinations and ambiguity.
Trustworthy for medical understanding or use.

---

**Important Notes:**
- Do not reward verbosity over correctness.
- Short answers that are correct should receive **high scores**.
- Semantic equivalence should be treated as equal even with different wording.
- Do not favor model answers just because they explain more.
- Penalize factual errors, not brevity.

---

Question: {question}

Human Answer: {human_answer}
Model Answer: {model_answer}


- Use intermediate values (e.g., 0.3, 0.65, 0.87) to reflect partial satisfaction of the criteria.

Compare the two answers briefly and return a JSON like this format:
{{
  "human_score": float between 0 and 1,
  "model_score": float between 0 and 1
}}

Only output valid JSON.
Remember:
- Be fair to both answers.
- Avoid bias in favor of brevity or model-style phrasing.
- Focus on **semantic accuracy**, **medical soundness**, and **usefulness**.
- Penalize hallucinated facts or medically inaccurate statements.
- Return only the valid JSON. Do not add comments, notes, or extra text.
"""

# Utility to extract and fix potentially malformed JSON from LLM output
def extract_json(text):
    logging.debug(f"Raw model response: {text}")
    try:
        # Try to extract all JSON-like blocks from text
        matches = re.findall(r"\{.*?\}", text, re.DOTALL)
        for match in matches:
            cleaned = match.strip()

            # Optional: Fix common issue like trailing quotes after numbers
            cleaned = re.sub(r'("?:\s*\d+\.?\d*)["\'](?=\s*[},])', r'\1', cleaned)

            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue  # Try next match

        raise ValueError("No valid JSON object found")

    except Exception as e:
        logging.warning(f"Unexpected parsing failure: {e}\nRaw text: {text}")
        return {}

# Query Together AI model for either generation or verification
def query_together(prompt, model, expect_json=False, max_tokens=256):
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
        "temperature": 0.3,
        "top_p": 0.8,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=data)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        if expect_json:
            parsed = extract_json(content)
            if not parsed:
                logging.warning(f"Empty or invalid JSON from model:\n{content}")
            return parsed
        else:
                return content

    except Exception as e:
        logging.error(f"API call failed: {e}")
        return {} if expect_json else ""
    

# Rerank multiple model-generated answers by scoring them and selecting the highest
def rerank_best_answer(question, human_answer, candidates):
    best = {"answer": "", "score": -1}
    for ans in candidates:
        verifier_prompt = build_verification_prompt(question, human_answer, ans)
        scores = query_together(verifier_prompt, model=verifier_model, expect_json=True)
        if scores.get("model_score", -1) > best["score"]:
            best = {"answer": ans, "score": scores["model_score"]}
    return best["answer"], best["score"]

# Main evaluation pipeline: generate answers, rerank, verify, and write results
def evaluate_pipeline(input_csv, output_csv, num_candidates=3):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        human_answer = row["answer"]

        # Generate multiple candidates
        candidates = []
        for _ in range(num_candidates):
            prompt = build_generation_prompt(question)
            model_answer = query_together(prompt, model=generator_model).strip()
            if model_answer not in candidates:
                candidates.append(model_answer)

        # Rerank
        best_model_answer, _ = rerank_best_answer(question, human_answer, candidates)

        # Final scoring
        final_prompt = build_verification_prompt(question, human_answer, best_model_answer)
        scores = query_together(final_prompt, model=verifier_model, expect_json=True)

        # Store results
        row_data = row.to_dict()
        row_data["model_answer"] = best_model_answer
        row_data["human_score"] = scores.get("human_score", -1)
        row_data["model_score"] = scores.get("model_score", -1)
        row_data["score_diff"] = abs(row_data["human_score"] - row_data["model_score"])

        results.append(row_data)

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")

# Compute mean scores and mean absolute error between human and model scores
def compute_mean_score(df):
    df["human_score"] = pd.to_numeric(df["human_score"], errors="coerce")
    df["model_score"] = pd.to_numeric(df["model_score"], errors="coerce")
    valid = df[(df["human_score"] >= 0) & (df["model_score"] >= 0)]
    if valid.empty:
        print("No valid scores")
        return

    mae = mean_absolute_error(valid["human_score"], valid["model_score"])
    print("\n--- Score Summary ---")
    print(f"Human Score Avg: {valid['human_score'].mean():.3f}")
    print(f"Model Score Avg: {valid['model_score'].mean():.3f}")
    print(f"MAE: {mae:.3f}")

# Run the full pipeline
if __name__ == "__main__":
    input_csv = "chem_que.csv"
    output_csv = "eval.csv"
    evaluate_pipeline(input_csv, output_csv, num_candidates=3)
    df = pd.read_csv(output_csv)
    compute_mean_score(df)
