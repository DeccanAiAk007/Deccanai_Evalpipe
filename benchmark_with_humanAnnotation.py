import torch
import re
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants
generator_model = "llama3.2"
# generator_model = "qwen:7b"
verifier_model = "llama3.2"
metrics = [
    "Relevance", "Succinctness", "Medical Correctness",
    "Hallucination", "Completeness", "Coherence"
]

def build_generation_prompt(question):
    return f"""You are a highly knowledgeable and trustworthy medical assistant.

Answer ONLY the specific question asked below. Be **factually correct**, **medically accurate**, and **concise**.
Do NOT include disclaimers, definitions, or unrelated information.
Do NOT hallucinate or make assumptions beyond the question.

Question: {question}
Answer:"""


def build_verification_prompt(question, ground_truth, model_output):
    return f"""
You are a medical evaluation expert responsible for assessing the quality of a model-generated answer in response to a medical question. You will be provided with:

- A medical question
- A gold-standard ground truth answer
- A model-generated answer

Your task is to evaluate the model answer against the ground truth using the six clinical evaluation metrics described below.

### Evaluation Metrics and Scoring Guidelines

Each metric should be scored on a scale from **0 to 3**, with 0 being the lowest and 3 being the highest.

1. **Relevance** – Does the response directly address the question posed?
   - 0: Irrelevant
   - 1: Weakly relevant
   - 2: Mostly relevant
   - 3: Fully relevant

2. **Succinctness** – Is the response concise, avoiding unnecessary detail?
   - 0: Excessively verbose or confusing
   - 1: Acceptable but could be more concise
   - 2: Mostly succinct
   - 3: Highly concise and to the point

3. **Medical Correctness** – Is the information clinically accurate?
   - 0: Contains dangerous errors
   - 1: Contains concerning inaccuracies
   - 2: Minor inaccuracies (benign)
   - 3: Fully medically accurate

4. **Hallucination** – Does the response avoid adding incorrect or fabricated information?
   - 0: Contains major fabricated facts
   - 1: Some unsupported claims
   - 2: Mostly evidence-based
   - 3: No hallucinations or unsupported claims

5. **Completeness** – Does the answer fully address all aspects of the question?
   - 0: Very incomplete
   - 1: Partially complete
   - 2: Mostly complete
   - 3: Fully complete

6. **Coherence** – Is the response clear, logically structured, and easy to follow?
   - 0: Incoherent
   - 1: Poorly structured
   - 2: Mostly coherent
   - 3: Very well-structured and clear

### Evaluation Task

Evaluate the model answer using the metrics above.

--- Input ---
Question: {question}

Gold Answer: {ground_truth}

Model Answer: {model_output}
--- End ---

### Output Format (STRICT)

Return **only** a valid JSON object with numeric scores (0–3) for each of the six metrics.

Example output:
{{
  "Relevance": 3,
  "Succinctness": 2,
  "Medical Correctness": 3,
  "Hallucination": 3,
  "Completeness": 2,
  "Coherence": 3
}}
"""

def extract_json(text):
    """
    Extract the first valid JSON object containing all expected metric keys.
    """
    stack = []
    start = None
    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start = i
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidate = text[start:i+1]
                    try:
                        obj = json.loads(candidate)
                        if all(k in obj and isinstance(obj[k], int) for k in metrics):
                            return obj
                        else:
                            logging.warning(f"Missing or invalid metric keys in extracted JSON: {candidate}")
                    except json.JSONDecodeError as e:
                        logging.warning(f"JSON decoding failed: {e}")
    logging.warning(f"No valid JSON found in:\n{text}")
    return {}



# Query ollama
def query_ollama(prompt, model="llama3.2", expect_json=False, max_tokens=512):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "repetition_penalty": 1.2
                }
            }
        )
        result = response.json()
        output = result.get("response", "").strip()
        return extract_json(output) if expect_json else output
    except Exception as e:
        logging.error(f"Error querying Ollama: {e}")
        return {} if expect_json else ""


def evaluate_pipeline(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        gold = row["ground_truth"]
        row_data = row.to_dict()

        g_prompt = build_generation_prompt(question)
        gen_answer = query_ollama(g_prompt, model=generator_model)

        if not gen_answer:
            logging.error(f" Empty generation for question:\n{question}")
            for m in metrics:
                row_data[f"{m}_llm"] = -1
            row_data["model_output"] = ""
            results.append(row_data)
            continue

        v_prompt = build_verification_prompt(question, gold, gen_answer)
        scores = query_ollama(v_prompt, model=verifier_model, expect_json=True)

        if not scores:
            logging.error(f" Verifier failed on:\nQ: {question}\nA: {gen_answer}\n")
            for m in metrics:
                row_data[f"{m}_llm"] = -1
        else:
            for m in metrics:
                row_data[f"{m}_llm"] = scores.get(m, -1)

        row_data["model_output"] = gen_answer
        results.append(row_data)

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\n Results saved to {output_csv}")

def compute_mae(df):
    maes = {}
    for m in metrics:
        col_human = f"{m}_human"
        col_llm = f"{m}_llm"
        if col_human in df.columns and col_llm in df.columns:
            valid = df[df[col_llm].between(0, 3)]
            if not valid.empty:
                maes[m] = mean_absolute_error(valid[col_human], valid[col_llm])
            else:
                maes[m] = "N/A"
    print("\n Mean Absolute Error (LLM vs Human):")
    for m, v in maes.items():
        print(f"{m}: {v}")

if __name__ == "__main__":
    input_csv = "non_binary_scoring.csv"
    output_csv = "zero_shot1.csv"
    evaluate_pipeline(input_csv, output_csv)
    df = pd.read_csv(output_csv)
    compute_mae(df)
