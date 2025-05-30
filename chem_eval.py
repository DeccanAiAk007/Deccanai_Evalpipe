# Imports
import pandas as pd
import requests
import re
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set logging level to INFO
logging.basicConfig(level=logging.INFO)

# API Key and Endpoint
TOGETHER_API_KEY = "your_api_key"
TOGETHER_API_URL = "https://api.together.xyz/v1/completions"

# List of models to evaluate
model_list = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
]

# Prompt to generate a final answer and explanation for chemistry questions
def build_generation_prompt(question):
    return f"""
You are a reliable and concise chemistry expert.

Answer the following chemistry question with:
1. A **short, final answer** (1-4 words, e.g., a chemical formula, value, or choice letter)
2. A **brief but accurate explanation** grounded in chemistry principles

Ensure:
- The answer is **factually correct**.
- Avoid vagueness or speculation.
- Use only verified chemical knowledge.
- For ambiguous or insufficient questions, say:
  Final Answer: Not enough information
  Explanation: The question lacks sufficient data to determine the answer.

---
Question: {question}

Respond in this format:
Final Answer: <your answer>
Explanation: <your explanation>
"""

# Call the Together AI API
def query_together(prompt, model, max_tokens=256):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant and strong problem solver."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "top_p": 0.8,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        # Handle multiple response styles
        if isinstance(result, dict):
            if "output" in result:  # Some Together formats
                return result["output"]
            elif "choices" in result:
                choice = result["choices"][0]
                return choice.get("text", choice.get("message", {}).get("content", "")).strip()
        logging.warning("Unexpected response format.")
        return ""
    except Exception as e:
        logging.error(f"Model: {model} | API call failed: {e}")
        return ""

def generate_answers(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        row_data = row.to_dict()

        def fetch_model_response(model):
            prompt = build_generation_prompt(question)
            response = query_together(prompt, model=model)
            final_answer, explanation = "", ""
            match = re.search(r"Final Answer:\s*(.*?)\nExplanation:\s*(.*)", response, re.DOTALL)
            if match:
                final_answer = match.group(1).strip()
                explanation = match.group(2).strip()
            return model, final_answer, explanation

        # Use ThreadPool to call multiple models concurrently
        with ThreadPoolExecutor(max_workers=len(model_list)) as executor:
            futures = [executor.submit(fetch_model_response, model) for model in model_list]
            for future in as_completed(futures):
                model, final_answer, explanation = future.result()
                row_data[f"{model}_final_answer"] = final_answer
                row_data[f"{model}_explanation"] = explanation

        results.append(row_data)

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")

# Run generation
if __name__ == "__main__":
    input_csv = "synchem.csv"
    output_csv = "synchem_ans2.csv"
    generate_answers(input_csv, output_csv)
