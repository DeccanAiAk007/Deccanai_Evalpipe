import pandas as pd
import re
import logging
from tqdm import tqdm
import requests

# Set logging level to INFO
logging.basicConfig(level=logging.INFO)

# List of models hosted on Hugging Face Inference API
model_list = [
    "tiiuae/falcon-rw-1b"
]

# Hugging Face API token (replace with your own token)
HF_API_TOKEN = "your_api_key"
HF_API_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{}"

# Prompt builder for chemistry questions
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

# Query model from Hugging Face Inference API
def query_huggingface_api(prompt, model_name):
    url = HF_API_URL_TEMPLATE.format(model_name)
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and 'generated_text' in result[0]:
            return result[0]['generated_text']
        elif isinstance(result, dict) and 'generated_text' in result:
            return result['generated_text']
        else:
            logging.warning(f"Unexpected response format from model {model_name}")
            return ""
    except Exception as e:
        logging.error(f"Model: {model_name} | API call failed: {e}")
        return ""

# Main pipeline for answering chemistry questions
def generate_answers(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        row_data = row.to_dict()

        for model in model_list:
            prompt = build_generation_prompt(question)
            response = query_huggingface_api(prompt, model)
            final_answer, explanation = "", ""
            match = re.search(r"Final Answer:\s*(.*?)\nExplanation:\s*(.*)", response, re.DOTALL)
            if match:
                final_answer = match.group(1).strip()
                explanation = match.group(2).strip()
            row_data[f"{model}_final_answer"] = final_answer
            row_data[f"{model}_explanation"] = explanation

        results.append(row_data)

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")

# Entry point
if __name__ == "__main__":
    input_csv = "synchem.csv"
    output_csv = "synchem_hf_api_answers.csv"
    generate_answers(input_csv, output_csv)
