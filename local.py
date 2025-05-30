import pandas as pd
import re
import logging
from tqdm import tqdm
import subprocess
import json

logging.basicConfig(level=logging.INFO)

model_list = ["llama3.2:latest"]

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

def query_ollama(prompt, model_name, max_tokens=256):
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180
        )
        output = result.stdout.decode("utf-8").strip()
        return output
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout when querying {model_name}")
        return "Final Answer: Error\nExplanation: Model timeout or failed."
    except Exception as e:
        logging.error(f"Error querying Ollama model: {e}")
        return "Final Answer: Error\nExplanation: Exception occurred."

def generate_answers(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        row_data = row.to_dict()

        for model in model_list:
            prompt = build_generation_prompt(question)
            response = query_ollama(prompt, model)
            final_answer, explanation = "", ""
            match = re.search(r"Final Answer:\s*(.*?)\nExplanation:\s*(.*)", response, re.DOTALL)
            if match:
                final_answer = match.group(1).strip()
                explanation = match.group(2).strip()
            else:
                final_answer = "Parse Error"
                explanation = "Failed to parse response."
            row_data[f"{model}_final_answer"] = final_answer
            row_data[f"{model}_explanation"] = explanation

        results.append(row_data)

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")

if __name__ == "__main__":
    input_csv = "synchem.csv"
    output_csv = "synchem_local_answers.csv"
    generate_answers(input_csv, output_csv)
