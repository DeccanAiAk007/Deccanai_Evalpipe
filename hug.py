import pandas as pd
import re
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set logging level to INFO
logging.basicConfig(level=logging.INFO)

# List of small open-source models to run locally
model_list = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
]

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

# Cache loaded models and tokenizers
MODEL_CACHE = {}
def load_model(model_name):
    if model_name not in MODEL_CACHE:
        logging.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, trust_remote_code=True)
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        MODEL_CACHE[model_name] = (tokenizer, model)
    return MODEL_CACHE[model_name]


# Query the local model
def query_local_model(prompt, model_name, max_tokens=256):
    tokenizer, model = load_model(model_name)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.3, top_p=0.8, do_sample=False)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result[len(prompt):].strip()

# Main pipeline for answering chemistry questions
def generate_answers(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        row_data = row.to_dict()

        for model in model_list:
            prompt = build_generation_prompt(question)
            response = query_local_model(prompt, model)
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
    output_csv = "synchem_local_answers.csv"
    generate_answers(input_csv, output_csv)