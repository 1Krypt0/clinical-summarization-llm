import os

import dotenv
import google.generativeai as genai
import pandas as pd

from common import ZERO_SHOT_PROMPT, load_test_data

data = load_test_data()

dotenv.load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-pro-latest")

results = []
TOTAL = 0

print("Beginning Zero-Shot Evaluation")

for _, row in data.iterrows():
    notes = row["notes"]

    prompt_parts = [
        ZERO_SHOT_PROMPT,
        "notes: " + notes,
        "summary: ",
    ]

    response = model.generate_content(prompt_parts)

    results.append(response)

    TOTAL += 1

    if TOTAL % 5 == 0:
        save_point = pd.DataFrame(results)
        save_point.to_json(
            "./outputs/zero-shot-responses-gemini-1.5.jsonl",
            lines=True,
            orient="records",
        )
        print(f"Evaluated {TOTAL} examples")

results = pd.DataFrame(results, columns=["summary"])

results.to_json(
    "./outputs/zero-shot-responses-gemini-1.5.jsonl", lines=True, orient="records"
)
