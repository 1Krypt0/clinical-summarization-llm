import os

import dotenv
import google.generativeai as genai
import pandas as pd

from common import (
    ZERO_SHOT_PROMPT,
    load_one_shot_notes,
    load_one_shot_summary,
    load_test_data,
    evaluate_summaries,
    save_scores,
)

data = load_test_data()

dotenv.load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-pro-latest")

results = []
TOTAL = 0

print("Beginning One-Shot Evaluation")

for _, row in data.iterrows():
    notes = row["notes"]

    prompt_parts = [
        ZERO_SHOT_PROMPT,
        "notes: " + load_one_shot_notes(),
        "summary: " + load_one_shot_summary(),
        "notes: " + notes,
        "summary: ",
    ]

    response = model.generate_content(prompt_parts)

    results.append(response)

    TOTAL += 1

    if TOTAL % 5 == 0:
        save_point = pd.DataFrame(results)
        save_point.to_json(
            "./outputs/one-shot-responses-gemini-1.5.jsonl",
            lines=True,
            orient="records",
        )
        print(f"Evaluated {TOTAL} examples")


results = pd.DataFrame(results, columns=["summary"])

results.to_json(
    "./outputs/one-shot-responses-gemini-1.5.jsonl", lines=True, orient="records"
)

predictions = []

for pred in results["summary"]:
    predictions.append(pred["text"])

rouge_score, bleu_score = evaluate_summaries(predictions=predictions)

save_scores("./gemini-one-shot.pkl", rouge_score, bleu_score)
