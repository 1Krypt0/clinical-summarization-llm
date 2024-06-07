import os
import dotenv
import pandas as pd
from openai import OpenAI
from common import load_test_data, ONE_SHOT_PROMPT, evaluate_summaries, save_scores

data = load_test_data()

dotenv.load_dotenv()

gpt_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

entries = []

for index, admission in data.iterrows():

    notes = admission["notes"]
    prompt = ONE_SHOT_PROMPT

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": notes},
    ]

    entry = {
        "custom_id": f"request-{index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "gpt-4-turbo", "messages": messages},
    }

    entries.append(entry)


entries = pd.DataFrame(entries)

##############################################################
# TO SEE HOW TO WORK WITH BATCHES, CHECK run_gpt3_zero_shot.py
##############################################################

predictions = pd.read_json("./outputs/one-shot-responses-gpt4.jsonl", lines=True)

predictions["custom_id"] = predictions["custom_id"].apply(
    lambda x: int(x.split("-")[1])
)
predictions = predictions.sort_values("custom_id")
predictions = predictions["response"]
predictions = predictions.apply(lambda x: x["body"]["choices"][0]["message"]["content"])

rouge_score, bleu_score = evaluate_summaries(predictions=predictions)

save_scores("./results/gpt4-one-shot.pkl", rouge_score, bleu_score)
