import pandas as pd
from common import load_test_data, ZERO_SHOT_PROMPT, evaluate_summaries, save_scores

data = load_test_data()

# Process the data in the batch style
entries = []

for index, admission in data.iterrows():

    notes = admission["notes"]
    prompt = ZERO_SHOT_PROMPT

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


predictions = pd.read_json("./outputs/zero-shot-responses-gpt4.jsonl", lines=True)

predictions["custom_id"] = predictions["custom_id"].apply(
    lambda x: int(x.split("-")[1])
)
predictions = predictions.sort_values("custom_id")
predictions = predictions["response"]
predictions = predictions.apply(lambda x: x["body"]["choices"][0]["message"]["content"])

rouge_score, bleu_score = evaluate_summaries(predictions=predictions)

save_scores("./results/gpt4-zero-shot.pkl", rouge_score, bleu_score)
