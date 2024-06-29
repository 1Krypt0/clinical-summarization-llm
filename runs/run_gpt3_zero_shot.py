import os
import dotenv
import pandas as pd
from openai import OpenAI
from common import load_test_data, ZERO_SHOT_PROMPT

data = load_test_data()

dotenv.load_dotenv()

gpt_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

entries = []

for idx, row in data.iterrows():

    notes = row["notes"]

    prompt = ZERO_SHOT_PROMPT

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": notes},
    ]

    entry = {
        "custom_id": f"request-{idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "gpt-3.5-turbo", "messages": messages},
    }

    entries.append(entry)


entries = pd.DataFrame(entries)

######################################################
# DON'T RUN THIS AGAIN. ONLY TO EXEMPLIFY HOW TO DO IT
######################################################

# Process data into batches
# for i in range(0, len(entries), 50):
#     entries[i : i + 50].to_json(
#         f"../data/zero-shot-requests-gpt3-part-{i // 50}.jsonl",
#         orient="records",
#         lines=True,
#     )

# batch_input_file = gpt_client.files.create(
#     file=open("../data/zero-shot-requests-gpt3-part-1.jsonl", "rb"),
#     purpose="batch",
# )

# batch_id = batch_input_file.id

# gpt_client.batches.create(
#     input_file_id=batch_id,
#     endpoint="/v1/chat/completions",
#     completion_window="24h",
#     metadata={"description": "Zero-shot summarization of clinical notes, with GPT 3"},
# )

# # Retrieve the file with the responses. Enter specific file ID
# output_file_id = ""

# res = gpt_client.files.content(output_file_id)
# content = res.content
# content = content.decode("utf-8")

# content = pd.read_json(content, lines=True)
# content.sort_values("custom_id", inplace=True)
# content.to_json(
#     "../data/zero-shot-responses-gpt3-part-1.jsonl", orient="records", lines=True
# )
