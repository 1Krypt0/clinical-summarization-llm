import pandas as pd
from common import load_test_data, ZERO_SHOT_PROMPT

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
