import pickle
import pandas as pd
import nltk
from typing import List
import re
from datetime import datetime
from evaluate import load
from blanc import BlancTune
from summac.model_summac import SummaCConv

DATA = pd.read_csv("../data/single-discharge-7.6k-test-formatted.csv")
REFERENCES = DATA["summary"]
NOTES_REFERENCES = DATA["notes"]

NUM_HEADERS = 18

rouge = load("rouge")
bleu = load("bleu")
bertscore = load("bertscore")
bleurt = load("bleurt")
blanc_tune = BlancTune(
    finetune_mask_evenly=False,
    show_progress_bar=True,
    device="cuda",
    inference_batch_size=8,
    finetune_batch_size=8,
)
model_conv = SummaCConv(
    models=["vitc"],
    bins="percentile",
    granularity="sentence",
    nli_labels="e",
    device="cuda",
    start_file="default",
    agg="mean",
)


def calculate_ttr(text):
    words = nltk.word_tokenize(text)
    unique_words = set(words)
    if len(words) == 0:
        return 0
    return len(unique_words) / len(words)


def calculate_avg_ttr(data) -> float:
    temp = []
    for entry in data:
        temp.append(calculate_ttr(entry))

    return sum(temp) / len(temp)


# Dict with all information
originals_headers = {
    "Service:": [],
    "Chief Complaint:": [],
    "Major Surgical or Invasive Procedure:": [],
    "History of Present Illness:": [],
    "Past Medical History:": [],
    "Allergies:": [],
    "Medications on Admission:": [],
    "Family History:": [],
    "Social History:": [],
    "Physical Exam:": [],
    "Pertinent Results:": [],
    "Brief Hospital Course:": [],
    "Discharge Diagnosis:": [],
    "Discharge Medications:": [],
    "Discharge Disposition:": [],
    "Discharge Instructions:": [],
    "Followup Instructions:": [],
    "Discharge Condition:": [],
}

summaries_headers = {
    "Service:": [],
    "Chief Complaint:": [],
    "Major Surgical or Invasive Procedure:": [],
    "History of present illness:": [],
    "Past medical history:": [],
    "Allergies:": [],
    "Medications on Admission:": [],
    "Family History:": [],
    "Social History:": [],
    "Physical Exam:": [],
    "Pertinent Results:": [],
    "Brief Hospital Course:": [],
    "Discharge Diagnosis:": [],
    "Discharge Medications:": [],
    "Discharge Disposition:": [],
    "Discharge Instructions:": [],
    "Follow-up Instructions:": [],
    "Discharge Condition:": [],
}


def separate_by_headers_summary(summary: str):
    pattern = r"(?P<header>Service:|Chief Complaint:|Major Surgical or Invasive Procedure:|History of present illness:|Past medical history:|Allergies:|Medications on Admission:|Family History:|Social History:|Physical Exam:|Pertinent Results:|Brief Hospital Course:|Discharge Diagnosis:|Discharge Medications:|Discharge Disposition:|Discharge Instructions:|Follow-up Instructions:|Discharge Condition:)\s*(?P<content>(?:.|\n)*?)(?=\n\n|$)"

    regex = re.compile(pattern, re.DOTALL)

    matches = regex.findall(summary)

    return matches


def separate_by_headers(summary: str):
    pattern = r"(?P<header>Service:|Chief Complaint:|Major Surgical or Invasive Procedure:|History of Present Illness:|Past Medical History:|Allergies:|Medications on Admission:|Family History:|Social History:|Physical Exam:|Pertinent Results:|Brief Hospital Course:|Discharge Diagnosis:|Discharge Medications:|Discharge Disposition:|Discharge Instructions:|Followup Instructions:|Discharge Condition:)\s*(?P<content>(?:.|\n)*?)(?=\n\n|$)"

    regex = re.compile(pattern, re.DOTALL)

    matches = regex.findall(summary)

    return matches


def separate_summaries(summaries):
    # If the header is not present, fill with empty string so len is always the same
    TOTAL = 0
    for summary in summaries:
        info = separate_by_headers_summary(summary, TOTAL)

        for header in summaries_headers.keys():
            if header not in [header for header, _ in info]:
                info.append((header, ""))

        for i in range(len(info)):
            if info[i][0] not in summaries_headers.keys():
                info.pop(i)

        # Finally, check for duplicates
        unique = [info[0]]

        # Go through each item and check if it already exists
        for i in range(1, len(info)):
            found = False
            for j in range(len(unique)):
                if info[i][0] == unique[j][0]:
                    found = True
                    break
            if not found:
                unique.append(info[i])

        info = unique

        assert (
            len(info) == NUM_HEADERS
        ), f"Expected {NUM_HEADERS} headers, got {len(info)} on {TOTAL}th entry"

        # Assert headers are the same
        headers = [header for header, _ in info]

        assert set(headers) == set(
            (summaries_headers.keys())
        ), f"Headers {headers} do not match"

        for header, content in info:
            summaries_headers[header].append(content)

        TOTAL += 1


# Fill the original headers dict
def add_originals(originals):
    TOTAL = 0
    for summary in originals:
        info = separate_by_headers(summary)

        for header in originals_headers.keys():
            if header not in [header for header, _ in info]:
                info.append((header, ""))

        # If it has a header that is not in the originals, discard it
        for i in range(len(info)):
            if info[i][0] not in originals_headers.keys():
                info.pop(i)

        # Finally, check for duplicates
        unique = [info[0]]

        for i in range(1, len(info)):
            found = False
            for j in range(len(unique)):
                if info[i][0] == unique[j][0]:
                    found = True
                    break
            if not found:
                unique.append(info[i])

        info = unique

        assert (
            len(info) == NUM_HEADERS
        ), f"Expected {NUM_HEADERS} headers, got {len(info)} on {TOTAL}th entry"

        # Assert headers are the same
        headers = [header for header, _ in info]

        assert set(headers) == set(
            (originals_headers.keys())
        ), f"Headers {headers} do not match on {TOTAL}th entry"

        for header, content in info:
            originals_headers[header].append(content)

        TOTAL += 1


def calculate_metrics():

    scores_headers = {}

    for header in originals_headers.keys():
        header_summary = header

        # Match because of discrepancies in the headers
        if header == "History of Present Illness:":
            header_summary = "History of present illness:"

        elif header == "Past Medical History:":
            header_summary = "Past medical history:"

        elif header == "Followup Instructions:":
            header_summary = "Follow-up Instructions:"

        print(f"Calculating metrics for {header}")

        rouge_results = rouge.compute(
            predictions=originals_headers[header],
            references=summaries_headers[header_summary],
        )

        print("Computed ROUGE for header {header}")

        bleu_results = bleu.compute(
            references=originals_headers[header],
            predictions=summaries_headers[header_summary],
        )

        print(f"Computed BLEU for header {header}")

        ttr_results = calculate_avg_ttr(summaries_headers[header_summary])

        print(f"Computed TTR for header {header}")

        bleurt_results = bleurt.compute(
            predictions=summaries_headers[header_summary],
            references=originals_headers[header],
        )

        print(f"Computed BLEURT for header {header}")

        bertscore_results = bertscore.compute(
            predictions=summaries_headers[header_summary],
            references=originals_headers[header],
            lang="en",
            verbose=True,
        )

        print(f"Computed BERTScore for header {header}")

        blanc_results = blanc_tune.eval_pairs(
            NOTES_REFERENCES,
            summaries_headers[header_summary],
        )

        print(f"Computed Blanc for header {header}")

        summac_results = model_conv.score(
            NOTES_REFERENCES,
            summaries_headers[header_summary],
        )

        print(f"Computed SummaC for header {header}")

        scores_headers[header] = {
            "rouge": rouge_results,
            "bleu": bleu_results,
            "ttr": ttr_results,
            "bleurt": bleurt_results["scores"],
            "bertscore": bertscore_results,
            "blanc": blanc_results,
            "summac": summac_results["scores"],
        }

    return scores_headers


def reset_headers():
    for header in summaries_headers.keys():
        summaries_headers[header] = []


def compute_model(predictions, save_path):
    print("Started separating summaries")
    separate_summaries(predictions)

    print("Done separating summaries")

    scores = calculate_metrics()

    print("Done calculating metrics")

    with open(save_path, "wb") as f:
        pickle.dump(scores, f)

    print(f"Saved {save_path}")

    reset_headers()


# Add original headers
originals = add_originals(REFERENCES)


### Proprietary Models
## Zero-Shot
# Evaluate GPT-3 Zero-Shot

predictions = pd.read_json("../outputs/zero-shot-responses-gpt3.jsonl", lines=True)

predictions["custom_id"] = predictions["custom_id"].apply(
    lambda x: int(x.split("-")[1])
)
predictions = predictions.sort_values("custom_id")
predictions = predictions["response"]
predictions = predictions.apply(lambda x: x["body"]["choices"][0]["message"]["content"])

compute_model(predictions=predictions, save_path="gpt3-zero-shot-headers-blanc.pkl")

# GPT-4 Zero-Shot

predictions = pd.read_json("../outputs/zero-shot-responses-gpt4.jsonl", lines=True)

predictions["custom_id"] = predictions["custom_id"].apply(
    lambda x: int(x.split("-")[1])
)
predictions = predictions.sort_values("custom_id")
predictions = predictions["response"]
predictions = predictions.apply(lambda x: x["body"]["choices"][0]["message"]["content"])

compute_model(predictions=predictions, save_path="gpt4-zero-shot-headers-blanc.pkl")

# Gemini Zero-Shot
predictions = []
temp = pd.read_json(
    "../outputs/zero-shot-responses-gemini-1.5.jsonl",
    lines=True,
)

for pred in temp[0]:
    predictions.append(pred["text"])

compute_model(predictions=predictions, save_path="gemini-zero-shot-headers-blanc.pkl")

## One-Shot
predictions = pd.read_json("../outputs/one-shot-responses-gpt4.jsonl", lines=True)

predictions["custom_id"] = predictions["custom_id"].apply(
    lambda x: int(x.split("-")[1])
)
predictions = predictions.sort_values("custom_id")
predictions = predictions["response"]
predictions = predictions.apply(lambda x: x["body"]["choices"][0]["message"]["content"])

compute_model(predictions=predictions, save_path="gpt4-one-shot-headers-blanc.pkl")

# Gemini One-Shot
predictions = []
temp = pd.read_json(
    "../outputs/one-shot-responses-gemini-1.5.jsonl",
    lines=True,
)

for pred in temp[0]:
    predictions.append(pred["text"])

compute_model(predictions=predictions, save_path="gemini-one-shot-headers-blanc.pkl")
