import pickle
import nltk

import pandas as pd
from evaluate import load

rouge = load("rouge")
bleu = load("bleu")
bertscore = load("bertscore")
bleurt = load("bleurt")


def calculate_ttr(text):
    words = nltk.word_tokenize(text)
    unique_words = set(words)
    return len(unique_words) / len(words)


def calculate_avg_ttr(data) -> float:
    temp = []
    for entry in data:
        temp.append(calculate_ttr(entry))

    return sum(temp) / len(temp)


DATA = pd.read_csv("../data/single-discharge-7.6k-test-formatted.csv")
REFERENCES = DATA["summary"]
NOTES_REFERENCES = DATA["notes"]


def compute_scores(predictions: pd.Series, save_path: str) -> None:

    print("Entered compute")

    rouge_results = rouge.compute(predictions=predictions, references=NOTES_REFERENCES)

    print("Computed ROUGE")

    bleu_results = bleu.compute(predictions=predictions, references=NOTES_REFERENCES)

    print("Computed BLEU")

    ttr_results = calculate_avg_ttr(predictions)

    print("Computed TTR")

    bleurt_results = bleurt.compute(
        predictions=predictions, references=NOTES_REFERENCES
    )

    print("Computed BLEURT")

    bertscore_results = bertscore.compute(
        predictions=predictions,
        references=REFERENCES,
        verbose=True,
        lang="en",
    )

    print("Computed BERTScore")

    scores = {
        "rouge": rouge_results,
        "bleu": bleu_results,
        "ttr": ttr_results,
        "bleurt": bleurt_results["scores"],
        "bertscore": bertscore_results,
    }

    with open(save_path, "wb") as f:
        pickle.dump(scores, f)

    print("Saved Files")


### Open-Source Models
# Evaluate Mistral Baseline
predictions = pd.read_csv("../outputs/mistral_baseline_preds.csv")["summary"]

compute_scores(predictions=predictions, save_path="mistral-baseline-overlap.pkl")

print("Computed Mistral Baseline")

# Evaluate Mistral Finetuned

predictions = pd.read_csv("../outputs/mistral_finetuned_preds.csv")["summary"]

compute_scores(predictions=predictions, save_path="mistral-finetuned-overlap.pkl")

print("Computed Mistral Baseline")

# Evaluate Llama Baseline
predictions = pd.read_csv("../outputs/llama_baseline_preds.csv")["summary"]

compute_scores(predictions=predictions, save_path="llama-baseline-overlap.pkl")

print("Computed Llama Baseline")
# Evaluate Llama finetuned
predictions = pd.read_csv("../outputs/llama_finetuned_preds.csv")["summary"]

compute_scores(predictions=predictions, save_path="llama-finetuned-overlap.pkl")

print("Computed Llama Finetuned")

### Proprietary Models
## Zero-Shot
# Eval GPT-3.5
predictions = pd.read_json("../outputs/zero-shot-responses-gpt3.jsonl", lines=True)

predictions["custom_id"] = predictions["custom_id"].apply(
    lambda x: int(x.split("-")[1])
)
predictions = predictions.sort_values("custom_id")
predictions = predictions["response"]
predictions = predictions.apply(lambda x: x["body"]["choices"][0]["message"]["content"])

compute_scores(predictions=predictions, save_path="gpt3-zero-shot-overlap.pkl")

print("Computed GPT-3 Zero")

# Eval GPT-4
predictions = pd.read_json("../outputs/zero-shot-responses-gpt4.jsonl", lines=True)

predictions["custom_id"] = predictions["custom_id"].apply(
    lambda x: int(x.split("-")[1])
)
predictions = predictions.sort_values("custom_id")
predictions = predictions["response"]
predictions = predictions.apply(lambda x: x["body"]["choices"][0]["message"]["content"])

compute_scores(predictions=predictions, save_path="gpt4-zero-shot-overlap.pkl")

print("Computed GPT-4 Zero")

# Eval Gemini
predictions = []
temp = pd.read_json(
    "../outputs/zero-shot-responses-gemini-1.5.jsonl",
    lines=True,
)

for pred in temp[0]:
    predictions.append(pred["text"])

compute_scores(predictions=predictions, save_path="gemini-zero-shot-overlap.pkl")

print("Computed Gemini Zero")

## One-Shot
# Eval GPT-4
predictions = pd.read_json("../outputs/one-shot-responses-gpt4.jsonl", lines=True)

predictions["custom_id"] = predictions["custom_id"].apply(
    lambda x: int(x.split("-")[1])
)
predictions = predictions.sort_values("custom_id")
predictions = predictions["response"]
predictions = predictions.apply(lambda x: x["body"]["choices"][0]["message"]["content"])

compute_scores(predictions=predictions, save_path="gpt4-one-shot-overlap.pkl")

print("Computed GPT-4 One")

# Eval Gemini
predictions = []
temp = pd.read_json(
    "../outputs/one-shot-responses-gemini-1.5.jsonl",
    lines=True,
)

for pred in temp[0]:
    predictions.append(pred["text"])

compute_scores(predictions=predictions, save_path="gemini-one-shot-overlap.pkl")

print("Computed Gemini One")
