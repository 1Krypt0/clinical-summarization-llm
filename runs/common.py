import pandas as pd
import evaluate
import pickle


def load_test_data():
    data = pd.read_csv("../data/single-discharge-7.6k-test-formatted.csv")
    return data


ZERO_SHOT_PROMPT = """You are an expert clinical assistant. You will receive a collection of clinical notes. You will summarize them in the style of a discharge summary, outputting the following fields, and no additional information:

Admission Date: [Admission Date]
Discharge Date: [Discharge Date]

Date of Birth: [Date of Birth]
Sex: [Sex]

Service: [Hospital Service used]

Chief Complaint: [Chief Complaint]

Major Surgical or Invasive Procedure: [Major Surgical or Invasive Procedures conducted]

History of present illness: [Description of the motives for admission, explaining what happened to the patient for admission]

Past medical history: [Previous medical conditions]

Allergies: [Allergies the patient might have]

Medications on Admission: [Medications on Admission]

Family History: [History of illness in the family]

Social History: [Social Conditions of the patient]

Physical Exam: [Description of conditions on Admission]

Pertinent Results: [Relevant Results from the exams]

Brief Hospital Course: [Description of hospital course]

Discharge Diagnosis: [Diagnosis on Discharge]

Discharge Medications: [Medications on Discharge]

Discharge Disposition: [Discharge Disposition]

Discharge Instructions: [Discharge Instructions]

Follow-up Instructions: [Follow-up Instructions]

Discharge Condition: [Discharge Condition]
"""


def load_one_shot_notes() -> str:
    with open("./one-shot-notes.txt", "r") as text_file:
        notes = "".join(text_file.readlines())

    return notes


def load_one_shot_summary() -> str:
    with open("./one-shot-summary.txt", "r") as summary:
        summary = "".join(summary.readlines())

    return summary


ONE_SHOT_PROMPT = (
    ZERO_SHOT_PROMPT
    + f"""
For example, given the notes:

### NOTES START ###
{load_one_shot_notes()}
### NOTES END ###

You would summarize them as:

### SUMMARY START ###
{load_one_shot_summary()}
### SUMMARY END ###
"""
)


def evaluate_summaries(predictions):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    references = load_test_data()["summary"]

    rouge_score = rouge.compute(predictions=predictions, references=references)
    bleu_score = bleu.compute(predictions=predictions, references=references)

    return rouge_score, bleu_score


def save_scores(path: str, rouge_score: dict, bleu_score: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump([rouge_score, bleu_score], f)
