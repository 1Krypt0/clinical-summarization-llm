from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import pandas as pd
from datasets import Dataset

test_data = Dataset.from_csv("./data/single-discharge-7.6k-test-formatted.csv")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device is {device}")

DEFAULT_SYSTEM_PROMPT = """
You are an expert clinical assistant. You will receive a collection of clinical notes. You will summarize them in the style of a discharge summary.
""".strip()


def generate_testing_prompt_mistral(
    notes: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    return f"""[INST] {system_prompt}

### Input:

{notes.strip()}

### Summary:

[/INST]
""".strip()


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(
    "theKrypt0/Mistral-FT",
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)

tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    "theKrypt0/Mistral-FT", quantization_config=bnb_config
)

model.eval()

results = []
TOTAL = 0

print("Beginning Finetuned Mistral Evaluation")

for entry in test_data:

    # Generate prompt without the response part
    input_text = generate_testing_prompt_mistral(entry["notes"])
    input_ids = tokenizer(input_text, return_tensors="pt").to(device)
    inputs_length = len(input_ids["input_ids"][0])

    with torch.inference_mode():
        outputs = model.generate(**input_ids, max_new_tokens=4096, pad_token_id=2)
        summary = tokenizer.decode(
            outputs[0][inputs_length:], skip_special_tokens=True
        ).strip()

    results.append(summary)

    TOTAL += 1

    if TOTAL % 10 == 0:
        print(f"Evaluated {TOTAL} admissions")

    if TOTAL == 5:
        break


results = pd.DataFrame(results, columns=["summary"])

results.to_csv("./outputs/mistral_finetuned_preds.csv", index=False)
