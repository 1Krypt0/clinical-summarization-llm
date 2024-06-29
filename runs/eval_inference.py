import torch
import random
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaConfig,
)
from datasets import Dataset
import numpy as np

test_data = Dataset.from_csv("./data/single-discharge-7.6k-test-formatted.csv")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device is {device}")

DEFAULT_SYSTEM_PROMPT = """
You are an expert clinical assistant. You will receive a collection of clinical notes. You will summarize them in the style of a discharge summary.
""".strip()


def generate_testing_prompt_llama(
    notes: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    return f"""[INST] <<SYS>> {system_prompt} <</SYS>>

### Input:

{notes.strip()}

### Summary:

[/INST]
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


# Load the model and tokenizer
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer_llama = AutoTokenizer.from_pretrained(
    "theKrypt0/Llama-FT-2",
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)

tokenizer_llama.pad_token = tokenizer_llama.eos_token

config = LlamaConfig.from_pretrained("theKrypt0/Llama-FT-2")
config.rope_scaling = {"type": "linear", "factor": 4.0}

model_llama = AutoModelForCausalLM.from_pretrained(
    "theKrypt0/Llama-FT-2",
    quantization_config=bnb_config,
    config=config,
)

tokenizer_mistral = AutoTokenizer.from_pretrained(
    "theKrypt0/Mistral-FT",
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)

tokenizer_mistral.pad_token = tokenizer_mistral.eos_token


model_mistral = AutoModelForCausalLM.from_pretrained(
    "theKrypt0/Mistral-FT",
    quantization_config=bnb_config,
)


model_llama.eval()

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
    enable_timing=True
)

# Choose random note
random.seed(42)
dummy_input = random.choice(test_data)["note"]
dummy_input = generate_testing_prompt_llama(dummy_input)
input_ids = tokenizer_llama(dummy_input, return_tensors="pt").to(device)
inputs_length = len(input_ids["input_ids"][0])

repetitions = 100
warmup = 5
timings = np.zeros((repetitions, 1))

for _ in range(warmup):
    _ = model_llama.generate(**input_ids, max_new_tokens=4096, pad_token_id=2)

with torch.inference_mode():
    for rep in range(repetitions):
        starter.record()
        _ = model_llama.generate(**input_ids, max_new_tokens=4096, pad_token_id=2)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time


mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(f"Your mean time on Llama was {mean_syn}")
print(f"Your std deviation on Llama was {std_syn}")


# Repeat for Mistral
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
    enable_timing=True
)

# Choose random note
random.seed(42)
dummy_input = random.choice(test_data)["note"]
dummy_input = generate_testing_prompt_mistral(dummy_input)
input_ids = tokenizer_mistral(dummy_input, return_tensors="pt").to(device)
inputs_length = len(input_ids["input_ids"][0])

repetitions = 100
warmup = 5
timings = np.zeros((repetitions, 1))

for _ in range(warmup):
    _ = model_mistral.generate(**input_ids, max_new_tokens=4096, pad_token_id=2)

with torch.inference_mode():
    for rep in range(repetitions):
        starter.record()
        _ = model_mistral.generate(**input_ids, max_new_tokens=4096, pad_token_id=2)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time


mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(f"Your mean time on Mistral was {mean_syn}")
print(f"Your std deviation on Mistral was {std_syn}")
