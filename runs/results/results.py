import pickle

with open("./gemini-zero-shot.pkl", "rb") as f:
    gemini_zero = pickle.load(f)

with open("./gpt3-zero-shot.pkl", "rb") as f:
    gpt3_zero = pickle.load(f)

with open("./gpt4-one-shot.pkl", "rb") as f:
    gpt4_one = pickle.load(f)

with open("./gpt4-zero-shot.pkl", "rb") as f:
    gpt4_zero = pickle.load(f)

with open("./gemini-one-shot.pkl", "rb") as f:
    gemini_one = pickle.load(f)

with open("./llama-baseline.pkl", "rb") as f:
    llama_baseline = pickle.load(f)

with open("./mistral-baseline.pkl", "rb") as f:
    mistral_baseline = pickle.load(f)

with open("./llama-finetuned.pkl", "rb") as f:
    llama_finetuned = pickle.load(f)

with open("./mistral-finetuned.pkl", "rb") as f:
    mistral_finetuned = pickle.load(f)

print("\n\nPROPRIETARY MODELS\n\n")

print("GPT-3 Zero-Shot Results")
print(gpt3_zero)
print("\nGPT-4 Zero-Shot Results")
print(gpt4_zero)
print("\nGemini 1.5 Pro Zero-Shot Results")
print(gemini_zero)
print("\nGPT-4 One-Shot Results")
print(gpt4_one)
print("\nGemini 1.5 Pro One-Shot Results")
print(gemini_one)

print("\n\nOPEN SOURCE MODELS\n\n")

print("Llama Baseline Results")
print(llama_baseline)
print("\nMistral Baseline Results")
print(mistral_baseline)
print("\nLlama Finetuned Results")
print(llama_finetuned)
print("\nMistral Finetuned Results")
print(mistral_finetuned)
