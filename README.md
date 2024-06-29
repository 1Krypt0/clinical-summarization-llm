# Discharge Summary Generation from Clinical Notes

This repo contains the code developed to evaluate a series of Large Language Models on Abstractive Summarization, with the end-goal of generating a discharge summary out of the notes written during a hospital admission.

We evaluate:

- Llama-2-7B
- Mistral-7B
- GPT-3.5
- GPT-4
- Gemini 1.5 Pro

It the project for my Master's Dissertation, titled "Extraction of Clinical Narratives from Medical Records using Large Language Models", for which I am also writing two papers. One is a Systematic Literature Review on Clinical Information Extraction using LLMs. It has been submitted to the [ACM Transactions on Computing for Helathcare](https://dl.acm.org/journal/health). The other explains our methodology, results and insights and is currently being written.

This project was developed by me, Tiago Rodrigues, under the supervision of Professor Carla Teixeira Lopes, from October 2023 to June 2024.

## Repo Structure

**runs/** - Contains the code for each evaluation, as well as the results obtained per model.

**torchtune_recipes/** - Contains the recipes used to fine-tune Llama and Mistral to our domain, which was achieved with the help of [torchtune](https://github.com/pytorch/torchtune), as well as an abstract instruct template, as our templates were built-in on the dataset.

**exploration.ipynb** - Simple exploration of the barebones MIMIC-III dataset and our filtered data.

**filter_data.ipynb** - Data filtering used to reduce the dataset.

**format_data.ipynb** - Dataset creation using a standard format for easier processing and model training.

## Contact

[Tiago Rodrigues](mailto:up201907021@up.pt)
