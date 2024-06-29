# Runs

Here we have the code to generate the output and evaluate it on the set of metrics I used. For each model, we computed:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- BLEU
- Type/Token Ration (TTR)
- BERTScore
- BLEURT
- [BLANC](https://github.com/PrimerAI/blanc)
- [SummaC](https://github.com/tingofurro/summac)

For the reasoning behind each one, check the publication.

Under each `run_*` file, we generate the output predictions. These are then passed through the eval scripts:

- `eval_results.py` - Computes the scores considering the document as a unit
- `eval_headers.py` - Computes the scores per section of a discharge summary
- `eval_overlap.py` - Checks the overlap between the generated summaries and the original notes
- `eval_inference.py` - Calculates the inference time of Llama and Mistral.
