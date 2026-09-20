# AP Fine-Grained Emotion ERC

**Appraisal Pattern Reasoning for Fine-Grained Emotion Recognition in Conversations**

This repository contains the code, prompts, appraisal patterns, datasets, and experimental outputs for studying **fine-grained emotion recognition with Large Language Models (LLMs)**.

The project compares conventional prompting approaches with an **Appraisal Pattern (AP)** framework that uses cognitive appraisal dimensions to support emotion reasoning and discrimination.

## Methods

We evaluate three prompting strategies:

- **Zero-shot (ZS):** directly predicts the target emotion from the input text.
- **Chain-of-Thought (CoT):** performs intermediate reasoning before emotion prediction.
- **Appraisal Pattern (AP):** reasons through cognitive appraisal dimensions and matches the resulting appraisal profile with predefined emotion-specific patterns.

The AP framework uses seven appraisal dimensions:

`attention`, `certainty`, `effort`, `pleasantness`, `responsibility`, `control`, and `circumstance`.

For conversational emotion recognition, AP follows two main phases:

1. **Phase I – Appraisal Extraction:** infer a seven-dimensional appraisal profile from the input with supporting textual evidence.
2. **Phase II – Emotion Prediction:** compare the inferred profile with emotion-specific appraisal patterns and use contextual evidence to resolve similar candidate emotions.

## Repository Structure

```text
AP_Fine_Grained_Emotion_ERC/
│
├── code/
│   ├── LLMs_AP_continuous_EnIS.py
│   ├── LLMs_AP_continuous_emp.py
│   ├── LLMs_CoT_ZeroShot_EnIS.py
│   └── LLMs_CoT_ZeroShot_emp.py
│
├── data/
│   ├── EnISEAR_corpus.csv
│   ├── EnISEAR_corpus_fixed.jsonl
│   ├── emp_chat.jsonl
│   └── empatheticdialogue
│
├── patterns/
│   ├── APatterm_emp32E.txt
│   └── APattern_enISEAR_7E.txt
│
├── prompts/
│   ├── prompt_emp_pha1.txt
│   ├── prompt_emp_pha2.txt
│   └── prompt_enISEAR.txt
│
├── results/
│   └── Experimental outputs from different LLMs and prompting methods
│
└── README.md
```

## Datasets

Experiments are conducted on:

- **EmpatheticDialogues** – conversational data with fine-grained emotion labels.
- **EnISEAR** – emotion-related situations with cognitive appraisal annotations.

Processed data used by the experiments is provided in the `data/` directory.

Please refer to the original dataset sources and licenses before redistributing the data.

## LLM Backbones

The experiments use multiple LLM families:

- **GPT-4o-mini**
- **Mistral Small**
- **Qwen**

For inference, `temperature = 0`; other decoding parameters use the corresponding API defaults.

## Running the Experiments

### EmpatheticDialogues

Zero-shot / CoT:

```bash
python code/LLMs_CoT_ZeroShot_emp.py
```

Appraisal Pattern:

```bash
python code/LLMs_AP_continuous_emp.py
```

### EnISEAR

Zero-shot / CoT:

```bash
python code/LLMs_CoT_ZeroShot_EnIS.py
```

Appraisal Pattern:

```bash
python code/LLMs_AP_continuous_EnIS.py
```

Before running the scripts, configure the required dataset paths and model API credentials.

## Prompts and Appraisal Patterns

Prompt templates are stored in:

```text
prompts/
```

Emotion-specific appraisal patterns are stored in:

```text
patterns/
```

These files contain the main prompting and appraisal knowledge used by the AP framework.

## Results

Model predictions and experimental outputs are stored in:

```text
results/
```

Results are organized across different combinations of:

```text
Dataset × LLM × Method
```

The primary evaluation metrics are **Accuracy** and **Macro-F1**, together with per-emotion Precision, Recall, and F1-score.

## Environment

The implementation is written in Python and mainly uses packages such as:

```text
pandas
numpy
openai
python-dotenv
scikit-learn
```

Example installation:

```bash
pip install pandas numpy openai python-dotenv scikit-learn
```

## API Keys

Do **not** store API keys directly in source files.

A local `.env` file can be used for credentials, for example:

```env
OPENAI_API_KEY=your_api_key
MISTRAL_API_KEY=your_api_key
QWEN_API_KEY=your_api_key
```

Make sure `.env` is excluded from Git tracking.

## Citation

If you use this repository in academic research, please cite the corresponding paper.

Citation information will be updated after publication.

## License

The code in this repository is provided for academic and research use.

Datasets and external models remain subject to the licenses and terms of their original providers.
