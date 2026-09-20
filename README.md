# AP Fine-Grained Emotion ERC

**Appraisal Pattern Reasoning for Fine-Grained Emotion Recognition in Conversations**

This repository contains the code, prompts, appraisal patterns, datasets, and experimental outputs for studying **fine-grained emotion recognition with Large Language Models (LLMs)**.

The project compares conventional prompting approaches with an **Appraisal Pattern (AP)** framework that uses cognitive appraisal dimensions to support emotion reasoning and discrimination.

---

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

---

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

## Datasets

Experiments are conducted on two datasets:

- **EmpatheticDialogues** – conversational data with fine-grained emotion labels.
- **EnISEAR** – emotion-related situations annotated with cognitive appraisal dimensions.

Processed data used in the experiments is provided in the `data/` directory.

Please refer to the original dataset sources and licenses before redistributing the data.

---

## LLM Backbones

Experiments are conducted using multiple LLM backbones:

- **GPT-4o-mini**
- **Mistral Small**
- **Qwen**

For all experiments, `temperature = 0`; other decoding parameters use the corresponding API defaults.

---

## Prompts and Appraisal Patterns

Prompt templates are stored in:

```text
prompts/

## Authors
Hoang-Diep Nguyen, Manh-Cuong Phan, Van-Quyet Nguyen, and Minh-Tien Nguyen
Hung Yen University of Technology and Engineering (UTEHY), Hung Yen, Vietnam.
Hoang-Diep Nguyen — diepnh@utehy.edu.vn
Manh-Cuong Phan — cuongpm@spkt.edu.vn
Van-Quyet Nguyen — quyetic@utehy.edu.vn · ORCID: 0000-0002-6898-4224
Minh-Tien Nguyen — tiennm@utehy.edu.vn · ORCID: 0000-0002-5028-0608
