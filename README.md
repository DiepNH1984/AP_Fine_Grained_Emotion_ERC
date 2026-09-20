# AP Fine-Grained Emotion ERC

**Appraisal Pattern Reasoning for Fine-Grained Emotion Recognition in Conversations**

This repository provides the code, prompts, appraisal patterns, datasets, and experimental outputs for **fine-grained emotion recognition using Large Language Models (LLMs)**.

We compare conventional prompting methods with an **Appraisal Pattern (AP)** framework that incorporates cognitive appraisal dimensions to improve emotion reasoning and discrimination.

## Methods

We evaluate three prompting strategies:

- **Zero-shot (ZS):** directly predicts the target emotion from the input.
- **Chain-of-Thought (CoT):** performs intermediate reasoning before emotion prediction.
- **Appraisal Pattern (AP):** infers cognitive appraisal dimensions and compares the resulting appraisal profile with predefined emotion-specific patterns.

The AP framework uses seven appraisal dimensions:

`attention`, `certainty`, `effort`, `pleasantness`, `responsibility`, `control`, and `circumstance`.

For conversational emotion recognition, AP consists of two phases:

1. **Phase I – Appraisal Extraction:** infer a seven-dimensional appraisal profile with supporting textual evidence.
2. **Phase II – Emotion Prediction:** compare the appraisal profile with emotion-specific patterns and use contextual evidence to distinguish similar emotions.

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
│   └── empatheticdialogue/
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
│   └── Experimental outputs
│
└── README.md
