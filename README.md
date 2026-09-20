# Appraisal Pattern Reasoning for Fine-Grained Emotion Recognition

This repository contains the code, prompts, appraisal patterns, datasets, and experimental results for **fine-grained emotion recognition using Large Language Models (LLMs)**.

The project investigates whether cognitive appraisal information can support LLMs in distinguishing fine-grained emotions, particularly when different emotions have similar textual expressions.

## Overview

We compare three prompting strategies:

- **Zero-shot (ZS):** directly predicts the emotion from the input text.
- **Chain-of-Thought (CoT):** uses intermediate reasoning before making the final emotion prediction.
- **Appraisal Pattern (AP):** represents the input using cognitive appraisal dimensions and compares the resulting appraisal profile with predefined emotion-specific appraisal patterns.

The Appraisal Pattern framework uses seven appraisal dimensions:

1. Attention
2. Certainty
3. Effort
4. Pleasantness
5. Responsibility
6. Control
7. Circumstance

Experiments are conducted on **EmpatheticDialogues** and **EnISEAR** using multiple LLM backbones.

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
│   └── Experimental outputs from GPT-4o-mini, Mistral, and Qwen
│
└── README.md
```

---

## Datasets

### EmpatheticDialogues

EmpatheticDialogues contains conversations associated with fine-grained emotion labels.

The files used in this repository are located in:

```text
data/emp_chat.jsonl
data/empatheticdialogue
```

The Appraisal Pattern experiments use predefined appraisal patterns for the emotion categories in:

```text
patterns/APatterm_emp32E.txt
```

### EnISEAR

EnISEAR contains emotion-related situations together with cognitive appraisal annotations.

The processed files used in this repository are:

```text
data/EnISEAR_corpus.csv
data/EnISEAR_corpus_fixed.jsonl
```

The corresponding appraisal patterns are stored in:

```text
patterns/APattern_enISEAR_7E.txt
```

---

## Appraisal Pattern Framework

The proposed approach consists of two main stages.

### Phase I: Appraisal Extraction

The LLM analyzes the input and produces an appraisal profile over seven cognitive appraisal dimensions:

```text
[attention,
 certainty,
 effort,
 pleasantness,
 responsibility,
 control,
 circumstance]
```

Each appraisal dimension is represented by a continuous score and supported by textual evidence.

### Phase II: Pattern Matching and Emotion Prediction

The inferred appraisal profile is compared with predefined emotion-specific appraisal patterns.

If one emotion has a clearly stronger match, it is selected directly. When several candidate emotions have similar appraisal profiles, the model additionally considers the original context, appraisal evidence, and emotion-specific discriminative cues.

---

## Prompt Files

Prompt templates are stored in the `prompts/` directory.

```text
prompts/
├── prompt_emp_pha1.txt
├── prompt_emp_pha2.txt
└── prompt_enISEAR.txt
```

For EmpatheticDialogues, the AP framework uses separate prompts for appraisal extraction and emotion prediction.

For EnISEAR, the corresponding prompt is provided in `prompt_enISEAR.txt`.

---

## Running the Experiments

### EmpatheticDialogues

Zero-shot and CoT:

```bash
python code/LLMs_CoT_ZeroShot_emp.py
```

Appraisal Pattern:

```bash
python code/LLMs_AP_continuous_emp.py
```

### EnISEAR

Zero-shot and CoT:

```bash
python code/LLMs_CoT_ZeroShot_EnIS.py
```

Appraisal Pattern:

```bash
python code/LLMs_AP_continuous_EnIS.py
```

Before running the scripts, update dataset paths and model/API configurations according to your local environment.

---

## LLM Backbones

The experiments include the following LLM families:

- **GPT-4o-mini**
- **Mistral**
- **Qwen**

The decoding temperature is set to:

```text
temperature = 0
```

Other decoding parameters use the default settings of the corresponding APIs.

---

## Results

Experimental predictions are stored in:

```text
results/
```

The directory contains outputs for different combinations of:

```text
Dataset × LLM × Prompting Method
```

including:

- Zero-shot
- Chain-of-Thought
- Appraisal Pattern

for GPT-4o-mini, Mistral, and Qwen.

The main evaluation metrics are:

- Accuracy
- Macro-F1
- Per-emotion Precision
- Per-emotion Recall
- Per-emotion F1-score

---

## Requirements

The main Python dependencies include:

```text
pandas
numpy
openai
python-dotenv
scikit-learn
```

Install the required packages with:

```bash
pip install pandas numpy openai python-dotenv scikit-learn
```

It is recommended to use Python 3.10 or later.

---

## API Configuration

Do not store API keys directly in the source code or upload them to GitHub.

API credentials can be stored in a local `.env` file:

```env
OPENAI_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
QWEN_API_KEY=your_key_here
```

The `.env` file should be excluded from Git using `.gitignore`.

---

## Reproducibility

To reproduce an experiment:

1. Prepare the corresponding dataset in `data/`.
2. Select the appropriate prompt from `prompts/`.
3. Select the appraisal pattern file from `patterns/` for AP experiments.
4. Configure the target LLM API.
5. Run the corresponding Python script in `code/`.
6. Save model predictions to `results/`.
7. Evaluate predictions using Accuracy and Macro-F1.

---

## Citation

If you use this repository in academic work, please cite the corresponding paper.

```bibtex
@inproceedings{appraisal_pattern_erc2026,
  title     = {Appraisal Pattern Reasoning for Fine-Grained Emotion Recognition},
  author    = {...},
  year      = {2026}
}
```

The citation information will be updated after publication.

---

## License

The source code in this repository is provided for research purposes.

The datasets remain subject to the licenses and terms of their original providers. Please refer to the original EmpatheticDialogues and EnISEAR resources before redistributing or using the datasets.

---

## Acknowledgements

This project builds on research in cognitive appraisal theory, emotion recognition, and Large Language Model reasoning.

We thank the authors and maintainers of the datasets and language models used in this study.
