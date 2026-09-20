•	Tiêu đề Appraisal Pattern Reasoning for Fine-Grained Emotion Recognition in Conversations
•	This repository contains: the code, prompts, appraisal patterns, datasets, and experimental results for fine-grained emotion recognition using Large Language Models (LLMs)
•	 
Build an Appraisal Pattern (AP) framework based on seven appraisal dimensions to enable LLMs to distinguish between closely related emotions in a structured, grounded manner, thereby improving performance compared to baseline, Zero-shot and CoT approaches. The Appraisal Pattern framework has 7 appraisal dimensions: 1. Attention 2. Certainty 3. Effort 4. Pleasantness 5. Responsibility 6. Control 7. Circumstance
Set up: The project is implemented in Python and developed using Visual Studio Code.
Installation
Requirements: Python 3.10+, Visual Studio Code, Git, and the VS Code Python extension.
git clone https://github.com/DiepNH1984/AP_Fine_Grained_Emotion_ERC.git
cd AP_Fine_Grained_Emotion_ERC

python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
Open the project in VS Code, select the .venv interpreter, and run:
python <filename>.py


Experiments are conducted on **EmpatheticDialogues** and **EnISEAR** using multiple LLM backbones.

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
