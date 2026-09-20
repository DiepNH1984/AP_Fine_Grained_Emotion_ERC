# Appraisal-Guided Fine-Grained Emotion Recognition

An AI-based framework for fine-grained emotion recognition using Large Language Models (LLMs) and cognitive appraisal patterns.

## Overview

This project investigates fine-grained emotion recognition from conversational text.
The proposed framework uses cognitive appraisal dimensions to support emotion reasoning
instead of directly mapping text to emotion labels.

The framework consists of two main phases:

1. **Appraisal Extraction**
   - Extracts seven appraisal dimensions from the conversation:
     attention, certainty, effort, pleasantness, responsibility, control, and circumstance.

2. **Emotion Prediction**
   - Compares the inferred appraisal profile with predefined emotion-specific appraisal patterns.
   - Uses conversational evidence to distinguish between similar emotions.

## Project Structure

```text
AP_Fine_Grained_Emotion_ERC/
├── code/
│   ├── LLMs_AP_continuous_emp.py
│   ├── LLMs_AP_continuous_EnIS.py
│   ├── LLMs_CoT_ZeroShot_emp.py
│   └── LLMs_CoT_ZeroShot_EnIS.py
│
├── data/
│   ├── EmpatheticDialogues/
│   └── EnISEAR/
│
├── prompts/
│   ├── phase1_prompt.txt
│   └── phase2_prompt.txt
│
├── patterns/
│   └── appraisal_patterns.json
│
├── results/
│
├── requirements.txt
└── README.md
