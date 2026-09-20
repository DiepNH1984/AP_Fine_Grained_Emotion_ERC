from importlib.resources import path
import json
from collections import defaultdict

import pandas as pd
import numpy as np

from openai import BadRequestError, OpenAI

import os
import traceback
from dotenv import load_dotenv


# ES dataset: conversation_id, label, appraisal
# Empathetic Dialogues dataset:
# conversation_id, label, situation, conversation


def read_jsonl_to_dataframe(    jsonl_file_path=r"G:\Python\emp_chat.jsonl"):
    data = []
    with open(jsonl_file_path, "r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    return df

load_dotenv()




def read_jsonl_to_dataframe(jsonl_file_path):
    data = []
    with open(jsonl_file_path, "r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    return df

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(
            f"Missing environment variable: {name}. Please check your .env file."
        )
    return value.strip()


def _messages(system_prompt, user_prompt):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _ensure_json_string(content, provider: str, model_name: str) -> str:
    """Normalize provider output and fail loudly if it is empty/invalid JSON."""
    if content is None:
        raise RuntimeError(f"[{provider}] model={model_name}: response content is None")

    if not isinstance(content, str):
        content = str(content)

    content = content.strip()
    if not content:
        raise RuntimeError(f"[{provider}] model={model_name}: response content is empty")

    # All downstream code expects json.loads(result_str) to work.
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"[{provider}] model={model_name}: returned non-JSON content: {content[:1000]!r}"
        ) from exc

    return content


def _azure_v1_client() -> OpenAI:
    api_key = _require_env("AZURE_OPENAI_API_KEY")
    endpoint = _require_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    if not endpoint.endswith("/openai/v1"):
        endpoint += "/openai/v1"
    return OpenAI(api_key=api_key, base_url=endpoint + "/", timeout=120.0, max_retries=2)

def call_open_ai_api_gpt5(model, system_prompt, user_prompt):
    model_name = (
        model
        or _require_env("OPENAI_GPT5_MODEL")
    ).strip()

    client = OpenAI(
        api_key=_require_env("OPENAI_API_KEY")
    )

    system_prompt = (
        system_prompt.rstrip()
        + "\n\nReturn the final answer as a valid JSON object only."
    )

    response = client.responses.create(
        model=model_name,

        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        text={
            "format": {
                "type": "json_object"
            }
        },

        max_output_tokens=4096,
    )

    output = (response.output_text or "").strip()

    if not output:
        raise RuntimeError(
            f"OpenAI model={model_name}: empty output"
        )

    return output


def call_azure_open_ai_api(model, system_prompt, user_prompt):
    """Azure GPT-4o/4.1 style model via Chat Completions -> JSON string."""
    deployment = model or _require_env("AZURE_OPENAI_DEPLOYMENT")
    client = _azure_v1_client()

    response = client.chat.completions.create(
        model=deployment,
        messages=_messages(system_prompt, user_prompt),
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    content = response.choices[0].message.content
    return _ensure_json_string(content, "Azure-Chat", deployment)

def call_open_ai_api_gpt4omini(model, system_prompt, user_prompt):
    """OpenAI GPT-4o-mini via Chat Completions -> JSON string."""
    model_name = model or os.getenv("OPENAI_GPT4O_MINI_MODEL", "gpt-4o-mini")
    client = OpenAI(
        api_key=_require_env("OPENAI_API_KEY"),
        timeout=120.0,
        max_retries=2,
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=_messages(system_prompt, user_prompt),
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    content = response.choices[0].message.content
    return _ensure_json_string(content, "OpenAI-Chat", model_name)
def call_azure_open_ai_GPT5api(model, system_prompt, user_prompt):
    """Azure OpenAI GPT-5 family via Responses API -> JSON string."""

    deployment = (
        model
        or _require_env("AZURE_OPENAI_GPT5_DEPLOYMENT")
    ).strip()

    client = _azure_v1_client()

    response = client.responses.create(
        model=deployment,

        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        text={
            "format": {
                "type": "json_object"
            }
        },

        max_output_tokens=4096,
    )

    status = getattr(response, "status", None)

    if status not in (None, "completed"):
        details = getattr(response, "incomplete_details", None)

        reason = (
            getattr(details, "reason", None)
            if details is not None
            else None
        )

        error = getattr(response, "error", None)

        if reason == "content_filter":
            return None

        raise RuntimeError(
            f"[Azure-GPT5] deployment={deployment}: "
            f"status={status}, reason={reason}, error={error}"
        )

    content = response.output_text

    if not content:
        raise RuntimeError(
            f"[Azure-GPT5] deployment={deployment}: empty output"
        )

    return _ensure_json_string(
        content,
        "Azure-GPT5",
        deployment
    )


def call_Qwen(model, system_prompt, user_prompt):
    """Qwen through Alibaba Model Studio OpenAI-compatible Chat API -> JSON string."""
    model_name = model or os.getenv("QWEN_MODEL", "qwen-plus")
    base_url = _require_env("QWEN_BASE_URL").rstrip("/")

    # QWEN_BASE_URL must already point to .../compatible-mode/v1
    client = OpenAI(
        api_key=_require_env("QWEN_API_KEY"),
        base_url=base_url + "/",
        timeout=120.0,
        max_retries=2,
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=_messages(system_prompt, user_prompt),
        response_format={"type": "json_object"},
        temperature=0.0,
        # JSON mode and Qwen thinking mode are not compatible for many Qwen3 models.
        extra_body={"enable_thinking": False},
    )

    content = response.choices[0].message.content
    return _ensure_json_string(content, "Qwen", model_name)


def call_Mistral(model, system_prompt, user_prompt):#lối 429 giới hạn 
    """Mistral official SDK JSON mode -> JSON string."""
    try:
        from mistralai.client import Mistral
    except ImportError as exc:
        raise RuntimeError(
            "Missing package 'mistralai'. Install with: pip install -U mistralai"
        ) from exc

    model_name = model or os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    client = Mistral(api_key=_require_env("MISTRAL_API_KEY"))

    response = client.chat.complete(
        model=model_name,
        messages=_messages(system_prompt, user_prompt),
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    if not getattr(response, "choices", None):
        raise RuntimeError(f"[Mistral] model={model_name}: response.choices is empty")

    content = response.choices[0].message.content
    if isinstance(content, str):
        text = content
    else:
        texts = []
        for item in content or []:
            if hasattr(item, "text"):
                texts.append(item.text)
            elif isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
        text = "".join(texts)

    return _ensure_json_string(text, "Mistral", model_name)
def call_groq(model, system_prompt, user_prompt):
    """Groq GPT-OSS via Groq Chat Completions -> JSON string."""

    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "Missing package 'groq'. Install with: pip install -U groq"
        ) from exc

    model_name = model or os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b"
    )

    client = Groq(
        api_key=_require_env("GROQ_API_KEY")
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=_messages(
            system_prompt,
            user_prompt
        ),
        response_format={
            "type": "json_object"
        },
        temperature=0.0,
    )

    if not getattr(response, "choices", None):
        raise RuntimeError(
            f"[Groq] model={model_name}: response.choices is empty"
        )

    content = response.choices[0].message.content

    return _ensure_json_string(
        content,
        "Groq",
        model_name
    )


def call_gemini(model, system_prompt, user_prompt):
    """Gemini google-genai SDK -> JSON string."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Missing package 'google-genai'. Install with: pip install -U google-genai"
        ) from exc

    model_name = model or os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
    client = genai.Client(api_key=_require_env("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    try:
        text = response.text
    except Exception as exc:
        candidates = getattr(response, "candidates", None)
        feedback = getattr(response, "prompt_feedback", None)
        raise RuntimeError(
            f"[Gemini] model={model_name}: no response.text. "
            f"prompt_feedback={feedback}, candidates={candidates}"
        ) from exc

    return _ensure_json_string(text, "Gemini", model_name)
def call_cerebras(model, system_prompt, user_prompt, max_retries=3):
    from cerebras.cloud.sdk import Cerebras

    model_name = model or os.getenv(
        "CEREBRAS_MODEL",
        "gpt-oss-120b"
    )

    client = Cerebras(
        api_key=_require_env("CEREBRAS_API_KEY")
    )

    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=_messages(system_prompt, user_prompt),
                response_format={"type": "json_object"},
                temperature=0.0,
                max_completion_tokens=4096,
                reasoning_effort="low",
                stream=False,
            )

            if not response.choices:
                raise RuntimeError("response.choices is empty")

            choice = response.choices[0]
            content = choice.message.content

            return _ensure_json_string(
                content,
                "Cerebras",
                model_name
            )

        except Exception as exc:
            last_error = exc

            print(
                f"[Cerebras] attempt {attempt + 1}/{max_retries} failed: {exc}"
            )

    raise RuntimeError(
        f"[Cerebras] failed after {max_retries} attempts: {last_error}"
    )
# Cerebras GPT-OSS-120B (ACTIVE)
client = call_cerebras
model = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
modelname = "cerebras_gpt_oss_120b"

# client = call_groq
# model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
# modelname = "groq_gpt_oss_20b"

# Ví dụ hiện tại: GPT-4o-mini ok
# client = call_open_ai_api_gpt4omini
# model = os.getenv("OPENAI_GPT4O_MINI_MODEL", "gpt-4o-mini")
# modelname = "azure_gpt4omini"
# modelname = "openai_gpt4omini"

# # OpenAI GPT-5 đã ok, return pha 1 ' ' thay vì " " => lỗi nhiều
# client = call_open_ai_api_gpt5
# model = os.getenv("OPENAI_GPT5_MODEL", "gpt-5o")
# modelname = "azure_gpt5"
# modelname = "open_ai_GPT5"

# Azure OpenAI GPT-5 ???
# client = call_azure_open_ai_GPT5api
# model = _require_env("AZURE_OPENAI_GPT5_DEPLOYMENT")
# modelname = "azure_gpt5"


# Azure OpenAI 4omini ok
# client = call_azure_open_ai_api
# model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
# modelname = "azure_gpt4omini"

# Qwen đc r Running Pha 1
# client = call_Qwen
# model = os.getenv("QWEN_MODEL", "qwen-plus")
# modelname = "QWEN_MODEL"

# Mistral từ chối request vì quota/rate limit 429=> nạp $ vẫn lỗi => tự nhiên đc sau nửa ngày
# client = call_Mistral
# model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
# modelname = "MISTRAL_MODEL"

# Gemini ok
# client = call_gemini
# model = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
# modelname = "GEMINI_MODEL"

print(f"LLM client: {client.__name__} | model: {model}")

# emotion_appraisal_relation = {}
emotion_list = ["afraid", "angry", "annoyed", "anticipating", "anxious", "apprehensive", "ashamed", "caring", "confident", "content", "devastated", "disappointed", "disgusted", "embarrassed", "excited", "faithful", "furious", "grateful", "guilty", "hopeful", "impressed", "jealous", "joyful", "lonely", "nostalgic", "prepared", "proud", "sad", "sentimental", "surprised", "terrified", "trusting"]
AppraisalDimension = {
    "attention":    {"score": 0.0, "reason": "", "evidence": ""},
    "certainty":    {"score": 0.0, "reason": "", "evidence": ""},
    "effort":       {"score": 0.0, "reason": "", "evidence": ""},
    "pleasantness": {"score": 0.0, "reason": "", "evidence": ""},
    "responsibility":{"score": 0.0, "reason": "", "evidence": ""},
    "control":      {"score": 0.0, "reason": "", "evidence": ""},
    "circumstance": {"score": 0.0, "reason": "", "evidence": ""}
}
AppraisalEmotionPattern_gpt={
  "afraid": {
    "attention": 0.88,
    "certainty": 0.35,
    "effort": 0.78,
    "pleasantness": 0.05,
    "responsibility": 0.15,
    "control": 0.20,
    "circumstance": 0.68
  },
  "angry": {
    "attention": 0.82,
    "certainty": 0.85,
    "effort": 0.78,
    "pleasantness": 0.05,
    "responsibility": 0.15,
    "control": 0.32,
    "circumstance": 0.18
  },
  "annoyed": {
    "attention": 0.62,
    "certainty": 0.82,
    "effort": 0.32,
    "pleasantness": 0.18,
    "responsibility": 0.15,
    "control": 0.52,
    "circumstance": 0.20
  },
  "anticipating": {
    "attention": 0.85,
    "certainty": 0.48,
    "effort": 0.35,
    "pleasantness": 0.52,
    "responsibility": 0.20,
    "control": 0.42,
    "circumstance": 0.35
  },
  "anxious": {
    "attention": 0.92,
    "certainty": 0.20,
    "effort": 0.78,
    "pleasantness": 0.05,
    "responsibility": 0.25,
    "control": 0.15,
    "circumstance": 0.62
  },
  "apprehensive": {
    "attention": 0.78,
    "certainty": 0.27,
    "effort": 0.50,
    "pleasantness": 0.12,
    "responsibility": 0.20,
    "control": 0.25,
    "circumstance": 0.55
  },
  "ashamed": {
    "attention": 0.48,
    "certainty": 0.82,
    "effort": 0.45,
    "pleasantness": 0.05,
    "responsibility": 0.88,
    "control": 0.28,
    "circumstance": 0.12
  },
  "caring": {
    "attention": 0.84,
    "certainty": 0.60,
    "effort": 0.52,
    "pleasantness": 0.55,
    "responsibility": 0.42,
    "control": 0.48,
    "circumstance": 0.30
  },
  "confident": {
    "attention": 0.60,
    "certainty": 0.92,
    "effort": 0.42,
    "pleasantness": 0.82,
    "responsibility": 0.68,
    "control": 0.88,
    "circumstance": 0.10
  },
  "content": {
    "attention": 0.28,
    "certainty": 0.86,
    "effort": 0.12,
    "pleasantness": 0.90,
    "responsibility": 0.25,
    "control": 0.62,
    "circumstance": 0.20
  },
  "devastated": {
    "attention": 0.88,
    "certainty": 0.86,
    "effort": 0.82,
    "pleasantness": 0.01,
    "responsibility": 0.12,
    "control": 0.05,
    "circumstance": 0.90
  },
  "disappointed": {
    "attention": 0.62,
    "certainty": 0.86,
    "effort": 0.38,
    "pleasantness": 0.08,
    "responsibility": 0.22,
    "control": 0.22,
    "circumstance": 0.58
  },
  "disgusted": {
    "attention": 0.38,
    "certainty": 0.90,
    "effort": 0.28,
    "pleasantness": 0.01,
    "responsibility": 0.10,
    "control": 0.46,
    "circumstance": 0.18
  },
  "embarrassed": {
    "attention": 0.72,
    "certainty": 0.72,
    "effort": 0.46,
    "pleasantness": 0.05,
    "responsibility": 0.72,
    "control": 0.28,
    "circumstance": 0.18
  },
  "excited": {
    "attention": 0.90,
    "certainty": 0.62,
    "effort": 0.58,
    "pleasantness": 0.92,
    "responsibility": 0.28,
    "control": 0.58,
    "circumstance": 0.20
  },
  "faithful": {
    "attention": 0.58,
    "certainty": 0.86,
    "effort": 0.48,
    "pleasantness": 0.66,
    "responsibility": 0.68,
    "control": 0.68,
    "circumstance": 0.15
  },
  "furious": {
    "attention": 0.96,
    "certainty": 0.90,
    "effort": 0.95,
    "pleasantness": 0.00,
    "responsibility": 0.10,
    "control": 0.12,
    "circumstance": 0.12
  },
  "grateful": {
    "attention": 0.62,
    "certainty": 0.90,
    "effort": 0.22,
    "pleasantness": 0.92,
    "responsibility": 0.10,
    "control": 0.32,
    "circumstance": 0.22
  },
  "guilty": {
    "attention": 0.78,
    "certainty": 0.88,
    "effort": 0.55,
    "pleasantness": 0.04,
    "responsibility": 0.96,
    "control": 0.42,
    "circumstance": 0.08
  },
  "hopeful": {
    "attention": 0.78,
    "certainty": 0.30,
    "effort": 0.45,
    "pleasantness": 0.80,
    "responsibility": 0.22,
    "control": 0.42,
    "circumstance": 0.48
  },
  "impressed": {
    "attention": 0.86,
    "certainty": 0.88,
    "effort": 0.22,
    "pleasantness": 0.86,
    "responsibility": 0.08,
    "control": 0.28,
    "circumstance": 0.18
  },
  "jealous": {
    "attention": 0.90,
    "certainty": 0.42,
    "effort": 0.68,
    "pleasantness": 0.04,
    "responsibility": 0.18,
    "control": 0.22,
    "circumstance": 0.42
  },
  "joyful": {
    "attention": 0.65,
    "certainty": 0.86,
    "effort": 0.28,
    "pleasantness": 0.98,
    "responsibility": 0.25,
    "control": 0.62,
    "circumstance": 0.18
  },
  "lonely": {
    "attention": 0.62,
    "certainty": 0.82,
    "effort": 0.50,
    "pleasantness": 0.04,
    "responsibility": 0.20,
    "control": 0.18,
    "circumstance": 0.64
  },
  "nostalgic": {
    "attention": 0.74,
    "certainty": 0.88,
    "effort": 0.22,
    "pleasantness": 0.52,
    "responsibility": 0.12,
    "control": 0.08,
    "circumstance": 0.84
  },
  "prepared": {
    "attention": 0.80,
    "certainty": 0.88,
    "effort": 0.76,
    "pleasantness": 0.64,
    "responsibility": 0.74,
    "control": 0.92,
    "circumstance": 0.08
  },
  "proud": {
    "attention": 0.62,
    "certainty": 0.92,
    "effort": 0.50,
    "pleasantness": 0.92,
    "responsibility": 0.90,
    "control": 0.78,
    "circumstance": 0.08
  },
  "sad": {
    "attention": 0.52,
    "certainty": 0.82,
    "effort": 0.40,
    "pleasantness": 0.04,
    "responsibility": 0.18,
    "control": 0.12,
    "circumstance": 0.80
  },
  "sentimental": {
    "attention": 0.68,
    "certainty": 0.86,
    "effort": 0.18,
    "pleasantness": 0.64,
    "responsibility": 0.12,
    "control": 0.28,
    "circumstance": 0.52
  },
  "surprised": {
    "attention": 0.96,
    "certainty": 0.08,
    "effort": 0.22,
    "pleasantness": 0.50,
    "responsibility": 0.08,
    "control": 0.18,
    "circumstance": 0.48
  },
  "terrified": {
    "attention": 0.98,
    "certainty": 0.28,
    "effort": 0.95,
    "pleasantness": 0.00,
    "responsibility": 0.08,
    "control": 0.03,
    "circumstance": 0.86
  },
  "trusting": {
    "attention": 0.48,
    "certainty": 0.92,
    "effort": 0.18,
    "pleasantness": 0.80,
    "responsibility": 0.12,
    "control": 0.58,
    "circumstance": 0.18
  }
}
EmotionsDicriminator= {
  "afraid": "Fear in response to a specific perceived threat or danger; less intense than terrified.",
  "angry": "Anger caused by a perceived wrong, harm, offense, or obstruction; stronger than annoyed but weaker than furious.",
  "annoyed": "Mild irritation or anger caused by inconvenience, disturbance, or bothersome behavior.",
  "anticipating": "Expecting or waiting for a future event, without necessarily implying a positive or negative outcome.",
  "anxious": "Persistent worry and distress caused by uncertainty; stronger and more sustained than apprehensive.",
  "apprehensive": "Uneasiness about a possible negative event that may happen in the future.",
  "ashamed": "Negative evaluation of the self as flawed, inadequate, or unworthy; focused on the self rather than a specific action.",
  "caring": "Concern for another person's well-being, often with a desire to help, protect, or support them.",
  "confident": "Strong belief in one's ability, judgment, or likelihood of success.",
  "content": "Calm satisfaction with the present situation; positive but low in emotional arousal.",
  "devastated": "Extreme distress or sorrow caused by a severe loss or highly negative event.",
  "disappointed": "Sadness or frustration because an outcome failed to meet prior expectations.",
  "disgusted": "Strong aversion or repulsion toward a person, object, behavior, or situation.",
  "embarrassed": "Social discomfort caused by perceived negative attention or evaluation from others.",
  "excited": "Highly energized positive emotion involving enthusiasm or eagerness about an event or opportunity.",
  "faithful": "Enduring loyalty and commitment to a person, relationship, promise, belief, or cause.",
  "furious": "Extremely intense anger caused by a serious perceived wrong, offense, or harm.",
  "grateful": "Thankfulness for a benefit, help, kindness, or valued outcome received from others.",
  "guilty": "Negative evaluation of a specific wrongful or harmful action for which one feels responsible.",
  "hopeful": "Positive expectation that a desired future outcome may occur despite uncertainty.",
  "impressed": "Positive evaluation of another person, object, action, ability, or achievement as remarkable.",
  "jealous": "Negative emotion caused by a perceived relational or comparative threat to something valued.",
  "joyful": "Clear and strong happiness in response to a positive experience or outcome.",
  "lonely": "Distress caused by a lack of desired social connection, closeness, or companionship.",
  "nostalgic": "Emotional longing for meaningful people, places, or experiences from the past, often mixing positive and negative feelings.",
  "prepared": "Feeling ready and capable of dealing with an expected future situation.",
  "proud": "Positive evaluation of one's own achievement, qualities, identity, or those of someone closely associated with oneself.",
  "sad": "General sorrow caused by loss or an undesirable outcome, without necessarily involving unmet expectations.",
  "sentimental": "Tender emotional attachment to a personally meaningful person, object, memory, or experience.",
  "surprised": "Reaction to an unexpected or expectation-violating event, which may be positive or negative.",
  "terrified": "Extreme and overwhelming fear in response to a serious or difficult-to-control perceived threat.",
  "trusting": "Belief that another person is reliable, safe, honest, or well-intentioned."
}
AppraisalDimensionsName = [    "attention",    "certainty",    "effort",    "pleasantness",    "responsibility",    "control",    "circumstance"]
output_jsonlFormat = {    "label": "...", "similarity":0.0,   "reason": "..."    }
EmotionsDicriminator_v2 = {

    "afraid":
    "Fear caused by a specific perceived threat or danger. Unlike anxious or apprehensive, the threat is relatively concrete; less intense than terrified.",

    "angry":
    "Anger caused by a perceived wrong, harm, offense, or obstruction. Stronger than annoyed but less extreme than furious.",

    "annoyed":
    "Mild irritation caused by inconvenience, disturbance, repetition, or bothersome behavior. A serious perceived wrong is not required; weaker than angry.",

    "anticipating":
    "Expecting or waiting for a future event. The defining cue is future-oriented expectation, without requiring a positive, negative, or desired outcome.",

    "anxious":
    "Sustained worry or distress under uncertainty, often about an unclear or future threat. Stronger and more persistent than apprehensive, and less threat-specific than afraid.",

    "apprehensive":
    "Mild uneasiness about a possible future negative event. Weaker and less persistent than anxious, and less tied to a concrete threat than afraid.",

    "ashamed":
    "Negative evaluation of the whole self or identity as flawed, inadequate, or unworthy. Unlike guilty, the focus is not merely a specific wrongful action; unlike embarrassed, social exposure is not required.",

    "caring":
    "Other-focused concern for someone's well-being with a desire to help, protect, comfort, or support them. Unlike faithful, it reflects concern rather than enduring commitment.",

    "confident":
    "Strong belief in one's own ability, judgment, or likelihood of success. Unlike proud, no completed achievement is required; unlike prepared, prior readiness or preparation is not required.",

    "content":
    "Calm, low-arousal satisfaction with the present situation. Unlike joyful or excited, it involves little emotional activation or eagerness.",

    "devastated":
    "Extreme distress or sorrow following a severe loss or highly negative event. Much stronger than sad or disappointed and typically associated with very low perceived control.",

    "disappointed":
    "Sadness or frustration specifically because an outcome failed to meet a prior expectation, hope, or standard. Without an unmet expectation, prefer sad.",

    "disgusted":
    "Strong repulsion, revulsion, or aversion toward a person, object, behavior, or situation. Mere negative judgment is disapproval, not disgust.",

    "embarrassed":
    "Social self-conscious discomfort caused by exposure, awkwardness, or perceived negative attention from others. Wrongdoing is not required; unlike ashamed, the core cue is social evaluation.",

    "excited":
    "High-arousal positive eagerness or enthusiasm about an event or opportunity. Unlike joyful, it is more activated and often future-oriented; unlike hopeful, uncertainty about a desired outcome is not essential.",

    "faithful":
    "Enduring loyalty and commitment to a person, relationship, promise, belief, or cause. Unlike trusting, it describes commitment rather than belief that another person is reliable.",

    "furious":
    "Extremely intense anger caused by a serious perceived wrong, offense, or harm. Stronger and more overwhelming than angry.",

    "grateful":
    "Thankfulness specifically for a benefit, help, kindness, or valued outcome received from another person. Unlike impressed, receiving a benefit is central.",

    "guilty":
    "Negative evaluation of a specific wrongful or harmful action for which one feels personally responsible. Unlike ashamed, the negative judgment targets the action rather than the whole self.",

    "hopeful":
    "Positive belief or desire that a favorable but uncertain future outcome may occur. Unlike anticipating, the future outcome must be desired; unlike confident, substantial uncertainty remains.",

    "impressed":
    "Positive evaluation of another person, object, ability, action, or achievement as remarkable or noteworthy. Unlike proud, it is not primarily self-focused; unlike grateful, no received benefit is required.",

    "jealous":
    "Negative emotion caused by a relational or comparative threat involving something valued, often including a rival, comparison, or fear of losing a valued relationship or status.",

    "joyful":
    "Strong happiness in response to a positive state, experience, or realized outcome. Unlike excited, it does not primarily involve future-oriented eagerness or high anticipatory activation.",

    "lonely":
    "Distress specifically caused by insufficient social connection, closeness, companionship, or belonging. Unlike general sadness, social disconnection is the defining cause.",

    "nostalgic":
    "Emotional longing or recollection directed toward meaningful people, places, or experiences from the past, often mixing positive and negative feelings. Past orientation is essential.",

    "prepared":
    "Feeling ready and capable of handling an expected future situation because of planning, knowledge, resources, or prior preparation. Unlike confident, readiness for a specific upcoming situation is central.",

    "proud":
    "Positive self-evaluation tied to one's own achievement, qualities, identity, or the achievement of someone closely associated with oneself. Achievement or self-relevance is required.",

    "sad":
    "General sorrow caused by loss or an undesirable outcome. Unlike disappointed, no failed prior expectation is required; less extreme than devastated.",

    "sentimental":
    "Tender emotional attachment to a personally meaningful person, object, memory, or experience. Unlike nostalgic, longing for or orientation toward the past is not required.",

    "surprised":
    "Immediate reaction to an unexpected or expectation-violating event. Unexpectedness is required; uncertainty alone is not sufficient, and the emotion may be positive or negative.",

    "terrified":
    "Extreme and overwhelming fear caused by a severe or difficult-to-control perceived threat. Much more intense and lower in perceived control than afraid.",

    "trusting":
    "Belief that another person is reliable, safe, honest, or well-intentioned. Unlike confident, the belief concerns another person; unlike faithful, it reflects perceived reliability rather than commitment."
}

AppraisalEmotionPattern_v2 = {

    "afraid": {
        "attention": 0.90,
        "certainty": 0.45,
        "effort": 0.70,
        "pleasantness": 0.05,
        "responsibility": 0.10,
        "control": 0.20,
        "circumstance": 0.75
    },

    "anxious": {
        "attention": 0.90,
        "certainty": 0.10,
        "effort": 0.70,
        "pleasantness": 0.05,
        "responsibility": 0.30,
        "control": 0.10,
        "circumstance": 0.40
    },

    "apprehensive": {
        "attention": 0.75,
        "certainty": 0.25,
        "effort": 0.40,
        "pleasantness": 0.15,
        "responsibility": 0.20,
        "control": 0.30,
        "circumstance": 0.50
    },

    "terrified": {
        "attention": 0.95,
        "certainty": 0.55,
        "effort": 0.95,
        "pleasantness": 0.00,
        "responsibility": 0.05,
        "control": 0.00,
        "circumstance": 0.90
    },

    "angry": {
        "attention": 0.80,
        "certainty": 0.90,
        "effort": 0.70,
        "pleasantness": 0.05,
        "responsibility": 0.10,
        "control": 0.50,
        "circumstance": 0.15
    },

    "furious": {
        "attention": 0.95,
        "certainty": 0.95,
        "effort": 0.95,
        "pleasantness": 0.00,
        "responsibility": 0.10,
        "control": 0.05,
        "circumstance": 0.10
    },

    "anticipating": {
        "attention": 0.90,
        "certainty": 0.40,
        "effort": 0.30,
        "pleasantness": 0.50,
        "responsibility": 0.10,
        "control": 0.35,
        "circumstance": 0.40
    },

    "ashamed": {
        "attention": 0.60,
        "certainty": 0.90,
        "effort": 0.55,
        "pleasantness": 0.05,
        "responsibility": 0.95,
        "control": 0.15,
        "circumstance": 0.05
    },

    "confident": {
        "attention": 0.60,
        "certainty": 0.90,
        "effort": 0.35,
        "pleasantness": 0.70,
        "responsibility": 0.45,
        "control": 0.95,
        "circumstance": 0.10
    },

    "proud": {
        "attention": 0.70,
        "certainty": 0.95,
        "effort": 0.50,
        "pleasantness": 0.95,
        "responsibility": 0.95,
        "control": 0.80,
        "circumstance": 0.05
    },

    "excited": {
        "attention": 0.95,
        "certainty": 0.60,
        "effort": 0.70,
        "pleasantness": 0.95,
        "responsibility": 0.20,
        "control": 0.55,
        "circumstance": 0.15
    },

    "joyful": {
        "attention": 0.75,
        "certainty": 0.90,
        "effort": 0.20,
        "pleasantness": 0.95,
        "responsibility": 0.20,
        "control": 0.70,
        "circumstance": 0.10
    },

    "devastated": {
        "attention": 0.85,
        "certainty": 0.80,
        "effort": 0.70,
        "pleasantness": 0.00,
        "responsibility": 0.10,
        "control": 0.10,
        "circumstance": 0.80
    },

    "prepared": {
        "attention": 0.80,
        "certainty": 0.85,
        "effort": 0.70,
        "pleasantness": 0.55,
        "responsibility": 0.60,
        "control": 0.85,
        "circumstance": 0.10
    },

    "sentimental": {
        "attention": 0.60,
        "certainty": 0.70,
        "effort": 0.15,
        "pleasantness": 0.60,
        "responsibility": 0.10,
        "control": 0.35,
        "circumstance": 0.40
    },

    "surprised": {
        "attention": 0.90,
        "certainty": 0.20,
        "effort": 0.15,
        "pleasantness": 0.50,
        "responsibility": 0.10,
        "control": 0.15,
        "circumstance": 0.60
    },

    "trusting": {
        "attention": 0.50,
        "certainty": 0.85,
        "effort": 0.15,
        "pleasantness": 0.75,
        "responsibility": 0.10,
        "control": 0.50,
        "circumstance": 0.30
    }
}
def build_Pha1prompt_appraisal(situation, conversation):
    sys_prompt = f"""
        You are an expert in annotation, Natural Language Processing, and psychology.

        Infer the speaker's Appraisal dimensions ONLY from the given situation and conversation by following these steps:
        1. Identify the speaker's core event and the objective fact. then, describe how the speaker subjectively interprets the event, including its perceived cause, outcome, and their thoughts, beliefs, expectations, or imaginings.
        2. Identify explicit and implicit appraisal indicators and extract concise evidence as a word, short phrase, or brief clause.
        3. Assign scores from 0.0 to 1.0 for each of the 7 Appraisal dimensions based on the strength of evidence, 
        Use the following values ​​as reference anchors: 0 = absent or unsupported; 0.25 = weak; 0.50 = moderate; 0.75 = strong; 1.0 = overwhelming.
        4. Provide a concise reason  explaining why the score was assigned.
      
        Appraisal Dimension:
        - attention: Desire to devote further attention to the event.
        - certainty: Certainty about what is happening.
        - effort: Mental or physical effort needed to handle the situation.
        - pleasantness: Perceived pleasantness of the event.
        - responsibility: Personal responsibility for the situation.
        - control: Personal ability to control or influence the situation.
        - circumstance: Belief that no one can influence or change the event.

        Important rules:
        - Use only the information provided in the situation, conversation.
        - Evaluate each Appraisal dimension independently.
        - Do not confuse responsibility with control:
            Responsibility concerns who caused the event or deserves credit or blame. 
            Control concerns the speaker's ability to influence or change it.
        -Do not confuse personal lack of control with an unchangeable circumstance:
            Low control does not necessarily mean the event is unavoidable or unchangeable; 
            High circumstance means the event is perceived as beyond anyone's influence.
        - Evidence must be grounded in the situation/dialogue
        - reason focus on situation/dialogue and the speaker's appraisal
        - Return ONLY one valid JSON object in EXACTLY this format:
        {AppraisalDimension}
    """
    #  reason (<12 words) => làm KQ giảm )-:
    user_prompt = f"""        
        situation:
        {situation}
        conversation:
        {conversation}
        let's think step by step.
       """
    return sys_prompt, user_prompt
def build_Pha1prompt_appraisal_01(situation,conversation):
    sys_prompt = f"""    
        You are an expert in annotation, Natural Language Processing, and psychology.

        conversation is:
        {conversation}

        Infer the Appraisal dimensions ONLY from the conversation by following these steps:
        1. Fact + Speaker's interpretation: Identify what occurred and how the speaker interprets the cause, outcome, and relevant beliefs.
        2. Evidence:  Extract explicit or implicit appraisal cues for each dimension as a word, phrase, or brief clause from the situation or conversation.
        3. Assign a continuous score from 0.0 to 1.0 based on evidence strength, ranging from unsupported to weak, moderate, strong, and very strong evidence.
        4. Provide a concise reason explaining why the score was assigned.
        
        Appraisal Dimension:
        - attention: Desire to devote further attention to the event.
        - certainty: Certainty about what is happening.
        - effort: Mental or physical effort needed to handle the situation.
        - pleasantness: Perceived pleasantness of the event.
        - responsibility: Personal responsibility for the situation.
        - control: Personal ability to control or influence the situation.
        - circumstance: Belief that no one can influence or change the event.

        Important rules:
        - Use only the information in the situation, conversation.
        - Do not fabricate unsupported information.
        - "evidence" must always be ONE JSON string.  If multiple cues are relevant, combine them into a single string separated by semicolons.
        - Treat the situation as contextual information summarizing the event.
        - Use the conversation as the primary source of evidence about the speaker's appraisal.
        - Evaluate each Appraisal dimension independently while considering the overall context. 
        - Do not infer high attention merely because the speaker gives a detailed description.  Attention refers to the speaker's psychological focus on the event,  not the length or detail of the sentence.
        - Return ONLY one valid JSON object in EXACTLY this format:
        {AppraisalDimension}
    """
    # strict JSON schema sửa lỗi kỹ thuật khi có nhiều evidence "evidence": "shaking in fear", "It was wild"
     
    user_prompt = f"""   
        situation:
        {situation}     
        conversation:
        {conversation}
       """
    return sys_prompt, user_prompt
def build_Pha1prompt_appraisal_01_v2(situation,conversation):
    sys_prompt = f"""    
        You are an expert in annotation, Natural Language Processing, and psychology.

        conversation is:
        {conversation}

        Infer the Appraisal dimensions ONLY from the conversation by following these steps:
        1. Fact + Speaker's interpretation: Identify what occurred and how the speaker interprets the cause, outcome, and relevant beliefs.
        2. Evidence:  Extract explicit or implicit appraisal cues for each dimension as a word, phrase, or brief clause from the situation or conversation.
        3. Assign a continuous score from 0.0 to 1.0 based on evidence strength, ranging from unsupported to weak, moderate, strong, and very strong evidence.
        4. Provide a concise reason explaining why the score was assigned.
        
        Appraisal Dimension:
        - attention: Desire to devote further attention to the event.
        - certainty: Certainty about what is happening.
        - effort: Mental or physical effort needed to handle the situation.
        - pleasantness: Perceived pleasantness of the event.
        - responsibility: Personal responsibility for the situation.
        - control: Personal ability to control or influence the situation.
        - circumstance: Belief that no one can influence or change the event.

        Important rules:
        - Use only information from the situation and conversation
        - do not infer or fabricate unsupported details.
        - Evaluate each Appraisal dimension independently while considering the overall context.
        - Ground every score and reason in evidence from the situation or conversation.
        - Keep each reason focused on the relevant event, textual evidence, and the speaker's appraisal.
        - Distinguish responsibility from control:
        Responsibility concerns who caused the event or deserves credit or blame.
        Control concerns the speaker's ability to influence, manage, or change the situation.
        - Distinguish low control from high circumstance:
        Low control means the speaker has limited ability to influence the situation.
        High circumstance means the event is largely determined by external conditions beyond anyone's influence.
        - Return ONLY one valid JSON object in EXACTLY this format:
        {AppraisalDimension}
    """
     
    user_prompt = f"""   
        situation:
        {situation}     
        conversation:
        {conversation}
       """
    return sys_prompt, user_prompt
def generate_text_jf(model, system_prompt, user_prompt):
    """Call selected provider and expose the REAL error instead of silently skipping rows."""
    try:
        print(f"CALL -> provider={client.__name__}, model={model}")
        result = client(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        if result is None:
            print(f"RETURN <- provider={client.__name__}, result=None")
            return None

        print(f"RETURN <- provider={client.__name__}, chars={len(result)}")
        return result

    except BadRequestError as error:
        error_text = str(error).lower()
        if "content_filter" in error_text or "content management policy" in error_text:
            print(f"CONTENT FILTER -> {type(error).__name__}: {error}")
            return None

        print(f"API ERROR -> {type(error).__name__}: {error}")
        traceback.print_exc()
        raise

    except Exception as error:
        print(
            f"API ERROR -> provider={client.__name__}, model={model}, "
            f"type={type(error).__name__}, error={error}"
        )
        traceback.print_exc()
        raise


def pha1_infer_appraisal_dimensions_jf(model, situation, conversation):
    sys_prompt, user_prompt = build_Pha1prompt_appraisal_01(
        situation=situation,
        conversation=conversation
    )
    # call API lấy response.outputtext kiểu str => chuyển về dict sd json.load
    result_str = generate_text_jf( #str
        model=model,
        system_prompt=sys_prompt,
        user_prompt=user_prompt        
    )
    
    if result_str is None:
        return None

    try:
        result_json = json.loads(result_str)
        return result_json

    except json.JSONDecodeError as error:
        print(f"JSON decode error: {error}")
        print(f"Raw output: {repr(result_str)}")
        return None
def runPha1_Text2Appraisal():
    # df=read_jsonl_to_dataframe(jsonl_file_path=r"G:\Python\emp_chat.jsonl")
    # error_file = r"G:\Python\APContinuous\ErrorRows" + "\\" +modelname+"_emp_APatternContinuous_pha1_Errors.jsonl"
    # output_file =r"G:\Python\APContinuous\Results" + "\\" +modelname+"_emp_APatternContinuous_pha1.jsonl"
# Rerun lỗi
    df=read_jsonl_to_dataframe(jsonl_file_path=r"G:\Python\APContinuous\ErrorRows" + "\\" +modelname+"_emp_APatternContinuous_pha1_L2.jsonl")
    error_file = r"G:\Python\APContinuous\ErrorRows" + "\\" +modelname+"_emp_APatternContinuous_pha1_L3.jsonl"
    output_file =r"G:\Python\APContinuous\Results" + "\\" +modelname+"_emp_APatternContinuous_pha1.jsonl"
 
    with (
        open(output_file, "a", encoding="utf-8") as f,
        open(error_file, "a", encoding="utf-8") as error_f):
        for index, row in df.iterrows():  # duyệt all dòng trong DataFrame
        # for index, row in df.iloc[0:3].iterrows():  
        # for index, row in df.iloc[3:len(df)].iterrows(): 
            print(f"\nProcessing row {index}:\n{output_file}")
            Conversation_ID = df['conversation_id'][index]
            label = df['label'][index]
            situation = df['situation'][index]
            conversation = df['conversation'][index]
            # print(situation, conversation)#ok
            try:
                result = pha1_infer_appraisal_dimensions_jf(model=model, situation=situation, conversation=conversation)
                if(result==None): print ("KQ trả về TRỐNG") #index0 row 1 bị content chặn ???
                else:   
                    # print("result: ",result)#ko ra đây
                    record = {
                        "conversation_id": Conversation_ID,
                        "label": label,
                        "situation": situation,
                        "conversation": conversation,
                        "appraisal":{
                            dimension: result.get( dimension, {} )
                            for dimension in [
                                "attention",
                                "certainty",
                                "effort",
                                "pleasantness",
                                "responsibility",
                                "control",
                                "circumstance"
                            ]
                        }}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")#ghi vào file đang mở một dòng JSON mới
                    f.flush() #đẩy dữ liệu từ bộ nhớ đệm xuống file ngay lập tức => tránh mất dữ liệu nếu chương trình bị dừng giữa chừng.

                    print(f"Conversation ID: {Conversation_ID}, label: {label} Saved => {output_file}")
            except Exception as e: #lỗi tại đây???
                error_record = {
                    "conversation_id": Conversation_ID,
                    "label": label,
                    "situation": situation,
                    "conversation": conversation,
                    "error_type": type(e).__name__,
                    "error_reason": str(e)
                }
                error_f.write( json.dumps(  error_record,  ensure_ascii=False  ) + "\n"  )

                error_f.flush()

                print(f"conversation_id ID: {Conversation_ID} gặp exception: {type(e).__name__}: {e}")
                print(f"Đã ghi row lỗi vào file {error_file}")#chỗ này
                continue  # bỏ qua chạy row tiếp
# runPha1_Text2Appraisal()

def buildPrompt_2Emotion_AppraisalEmotionPattern_ver2(situation,conversation, appraisal):#34
    sys_prompt = f"""
    You are an expert in Natural Language Processing and psychology

    AppraisalEmotionPattern:
    {AppraisalEmotionPattern_gpt}

    Emotion Discriminators:
    {EmotionsDicriminator}
    
    Task:
    Identify the primary emotion that best matches the observed appraisal.

    Instructions:
    1. Compare the observed appraisal with all emotion patterns and measure the similarity.
    2. Select the emotion with the highest similarity. 
    If several emotions are similarly well matched, use the situation, conversation, discriminating textual cues, and emotion discriminators {EmotionsDicriminator} to make the final choice.
    3. Explain the elected emotion , including:
    - key appraisal cues,
    - brief quoted phrases from the conversation,
    - why it's the best match.

    Rules:
    - Select exactly ONE emotion from:
    {emotion_list}
    - Return an exactly valid JSON object matching:
    {output_jsonlFormat}
    """

    user_prompt = f"""
    Identify the primary emotion based on:
    Situation:
    {situation}
    Conversation:
    {conversation}

    Appraisal:
    {appraisal}
    Let's Think step by step internally

    """
# in fewer than 45 words
    return sys_prompt, user_prompt
def buildPrompt_2Emotion_AppraisalEmotionPattern_ver2_2(situation,conversation, appraisal):#34
    sys_prompt = f"""
    You are an expert in Natural Language Processing and psychology

    AppraisalEmotionPattern:
    {AppraisalEmotionPattern_gpt}

    Emotion Discriminators:
    {EmotionsDicriminator}
    
    Task:
    Identify the primary emotion that best matches the observed appraisal.

    Instructions:
    1. Compare the observed appraisal with all emotion patterns and measure the similarity.
    2. Identify the candidate emotions whose similarity scores are substantially higher than the others.
    3. Among the candidate emotions:
   - If one emotion has clearly higher similarity than the remaining candidates, select it.
   - Otherwise, use the situation, conversation, discriminating textual cues, and {EmotionsDicriminator} to make the final choice. Prefer the emotion that is better supported by emotion-specific evidence, even if its similarity is slightly lower.
   3. Explain the elected emotion , including:
    - key appraisal cues,
    - brief quoted phrases from the conversation,
    - why it's the best match.

    Rules:
    - Select exactly ONE emotion from:
    {emotion_list}
    - Return an exactly valid JSON object matching:
    {output_jsonlFormat}
    """

    user_prompt = f"""
    Identify the primary emotion based on:
    Situation:
    {situation}
    Conversation:
    {conversation}

    Appraisal:
    {appraisal}
    Let's Think step by step internally

    """
# in fewer than 45 words
    return sys_prompt, user_prompt
def buildPrompt_2Emotion_AppraisalEmotionPattern_ver3(situation,conversation, appraisal):#42,56
    sys_prompt = f"""
    You are an expert in NLP, psychology and fine-grained emotion classification.
    
    Conversation:
    {conversation}
    AppraisalEmotionPattern:
    {AppraisalEmotionPattern_gpt}
    
    Task: Identify the single primary emotion that best matches the conversation and appraisal.

    Instructions:
    1. Compare the observed appraisal with all emotion patterns and measure their similarity.
    2. If one emotion has clearly higher similarity than all others, select it.
    3. If the top similarities are close, compare the competing emotions using: the conversation and situation, discriminating textual cues, and {EmotionsDicriminator}. prefer the emotion better supported by
emotion-specific evidence, even if its similarity is slightly lower.
    4. Explain briefly why the selected emotion best matches the appraisal and conversation evidence.

    Rules:
    - Select exactly ONE emotion from:
    {emotion_list}
    - Use only information from the conversation.
    - Return exactly one valid JSON object parseable by json.loads():
    {output_jsonlFormat}
    """

    user_prompt = f"""
    Identify the primary emotion based on:
    Situation:
    {situation}
    Conversation:
    {conversation}

    Appraisal:
    {appraisal}

    """
# in fewer than 45 words
    return sys_prompt, user_prompt
def buildPrompt_2Emotion_AppraisalEmotionPattern_ver3_v2(situation,conversation, appraisal):#42,56
    sys_prompt = f"""
    You are an expert in NLP, psychology and fine-grained emotion classification.
    
    Conversation:
    {conversation}
    AppraisalEmotionPattern:
    {AppraisalEmotionPattern_v2}
    
    Task: Identify the single primary emotion that best matches the conversation and appraisal.

    Instructions:
    1. Compare the observed appraisal with all emotion patterns and measure their similarity.
    2. If one emotion has sufficiently higher similarity than all others, select it.
    otherwise, compare the competing emotions using: the conversation, situation, discriminating textual cues, {EmotionsDicriminator_v2}.
      prefer the emotion better supported by emotion-specific evidence, even if its similarity is slightly lower.
    3. Explain briefly why the selected emotion is the best.

    Rules:
    - Select exactly ONE emotion from:
    {emotion_list}
    - Use only information from the conversation.
    - Return exactly one valid JSON object parseable by json.loads():
    {output_jsonlFormat}
    """

    user_prompt = f"""
    Identify the primary emotion based on:
    Situation:
    {situation}
    Conversation:
    {conversation}

    Appraisal:
    {appraisal}

    """
# in fewer than 45 words
    return sys_prompt, user_prompt
def buildPrompt_2Emotion_AppraisalEmotionPattern_ver4(situation,conversation, appraisal):#dùng 41,19
    sys_prompt = f"""
    You are an expert in NLP, psychology and fine-grained emotion classification.
    
    Conversation:
    {conversation}
    AppraisalEmotionPattern:
    {AppraisalEmotionPattern_gpt}
    
    Task: Identify the single primary emotion that best matches the conversation and appraisal.

    Instructions:
    1. Compare the observed appraisal with all emotion patterns and measure their similarity.
    2. If one emotion has sufficiently higher similarity than all others, select it. 
    Otherwise, evaluate the competing emotions using appraisal similarity, conversation context, discriminating textual cues, and {EmotionsDicriminator}, then select the emotion best supported by the overall evidence.
    3. Explain briefly why the selected emotion is the best overall match.

    Rules:
    - Select exactly ONE emotion from:
    {emotion_list}
    - Use only information from the conversation.
    - Return exactly one valid JSON object parseable by json.loads():
    {output_jsonlFormat}
    """
# sufficiently mô tả đúng “đủ để ra quyết định”, meaningfully mô tả “có ý nghĩa”, còn clearly chỉ mô tả “rõ ràng”.
    user_prompt = f"""
    Identify the primary emotion based on:
    Situation:
    {situation}
    Conversation:
    {conversation}

    Appraisal:
    {appraisal}

    """
# in fewer than 45 words
    return sys_prompt, user_prompt

def pha2_infer_appraisal_dimensions_jf(model, situation, conversation, appraisal):
    sys_prompt, user_prompt = buildPrompt_2Emotion_AppraisalEmotionPattern_ver2_2(
    # sys_prompt, user_prompt = buildPrompt_2Emotion_AppraisalEmotionPattern_ver2(
    # sys_prompt, user_prompt = buildPrompt_2Emotion_AppraisalEmotionPattern_gemi(
        situation=situation,
        conversation=conversation,
        appraisal=appraisal
    )
    # call API lấy response.outputtext kiểu str => chuyển về dict sd json.load
    result_str = generate_text_jf(
        model=model,
        system_prompt=sys_prompt,
        user_prompt=user_prompt        
    )
    if result_str is None:
            return None
    
    try:
        result_json = json.loads(result_str)
        return result_json

    except json.JSONDecodeError as error:
        print(f"JSON decode error: {error}")
        print(f"Raw output: {repr(result_str)}")
        return None
def runPha2End_Text2Appraisal():        
    df2=read_jsonl_to_dataframe(jsonl_file_path = r"G:\Python\APContinuous\Results" + "\\" +modelname+"_emp_APatternContinuous_pha1.jsonl")
    # df2=read_jsonl_to_dataframe(jsonl_file_path = r"G:\Python\APContinuous\Results\azure_gpt4omini_emp_APatternContinuous_pha1.jsonl")
    # # azure_gpt4omini_emp_APatternContinuous_pha1
    error_file = r"G:\Python\APContinuous\ErrorRows" + "\\" +modelname+"_emp_APatternContinuous_Pha2.jsonl"
    output_file =r"G:\Python\APContinuous\Results" + "\\" +modelname+"_emp_APatternContinuous_Pha2.jsonl"
    # XLY row lỗi
    # XLy file lỗi
    # df2=read_jsonl_to_dataframe(jsonl_file_path=r"G:\Python\APContinuous\ErrorRows" + "\\" +modelname+"_emp_APattenContinuous_Pha2_L2.jsonl")
    # df2=read_jsonl_to_dataframe(jsonl_file_path=r"G:\Python\APContinuous\ErrorRows" + "\\" +modelname+"_emp_APattenContinuous_Pha2_L2.jsonl")

    # error_file = r"G:\Python\APContinuous\ErrorRows" + "\\" +modelname+"_emp_APatternContinuous_Pha2_L3.jsonl"
    # output_file =r"G:\Python\APContinuous\Results" + "\\" +modelname+"_emp_APatternContinuous_Pha2.jsonl"
        # Mở file để ghi kết quả
    with (
        open(output_file, "a", encoding="utf-8") as f,
        open(error_file, "a", encoding="utf-8") as error_f):
        # for index in range(0,3):
        # for index in range(900,1700):
        for index in range(3,len(df2)):
            
            # conversation_id=df2['conversation_id'][index] 
            conversation_id = df2['conversation_id'].iloc[index]
            print(f"\nProcessing row {index} {conversation_id}: {output_file}")   
            label=df2['label'][index]
           
            situation = df2['situation'].iloc[index]
            conversation = df2['conversation'].iloc[index]
            appraisal = df2['appraisal'].iloc[index]
            try:
                result=pha2_infer_appraisal_dimensions_jf(model=model, situation=situation, conversation=conversation, appraisal=appraisal)#,AppraisalResult=appraisal)
                #Xly case result=none generate_text_jf() trả về None/json.loads() bị lỗi (JSONDecodeError)/Có exception khác khiến bạn return None.

                # print(f"conversation_id ID: {conversation_id}, lable: {label}")
                # print(f"Model Output: {result}")
    # generate_text_jf trả về None khi gặp azure chặn content_filter
                if result is None:
                    error_record = {
                        "conversation_id": conversation_id,
                        "label": label,
                        "situation": situation,
                        "conversation": conversation,
                        "appraisal": appraisal,
                        "error_type": type(e).__name__,
                        "error_reason": str(e)
                    }
                    error_f.write( json.dumps(  error_record,  ensure_ascii=False  ) + "\n"  )                    
                    error_f.flush()
                    print(f"Đã ghi row lỗi NONE vào file {error_file}")#chỗ này
                    continue  # bỏ qua chạy row tiếp
                else:   
                # Ghi mỗi dòng là một JSON => Tạo dữ liệu ghi ra file jsonl
                    output_fields = {
                        "conversation_id": conversation_id,
                        "true_label": label,
                        "pred_label": result.get("label", ""),                
                        "reason": result.get("reason", ""),
                        "similarity":result.get("similarity","")
                    }
    
                    f.write(json.dumps(output_fields, ensure_ascii=False) + "\n")#Case lỗi vẫn tạo json file đúng format để ghi file
                    f.flush()#ghi ngay file tránh mất dữ liệu khi crash/lost connection...
                    print(
                        f"SAVED -> row {index}, "
                        f"conversation_id={conversation_id}, "
                        f"pred_label={output_fields['pred_label']}"
                    )
            except Exception as e:#Xử lý đối với các ngoại lệ content_filter bị lọc, bỏ qua chạy rơ tiếp
                error_record = {
                    "conversation_id": conversation_id,
                    "label": label,
                    "situation": situation,
                    "conversation": conversation,
                    "appraisal": appraisal,
                    "error_type": type(e).__name__,
                    "error_reason": str(e)
                }
                error_f.write( json.dumps(  error_record,  ensure_ascii=False  ) + "\n"  )

                error_f.flush()

                print(f"conversation_id ID: {conversation_id} gặp exception: {type(e).__name__}: {e}")
                print(f"Đã ghi row lỗi vào file {error_file}")#chỗ này
                continue  # bỏ qua chạy row tiếp
runPha2End_Text2Appraisal()
