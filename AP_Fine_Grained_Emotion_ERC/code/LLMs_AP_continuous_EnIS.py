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
def read_enisear(path):
    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)

            if "pleasantness" not in x:
                x["pleasantness"] = x.get("pleasant")

            records.append({
                "sentence_id": x.get("sentence_id"),
                "true_label": x.get("true_label"),
                "sentence": x.get("sentence"),
                "appraisal": {
                    d: x.get(d)
                    for d in AppraisalDimensionsName
                }
            })

    return pd.DataFrame(records)



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
def call_cerebras(model, system_prompt, user_prompt):
    """Cerebras GPT-OSS-120B via Cerebras Chat Completions -> JSON string."""

    try:
        from cerebras.cloud.sdk import Cerebras
    except ImportError as exc:
        raise RuntimeError(
            "Missing package 'cerebras-cloud-sdk'. "
            "Install with: pip install -U cerebras-cloud-sdk"
        ) from exc

    model_name = model or os.getenv(
        "CEREBRAS_MODEL",
        "gpt-oss-120b"
    )

    client = Cerebras(
        api_key=_require_env("CEREBRAS_API_KEY")
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
        max_completion_tokens=2048,
        reasoning_effort="low",
        stream=False,
    )

    if not getattr(response, "choices", None):
        raise RuntimeError(
            f"[Cerebras] model={model_name}: response.choices is empty"
        )

    content = response.choices[0].message.content

    return _ensure_json_string(
        content,
        "Cerebras",
        model_name
    )




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

# Gemini ok
# client = call_gemini
# model = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
# modelname = "GEMINI_MODEL"

# client = call_groq
# model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
# modelname = "groq_gpt_oss_20b"

# Cerebras GPT-OSS-120B (ACTIVE)
# client = call_cerebras
# model = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
# modelname = "cerebras_gpt_oss_120b"

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
client = call_Mistral
model = os.getenv("MISTRAL_MODEL", "mistral-small-2603")
modelname = "MISTRAL_MODEL"


print(f"LLM client: {client.__name__} | model: {model}")

# emotion_appraisal_relation = {}
emotion_list = [    "Anger",    "Disgust",    "Fear",    "Guilt",    "Joy",    "Sadness",    "Shame"]
AppraisalDimension = {
    "attention":    {"score": 0.0, "reason": "", "evidence": ""},
    "certainty":    {"score": 0.0, "reason": "", "evidence": ""},
    "effort":       {"score": 0.0, "reason": "", "evidence": ""},
    "pleasantness": {"score": 0.0, "reason": "", "evidence": ""},
    "responsibility":{"score": 0.0, "reason": "", "evidence": ""},
    "control":      {"score": 0.0, "reason": "", "evidence": ""},
    "circumstance": {"score": 0.0, "reason": "", "evidence": ""}
}
AppraisalDimensionsName = [    "attention",    "certainty",    "effort",    "pleasantness",    "responsibility",    "control",    "circumstance"]
output_jsonlFormat = {    "label": "...", "similarity":0.0,   "reason": "..."    }
EmotionsDiscriminator = {
    "Anger": "Negative emotion caused by perceived wrongdoing, unfairness, harm, or obstruction, typically involving blame and a desire to confront or correct it.",

    "Disgust": "Strong aversion or repulsion toward something perceived as offensive, contaminating, immoral, or deeply unpleasant, with a tendency to reject or avoid it.",

    "Fear": "Negative emotion caused by a perceived threat or danger, characterized by vulnerability, uncertainty, and a desire to escape or protect oneself.",

    "Guilt": "Self-conscious negative emotion caused by judging a specific action or omission of one's own as wrong or harmful, often motivating repair or apology.",

    "Joy": "Positive emotion caused by a desirable event, achievement, gain, or valued experience that is appraised as pleasant and goal-consistent.",

    "Sadness": "Negative emotion caused by loss, separation, failure, or an undesirable outcome that is experienced as difficult or impossible to change.",

    "Shame": "Self-conscious negative emotion caused by evaluating oneself as inadequate, flawed, exposed, or socially unacceptable, often motivating withdrawal or hiding."
}
AppraisalEmotionPattern = {
    "Anger": {
        "attention": 0.80,
        "certainty": 0.75,
        "effort": 0.50,
        "pleasantness": 0.00,
        "responsibility": 0.10,
        "control": 0.05,
        "circumstance": 0.10
    },

    "Disgust": {
        "attention": 0.45,
        "certainty": 0.85,
        "effort": 0.35,
        "pleasantness": 0.00,
        "responsibility": 0.15,
        "control": 0.15,
        "circumstance": 0.20
    },

    "Fear": {
        "attention": 0.80,
        "certainty": 0.15,
        "effort": 0.80,
        "pleasantness": 0.00,
        "responsibility": 0.30,
        "control": 0.20,
        "circumstance": 0.45
    },

    "Guilt": {
        "attention": 0.40,
        "certainty": 0.80,
        "effort": 0.30,
        "pleasantness": 0.00,
        "responsibility": 0.90,
        "control": 0.60,
        "circumstance": 0.10
    },

    "Joy": {
        "attention": 0.95,
        "certainty": 0.90,
        "effort": 0.05,
        "pleasantness": 0.95,
        "responsibility": 0.40,
        "control": 0.30,
        "circumstance": 0.20
    },

    "Sadness": {
        "attention": 0.75,
        "certainty": 0.65,
        "effort": 0.55,
        "pleasantness": 0.00,
        "responsibility": 0.05,
        "control": 0.05,
        "circumstance": 0.70
    },

    "Shame": {
        "attention": 0.30,
        "certainty": 0.75,
        "effort": 0.35,
        "pleasantness": 0.00,
        "responsibility": 0.75,
        "control": 0.45,
        "circumstance": 0.10
    }
}
AppraisalEmotionPattern_gpt=AppraisalEmotionPattern = {
    "Anger": {
        "attention": 0.90,
        "certainty": 0.83,
        "effort": 0.42,
        "pleasantness": 0.00,
        "responsibility": 0.06,
        "control": 0.01,
        "circumstance": 0.03
    },

    "Disgust": {
        "attention": 0.47,
        "certainty": 0.94,
        "effort": 0.28,
        "pleasantness": 0.01,
        "responsibility": 0.10,
        "control": 0.08,
        "circumstance": 0.17
    },

    "Fear": {
        "attention": 0.90,
        "certainty": 0.09,
        "effort": 0.85,
        "pleasantness": 0.03,
        "responsibility": 0.30,
        "control": 0.13,
        "circumstance": 0.46
    },

    "Guilt": {
        "attention": 0.38,
        "certainty": 0.92,
        "effort": 0.25,
        "pleasantness": 0.00,
        "responsibility": 0.93,
        "control": 0.62,
        "circumstance": 0.08
    },

    "Joy": {
        "attention": 0.97,
        "certainty": 0.98,
        "effort": 0.03,
        "pleasantness": 0.99,
        "responsibility": 0.45,
        "control": 0.29,
        "circumstance": 0.17
    },

    "Sadness": {
        "attention": 0.85,
        "certainty": 0.78,
        "effort": 0.62,
        "pleasantness": 0.01,
        "responsibility": 0.05,
        "control": 0.01,
        "circumstance": 0.68
    },

    "Shame": {
        "attention": 0.22,
        "certainty": 0.78,
        "effort": 0.36,
        "pleasantness": 0.01,
        "responsibility": 0.74,
        "control": 0.47,
        "circumstance": 0.08
    }
}
def build_Pha1prompt_appraisal(sentence):#score 0.25 0.5 ...
    sys_prompt = f"""
        You are an expert in annotation, Natural Language Processing, and psychology.

        Infer the speaker's Appraisal dimensions ONLY from the given sentence by following these steps:
        1. Identify the core event and objective fact, then describe how the speaker subjectively interprets
           the event, including its perceived cause, outcome, thoughts, beliefs, expectations, or imaginings.
        2. Identify explicit and implicit appraisal indicators and extract concise evidence as a word,
           short phrase, or brief clause from the sentence.
        3. Assign scores from 0.0 to 1.0 for each of the 7 Appraisal dimensions based on evidence strength.
           Use these reference anchors:
           0.00 = absent or unsupported
           0.25 = weak
           0.50 = moderate
           0.75 = strong
           1.00 = overwhelming
        4. Provide a concise reason explaining why each score was assigned.

        Appraisal Dimensions:
        - attention: Desire to devote further attention to the event.
        - certainty: Certainty about what is happening.
        - effort: Mental or physical effort the SPEAKER personally needs to handle the situation.
        - pleasantness: Perceived pleasantness of the event.
        - responsibility: The SPEAKER's personal responsibility for the situation.
        - control: The SPEAKER's personal ability to control or influence the situation.
        - circumstance: Belief that no one can influence or change the event.

        Important rules:
        - Use only information provided in the sentence.
        - Do not fabricate unsupported details.
        - Evaluate each Appraisal dimension independently.
        - For effort, evaluate only the speaker's own effort, not effort made by other people.
        - Do not confuse responsibility with control:
          Responsibility concerns whether the speaker caused the event or deserves credit or blame.
          Control concerns the speaker's ability to influence or change the event.
        - Do not confuse personal lack of control with an unchangeable circumstance:
          Low control does not necessarily mean the event is unavoidable or unchangeable.
          High circumstance means the event is perceived as beyond anyone's influence.
        - Evidence must be grounded in the sentence.
        - Return ONLY one valid JSON object in EXACTLY this format:
        {AppraisalDimension}
    """

    user_prompt = f"""
        Sentence:
        {sentence}

        Let's think step by step internally.
    """
    return sys_prompt, user_prompt


def build_Pha1prompt_appraisal_01(sentence):
    sys_prompt = f"""
        You are an expert in annotation, Natural Language Processing, and psychology.

        Infer the Appraisal dimensions ONLY from the sentence by following these steps:
        1. Fact + speaker interpretation:
           Identify what occurred and how the speaker interprets the cause, outcome, and relevant beliefs.
        2. Evidence:
           Extract explicit or implicit appraisal cues for each dimension as a word, phrase,
           or brief clause from the sentence.
        3. Assign a continuous score from 0.0 to 1.0 based on evidence strength,
           ranging from unsupported to weak, moderate, strong, and very strong evidence.
        4. Provide a concise reason explaining why the score was assigned.

        Appraisal Dimensions:
        - attention: Desire to devote further attention to the event.
        - certainty: Certainty about what is happening.
        - effort: Mental or physical effort the SPEAKER personally needs to handle the situation.
        - pleasantness: Perceived pleasantness of the event.
        - responsibility: The SPEAKER's personal responsibility for the situation.
        - control: The SPEAKER's personal ability to control or influence the situation.
        - circumstance: Belief that no one can influence or change the event.

        Important rules:
        - Use only information in the sentence.
        - Do not infer or fabricate unsupported information.
        - Evaluate each Appraisal dimension independently while considering the whole sentence.
        - For effort, responsibility, and control, evaluate the SPEAKER, not other people.
        - Evidence, scores, and reasons must be grounded in the sentence.
        - Return ONLY one valid JSON object in EXACTLY this format:
        {AppraisalDimension}
    """

    user_prompt = f"""
        Sentence:
        {sentence}
    """
    return sys_prompt, user_prompt


def build_Pha1prompt_appraisal_01_v2(sentence):#gpt cải tiến v1
    sys_prompt = f"""
        You are an expert in annotation, Natural Language Processing, and psychology.

        Infer the Appraisal dimensions ONLY from the sentence by following these steps:
        1. Fact + speaker interpretation:
           Identify what occurred and how the speaker interprets the cause, outcome, and relevant beliefs.
        2. Evidence:
           Extract explicit or implicit appraisal cues for each dimension as a word, phrase,
           or brief clause from the sentence.
        3. Assign a continuous score from 0.0 to 1.0 based on evidence strength,
           ranging from unsupported to weak, moderate, strong, and very strong evidence.
        4. Provide a concise reason explaining why the score was assigned.

        Appraisal Dimensions:
        - attention: Desire to devote further attention to the event.
        - certainty: Certainty about what is happening.
        - effort: Mental or physical effort the SPEAKER personally needs to handle the situation.
        - pleasantness: Perceived pleasantness of the event.
        - responsibility: The SPEAKER's personal responsibility for the situation.
        - control: The SPEAKER's personal ability to control or influence the situation.
        - circumstance: Belief that no one can influence or change the event.

        Important rules:
        - Use only information from the sentence.
        - Do not infer or fabricate unsupported details.
        - Evaluate each Appraisal dimension independently while considering the overall context.
        - Ground every score and reason in evidence from the sentence.
        - Keep each reason focused on the relevant event, textual evidence, and the speaker's appraisal.
        - For effort, evaluate only effort personally expended by the speaker.
        - Distinguish responsibility from control:
          Responsibility concerns whether the speaker caused the event or deserves credit or blame.
          Control concerns the speaker's ability to influence, manage, or change the situation.
        - Distinguish low control from high circumstance:
          Low control means the speaker has limited ability to influence the situation.
          High circumstance means the event is largely determined by conditions beyond anyone's influence.
        - Return ONLY one valid JSON object in EXACTLY this format:
        {AppraisalDimension}
    """

    user_prompt = f"""
        Sentence:
        {sentence}
    """
    return sys_prompt, user_prompt


def generate_text_jf(model, system_prompt, user_prompt):
    """Call selected provider and expose the real error instead of silently skipping rows."""
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


def pha1_infer_appraisal_dimensions_jf(model, sentence):
    sys_prompt, user_prompt = build_Pha1prompt_appraisal_01_v2(
        sentence=sentence
    )

    result_str = generate_text_jf(
        model=model,
        system_prompt=sys_prompt,
        user_prompt=user_prompt
    )

    if result_str is None:
        return None

    try:
        return json.loads(result_str)
    except json.JSONDecodeError as error:
        print(f"JSON decode error: {error}")
        print(f"Raw output: {repr(result_str)}")
        return None


def runPha1_Text2Appraisal():
    # EnISEAR input fields used here: sentence_id, sentence, true_label
    # df = read_jsonl_to_dataframe( jsonl_file_path=r"G:\Python\Datasets\EnISEAR\EnISEAR_corpus_fixed.jsonl")
    # df = df[["sentence_id", "sentence", "true_label"]].copy()

    # error_file = ( r"G:\Python\EnISEAR_AP01\ErrorRows"  + "\\"  + modelname  + "_EnIS_APatternContinuous_pha1_Errors.jsonl")
    # output_file = ( r"G:\Python\EnISEAR_AP01\Results"  + "\\"+ modelname+ "_EnIS_APatternContinuous_pha1.jsonl"    )

    # Rerun errors example:
    # df = read_jsonl_to_dataframe( r"G:\Python\EnISEAR_AP01\ErrorRows"  + "\\"   + modelname + "_EnIS_APatternContinuous_pha1_Errors.jsonl" )
# Rerun lỗi
    df=read_jsonl_to_dataframe(jsonl_file_path=r"G:\Python\EnISEAR_AP01\ErrorRows" + "\\" +modelname+"_EnIS_APatternContinuous_pha1_L2.jsonl")
    error_file = r"G:\Python\EnISEAR_AP01\ErrorRows" + "\\" +modelname+"_EnIS_APatternContinuous_pha1_L3.jsonl"
    output_file =r"G:\Python\EnISEAR_AP01\Results" + "\\" +modelname+"_EnIS_APatternContinuous_pha1.jsonl"

    with (
        open(output_file, "a", encoding="utf-8") as f,
        open(error_file, "a", encoding="utf-8") as error_f
    ):
        for index, row in df.iterrows():
        # for index, row in df.iloc[0:1].iterrows():  
        # for index, row in df.iloc[3:len(df)].iterrows(): 
            sentence_id = row["sentence_id"]
            label = row["true_label"]
            sentence = row["sentence"]

            print(f"\nProcessing row {index}, sentence_id={sentence_id}:\n{output_file}")

            try:
                result = pha1_infer_appraisal_dimensions_jf(model=model,sentence=sentence                )

                if result is None:
                    error_record = {
                        "sentence_id": sentence_id,
                        "true_label": label,
                        "sentence": sentence,
                        "error_type": "EmptyResult",
                        "error_reason": "Model returned None or invalid JSON."
                    }
                    error_f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                    error_f.flush()
                    print(f"Empty result -> saved to {error_file}")
                    continue

                record = {
                    "sentence_id": sentence_id,
                    "true_label": label,
                    "sentence": sentence,
                    "appraisal": {
                        dimension: result.get(dimension, {})
                        for dimension in AppraisalDimensionsName
                    }
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                print(
                    f"Sentence ID: {sentence_id}, label: {label} "
                    f"Saved => {output_file}"
                )

            except Exception as e:
                error_record = {
                    "sentence_id": sentence_id,
                    "true_label": label,
                    "sentence": sentence,
                    "error_type": type(e).__name__,
                    "error_reason": str(e)
                }
                error_f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                error_f.flush()

                print(
                    f"sentence_id={sentence_id} exception: "
                    f"{type(e).__name__}: {e}"
                )
                print(f"Saved error row to {error_file}")
                continue

# runPha1_Text2Appraisal()
def buildPrompt_2Emotion_AppraisalEmotionPattern_ver2_2(sentence, appraisal):
    sys_prompt = f"""
    You are an expert in Natural Language Processing and psychology.

    AppraisalEmotionPattern:
    {AppraisalEmotionPattern}

    Emotion Discriminators:
    {EmotionsDiscriminator}

    Task:
    Identify the primary emotion that best matches the observed appraisal.

    Instructions:
        Instructions:
    1. Compare the observed appraisal with all emotion patterns and measure the Cosine similarity.
    2. Select the emotion with the highest Cosine similarity. 
    If several emotions are similarly well matched, use the situation, conversation, discriminating textual cues, and emotion discriminators {EmotionsDiscriminator} to make the final choice.
    3. Explain the elected emotion , including:
    - key appraisal cues,
    - brief quoted phrases from the conversation,
    - why it's the best match.

    Rules:
    - Select exactly ONE emotion from:
      {emotion_list}
    - Use only information from the sentence.
    - Return exactly one valid JSON object matching:
      {output_jsonlFormat}
    """

    user_prompt = f"""
    Identify the primary emotion based on:

    Sentence:
    {sentence}

    Appraisal:
    {appraisal}
    """
    return sys_prompt, user_prompt


def buildPrompt_2Emotion_AppraisalEmotionPattern_v1_1(sentence, appraisal):
    sys_prompt = f"""
    You are an expert in NLP, psychology, and fine-grained emotion classification.

    AppraisalEmotionPattern:
    {AppraisalEmotionPattern}

    Task:
    Identify the single primary emotion that best matches the sentence and appraisal.

    Instructions:
    Instructions:
        1. Compare the observed appraisal with all emotion patterns and measure the Cosine Similarity.
        2. Select the emotion with the highest Cosine similarity. 
        If several emotions are similarly well matched, use the situation, conversation, discriminating textual cues, and emotion discriminators {EmotionsDiscriminator} to make the final choice.
        3. Explain the elected emotion , including:
        - key appraisal cues,
        - brief quoted phrases from the conversation,
        - why it's the best match.

    Rules:
    - Select exactly ONE emotion from:
      {emotion_list}
    - Use only information from the sentence.
    - Return exactly one valid JSON object parseable by json.loads():
      {output_jsonlFormat}
    """

    user_prompt = f"""
    Sentence:
    {sentence}

    Appraisal:
    {appraisal}
    """
    return sys_prompt, user_prompt


def buildPrompt_2Emotion_AppraisalEmotionPattern_v1gpt(sentence, appraisal):
    sys_prompt = f"""
    You are an expert in NLP, psychology, and fine-grained emotion classification.

    AppraisalEmotionPattern:
    {AppraisalEmotionPattern_gpt}

    Task:
    Identify the single primary emotion that best matches the sentence and appraisal.

    Instructions:
    Instructions:
        1. Compare the observed appraisal with all emotion patterns and measure the similarity.
        2. Select the emotion with the highest similarity. 
        If several emotions are similarly well matched, use the situation, conversation, discriminating textual cues, and emotion discriminators {EmotionsDiscriminator} to make the final choice.
        3. Explain the elected emotion , including:
        - key appraisal cues,
        - brief quoted phrases from the conversation,
        - why it's the best match.

    Rules:
    - Select exactly ONE emotion from:
      {emotion_list}
    - Use only information from the sentence.
    - Return exactly one valid JSON object parseable by json.loads():
      {output_jsonlFormat}
    """

    user_prompt = f"""
    Sentence:
    {sentence}

    Appraisal:
    {appraisal}
    """
    return sys_prompt, user_prompt

def buildPrompt_2Emotion_AppraisalEmotionPattern_ver3_v2_2603(sentence, appraisal):#mặc định
    sys_prompt = f"""
    You are an expert in NLP, psychology, and fine-grained emotion classification.

    AppraisalEmotionPattern:
    {AppraisalEmotionPattern}

    Task:
    Identify the single primary emotion that best matches the sentence and appraisal.

    Instructions:
    1. Compare the observed appraisal with all emotion patterns and measure their similarity.
    2. Identify the candidate emotions whose similarity scores are markedly higher than the others.
    3. Among these candidates:
       - If one emotion has substantially higher similarity than the remaining candidates, select it.
       - Otherwise, use the sentence, discriminating textual cues, and
         {EmotionsDiscriminator} to make the final choice.
         Prefer the emotion better supported by emotion-specific evidence,
         even if its similarity is slightly lower.
    4. Explain briefly why the selected emotion is the best overall match.

    Rules:
    - Select exactly ONE emotion from:
      {emotion_list}
    - Use only information from the sentence.
    - Return exactly one valid JSON object parseable by json.loads():
      {output_jsonlFormat}
    """

    user_prompt = f"""
    Sentence:
    {sentence}

    Appraisal:
    {appraisal}
    """
    return sys_prompt, user_prompt


def buildPrompt_2Emotion_AppraisalEmotionPattern_ver4(sentence, appraisal):
    sys_prompt = f"""
    You are an expert in NLP, psychology, and fine-grained emotion classification.

    AppraisalEmotionPattern:
    {AppraisalEmotionPattern}

    Task:
    Identify the single primary emotion that best matches the sentence and appraisal.

    Instructions:
    1. Compare the observed appraisal with all emotion patterns and measure their similarity.
    2. Identify the candidate emotions whose similarity scores are markedly higher than the others.
    3. If one candidate has substantially higher similarity than the remaining candidates, select it.
       Otherwise, evaluate the competing emotions using appraisal similarity, the sentence,
       discriminating textual cues, and {EmotionsDiscriminator}, then select the emotion
       best supported by the overall evidence.
    4. Explain briefly why the selected emotion is the best overall match.

    Rules:
    - Select exactly ONE emotion from:
      {emotion_list}
    - Use only information from the sentence.
    - Return exactly one valid JSON object parseable by json.loads():
      {output_jsonlFormat}
    """

    user_prompt = f"""
    Sentence:
    {sentence}

    Appraisal:
    {appraisal}
    """
    return sys_prompt, user_prompt


def pha2_infer_appraisal_dimensions_jf_v3(model, sentence, appraisal):#mặc định
    sys_prompt, user_prompt = buildPrompt_2Emotion_AppraisalEmotionPattern_ver3_v2_2603(
        sentence=sentence,
        appraisal=appraisal
    )

    result_str = generate_text_jf(
        model=model,
        system_prompt=sys_prompt,
        user_prompt=user_prompt
    )

    if result_str is None:
        return None

    try:
        return json.loads(result_str)
    except json.JSONDecodeError as error:
        print(f"JSON decode error: {error}")
        print(f"Raw output: {repr(result_str)}")
        return None


def pha2_infer_appraisal_dimensions_jf_v2(model, sentence, appraisal):
    sys_prompt, user_prompt = buildPrompt_2Emotion_AppraisalEmotionPattern_ver2_2(
        sentence=sentence,
        appraisal=appraisal
    )

    result_str = generate_text_jf(
        model=model,
        system_prompt=sys_prompt,
        user_prompt=user_prompt
    )

    if result_str is None:
        return None

    try:
        return json.loads(result_str)
    except json.JSONDecodeError as error:
        print(f"JSON decode error: {error}")
        print(f"Raw output: {repr(result_str)}")
        return None


def pha2_infer_appraisal_dimensions_jf_v1(model, sentence, appraisal):
    sys_prompt, user_prompt = buildPrompt_2Emotion_AppraisalEmotionPattern_v1_1(
        sentence=sentence,
        appraisal=appraisal
    )

    result_str = generate_text_jf(
        model=model,
        system_prompt=sys_prompt,
        user_prompt=user_prompt
    )

    if result_str is None:
        return None

    try:
        return json.loads(result_str)
    except json.JSONDecodeError as error:
        print(f"JSON decode error: {error}")
        print(f"Raw output: {repr(result_str)}")
        return None


def runPha2End_Text2Appraisal():
    # Sd gold làm pha 1
    df2=read_enisear(path=r"G:\Python\Datasets\EnISEAR\EnISEAR_corpus_fixed.jsonl" )
    error_file = (  r"G:\Python\EnISEAR_AP01\ErrorRows" + "\\"+ modelname+ "_EnIS_APatternContinuous_GOLDPha2_Errors_v3_2.jsonl" )
    output_file = ( r"G:\Python\EnISEAR_AP01\Results"+ "\\" + modelname+ "_EnIS_APatternContinuous_GOLDPha2_v3_2603.jsonl")
    # XLy GOLD Errors
    # df2 = read_jsonl_to_dataframe( jsonl_file_path=r"G:\Python\EnISEAR_AP01\ErrorRows" + "\\"+ modelname+ "_EnIS_APatternContinuous_GOLDPha2_Errors_v3.jsonl"  )
    # error_file = (  r"G:\Python\EnISEAR_AP01\ErrorRows" + "\\"+ modelname+ "_EnIS_APatternContinuous_GOLDPha2_Errors_v3.jsonl" )
    # output_file = ( r"G:\Python\EnISEAR_AP01\Results"+ "\\" + modelname+ "_EnIS_APatternContinuous_GOLDPha2_v3.jsonl")
    # print(df2.head)

    # # Sd gold làm pha 1
    # df2 = read_jsonl_to_dataframe( jsonl_file_path=(r"G:\Python\EnISEAR_AP01\Results" + "\\"+ modelname + "_EnIS_APatternContinuous_pha1.jsonl"  )    )

    # error_file = (  r"G:\Python\EnISEAR_AP01\ErrorRows" + "\\"+ modelname+ "_EnIS_APatternContinuous_Pha2_Errors.jsonl" )
    # output_file = ( r"G:\Python\EnISEAR_AP01\Results"+ "\\" + modelname+ "_EnIS_APatternContinuous_Pha2_v1gpt.jsonl")

    with (
        open(output_file, "a", encoding="utf-8") as f,
        open(error_file, "a", encoding="utf-8") as error_f
    ):
        for index, row in df2.iterrows():
        # for index, row in df2.iloc[0:20].iterrows():  
        # for index, row in df2.iloc[20:len(df2)].iterrows(): 
            sentence_id = row["sentence_id"]
            label = row["true_label"]
            sentence = row["sentence"]
            appraisal = row["appraisal"]

            print(
                f"\nProcessing row {index}, sentence_id={sentence_id}: "
                f"{output_file}"
            )

            try:
                # result = pha2_infer_appraisal_dimensions_jf_v2(model=model,sentence=sentence,  appraisal=appraisal )
                result = pha2_infer_appraisal_dimensions_jf_v3(model=model,sentence=sentence,  appraisal=appraisal )
                # result = pha2_infer_appraisal_dimensions_jf_v1(model=model,sentence=sentence,  appraisal=appraisal )

                if result is None:
                    error_record = {
                        "sentence_id": sentence_id,
                        "true_label": label,
                        "sentence": sentence,
                        "appraisal": appraisal,
                        "error_type": "EmptyResult",
                        "error_reason": "Model returned None or invalid JSON."
                    }
                    error_f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                    error_f.flush()
                    print(f"Empty result -> saved to {error_file}")
                    continue

                output_fields = {
                    "sentence_id": sentence_id,
                    "true_label": label,
                    "pred_label": result.get("label", ""),
                    "reason": result.get("reason", ""),
                    "similarity": result.get("similarity", "")
                }

                f.write(json.dumps(output_fields, ensure_ascii=False) + "\n")
                f.flush()

                print(
                    f"SAVED -> row {index}, "
                    f"sentence_id={sentence_id}, "
                    f"pred_label={output_fields['pred_label']}"
                )

            except Exception as e:
                error_record = {
                    "sentence_id": sentence_id,
                    "true_label": label,
                    "sentence": sentence,
                    "appraisal": appraisal,
                    "error_type": type(e).__name__,
                    "error_reason": str(e)
                }
                error_f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                error_f.flush()

                print(f"sentence_id={sentence_id} exception: "     f"{type(e).__name__}: {e}" )
                print(f"Saved error row to {error_file}")
                continue
runPha2End_Text2Appraisal()
