from importlib.resources import path
import json
from collections import defaultdict
from operator import index

import pandas as pd
import numpy as np

from openai import BadRequestError, OpenAI

import os
import traceback
from dotenv import load_dotenv


# ES dataset: conversation_id, label, appraisal
# Empathetic Dialogues dataset:
# conversation_id, label, situation, conversation


def read_jsonl_to_dataframe(    jsonl_file_path):
    data = []
    with open(jsonl_file_path, "r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    return df

load_dotenv()
# ============================================================
# MULTI-LLM CONFIGURATION
# ============================================================
# Tất cả API key / endpoint / model được đọc từ .env.
# Chọn LLM bằng cách gán trực tiếp 1 hàm cho biến client.
# ============================================================

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
    return OpenAI(
        api_key=api_key,
        base_url=endpoint + "/",
        timeout=120.0,
        max_retries=2,
    )


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


def call_azure_open_ai_GPT5api(model, system_prompt, user_prompt):
    """Azure OpenAI GPT-5 family via Responses API -> JSON string."""
    deployment = (
        model
        or _require_env("AZURE_OPENAI_GPT5_DEPLOYMENT")
    ).strip()

    client = _azure_v1_client()

    response = client.responses.create(
        model=deployment,
        input=f"{system_prompt}\n\n{user_prompt}",
    )

    status = getattr(response, "status", None)
    if status not in (None, "completed"):
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None) if details is not None else None
        error = getattr(response, "error", None)

        if reason == "content_filter":
            return None

        raise RuntimeError(
            f"[Azure-GPT5] deployment={deployment}: "
            f"status={status}, reason={reason}, error={error}"
        )

    return _ensure_json_string(
        response.output_text,
        "Azure-GPT5",
        deployment
    )


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
def call_open_ai_api_gpt5(model, system_prompt, user_prompt):

    json_instruction = """
Return ONLY a valid JSON object.
The response must be parseable by json.loads().
Do not use Markdown code fences.
Do not output explanations outside the JSON object.
"""

    response = client.responses.create(
        model=model,

        input=[
            {
                "role": "system",
                "content": system_prompt + "\n\n" + json_instruction
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

    return response.output_text

def call_Qwen(model, system_prompt, user_prompt):
    """Qwen through Alibaba Model Studio OpenAI-compatible Chat API -> JSON string."""
    model_name = model or os.getenv("QWEN_MODEL", "qwen-plus")
    base_url = _require_env("QWEN_BASE_URL").rstrip("/")

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
        extra_body={"enable_thinking": False},
    )

    content = response.choices[0].message.content
    return _ensure_json_string(content, "Qwen", model_name)


def call_Mistral(model, system_prompt, user_prompt):  # lỗi 429 giới hạn
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
# Cerebras GPT-OSS-120B (ACTIVE)
# client = call_cerebras
# model = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
# modelname = "cerebras_gpt_oss_120b"
# ============================================================
# CHỌN LLM
# Chỉ bật 1 client/model tại một thời điểm.
# ============================================================

# OpenAI GPT-4o-mini - mặc định
# client = call_open_ai_api_gpt4omini
# model = os.getenv("OPENAI_GPT4O_MINI_MODEL", "gpt-4o-mini")
# modelname = "azure_gpt4omini"
# modelname = "openai_gpt4omini"

# OpenAI GPT-5 XL lỗi
# client = call_open_ai_api_gpt5
# model = os.getenv("OPENAI_GPT5_MODEL", "gpt-5-mini")
# modelname = "azure_gpt5"
# modelname = "open_ai_GPT5"

# Azure OpenAI GPT-5
# client = call_azure_open_ai_GPT5api
# model = _require_env("AZURE_OPENAI_GPT5_DEPLOYMENT")
# modelname = "azure_gpt5"

# Azure OpenAI GPT-4o-mini
# client = call_azure_open_ai_api
# model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
# modelname = "azure_gpt4omini"

# Qwen
client = call_Qwen
model = os.getenv("QWEN_MODEL", "qwen-plus")
modelname = "QWEN_MODEL"

# # Mistral
# client = call_Mistral
# model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
# modelname = "MISTRAL_MODEL"

# Gemini
# client = call_gemini
# model = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
# modelname = "GEMINI_MODEL"

print(f"LLM client: {client.__name__} | model: {model}")



# ============================================================
# EnISEAR CONFIGURATION
# Input fields: sentence_id, true_label, sentence
# ============================================================
emotion_list = [    "Anger",    "Disgust",    "Fear",    "Guilt",    "Joy",    "Sadness",    "Shame"]


EmotionsDiscriminator = {
    "Anger": "Negative emotion caused by perceived wrongdoing, unfairness, harm, or obstruction, typically involving blame and a desire to confront or correct it.",
    "Disgust": "Strong aversion or repulsion toward something perceived as offensive, contaminating, immoral, or deeply unpleasant, with a tendency to reject or avoid it.",
    "Fear": "Negative emotion caused by a perceived threat or danger, characterized by vulnerability, uncertainty, and a desire to escape or protect oneself.",
    "Guilt": "Self-conscious negative emotion caused by judging a specific action or omission of one's own as wrong or harmful, often motivating repair or apology.",
    "Joy": "Positive emotion caused by a desirable event, achievement, gain, or valued experience that is appraised as pleasant and goal-consistent.",
    "Sadness": "Negative emotion caused by loss, separation, failure, or an undesirable outcome that is experienced as difficult or impossible to change.",
    "Shame": "Self-conscious negative emotion caused by evaluating oneself as inadequate, flawed, exposed, or socially unacceptable, often motivating withdrawal or hiding."
}

output_jsonlFormat = {
    "label": "...",
    "reason": "..."
}
output_jsonlFormat_JSON = json.dumps(
    output_jsonlFormat,
    ensure_ascii=False,
    indent=2
)


def ZeroShot_BuildPrompt_2Emotion(sentence):#cao quá 75.62
    sys_prompt = f"""
    You are an expert in Natural Language Processing and psychology.

    Identify the single primary emotion expressed by the speaker.
    Explain the decision briefly using evidence from the sentence.

    Select exactly ONE emotion from:
    {emotion_list}

    STRICT OUTPUT RULES:
    - Use only information from the sentence.
    - Return ONLY one valid JSON object.
    - Use double quotes for all JSON keys and string values.
    - Do NOT use single quotes, Markdown, or code fences.
    - Do NOT add text before or after the JSON object.
    - The output must be directly parseable by json.loads().
    - Return exactly this JSON structure:
    {output_jsonlFormat_JSON}
    """

    user_prompt = f"""
    Classify the primary emotion expressed in this sentence:

    Sentence:
    {sentence}
    """

    return sys_prompt, user_prompt


def ZeroShot_BuildPrompt_2Emotion_v3L(sentence):#v2L 75.3 v3L
    sys_prompt = f"""
    You are an expert in NLP and psychology.

    Identify the exactly ONE emotion from:
    {emotion_list}

    STRICT OUTPUT RULES:
    - Use double quotes for all JSON keys and string values.
    - Do NOT use single quotes, Markdown, or code fences.
    - Do NOT add text before or after the JSON object.
    - The output must be directly parseable by json.loads().
    - Return exactly this JSON structure:
    {output_jsonlFormat_JSON}
    """

    user_prompt = f"""
    Sentence:
    {sentence}
    """

    return sys_prompt, user_prompt

def ZeroShot_BuildPrompt_2Emotion_v4L(sentence):
    sys_prompt = f"""
    You are an expert in NLP

    Identify the exactly ONE emotion from:
    {emotion_list}

    STRICT OUTPUT RULES:
    - Use double quotes for all JSON keys and string values.
    - Do NOT use single quotes, Markdown, or code fences.
    - Do NOT add text before or after the JSON object.
    - The output must be directly parseable by json.loads().
    - Return exactly this JSON structure:
    {output_jsonlFormat_JSON}
    """

    user_prompt = f"""
    Sentence:
    {sentence}
    """

    return sys_prompt, user_prompt
def ZeroShot_BuildPrompt_2Emotion_v5L(sentence):
    sys_prompt = f"""
    Identify the exactly ONE emotion from:
    {emotion_list}

    STRICT OUTPUT RULES:
    - Use double quotes for all JSON keys and string values.
    - Do NOT use single quotes, Markdown, or code fences.
    - Do NOT add text before or after the JSON object.
    - The output must be directly parseable by json.loads().
    - Return exactly this JSON structure:
    {output_jsonlFormat_JSON}
    """

    user_prompt = f"""
    Sentence:
    {sentence}
    """

    return sys_prompt, user_prompt
def CoT_buildPrompt_2Emotion_v2L(sentence):
    sys_prompt = f"""
    You are an expert in Natural Language Processing and psychology.

    Identify the single primary emotion expressed by the speaker.

    Important rules:
    - Think step by step internally before making the final decision.
    - Consider the event described, the speaker's interpretation, and emotional expression in the sentence.
    - Use only information from the sentence.
    - Select exactly ONE primary emotion from:
    {emotion_list}
    - Return ONLY one valid JSON object.
    - Use double quotes for all JSON keys and string values.
    - Do NOT use single quotes, Markdown, or code fences.
    - Do NOT add text before or after the JSON object.
    - The output must be directly parseable by json.loads().
    - Return exactly this JSON structure:
    {output_jsonlFormat_JSON}
    """

    user_prompt = f"""
    Sentence:
    {sentence}
    """

    return sys_prompt, user_prompt
def CoT_buildPrompt_2Emotion_ver2(sentence):
    sys_prompt = f"""
    You are an expert in Natural Language Processing and psychology.

    Identify the single primary emotion expressed by the speaker.

    Important rules:
    - Think step by step internally before making the final decision.
    - Consider the event described, the speaker's interpretation, and emotional expression in the sentence.
    - Use only information from the sentence.
    - Select exactly ONE primary emotion from:
    {emotion_list}
    - Return ONLY one valid JSON object.
    - Use double quotes for all JSON keys and string values.
    - Do NOT use single quotes, Markdown, or code fences.
    - Do NOT add text before or after the JSON object.
    - The output must be directly parseable by json.loads().
    - Return exactly this JSON structure:
    {output_jsonlFormat_JSON}
    """

    user_prompt = f"""
    Identify the primary emotion expressed in this sentence:

    Sentence:
    {sentence}

    Think step by step internally.
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


def text_infer_emotion_jf_coT(model, sentence):
    sys_prompt, user_prompt = CoT_buildPrompt_2Emotion_v2L(
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


def text_infer_emotion_jf_zeroshot(model, sentence):
    sys_prompt, user_prompt = ZeroShot_BuildPrompt_2Emotion_v5L(
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


def Text2Appraisal():
    df = read_jsonl_to_dataframe( jsonl_file_path=r"G:\Python\Datasets\EnISEAR\EnISEAR_corpus_fixed.jsonl")
    # df = df[["sentence_id",  "true_label","sentence"]].copy()
    # # ACTIVE MODE: CoT
    # df=read_jsonl_to_dataframe(jsonl_file_path=r"G:\Python\EnISEAR_AP01\ErrorRows" + "\\" +modelname+"_EnIS_CoT_Errors_v2L.jsonl")
    # error_file = ( r"G:\Python\EnISEAR_AP01\ErrorRows"  + "\\"  + modelname  + "_EnIS_CoT_Errors_v2L_L3.jsonl")
    # output_file = ( r"G:\Python\EnISEAR_AP01\Results"  + "\\"+ modelname+ "_EnIS_CoT_v2L.jsonl"    )

    # df=read_jsonl_to_dataframe(jsonl_file_path=r"G:\Python\EnISEAR_AP01\ErrorRows" + "\\" +modelname+"_EnIS_ZeroShot_Errors_v2L.jsonl")
    error_file = ( r"G:\Python\EnISEAR_AP01\ErrorRows"  + "\\"  + modelname  + "_EnIS_ZeroShot_Errors_v5L.jsonl")
    output_file = ( r"G:\Python\EnISEAR_AP01\Results"  + "\\"+ modelname+ "_EnIS_ZeroShot_v5L.jsonl"    )
  

    with (
        open(output_file, "a", encoding="utf-8") as f,
        open(error_file, "a", encoding="utf-8") as error_f
    ):
        print(output_file)
        # for index, row in df.iloc[0:3].iterrows():  
        # for index, row in df.iloc[3:len(df)].iterrows(): 
        for index, row in df.iterrows():
            sentence_id = row["sentence_id"]
            true_label = row["true_label"]
            sentence = row["sentence"]

            print(  f"\nProcessing row {index}, "      f"sentence_id={sentence_id}:\n{output_file}")

            try:
                # ACTIVE: CoT
                # result = text_infer_emotion_jf_coT( model=model, sentence=sentence  )

                # ZeroShot:
                result = text_infer_emotion_jf_zeroshot( model=model, sentence=sentence  )


                if result is None:
                    error_record = {
                        "sentence_id": sentence_id,
                        "true_label": true_label,
                        "sentence": sentence,
                        "error_type": "EmptyOrInvalidResult",
                        "error_reason": "Model returned None or invalid JSON."
                    }
                    error_f.write(
                        json.dumps(error_record, ensure_ascii=False) + "\n"
                    )
                    error_f.flush()
                    print(f"EMPTY/INVALID -> saved to {error_file}")
                    continue

                record = {
                    "sentence_id": sentence_id,
                    "true_label": true_label,
                    "pred_label": result.get("label", ""),
                    "reason": result.get("reason", "")
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                print(
                    f"SAVED -> sentence_id={sentence_id}, "
                    f"true_label={true_label}, "
                    f"pred_label={record['pred_label']}"
                )

            except Exception as e:
                error_record = {
                    "sentence_id": sentence_id,
                    "true_label": true_label,
                    "sentence": sentence,
                    "error_type": type(e).__name__,
                    "error_reason": str(e)
                }

                error_f.write(
                    json.dumps(error_record, ensure_ascii=False) + "\n"
                )
                error_f.flush()

                print(
                    f"sentence_id={sentence_id} gặp exception: "
                    f"{type(e).__name__}: {e}"
                )
                print(f"Đã ghi row lỗi vào file {error_file}")
                continue


Text2Appraisal()
