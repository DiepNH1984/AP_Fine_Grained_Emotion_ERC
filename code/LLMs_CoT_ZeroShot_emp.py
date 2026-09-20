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


def read_jsonl_to_dataframe(    jsonl_file_path=r"G:\Python\emp_chat.jsonl"):
    data = []
    with open(jsonl_file_path, "r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    return df
# df = read_jsonl_to_dataframe()


# ============================================================
# MULTI-LLM CONFIGURATION
# ============================================================
# Tất cả API key / endpoint / model được đọc từ .env.
# Chọn LLM bằng cách gán trực tiếp 1 hàm cho biến client.
# ============================================================

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
client = call_cerebras
model = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
modelname = "cerebras_gpt_oss_120b"
# ============================================================
# CHỌN LLM
# Chỉ bật 1 client/model tại một thời điểm.
# ============================================================

# OpenAI GPT-4o-mini - mặc định
# client = call_open_ai_api_gpt4omini
# model = os.getenv("OPENAI_GPT4O_MINI_MODEL", "gpt-4o-mini")
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
# client = call_Qwen
# model = os.getenv("QWEN_MODEL", "qwen-plus")
# modelname = "QWEN_MODEL"

# Mistral
# client = call_Mistral
# model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
# modelname = "MISTRAL_MODEL"

# Gemini
# client = call_gemini
# model = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
# modelname = "GEMINI_MODEL"

print(f"LLM client: {client.__name__} | model: {model}")

# emotion_appraisal_relation = {}
emotion_list = ["afraid", "angry", "annoyed", "anticipating", "anxious", "apprehensive", "ashamed", "caring", "confident", "content", "devastated", "disappointed", "disgusted", "embarrassed", "excited", "faithful", "furious", "grateful", "guilty", "hopeful", "impressed", "jealous", "joyful", "lonely", "nostalgic", "prepared", "proud", "sad", "sentimental", "surprised", "terrified", "trusting"]

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
output_jsonlFormat = {    "label": "...",  "reason": "..."    }
output_jsonlFormat_JSON = json.dumps(output_jsonlFormat, ensure_ascii=False, indent=2)



def ZeroShot_BuildPrompt_2Emotion(situation,conversation):
    sys_prompt = f"""
        You are an expert in Natural Language Processing expert and psychology.
        Identify the primary emotion and explain how it is formed using short quotes from the conversation (< 60 words)
        
        Select emotion ONLY from {emotion_list}

        STRICT OUTPUT RULES:
        - Return ONLY one valid JSON object.
        - Use double quotes for all JSON keys and string values.
        - Do NOT use single quotes, Markdown, or code fences.
        - Do NOT add text before or after the JSON object.
        - The output must be directly parseable by json.loads().
        - Return exactly this JSON structure:
        {output_jsonlFormat_JSON}
                """ 
    user_prompt =  f"""
            Classify the primary emotion base on
            situation:
            {situation}
            Dialogue:
            {conversation}
                """
    return sys_prompt, user_prompt


def CoT_buildPrompt_2Emotion_ver2(situation,conversation):#XL ổn hơn 46%
    sys_prompt = f""" 
    You are an expert in Natural Language Processing and psychology. 

    Identify the primary emotion and explain how it is formed using short quotes from the conversation 

    important rule: 
    - Let's Think step by step internally before making the final decision Use the following information 
    - The context, conversation situation, and The speaker's emotional expression. 
    - you must exactly one primary emotion from the following emotion list: 
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
    Identify the primary emotion. 
    situation:
    {situation}
    Dialogue: 
    {conversation} 
    Let's Think step by step internally
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

def text_infer_emotion_jf_coT(model, situation,conversation):
    # sys_prompt, user_prompt = ZeroShot_BuildPrompt_2Emotion(
    sys_prompt, user_prompt = CoT_buildPrompt_2Emotion_ver2(
        situation= situation,
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
def text_infer_emotion_jf_zeroshot(model, situation,conversation):
    sys_prompt, user_prompt = ZeroShot_BuildPrompt_2Emotion(
    # sys_prompt, user_prompt = CoT_buildPrompt_2Emotion_ver2(
        situation= situation,
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
def Text2Appraisal():
    df=read_jsonl_to_dataframe(jsonl_file_path=r"G:\Python\emp_chat.jsonl")
    # error_file = r"G:\Python\APContinuous\ErrorRows" + "\\" +modelname+"_emp_ZeroShot_rowerror.jsonl"
    # output_file =r"G:\Python\APContinuous\Results" + "\\" +modelname+"_emp_ZeroShot.jsonl"


    error_file = r"G:\Python\APContinuous\ErrorRows" + "\\" +modelname+"_emp_CoT.jsonl"
    output_file =r"G:\Python\APContinuous\Results" + "\\" +modelname+"_emp_CoT_test.jsonl"
# Rows Errors
    # df2=read_jsonl_to_dataframe(jsonl_file_path=r"G:\Python\APContinuous\ErrorRows" + "\\" +modelname+"_emp_CoT_L2.jsonl")
    # error_file = r"G:\Python\APContinuous\ErrorRows" + "\\" +modelname+"_emp_CoT_L3.jsonl"
    # output_file =r"G:\Python\APContinuous\Results" + "\\" +modelname+"_emp_CoT.jsonl"
    # df2=read_jsonl_to_dataframe(jsonl_file_path=r"G:\Python\APContinuous\ErrorRows\azure_gpt5_emp_CoT_L2.jsonl")
    # df=read_jsonl_to_dataframe(jsonl_file_path=r"G:\Python\APContinuous\ErrorRows\azure_gpt5_emp_ZeroShot_L2.jsonl")
    # print(df)
    # error_file = r"G:\Python\APContinuous\ErrorRows" + "\\" +modelname+"_emp_ZeroShot_L3.jsonl"
    # output_file =r"G:\Python\APContinuous\Results" + "\\" +modelname+"_emp_ZeroShot.jsonl"

    with (
        open(output_file, "a", encoding="utf-8") as f,
        open(error_file, "a", encoding="utf-8") as error_f):
        # for index, row in df.iterrows():  # duyệt all dòng trong DataFram
        print(output_file)
        # for index, row in df.iloc[0:2].iterrows():  
        # for index, row in df.iloc[600:len(df)].iterrows():  
        for index, row in df.iloc[0:len(df)].iterrows():  
            print(f"\nProcessing row {index}:\n{output_file}")
            Conversation_ID = df['conversation_id'][index]
            label = df['label'][index]
            situation = df['situation'][index]
            ConversationText = df['conversation'][index]

            try:
                # result = text_infer_emotion_jf_zeroshot( model=model,situation=situation, conversation=ConversationText)
                result = text_infer_emotion_jf_coT( model=model,situation=situation, conversation=ConversationText)
                # print(result)#ok có data
                record = {
                    "conversation_id": Conversation_ID,
                    "true_label": label,
                    "pred_label": result.get("label","") ,
                    "reason": result.get("reason","") ,                 
                    }
                # print(record) #ok có data
                f.write(json.dumps(record, ensure_ascii=False) + "\n")#ghi vào file đang mở một dòng JSON mới
                f.flush() #đẩy dữ liệu từ bộ nhớ đệm xuống file ngay lập tức => tránh mất dữ liệu nếu chương trình bị dừng giữa chừng.
                print(f" label: {label} Saved to file {output_file}")
            except Exception as e: #lỗi tại đây???
                error_record = {
                     "conversation_id": Conversation_ID,
                     "label": label,
                     "situation": situation,
                     "conversation": ConversationText,
                     "error_type": type(e).__name__,
                     "error_reason": str(e)
                 }
                error_f.write( json.dumps(  error_record,  ensure_ascii=False  ) + "\n"  )
 
                error_f.flush()

                print(f"conversation_id ID: {Conversation_ID} gặp exception: {type(e).__name__}: {e}")
                print(f"Đã ghi row lỗi vào file {error_file}")#chỗ này
                continue  # bỏ qua chạy row tiếp
Text2Appraisal()