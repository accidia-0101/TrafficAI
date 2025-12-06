# import textwrap
# import requests
# from datetime import datetime
#
# OPENROUTER_API_KEY = "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"
# OPENROUTER_MODEL = "anthropic/claude-4.5-sonnet"
# OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
#
#
# def set_openrouter_config(api_key: str, model: str = "anthropic/claude-3.5-sonnet"):
#     global OPENROUTER_API_KEY, OPENROUTER_MODEL
#     OPENROUTER_API_KEY = api_key
#     OPENROUTER_MODEL = model
#
#
# def build_system_prompt(query_type: str) -> str:
#     now = datetime.now().strftime("%Y-%m-%d %H:%M")
#
#     base_prompt = f"""You are a traffic incident analysis assistant. Current date/time: {now}.
# Answer based ONLY on the provided evidence. Be concise and precise."""
#
#     type_specific = {
#         "factual": """
# For factual queries:
# - State the answer directly with specific details (timestamps, camera IDs, counts)
# - If no relevant data exists, say "No relevant incident found."
# - Include exact numbers when available""",
#
#         "filtered": """
# For filtered queries:
# - List matching events with timestamps and details
# - Group by relevant criteria if helpful
# - State the total count of matching events
# - If no events match the filter, clearly state that""",
#
#         "aggregation": """
# For aggregation queries:
# - Provide the exact count or statistic requested
# - Reference specific data points that support the answer
# - If asking about "most recent" or "earliest", include the timestamp
# - For "which camera" questions, compare counts directly""",
#
#         "comparison": """
# For comparison queries:
# - State both values being compared with exact numbers
# - Calculate and state the difference explicitly
# - Use phrases like "X more than Y" or "increased/decreased by Z"
# - Include percentages if meaningful"""
#     }
#
#     return base_prompt + type_specific.get(query_type, type_specific["factual"])
#
#
# def generate_local_answer(question: str, context: str) -> str:
#     from rag.rag_service import detect_query_type, build_context_for_llm
#
#     query_type = detect_query_type(question)
#
#     if isinstance(context, str) and "=== Statistics ===" not in context:
#         context = build_context_for_llm(question)
#
#     system_prompt = build_system_prompt(query_type)
#
#     user_content = textwrap.dedent(f"""
#     Evidence:
#     {context}
#
#     Question: {question}
#
#     Answer:""").strip()
#
#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json",
#     }
#
#     payload = {
#         "model": OPENROUTER_MODEL,
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_content},
#         ],
#         "max_tokens": 500,
#         "temperature": 0.5,
#     }
#
#     response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload)
#     response.raise_for_status()
#
#     result = response.json()
#     return result["choices"][0]["message"]["content"].strip()


import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

_MODEL = None
_TOKENIZER = None


def load_local_llm(model_dir: str = r"Qwen/Qwen3-4B"):
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        _MODEL.eval()
    return _TOKENIZER, _MODEL


def build_system_prompt(query_type: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    base_prompt = f"""You are a traffic incident analysis assistant. Current date/time: {now}.
Answer based ONLY on the provided evidence. Be concise and precise."""

    type_specific = {
        "factual": """
For factual queries:
- State the answer directly with specific details (timestamps, camera IDs, counts)
- If no relevant data exists, say "No relevant incident found."
- Include exact numbers when available""",

        "filtered": """
For filtered queries:
- List matching events with timestamps and details
- Group by relevant criteria if helpful
- State the total count of matching events
- If no events match the filter, clearly state that""",

        "aggregation": """
For aggregation queries:
- Provide the exact count or statistic requested
- Reference specific data points that support the answer
- If asking about "most recent" or "earliest", include the timestamp
- For "which camera" questions, compare counts directly""",

        "comparison": """
For comparison queries:
- State both values being compared with exact numbers
- Calculate and state the difference explicitly
- Use phrases like "X more than Y" or "increased/decreased by Z"
- Include percentages if meaningful"""
    }

    return base_prompt + type_specific.get(query_type, type_specific["factual"])


def generate_local_answer(question: str, context: str) -> str:
    from rag.rag_service import detect_query_type, build_context_for_llm

    tok, model = load_local_llm()
    query_type = detect_query_type(question)

    if isinstance(context, str) and "=== Statistics ===" not in context:
        context = build_context_for_llm(question)

    system_prompt = build_system_prompt(query_type)

    user_content = textwrap.dedent(f"""
    Evidence:
    {context}

    Question: {question}

    Answer:""").strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tok(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.5,
            top_p=0.85,
            do_sample=True,
        )

    gen_ids = outputs[0][inputs.input_ids.shape[1]:]
    result = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return result