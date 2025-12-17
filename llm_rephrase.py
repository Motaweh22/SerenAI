from llm_client_unsloth import generate_llm_answer

def build_rephrase_prompt(query: str, retrieved_answer: str):
    return f"""
You are a careful, empathetic mental health assistant.

INSTRUCTIONS:

- First, analyze the user's question and the retrieved answer.
- If the retrieved answer is relevant to the user's question (even partially), rephrase it in a calm, supportive, and non-judgmental tone.
- If the retrieved answer is weakly related, try to rephrase only the parts that are relevant and omit unrelated content.
- If the retrieved answer is completely unrelated to the user's question:
- Do NOT force relevance.
- If the user's question is safe, non-harmful, and answerable, provide a general, neutral, and supportive response based on common mental health knowledge.
- If the question requires professional expertise or cannot be answered safely, clearly state that and encourage seeking professional help.
- DO NOT invent specific facts, statistics, or claims that are not grounded in the retrieved answer or general well-known mental health guidance.
- If the user's content contains signs of severe distress, self-harm, or suicidal ideation, gently and clearly encourage them to seek immediate help from a licensed mental health professional or a local crisis hotline.
- Keep the response clear, factual, empathetic, and as brief as possible, without being dismissive.

User question:
{query}

Original retrieved answer:
{retrieved_answer}

Rewrite below:
"""

def rephrase_answer(query: str, retrieved_answer: str, max_new_tokens: int = 256, temperature: float = 0.25, top_p: float = 0.9, system_prompt: str = None, enable_safety_prompt: bool = True):
    # optional system_prompt and safety are handled by caller; keep interface flexible
    prompt = build_rephrase_prompt(query, retrieved_answer)
    return generate_llm_answer(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
