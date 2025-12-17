from llm_client_unsloth import generate_llm_answer

def build_rephrase_prompt(query: str, retrieved_answer: str):
    return f"""
You are a careful, empathetic mental health assistant.

TASK:
Decide how to respond based on the relationship between the user's question and the retrieved answer.

STEP 1 — RELEVANCE CHECK:
Classify the relevance as:
- Strongly relevant
- Partially relevant
- Not relevant

STEP 2 — RESPONSE STRATEGY:

• Strongly relevant:
  - Rephrase the retrieved answer calmly and empathetically.
  - Do NOT add new facts.

• Partially relevant:
  - Use only relevant parts.
  - Omit unrelated content.

• Not relevant:
  - Ignore the retrieved answer.
  - If safe, provide a general supportive response.
  - If unsafe or requires diagnosis, encourage professional help.

SAFETY RULES:
- Do NOT invent facts or diagnoses.
- If there are signs of self-harm or suicidal ideation, encourage immediate professional help.

STYLE:
- Calm, empathetic, non-judgmental
- Clear and brief

IMPORTANT OUTPUT RULES:
- Output ONLY the final user-facing answer.
- NO analysis, NO notes, NO explanations, NO response strategy.
- NO internal reasoning or safety justification.

User question:
{query}

Retrieved answer:
{retrieved_answer}

Rewrite below:
"""

def rephrase_answer(query: str, retrieved_answer: str, max_new_tokens: int = 256, temperature: float = 0.25, top_p: float = 0.9, system_prompt: str = None, enable_safety_prompt: bool = True):
    # optional system_prompt and safety are handled by caller; keep interface flexible
    prompt = build_rephrase_prompt(query, retrieved_answer)
    return generate_llm_answer(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
