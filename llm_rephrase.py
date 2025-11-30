from llm_client_unsloth import generate_llm_answer

def build_rephrase_prompt(query: str, retrieved_answer: str):
    """
    Prompt that tells the model:
    - Rephrase the original retrieved answer
    - Never invent new facts
    - Keep safety rules
    - Encourage seeking help if distress is severe
    """

    return f"""
You are a professional mental health assistant.

Your job:
- Rewrite the following answer in a clearer, more empathetic, supportive and safe tone. 
- DO NOT invent any new information. Only rephrase and clarify.
- Keep the same meaning as the original answer.
- Your style must be calm, non-judgmental, and encouraging.
- NEVER diagnose the user.
- NEVER dismiss their feelings.

CRITICAL SAFETY RULE:
If the user's emotional state appears extremely severe 
(self-harm signs, extreme distress, hopelessness, inability to cope),
gently but clearly advise them to reach out to a licensed mental health 
professional or crisis hotline immediately.
Do NOT sound alarming. Be supportive and grounding.

User question:
{query}

Original retrieved answer:
{retrieved_answer}

Rewrite the answer below:
"""

def rephrase_answer(query: str, retrieved_answer: str):
    prompt = build_rephrase_prompt(query, retrieved_answer)
    return generate_llm_answer(prompt, max_new_tokens=256, temperature=0.3)
