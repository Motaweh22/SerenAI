from llm_client_unsloth import generate_answer

def build_rephrase_prompt(query, retrieved_answer, system_prompt, enable_safety_prompt):
    safety_block = ""
    if enable_safety_prompt:
        safety_block = """
CRITICAL SAFETY RULE:
If the user's message shows signs of severe emotional distress, self-harm, hopelessness,
or inability to cope, gently but clearly advise them to seek immediate help from a
licensed mental health professional or a crisis hotline. 
"""

    return f"""
{system_prompt}

{safe_block if enable_safety_prompt else ""}

User question:
{query}

Original retrieved answer:
{retrieved_answer}

Rewrite below:
"""

def rephrase_answer(
    query, 
    retrieved_answer, 
    system_prompt,
    temperature=0.2, 
    top_p=0.9, 
    max_new_tokens=256,
    enable_safety_prompt=True
):
    prompt = build_rephrase_prompt(
        query=query, 
        retrieved_answer=retrieved_answer, 
        system_prompt=system_prompt,
        enable_safety_prompt=enable_safety_prompt
    )
    
    return generate_answer(
        prompt, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_p=top_p
    )
