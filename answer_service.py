from retriever import retriever
from config import SELFHARM_KEYWORDS, ABUSE_KEYWORDS, MAX_DISPLAY_CHARS
from llm_rephrase import rephrase_answer

class AnswerService:

    def safety_check(self, text):
        t = text.lower()
        if any(k in t for k in SELFHARM_KEYWORDS):
            return {"level": "high", "reason": "self harm"}
        if any(k in t for k in ABUSE_KEYWORDS):
            return {"level": "medium", "reason": "abuse"}
        return {"level": "ok", "reason": ""}

    def extract_response(self, pair_text):
        if "Response:" in pair_text:
            return pair_text.split("Response:",1)[1].strip()
        return pair_text.strip()

    def clean(self, text):
        text = " ".join(text.split())
        return text if len(text) <= MAX_DISPLAY_CHARS else text[:MAX_DISPLAY_CHARS] + "..."

    def answer(
        self, 
        query, 
        k=5, 
        use_llm=True, 
        system_prompt="You are a helpful mental health AI.",
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=256,
        enable_safety_prompt=True,
    ):

        candidates = retriever.retrieve(query, k)

        processed = []
        for c in candidates:
            resp_raw = self.extract_response(c["pair_text"])
            safe = self.safety_check(resp_raw)
            cleaned = self.clean(resp_raw)
            processed.append({
                "score": c["score"],
                "raw_response": resp_raw,
                "cleaned_response": cleaned,
                "metadata": c["metadata"],
                "safety": safe
            })

        best = processed[0]
        
        if use_llm:
            llm_ans = rephrase_answer(
                query=query,
                retrieved_answer=best["raw_response"],
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                enable_safety_prompt=enable_safety_prompt
            )
        else:
            llm_ans = best["cleaned_response"]

        return {
            "query": query,
            "retrieved_answer": best["cleaned_response"],
            "llm_answer": llm_ans,
            "safety": best["safety"],
            "candidates": processed
        }

answer_service = AnswerService()
