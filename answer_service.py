from retriever import retriever
from config import (
    SELFHARM_KEYWORDS, ABUSE_KEYWORDS, MAX_DISPLAY_CHARS
)
from llm_rephrase import rephrase_answer


class AnswerService:
    # -------- Safety --------
    def safety_check(self, text):
        t = text.lower()
        if any(k in t for k in SELFHARM_KEYWORDS):
            return {"level": "high", "reason": "self harm detected"}
        if any(k in t for k in ABUSE_KEYWORDS):
            return {"level": "medium", "reason": "abuse detected"}
        return {"level": "ok", "reason": ""}

    # -------- Extract Response --------
    def extract_response(self, pair_text):
        if "Response:" in pair_text:
            return pair_text.split("Response:",1)[1].strip()
        return pair_text.strip()

    # -------- Clean --------
    def clean(self, text):
        s = " ".join(text.split())
        return s if len(s) <= MAX_DISPLAY_CHARS else s[:MAX_DISPLAY_CHARS] + "..."

    # -------- Pipeline --------
    def answer(self, query, k=5, use_llm=True):
        # 1) retrieve top-k pair chunks
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

        # 2) best candidate
        best = processed[0] if processed else None
        if not best:
            return {"error": "no_response_available"}

        # 3) LLM rephrase layer
        if use_llm:
            rephrased = rephrase_answer(query, best["raw_response"])
        else:
            rephrased = best["cleaned_response"]

        # 4) Final output
        return {
            "query": query,
            "retrieved_answer": best["cleaned_response"],
            "llm_answer": rephrased,
            "safety": best["safety"],
            "candidates": processed
        }


answer_service = AnswerService()
