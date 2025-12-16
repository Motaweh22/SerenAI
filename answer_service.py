from retriever import retriever
from config import SELFHARM_KEYWORDS, ABUSE_KEYWORDS, MAX_DISPLAY_CHARS
from llm_rephrase import rephrase_answer


class AnswerService:
    """
    Orchestrates the RAG pipeline:
    - Retrieve similar therapist Q&A pairs
    - Run basic safety checks
    - Optionally rephrase using an LLM
    """

    def safety_check(self, text: str):
        t = text.lower()
        if any(k in t for k in SELFHARM_KEYWORDS):
            return {"level": "high", "reason": "self harm"}
        if any(k in t for k in ABUSE_KEYWORDS):
            return {"level": "medium", "reason": "abuse"}
        return {"level": "ok", "reason": ""}

    def extract_response(self, pair_text: str):
        """
        Extracts the response part from a stored Q/A pair.
        """
        if "Response:" in pair_text:
            return pair_text.split("Response:", 1)[1].strip()
        return pair_text.strip()

    def clean(self, text: str):
        """
        Normalizes whitespace and limits output length for UI display.
        """
        text = " ".join(text.split())
        if len(text) <= MAX_DISPLAY_CHARS:
            return text
        return text[:MAX_DISPLAY_CHARS] + "..."

    def answer(
        self,
        query: str,
        k: int = 5,
        use_llm: bool = True,
        system_prompt: str = "You are a helpful mental health AI.",
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
        enable_safety_prompt: bool = True,
    ):
        """
        Main pipeline entry point.
        """

        # -------------------------
        # 1) Retrieval
        # -------------------------
        candidates = retriever.retrieve(query, k)

        if not candidates:
            return {"error": "no_results"}

        processed = []
        for c in candidates:
            raw_response = self.extract_response(c["pair_text"])
            safety = self.safety_check(raw_response)
            cleaned = self.clean(raw_response)

            processed.append({
                "score": c["score"],
                "raw_response": raw_response,
                "cleaned_response": cleaned,
                "metadata": c.get("metadata", {}),
                "safety": safety
            })

        best = processed[0]

        # -------------------------
        # 2) LLM Rephrase (optional)
        # -------------------------
        if use_llm:
            llm_answer = rephrase_answer(
                query=query,
                retrieved_answer=best["raw_response"],
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                system_prompt=system_prompt,
                enable_safety_prompt=enable_safety_prompt
            )
        else:
            llm_answer = best["cleaned_response"]

        # -------------------------
        # 3) Return final output
        # -------------------------
        return {
            "query": query,
            "retrieved_answer": best["cleaned_response"],
            "llm_answer": llm_answer,
            "safety": best["safety"],
            "candidates": processed
        }


# Singleton instance
answer_service = AnswerService()
