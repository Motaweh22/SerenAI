# app.py
import streamlit as st
import json
from datetime import datetime

# حاول تستورد الـ service؛ لو حصل خطأ اعرض رسالة واضحة
try:
    from answer_service import answer_service
except Exception as e:
    st.error("فشل استيراد answer_service. تأكد إن الملفات loader/retriever/answer_service موجودة ومُحمّلة.")
    st.stop()

st.set_page_config(page_title="Mental Health Assistant", page_icon="🧠", layout="wide")

st.title("🧠 Mental Health Assistant — RAG + LLM")
st.caption("Retrieval (pair-embeddings) + LLM rephrase (Unsloth) + Safety checks")

# -------------------------
# Sidebar: Retrieval + LLM settings (old + new)
# -------------------------
with st.sidebar:
    st.header("Retrieval Settings")
    k = st.slider("Top-K retrieval", 1, 20, 5)
    use_llm = st.checkbox("Use LLM Rephrase", value=True)
    show_context = st.checkbox("Show Retrieved Context", value=False)
    show_llm_raw = st.checkbox("Show raw LLM output", value=False)

    st.markdown("---")

    st.header("LLM Settings")
    system_prompt = st.text_area(
        "System Prompt",
        value=(
            "You are a helpful, calm, supportive mental health assistant. "
            "You respond in a comforting tone and avoid giving medical or diagnostic claims."
        ),
        height=120
    )

    temperature = st.slider("Temperature", 0.0, 2.0, 0.4, step=0.05)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.9, step=0.01)
    max_new_tokens = st.slider("Max New Tokens", 16, 2048, 256, step=16)
    enable_safety_prompt = st.checkbox("Enable Safety Block", value=True)

    st.markdown("---")
    st.caption("Tip: استخدم الإعدادات بعناية عند اختبار الأداء أو التكلفة.")

# -------------------------
# Main UI: input + run
# -------------------------
st.subheader("Write your message")
query = st.text_area("Your message:", height=160)

col1, col2 = st.columns([1, 1])
with col1:
    run_button = st.button("Get Answer")
with col2:
    clear_history = st.button("Clear History")

if clear_history:
    st.session_state.pop("history", None)
    st.success("Conversation history cleared.")

# initialize history
if "history" not in st.session_state:
    st.session_state["history"] = []

# -------------------------
# Run the pipeline when asked
# -------------------------
if run_button:
    if not query.strip():
        st.warning("Please enter a message first.")
    else:
        try:
            with st.spinner("Processing — retrieving and (optionally) rephrasing..."):
                out = answer_service.answer(
                    query=query,
                    k=k,
                    use_llm=use_llm,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    enable_safety_prompt=enable_safety_prompt,
                )
        except Exception as e:
            st.exception(f"Error running pipeline: {e}")
            out = None

        if out:
            # Final LLM answer (or cleaned retrieved if LLM disabled)
            st.subheader("💡 Assistant Response")
            llm_text = out.get("llm_answer") or out.get("retrieved_answer") or "No answer returned."
            st.markdown(llm_text)

            # Safety block
            st.subheader("🛡 Safety Check")
            safety = out.get("safety", {"level": "ok", "reason": ""})
            level = safety.get("level", "ok")
            reason = safety.get("reason", "")

            if level == "ok":
                st.success("No risk detected.")
            elif level == "medium":
                st.warning(f"Emotional distress detected. {reason}")
            elif level == "high":
                st.error("⚠ Severe risk detected. Encourage immediate professional help. " + (reason or ""))
            else:
                st.info(f"Status: {level}. {reason}")

            # Show retrieved context if requested
            if show_context:
                st.subheader("📚 Retrieved Context (Top-K)")
                for idx, c in enumerate(out.get("candidates", []), start=1):
                    st.markdown(f"**Rank {idx} — Score {c.get('score', 0):.4f}**")
                    # show retrieved raw_response & metadata
                    raw = c.get("raw_response") or ""
                    md = c.get("metadata") or {}
                    st.code(raw, language="text")
                    st.caption(f"Metadata: {json.dumps(md, ensure_ascii=False)}")
                    st.markdown("---")

            # show raw llm output if requested
            if show_llm_raw:
                st.subheader("🔍 Raw LLM Output")
                st.code(llm_text, language="text")

            # Add to session history
            st.session_state["history"].append({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "query": query,
                "final_answer": llm_text,
                "retrieved_best": out.get("retrieved_answer"),
                "safety": safety,
            })

            # Provide download of single result
            st.download_button(
                "Download this result (JSON)",
                data=json.dumps(out, ensure_ascii=False, indent=2),
                file_name="result.json",
                mime="application/json",
            )

# -------------------------
# Conversation History display + export all
# -------------------------
st.markdown("---")
st.subheader("📜 Conversation History")
if st.session_state["history"]:
    for i, item in enumerate(reversed(st.session_state["history"]), start=1):
        st.markdown(f"**{i}. You:** {item['query']}")
        st.markdown(f"**Assistant:** {item['final_answer']}")
        st.caption(f"Safety: {item['safety'].get('level')}  •  Time: {item['timestamp']}")
        st.markdown("---")

    # download full history
    if st.button("Download Full History as JSON"):
        st.download_button(
            "Click to download history",
            data=json.dumps(st.session_state["history"], ensure_ascii=False, indent=2),
            file_name="history.json",
            mime="application/json",
        )
else:
    st.info("No history yet. Ask something to get started.")

# -------------------------
# Footer / health-check
# -------------------------
st.markdown("---")
st.caption("If you face latency issues, consider switching device (GPU) or lowering Max New Tokens / Temperature.")
