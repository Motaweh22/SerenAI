# app.py
import streamlit as st
import json
from datetime import datetime
import os

# -------------------------
# Try importing services
# -------------------------
try:
    from answer_service import answer_service
    from voice.stt import speech_to_text      # Whisper tiny
    from voice.tts import text_to_speech      # TTS
except Exception:
    st.error("فشل استيراد answer_service أو voice modules.")
    st.stop()

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Mental Health Assistant", page_icon="🧠", layout="wide")

st.title("🧠 Mental Health Assistant — RAG + LLM + Voice")
st.caption("Retrieval (pair-embeddings) + LLM rephrase (Unsloth) + Safety checks + Voice")

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Retrieval Settings")
    k = st.slider("Top-K retrieval", 1, 20, 5)
    use_llm = st.checkbox("Use LLM Rephrase", value=True)
    show_context = st.checkbox("Show Retrieved Context", value=False)
    show_llm_raw = st.checkbox("Show raw LLM output", value=False)

    st.markdown("---")

    st.header("LLM Settings")

    # Model selector (UI only)
    selected_model = st.selectbox(
        "Choose LLM Model",
        [
            "Llama-3.2-3B-Instruct (Active)",
            "Gemma-3-4B (Demo)",
            "Mistral-7B (Demo)"
        ],
        index=0
    )

    st.caption(
        "ℹ️ Model switching is UI-only. "
        "Backend currently runs a single preloaded model."
    )

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

    st.header("Voice Settings")
    use_voice_input = st.checkbox("🎙️ Voice Input", value=False)
    use_voice_output = st.checkbox("🔊 Voice Output", value=True)

    st.markdown("---")
    st.caption("Tip: استخدم الإعدادات بعناية عند اختبار الأداء أو التكلفة.")

# -------------------------
# Main Input Section
# -------------------------
st.subheader("Write or Record your message")

query = ""

if use_voice_input:
    st.caption("You can record directly from your microphone or upload an audio file.")

    # -------- Record from microphone --------
    audio_mic = st.audio_input("🎙️ Record from microphone")

    # -------- Upload audio file --------
    audio_upload = st.file_uploader(
        "📂 Or upload an audio file",
        type=["wav", "mp3", "m4a"]
    )

    audio_path = None

    if audio_mic:
        audio_path = "input_audio_mic.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_mic.read())

    elif audio_upload:
        audio_path = "input_audio_upload.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_upload.read())

    if audio_path:
        with st.spinner("Transcribing audio..."):
            query = speech_to_text(audio_path)

        st.markdown("### 📝 Transcribed Text")
        st.info(query if query else "No speech detected.")

else:
    query = st.text_area("Your message:", height=160)

col1, col2 = st.columns([1, 1])
with col1:
    run_button = st.button("Get Answer")
with col2:
    clear_history = st.button("Clear History")

# -------------------------
# History handling
# -------------------------
if clear_history:
    st.session_state.pop("history", None)
    st.success("Conversation history cleared.")

if "history" not in st.session_state:
    st.session_state["history"] = []

# -------------------------
# Run pipeline
# -------------------------
if run_button:
    if not query.strip():
        st.warning("Please enter or record a message first.")
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
            st.subheader("💡 Assistant Response")
            llm_text = out.get("llm_answer") or out.get("retrieved_answer") or "No answer returned."
            st.markdown(llm_text)

            # -------- Voice Output --------
            if use_voice_output:
                with st.spinner("Generating voice response..."):
                    audio_out = text_to_speech(llm_text)

                if audio_out and os.path.exists(audio_out):
                    st.audio(audio_out)

            # -------- Safety --------
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

            # -------- Retrieved Context --------
            if show_context:
                st.subheader("📚 Retrieved Context (Top-K)")
                for idx, c in enumerate(out.get("candidates", []), start=1):
                    st.markdown(f"**Rank {idx} — Score {c.get('score', 0):.4f}**")
                    st.code(c.get("raw_response", ""), language="text")
                    st.caption(f"Metadata: {json.dumps(c.get('metadata', {}), ensure_ascii=False)}")
                    st.markdown("---")

            if show_llm_raw:
                st.subheader("🔍 Raw LLM Output")
                st.code(llm_text, language="text")

            # -------- Save history --------
            st.session_state["history"].append({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "query": query,
                "final_answer": llm_text,
                "safety": safety,
            })

# -------------------------
# Conversation History
# -------------------------
st.markdown("---")
st.subheader("📜 Conversation History")

if st.session_state["history"]:
    for i, item in enumerate(reversed(st.session_state["history"]), start=1):
        st.markdown(f"**{i}. You:** {item['query']}")
        st.markdown(f"**Assistant:** {item['final_answer']}")
        st.caption(f"Safety: {item['safety'].get('level')} • Time: {item['timestamp']}")
        st.markdown("---")

    st.download_button(
        "Download Full History as JSON",
        data=json.dumps(st.session_state["history"], ensure_ascii=False, indent=2),
        file_name="history.json",
        mime="application/json",
    )
else:
    st.info("No history yet. Ask something to get started.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption(
    "⚠️ This tool is not a medical diagnostic system. "
    "If you are in immediate danger, please seek professional help."
)
