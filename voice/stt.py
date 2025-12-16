import whisper

_model = whisper.load_model("tiny")

def speech_to_text(audio_path: str) -> str:
    result = _model.transcribe(audio_path)
    return result.get("text", "").strip()
