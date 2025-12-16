import asyncio
import edge_tts

VOICE = "en-US-JennyNeural"  # صوت هادي وداعم

async def _generate_speech(text: str, out_path: str):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(out_path)

def text_to_speech(text: str, out_path: str = "response.mp3") -> str:
    """
    Converts text to speech using Edge TTS (lightweight).
    """
    if not text.strip():
        return None

    asyncio.run(_generate_speech(text, out_path))
    return out_path
