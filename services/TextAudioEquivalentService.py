import re
import io
import tempfile
import os
import openai
from fastapi import UploadFile
from difflib import SequenceMatcher

class TextAudioEquivalentService:
    def __init__(self, threshold: float = 0.45):
        self.threshold = threshold

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\sáéíóúüñ]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    async def verify(self, audio_bytes: bytes, provided_text: str):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            with open(tmp_path, "rb") as f:
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            transcript = transcription.text
            cleaned_transcript = self.clean_text(transcript)
            cleaned_provided = self.clean_text(provided_text)
            sim = self.similarity(cleaned_transcript, cleaned_provided)
            match = sim >= self.threshold
            return {"match": match, "similaridad": sim, "transcripcion": transcript}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass