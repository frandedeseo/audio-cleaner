import os
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types
from .AudioEvaluationStrategy import AudioEvaluationStrategy

load_dotenv()


class Criterio(BaseModel):
    nivel: str
    comentario: str


class EvaluacionLectura(BaseModel):
    estrategia_silabica: Criterio
    manejo_ritmo: Criterio
    manejo_respiracion: Criterio
    precision: Criterio
    fluidez_lectora: Criterio


class GeminiEvaluationStrategy(AudioEvaluationStrategy):
    """Estrategia usando Gemini 3.1 Preview (stable, rápido)."""

    MODEL = "gemini-3.1-preview"

    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    async def evaluate(self, text: str, wpm: float, audio_bytes: bytes) -> dict:
        try:
            response = self.client.models.generate_content(
                model=self.MODEL,
                contents=[
                    f"{self._get_system_instructions()}\n\nTexto: {text}\nWPM: {wpm:.1f}",
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EvaluacionLectura,
                ),
            )
            return response.parsed.model_dump() if response.parsed else __import__("json").loads(response.text)
        except Exception as e:
            raise Exception(f"Error Gemini Flash: {str(e)}")