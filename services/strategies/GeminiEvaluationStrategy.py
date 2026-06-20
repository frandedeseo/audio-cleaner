import os
import json
import base64
import google.generativeai as genai
from dotenv import load_dotenv
from .AudioEvaluationStrategy import AudioEvaluationStrategy

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiEvaluationStrategy(AudioEvaluationStrategy):
    """Estrategia usando Gemini 3.5 Flash (stable, rápido)."""
    
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-3.5-flash")
    
    async def evaluate(self, text: str, wpm: float, audio_bytes: bytes) -> dict:
        try:
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            system_instructions = self._get_system_instructions()
            
            response = self.model.generate_content([
                f"{system_instructions}\n\nTexto: {text}\nWPM: {wpm:.1f}\n\nDevuelve solo JSON.",
                {"mime_type": "audio/wav", "data": audio_b64}
            ])
            
            response_text = response.text
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            return json.loads(response_text.strip())
        except Exception as e:
            raise Exception(f"Error Gemini Flash: {str(e)}")
