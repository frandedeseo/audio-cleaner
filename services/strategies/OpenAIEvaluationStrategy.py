import os
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv
from .AudioEvaluationStrategy import AudioEvaluationStrategy

load_dotenv()

class OpenAIEvaluationStrategy(AudioEvaluationStrategy):
    """
    Estrategia usando GPT-Audio-1.5 (multimodal nativo).
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def evaluate(self, text: str, wpm: float, audio_bytes: bytes) -> dict:
        """Evalúa lectura con GPT-Audio-1.5."""
        try:
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            response = self.client.chat.completions.create(
                model="gpt-audio-1.5",
                modalities=["text", "audio"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{self._get_system_instructions()}\n\nTexto: {text}\nWPM: {wpm:.1f}\n\nDevuelve solo JSON."},
                            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
                        ]
                    }
                ],
                temperature=0
            )
            
            response_text = response.choices[0].message.content
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            return json.loads(response_text.strip())
            
        except Exception as e:
            raise Exception(f"Error OpenAI: {str(e)}")
