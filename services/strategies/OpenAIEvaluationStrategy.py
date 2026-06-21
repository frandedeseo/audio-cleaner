import os
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv
from .AudioEvaluationStrategy import AudioEvaluationStrategy

load_dotenv()

EVALUAR_TOOL = {
    "type": "function",
    "function": {
        "name": "evaluar_lectura",
        "description": "Devuelve evaluación según rúbrica de lectura infantil",
        "parameters": {
            "type": "object",
            "properties": {
                k: {
                    "type": "object",
                    "properties": {"nivel": {"type": "string"}, "comentario": {"type": "string"}},
                    "required": ["nivel", "comentario"],
                }
                for k in [
                    "estrategia_silabica", "manejo_ritmo",
                    "manejo_respiracion", "precision", "fluidez_lectora",
                ]
            },
            "required": [
                "estrategia_silabica", "manejo_ritmo",
                "manejo_respiracion", "precision", "fluidez_lectora",
            ],
        },
    },
}


class OpenAIEvaluationStrategy(AudioEvaluationStrategy):
    """Estrategia usando gpt-audio-1.5 (audio in, texto out)."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def evaluate(self, text: str, wpm: float, audio_bytes: bytes) -> dict:
        try:
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            response = self.client.chat.completions.create(
                model="gpt-audio-1.5",
                # OJO: sin "audio" en modalities -> no generamos audio de salida,
                # solo necesitamos texto/JSON.
                modalities=["text"],
                messages=[
                    {"role": "system", "content": self._get_system_instructions()},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Texto a leer: {text}\nWPM: {wpm:.1f}"},
                            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                        ],
                    },
                ],
                tools=[EVALUAR_TOOL],
                tool_choice={"type": "function", "function": {"name": "evaluar_lectura"}},
                temperature=0,
            )

            tool_call = response.choices[0].message.tool_calls[0]
            return json.loads(tool_call.function.arguments)

        except Exception as e:
            raise Exception(f"Error OpenAI: {str(e)}")