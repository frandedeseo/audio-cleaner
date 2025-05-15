import os
import json
import base64
import openai
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Instrucciones base
SYSTEM_INSTRUCTIONS = """
Sos una psicopedagoga experta en evaluación lectora infantil. Vas a recibir dos elementos: 
un texto que el estudiante debía leer y las métricas de lectura.  
Tu tarea es analizar la lectura y evaluar el desempeño del estudiante en base a los siguientes 5 criterios, usando esta rúbrica:

Rúbrica de lectura (por niveles de desempeño):

1. Estrategia silábica
• Inicial: Aún no logra asociar los sonidos en una sílaba.
• En proceso: Puede asociar consonantes, continuar con vocales y empezar a leer sílabas.
• Logrado: Utiliza la estrategia silábica para la lectura.
• Avanzado: No necesita utilizar la estrategia silábica.

2. Manejo del ritmo
• Inicial: Lee en forma silabeante y sin entonar.
• En proceso: Lee en forma monótona pero sin silabear las palabras.
• Logrado: Entona para abajo en los puntos y comas.
• Avanzado: Lee el texto con un adecuado cambio de entonación.

3. Manejo de la respiración
• Inicial: No hace pausas en puntos y comas.
• En proceso: Realiza pausas entre palabra y palabra.
• Logrado: Hace pausas en puntos y comas.
• Avanzado: Hace pausas en puntos, comas y antes de palabras cortas, cuando las oraciones son muy largas.

4. Precisión
• Inicial: Cambia letras por otras: sustituye, omite o añade letras.
• En proceso: Cambia palabras por otras o adivina en forma incorrecta.
• Logrado: Logra leer oraciones cometiendo uno o dos errores aislados, no fonológicos.
• Avanzado: Logra leer párrafos o textos sin errores.

5. Fluidez Lectora (se mide en palabras por minuto)
• Inicial: 0-49 palabras por minuto = Etapa fonológica.
• En proceso: 50-70 palabras por minuto = Etapa ortográfica.
• Logrado: 71-90 palabras por minuto = Etapa de Transición hacia Expresiva.
• Avanzado: 90-200 palabras por minuto = Etapa Expresiva Consolidada.

Ejemplo de salida válida:
{
  "estrategia_silabica": {"nivel": "Logrado", "comentario": "El alumno..."},
  "manejo_ritmo": {"nivel": "En proceso", "comentario": "Lee de forma monótona..."},
  "manejo_respiracion": {"nivel": "Inicial", "comentario": "No hace pausas en puntos..."},
  "precision": {"nivel": "Avanzado", "comentario": "Lee sin errores..."},
  "fluidez_lectora": {"nivel": "En proceso", "comentario": "80 palabras en 1m20s => 60 WPM"}
}

IMPORTANTE: Devuelve **solo** este objeto JSON, sin texto libre, sin claves extra, sin comillas alrededor del json.
"""

# Esquema de función para forcing JSON
EVALUAR_FUNC = {
    "name": "evaluar_lectura",
    "description": "Devuelve evaluación según rúbrica de lectura infantil",
    "parameters": {
        "type": "object",
        "properties": {
            "estrategia_silabica": {"type": "object",
                "properties": {"nivel": {"type": "string"}, "comentario": {"type": "string"}},
                "required": ["nivel","comentario"]
            },
            "manejo_ritmo": {"type": "object",
                "properties": {"nivel": {"type": "string"}, "comentario": {"type": "string"}},
                "required": ["nivel","comentario"]
            },
            "manejo_respiracion": {"type": "object",
                "properties": {"nivel": {"type": "string"}, "comentario": {"type": "string"}},
                "required": ["nivel","comentario"]
            },
            "precision": {"type": "object",
                "properties": {"nivel": {"type": "string"}, "comentario": {"type": "string"}},
                "required": ["nivel","comentario"]
            },
            "fluidez_lectora": {"type": "object",
                "properties": {"nivel": {"type": "string"}, "comentario": {"type": "string"}},
                "required": ["nivel","comentario"]
            }
        },
        "required": ["estrategia_silabica","manejo_ritmo","manejo_respiracion","precision","fluidez_lectora"]
    }
}


class GptService:
    async def evaluate(self, text: str, wpm: float, audio_bytes: bytes) -> dict:
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": [
                {"type": "text", "text": f"Texto a leer: {text}\nWPM: {wpm:.1f}"},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
            ]}
        ]
        response = openai.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=messages,
            functions=[EVALUAR_FUNC],
            function_call={"name": "evaluar_lectura"},
            temperature=0
        )
        args = json.loads(response.choices[0].message.function_call.arguments)
        return args