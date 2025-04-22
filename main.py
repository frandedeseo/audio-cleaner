import os
import io
import base64
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydub import AudioSegment, silence
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
# Permitir CORS (opcional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrucciones base
INSTRUCCIONES_SYSTEM = """
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
• Inicial: 0-50 palabras por minuto.
• En proceso: 50-90 palabras por minuto.
• Logrado: 90-130 palabras por minuto.
• Avanzado: 130-200 palabras por minuto.

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

# Función para calcular duración activa y WPM

def calcular_duracion_activa(audio_bytes: bytes,
                              min_silence_len: int = 2000,
                              silence_thresh: int = -70) -> float:
    
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
    silent_ranges = silence.detect_silence(audio,
                                           min_silence_len=min_silence_len,
                                           silence_thresh=silence_thresh)
    total_silence = sum((end - start) for start, end in silent_ranges)
    duracion_total_ms = len(audio)
    print(total_silence)
    print(f"Duración total: {duracion_total_ms / 1000:.2f} segundos")
    duracion_activa_ms = max(0, duracion_total_ms - total_silence)
    return duracion_activa_ms / 1000  # en segundos

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

@app.post("/evaluar-lectura")
async def evaluar_lectura(text: str = Form(...), audio: UploadFile = File(...)):
    # Leer y procesar audio
    audio_bytes = await audio.read()
    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
    duracion_activa = calcular_duracion_activa(audio_bytes)
    print(f"Duración activa: {duracion_activa:.2f} segundos")
    cant_palabras = len(text.split())
    print(cant_palabras)
    wpm = cant_palabras / (duracion_activa / 60) if duracion_activa > 0 else 0
    print(f"WPM: {wpm:.1f}")

    # Preparar llamada
    messages = [
        {"role": "system", "content": INSTRUCCIONES_SYSTEM},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Texto a leer: {text}\nWPM: {wpm:.1f}"
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64_audio,
                        "format": "wav"
                    }
                }
            ]
        },
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
    args = response.choices[0].message.function_call.arguments
    return json.loads(args)

