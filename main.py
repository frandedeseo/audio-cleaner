import os
import io
import base64
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydub import AudioSegment, silence
from difflib import SequenceMatcher
import tempfile
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


# Función para calcular duración activa y WPM
def getWpm(audio_bytes: bytes,
                             text: str,
                              min_silence_len: int = 2000,
                              silence_thresh: int = -70) -> float:
    
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
    silent_ranges = silence.detect_silence(audio,
                                           min_silence_len=min_silence_len,
                                           silence_thresh=silence_thresh)
    total_silence = sum((end - start) for start, end in silent_ranges)
    duracion_total_ms = len(audio)
    duracion_activa_s = max(0, duracion_total_ms - total_silence) / 1000  # en segundos
    cant_palabras = len(text.split())
    wpm = cant_palabras / (duracion_activa_s / 60) if duracion_activa_s > 0 else 0
    return wpm

def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

async def verificar_coincidencia_audio_texto(audio_bytes: bytes, texto_proporcionado: str, umbral: float = 0.7):
    # Guardar el audio en un archivo temporal con extensión .wav
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Enviar archivo a Whisper para transcripción
    with open(tmp_path, "rb") as f:
        transcription = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    print(transcription)
    transcript_text = transcription.text

    # Calcular similaridad
    similarity = text_similarity(transcript_text, texto_proporcionado)

    # Verificar si está por debajo del umbral
    if similarity < umbral:
        return {
            "match": False,
            "similaridad": similarity,
            "transcripcion": transcript_text
        }

    return {
        "match": True,
        "similaridad": similarity,
        "transcripcion": transcript_text
    }

@app.post("/evaluar-lectura")
async def evaluar_lectura(text: str = Form(...), audio: UploadFile = File(...)):
    # Leer y procesar audio
    audio_bytes = await audio.read()

    # Verificar si el audio corresponde al texto
    verificacion = await verificar_coincidencia_audio_texto(audio_bytes, text)
    if not verificacion["match"]:
        return {
            "error": "El texto proporcionado no coincide con el audio.",
            "similaridad": verificacion["similaridad"],
            "transcripcion_detectada": verificacion["transcripcion"]
        }

    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
    wpm = getWpm(audio_bytes, text)


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

