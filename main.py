import os
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
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

# 📌 Rúbrica embebida en el backend
RUBRICA_EVALUACION = """
Hola chat, quiero que te situes en la posición de una psicopedagoga. Tenes que evaluar al lector del audio en 5 tópicos distintos y en cada uno clasificarlo en 4 niveles distintos. Ahora te escribo la información para que puedas ejecutar tu tarea de manera correcta:

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

Retorna los datos solo con un json de esta forma:
{
  "estrategia_silabica": {
    "nivel": "...",
    "comentario": "..."
  },
  "manejo_ritmo": {
    "nivel": "...",
    "comentario": "..."
  },
  "manejo_respiracion": {
    "nivel": "...",
    "comentario": "..."
  },
  "precision": {
    "nivel": "...",
    "comentario": "..."
  },
  "fluidez_lectora": {
    "nivel": "...",
    "comentario": "..."
  }
}
Los espacios donde hay ... son para que el modelo complete con la evaluación.

"""

@app.post("/evaluar-lectura")
async def evaluar_lectura(
    text: str = Form(...),
    audio: UploadFile = File(...)
):
    # Leer los bytes del archivo
    audio_bytes = await audio.read()

    # Convertir audio a Base64
    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

    # Construir mensajes para GPT-4o Audio Preview
    messages = [
        {
            "role": "system",
            "content": "Sos una psicopedagoga experta en evaluación lectora. Vas a recibir un texto, un audio y una rúbrica para emitir una evaluación precisa en base a los criterios."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{RUBRICA_EVALUACION}\n\nTexto que el estudiante debía leer:\n{text}\n"
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64_audio,
                        "format": "wav"
                    }
                }
            ]
        }
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=messages,
    )

    return {
        "evaluacion": response.choices[0].message.audio.transcript
    }


