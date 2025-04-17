# main.py
import os
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

@app.post("/transcribe-and-chat")
async def transcribe_and_chat(file: UploadFile = File(...)):
    # Step 1: Transcribe audio with Whisper (via GPT-4o-audio-preview)
    audio_bytes = await file.read()

    transcription = openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_bytes,
        response_format="text"
    )

    print("Transcription:", transcription)

    # Step 2: Send transcription to GPT-4o for response
    chat_response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": transcription}
        ]
    )

    return {
        "transcription": transcription,
        "response": chat_response.choices[0].message.content
    }
