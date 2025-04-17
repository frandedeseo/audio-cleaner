import parselmouth
import numpy as np
from pathlib import Path
import openai

# Configurar clave de API de OpenAI
openai.api_key = "tu_clave_api"  # <-- Reemplazá con tu clave real

# Cargar archivos
base_dir = Path("praat")
audio_path = base_dir / "lectura.wav"
texto_path = base_dir / "lectura.txt"

# Cargar audio y texto
sound = parselmouth.Sound(str(audio_path))
with open(texto_path, "r", encoding="utf-8") as f:
    texto = f.read().strip()

# Extraer estadísticas de audio
pitch = sound.to_pitch()
intensity = sound.to_intensity()

# Filtrar valores válidos de pitch
pitch_values = pitch.selected_array['frequency']
valid_pitch = pitch_values[~np.isnan(pitch_values)]

# Calcular métricas
frecuencia_promedio = round(np.mean(valid_pitch), 2)
frecuencia_maxima = round(np.max(valid_pitch), 2)
frecuencia_minima = round(np.min(valid_pitch), 2)
intensidad_media = round(intensity.get_average(), 2)
duracion_total = round(sound.get_total_duration(), 2)
cantidad_silencios = sum(1 for t in pitch.xs() if pitch.get_value_at_time(t) is None)

# Armar el prompt para OpenAI
prompt = f"""
Analiza la fluidez de una lectura oral a partir de las siguientes estadísticas:

- Duración total: {duracion_total} segundos
- Frecuencia promedio: {frecuencia_promedio} Hz
- Frecuencia mínima: {frecuencia_minima} Hz
- Frecuencia máxima: {frecuencia_maxima} Hz
- Intensidad promedio: {intensidad_media} dB
- Cantidad de pausas (por ausencia de pitch): {cantidad_silencios}

Texto leído:
\"\"\"{texto}\"\"\"

Usa esta rúbrica para evaluar:
1. Estrategia silábica
2. Manejo del ritmo
3. Control de la respiración
4. Precisión

Para cada criterio, clasifica en: Inicial, En Proceso, Logrado o Avanzado.

Devuelve una respuesta clara con una pequeña explicación por criterio.
"""

# Enviar a OpenAI
response = openai.ChatCompletion.create(
    model="gpt-4",  # podés usar "gpt-3.5-turbo" si querés algo más rápido/barato
    messages=[
        {"role": "system", "content": "Sos un evaluador de lectura oral."},
        {"role": "user", "content": prompt}
    ]
)

# Mostrar la respuesta
print(response["choices"][0]["message"]["content"])
