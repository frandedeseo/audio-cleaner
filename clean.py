import parselmouth
import numpy as np
from pathlib import Path
import re

# Cargar archivos
base_dir = Path("praat")
audio_path = base_dir / "20736.wav"
texto_path = base_dir / "buho.txt"

sound = parselmouth.Sound(str(audio_path))
pitch = sound.to_pitch()
intensity = sound.to_intensity()

with open(texto_path, "r", encoding="utf-8") as f:
    texto = f.read().strip()

# --- Estadísticas generales ---
pitch_values = pitch.selected_array['frequency']
valid_pitch = pitch_values[~np.isnan(pitch_values)]

frecuencia_promedio = round(np.mean(valid_pitch), 2)
frecuencia_maxima = round(np.max(valid_pitch), 2)
frecuencia_minima = round(np.min(valid_pitch), 2)
intensidad_media = round(intensity.get_average(), 2)
duracion_total = round(sound.get_total_duration(), 2)

# --- Detectar pausas (frecuencia y/o intensidad) ---
cantidad_silencios = 0
pausas_largas = []
threshold_pitch = 50  # Umbral para detectar pausas basado en frecuencia (Hz)
threshold_intensidad = 20  # Umbral para detectar pausas basado en intensidad (dB)
threshold_segundos = 0.4  # Pausa mínima de 400ms
# Ajusta estos valores según sea necesario
threshold_pitch = 140  # Umbral para detectar pausas basado en frecuencia (Hz)
threshold_intensidad = 50  # Umbral para detectar pausas basado en intensidad (dB)
threshold_segundos = 0.05  # Pausa mínima de 400ms

t_start = None
time_stamps = pitch.xs()

for t in time_stamps:
    pitch_value = pitch.get_value_at_time(t)
    
    # Obtener valor de intensidad en el tiempo t
    intensity_value = intensity.get_value_at_xy(t, 0)  # Pasamos tiempo 't' y '0' como argumento para obtener la intensidad

    # Detectar pausas: pitch y/o intensidad baja
    if pitch_value is None or pitch_value < threshold_pitch or intensity_value < threshold_intensidad:
        if t_start is None:
            t_start = t
    else:
        if t_start is not None:
            dur = t - t_start
            if dur >= threshold_segundos:
                pausas_largas.append((round(t_start, 2), round(t, 2), round(dur, 2)))
            cantidad_silencios += 1
            t_start = None

# --- Puntuación en el texto ---
puntuaciones = [(m.start(), m.group()) for m in re.finditer(r'[.,;:!?]', texto)]

# --- Armado del prompt para OpenAI ---
prompt = f"""
Analiza la fluidez de una lectura oral a partir de las siguientes estadísticas:

- Duración total: {duracion_total} segundos
- Frecuencia promedio: {frecuencia_promedio} Hz
- Frecuencia mínima: {frecuencia_minima} Hz
- Frecuencia máxima: {frecuencia_maxima} Hz
- Intensidad promedio: {intensidad_media} dB
- Cantidad total de pausas (por ausencia de pitch o baja intensidad): {cantidad_silencios}
- Pausas largas (> {threshold_segundos}s): {len(pausas_largas)}

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

# --- Enviar a OpenAI (comentado por ahora) ---
# import openai
# openai.api_key = "tu_clave_api"
# response = openai.ChatCompletion.create(
#     model="gpt-4",
#     messages=[
#         {"role": "system", "content": "Sos un evaluador de lectura oral."},
#         {"role": "user", "content": prompt}
#     ]
# )
# print(response["choices"][0]["message"]["content"])

# --- Mostrar solo el prompt por ahora ---
print(prompt)

# --- Pausas detectadas (debug útil) ---
print("\n🔎 Pausas largas detectadas:")
for i, (start, end, dur) in enumerate(pausas_largas, 1):
    print(f"  Pausa {i}: desde {start}s hasta {end}s - duración {dur}s")

print("\n📌 Signos de puntuación en texto:")    
print("".join(c if c in '.,;:!?' else " " for c in texto))
