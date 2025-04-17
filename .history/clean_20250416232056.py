import parselmouth
from pathlib import Path

# Rutas
base_dir = Path("praat")
audio_path = base_dir / "20736.wav"
texto_path = base_dir / "buho.txt"

# Cargar audio
sound = parselmouth.Sound(str(audio_path))

# Cargar texto leído (si lo necesitas para comparar duración, sílabas, etc.)
with open(texto_path, "r", encoding="utf-8") as f:
    texto = f.read().strip()

pitch = sound.to_pitch()
intensity = sound.to_intensity()

data = {
    "duracion_total": round(sound.get_total_duration(), 2),  # segundos
    "frecuencia_promedio": round(pitch.get_mean(), 2),       # Hz
    "frecuencia_maxima": round(pitch.get_maximum(), 2),
    "frecuencia_minima": round(pitch.get_minimum(), 2),
    "intensidad_media": round(intensity.get_average(), 2),   # dB
    "cantidad_silencios": sum(1 for t in pitch.xs() if pitch.get_value_at_time(t) is None),
    "texto_leido": texto
}
print(data)
