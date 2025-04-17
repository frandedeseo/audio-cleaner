import parselmouth

# Cargar archivo de audio
sound = parselmouth.Sound("lectura.wav")

# Extraer el pitch (frecuencia fundamental)
pitch = sound.to_pitch()

# Imprimir algunos datos
for t in pitch.xs():
    freq = pitch.get_value_at_time(t)
    if freq:  # puede ser None si hay silencio
        print(f"Tiempo: {t:.2f} s | Frecuencia: {freq:.2f} Hz")
