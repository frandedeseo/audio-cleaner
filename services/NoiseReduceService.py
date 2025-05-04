import io
import librosa
import noisereduce as nr
import soundfile as sf

class NoiseReduceService:
    @staticmethod
    def reducir_ruido(audio_bytes: bytes) -> bytes:
        """
        Reduce noise from the given audio bytes.

        :param audio_bytes: Audio data in bytes.
        :return: Audio data with reduced noise in bytes.
        """
        # Load the audio
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

        # Assume the first second is noise
        noise_sample = y[:sr]

        # Apply noise reduction
        reduced_noise = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)

        # Save back to bytes
        out_io = io.BytesIO()
        sf.write(out_io, reduced_noise, sr, format='WAV')
        out_io.seek(0)

        return out_io.read()
