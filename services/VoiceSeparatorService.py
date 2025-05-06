import io
import torchaudio
from speechbrain.inference.separation import SepformerSeparation
from speechbrain.utils.fetching import LocalStrategy

class VoiceSeparatorService:
    def __init__(self):
        self.separator = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-whamr",
            savedir="tmpdir",
            local_strategy=LocalStrategy.COPY,  # Force copy, no symlink
            run_opts={"use_symlink": False},    # Backup
        )

    def separar_voces(self, audio_bytes: bytes) -> bytes:
        # Save the bytes to a temporary file
        with open("temp_input.wav", "wb") as f:
            f.write(audio_bytes)

        # Load audio and resample to 8 kHz if necessary
        waveform, sample_rate = torchaudio.load("temp_input.wav")
        if sample_rate != 8000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=8000)
            sample_rate = 8000

        # Save the resampled waveform
        torchaudio.save("temp_input_resampled.wav", waveform, sample_rate)

        # Separate audio sources
        est_sources = self.separator.separate_file(path="temp_input_resampled.wav")

        # Select the source with the highest energy
        energy_1 = est_sources[:, :, 0].pow(2).mean()
        energy_2 = est_sources[:, :, 1].pow(2).mean()
        voz_cercana = est_sources[:, :, 0] if energy_1 > energy_2 else est_sources[:, :, 1]

        # Save the result to a buffer
        buffer = io.BytesIO()
        torchaudio.save(buffer, voz_cercana.detach().cpu(), 8000, format='wav')
        buffer.seek(0)
        return buffer.read()