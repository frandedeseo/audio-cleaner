#from pydub import AudioSegment, silence
import io

class WpmService:
    def calculate(self, audio_bytes: bytes, text: str, min_silence_len: int = 2000, silence_thresh: int = -70) -> float:
        #audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
        #silent_ranges = silence.detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        total_silence = 0
        dur_total_ms = 10000
        active_s = max(0, dur_total_ms - total_silence) / 1000
        word_count = len(text.split())
        return word_count / (active_s / 60) if active_s > 0 else 0