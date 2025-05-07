import json
import io
import os
from pydub import AudioSegment, silence
import numpy as np

class WpmService:
    def calculate(self, audio_bytes: bytes, text: str,
                  min_silence_len: int = 2000,
                  silence_thresh: int = -70) -> float:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
        silent_ranges = silence.detect_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        total_silence_ms = sum((end - start) for start, end in silent_ranges)
        total_ms = len(audio)
        active_seconds = max(0, total_ms - total_silence_ms) / 1000.0
        word_count = len(text.split())

        return word_count / (active_seconds / 60.0) if active_seconds > 0 else 0.0

def main():
    # adjust these paths if needed
    JSON_PATH     = 'result.json'
    AUDIO_DIR     = 'test-audios'

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    service = WpmService()
    errors = []

    print(f"{'file':20s}  {'true':>6s}  {'pred':>6s}  {'err':>6s}")
    print("-" * 44)
    for case in test_cases:
        fname     = case['audio']
        true_wpm  = case['output']['fluidez_lectora']
        text      = case['text']
        path      = os.path.join(AUDIO_DIR, fname)

        with open(path, 'rb') as af:
            audio_bytes = af.read()

        pred_wpm = service.calculate(audio_bytes, text)



        err      = pred_wpm - true_wpm
        errors.append(err**2)

        print(f"{fname:20s}  {true_wpm:6.1f}  {pred_wpm:6.1f}  {err:6.1f}")

    mse = np.mean(errors) if errors else float('nan')
    print("\nMean Squared Error (MSE):", round(mse, 2))

if __name__ == '__main__':
    main()
