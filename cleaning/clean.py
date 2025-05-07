
#!/usr/bin/env python3
import os
import io
import asyncio
import json
from pathlib import Path

# Import your services
from services.NoiseReduceService import NoiseReduceService
from services.VoiceSeparatorService import VoiceSeparatorService
from services.TextAudioEquivalentService import TextAudioEquivalentService

async def verify_and_save(service, audio_bytes, provided_text, output_dir, filename):
    """
    Calls the text-audio verification service and saves the audio file if it passes.
    """
    try:
        result = await service.verify(audio_bytes, provided_text)
    except Exception as e:
        print(f"[ERROR] Verification failed for {filename}: {e}")
        return False

    if result.get("match"):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / filename).write_bytes(audio_bytes)
        print(f"[SAVED] {filename} -> {output_dir}")
        return True

    return False

async def process_file(file_path: Path, services, dirs, text_map):
    """
    Process a single .wav file: lookup text, verify original, apply transformations, verify transformed.
    """
    filename = file_path.name
    if filename not in text_map:
        print(f"[SKIP] No text mapping for {filename}, skipping.")
        return filename, False, False

    provided_text = text_map[filename]
    print(f"Processing {filename}...")
    audio_bytes = file_path.read_bytes()

    # 1) Verify original
    orig_ok = await verify_and_save(
        services['text'], audio_bytes, provided_text,
        dirs['original'], filename
    )

    # 2) Apply noise reduction and voice separation
    reduced = services['noise'].reducir_ruido(audio_bytes)
    #separated = services['separator'].separar_voces(audio_bytes)

    # 3) Verify transformed
    trans_ok = await verify_and_save(
        services['text'], reduced, provided_text,
        dirs['transformed'], filename
    )

    return filename, orig_ok, trans_ok

async def main():
    # Determine paths relative to script location
    base_dir = Path(__file__).parent.resolve()
    input_dir = base_dir / 'audio_files'
    output_original = base_dir / 'output' / 'original'
    output_transformed = base_dir / 'output' / 'transformed'
    mapping_file = base_dir / 'textToAudio.json'

    # Load JSON mapping of audio filenames to text
    if not mapping_file.exists():
        print(f"[ERROR] Mapping file not found: {mapping_file}")
        return
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    text_map = {}
    # Build a dict: filename -> text
    for entry in data:
        text = entry.get('text', '').strip()
        for audio_name in entry.get('audio', []):
            text_map[audio_name] = text

    # Initialize services
    noise = NoiseReduceService()
    separator = VoiceSeparatorService()
    text_service = TextAudioEquivalentService(threshold=0.45)
    services = {'noise': noise, "separator": separator, 'text': text_service}

    # Create tasks for each .wav file in the input folder
    tasks = []
    for wav_file in input_dir.glob('*.wav'):
        tasks.append(
            process_file(
                wav_file,
                services,
                {'original': output_original, 'transformed': output_transformed},
                text_map
            )
        )

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Summary
    print("\nSummary:")
    for fname, orig_ok, trans_ok in results:
        status = (
            f"original_pass={'✔' if orig_ok else '✘'}; "
            f"transformed_pass={'✔' if trans_ok else '✘'}"
        )
        print(f" - {fname}: {status}")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
