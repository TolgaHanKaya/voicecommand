#!/usr/bin/env python3
"""record_data.py

CLI script to record short audio clips using sounddevice and scipy.io.wavfile.

- Creates `dataset/evet` and `dataset/hayir` directories if they don't exist.
- Press 'e' + Enter to record a 1.2 s "yes" clip (saved as `dataset/evet/evet_<n>.wav`).
- Press 'h' + Enter to record a 1.2 s "no" clip (saved as `dataset/hayir/hayir_<n>.wav`).
- Press 'q' + Enter to quit.

The recordings are mono, 16 kHz sample rate.
"""

import os
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# Configuration
SAMPLE_RATE = 16000  # Hz
DURATION = 1.2        # seconds
CHANNELS = 1          # mono

# Paths
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "dataset")
YES_DIR = os.path.join(DATASET_ROOT, "evet")
NO_DIR = os.path.join(DATASET_ROOT, "hayir")

def ensure_dirs():
    """Create required directories if they do not exist."""
    for d in (YES_DIR, NO_DIR):
        os.makedirs(d, exist_ok=True)

def next_index(directory: str, prefix: str) -> int:
    """Return the next incremental index for files named `<prefix>_<n>.wav`."""
    max_idx = 0
    if not os.path.isdir(directory):
        return 1
    for fname in os.listdir(directory):
        if fname.startswith(prefix) and fname.endswith('.wav'):
            try:
                idx = int(fname[len(prefix) + 1:-4])
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                continue
    return max_idx + 1

def record_clip() -> np.ndarray:
    """Record a clip of ``DURATION`` seconds and return the audio data."""
    print(f"Recording {DURATION}s...", flush=True)
    rec = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                 channels=CHANNELS, dtype='float32')
    sd.wait()
    return np.squeeze(rec)

def save_wav(data: np.ndarray, path: str):
    """Save *data* as a 16‑bit PCM WAV file."""
    if data.dtype == np.float32:
        data = np.clip(data, -1.0, 1.0)
        data = (data * 32767).astype(np.int16)
    wavfile.write(path, SAMPLE_RATE, data)
    print(f"Saved: {path}")

def main():
    ensure_dirs()
    print("Press 'e' + Enter to record YES, 'h' + Enter to record NO, 'q' to quit.")
    while True:
        try:
            choice = input('> ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print('\nExiting.')
            break
        if choice == 'q':
            print('Good bye!')
            break
        elif choice == 'e':
            idx = next_index(YES_DIR, 'evet')
            path = os.path.join(YES_DIR, f"evet_{idx}.wav")
            save_wav(record_clip(), path)
        elif choice == 'h':
            idx = next_index(NO_DIR, 'hayir')
            path = os.path.join(NO_DIR, f"hayir_{idx}.wav")
            save_wav(record_clip(), path)
        else:
            print("Unrecognized command. Use 'e', 'h', or 'q'.")

if __name__ == '__main__':
    main()
