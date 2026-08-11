# live_test.py
"""Live microphone test script.

Continuously records 1.2‑second audio snippets when the user presses **Enter**,
extracts 13‑dimensional MFCC features (same as `extract_features.py`),
scales them with the saved scaler, and predicts the command using the trained
model. The prediction and its confidence are printed with coloured output.
Press **q** (followed by Enter) to exit.
"""

import os
import sys
import numpy as np
import joblib
import sounddevice as sd
import librosa

# Load model and scaler
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "voice_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

if not os.path.isfile(model_path) or not os.path.isfile(scaler_path):
    sys.exit("Model or scaler file not found. Ensure training was completed.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Configuration matching extract_features.py
SR = 16000          # Sampling rate
DURATION = 1.2      # seconds
N_MFCC = 13

def record_audio():
    """Record a short audio clip from the default microphone."""
    print("\nPress Enter to record, or 'q' + Enter to quit.")
    key = input().strip().lower()
    if key == "q":
        return None
    print(f"Recording {DURATION}s…")
    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()

def extract_mfcc_features(signal: np.ndarray) -> np.ndarray:
    """Compute the mean 13‑dimensional MFCC vector for *signal*."""
    mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)

def colour_text(text: str, colour: str) -> str:
    colours = {"black":"30","red":"31","green":"32","yellow":"33","blue":"34","magenta":"35","cyan":"36","white":"37"}
    code = colours.get(colour.lower(), "37")
    return f"\033[{code}m{text}\033[0m"

def main():
    while True:
        audio = record_audio()
        if audio is None:
            print("Exiting live test.")
            break
        try:
            features = extract_mfcc_features(audio)
        except Exception as e:
            print(colour_text(f"Error extracting features: {e}", "red"))
            continue
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)[0]
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(features_scaled)[0][int(pred)] * 100
        else:
            prob = 100.0
        label = "EVET" if pred == 1 else "HAYIR"
        colour = "green" if pred == 1 else "red"
        print(colour_text(f"Tahmin: {label} ({prob:.1f}%)", colour))

if __name__ == "__main__":
    main()
