import os
import time
import numpy as np
import joblib
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Constants matching the training / live test configuration
SR = 16000  # Sampling rate
N_MFCC = 13

app = FastAPI(title="Voice Command Verification Service", version="1.0")

# Allow cross‑origin requests (CORS) – permissive for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and scaler at startup
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "voice_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

if not os.path.isfile(model_path) or not os.path.isfile(scaler_path):
    raise RuntimeError("Model or scaler file not found. Ensure training was completed.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


def extract_mfcc_features(signal: np.ndarray) -> np.ndarray:
    """Compute the mean MFCC vector for a raw audio signal."""
    mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)


class VerificationResult(BaseModel):
    prediction: str
    confidence: float
    processing_time_ms: int


@app.post("/api/v1/voice/verify", response_model=VerificationResult)
async def verify_voice(file: UploadFile = File(...)):
    # Simple validation – ensure the uploaded file is a WAV
    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/wave"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only WAV audio files are supported.",
        )
    try:
        # Read file into memory
        raw_bytes = await file.read()
        # Load audio with librosa – it can read from bytes via a temporary file
        # Write to a temporary file inside the workspace (no external /tmp)
        tmp_path = os.path.join(BASE_DIR, "tmp_upload.wav")
        with open(tmp_path, "wb") as f:
            f.write(raw_bytes)
        signal, _ = librosa.load(tmp_path, sr=SR, mono=True)
        os.remove(tmp_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process audio file: {e}",
        )

    start = time.time()
    try:
        features = extract_mfcc_features(signal)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature extraction error: {e}",
        )
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)[0]
    # Compute confidence – fall back to 100% if model lacks predict_proba
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features_scaled)[0][int(pred)] * 100.0
    else:
        prob = 100.0
    label = "EVET" if pred == 1 else "HAYIR"
    elapsed_ms = int((time.time() - start) * 1000)
    return VerificationResult(prediction=label, confidence=round(prob, 2), processing_time_ms=elapsed_ms)
