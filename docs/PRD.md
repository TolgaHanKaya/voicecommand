# PRD – Voice Command Verification Service

## Document Control & Metadata

| Attribute | Details |
| :--- | :--- |
| **Document Title** | Voice Command Verification Service PRD |
| **PRD Version** | v1.0 (As-Built / Baseline) |
| **Target App Release** | v0.1.1 |
| **Author** | Tolga Han Kaya |
| **Status** | Approved / Baseline |
| **Last Updated** | 18.08.2026 |

---

## 1. Project Overview
The **Voice Command Verification Service** is a FastAPI micro‑service that receives a spoken command (as a `.wav` file) and returns a verification decision (`EVET` / `HAYIR`), confidence score and latency. It re‑uses the existing MFCC extraction, scaling and SVM inference pipeline from the repository.

## 2. Target Users
- Banking app users who need voice‑based transaction verification
- Fraud‑detection analysts reviewing verification attempts

## 3. Core Features
| Feature | Description |
|---------|-------------|
| **API Endpoint (FastAPI)** | `POST /api/v1/voice/verify` accepts a `.wav` file and returns verification result, confidence, and latency |
| **Audio Validation** | Reject files >2 MB or >10 s with HTTP 413 payload too large |
| **Feature Extraction** | MFCC extraction via `extract_features.py` pipeline |
| **Model Inference** | Scaled features fed to pre‑trained SVM model for prediction |

## 4. User Flow
1. Client uploads a `.wav` voice command to the `/api/v1/voice/verify` endpoint.
2. Service validates file size and duration.
3. Audio is processed to extract MFCC features.
4. Features are scaled and fed to the SVM model for prediction.
5. Service returns JSON with `prediction` (`EVET`/`HAYIR`), `confidence`, and `latency_ms`.

## 5. Technical Requirements
- Python 3.9+
- Packages: `fastapi`, `uvicorn`, `numpy`, `scipy`, `scikit-learn`, `pydantic`
- In‑memory audio processing only (no persistent storage)

## 6. Success Criteria
- End‑to‑end latency ≤ 250 ms for 95 % of requests.
- Verification accuracy ≥ 95 % on validation set.
- No audio files are persisted beyond processing.

## 7. Risks & Limitations
- Poor microphone quality may reduce verification accuracy.
- High‑load scenarios could increase latency; scaling strategies may be required.

## 8. User Stories
- **As a bank user**, I want to approve/disapprove my transactions with my voice.

## 9. Functional Requirements
1. The system shall expose `POST /api/v1/voice/verify` accepting `multipart/form-data` with a `.wav` file.
2. The system shall validate file size (≤2 MB) and duration (≤10 s), returning HTTP 413 on violations.
3. The system shall extract MFCC features using the existing pipeline.
4. The system shall scale features with the pre‑trained scaler.
5. The system shall perform prediction using the pre‑trained SVM model.
6. The system shall return JSON `{ "prediction": "EVET"|"HAYIR", "confidence": <0‑100>, "latency_ms": <int> }`.

## 10. Non‑Functional Requirements
- **Performance**: 95 % of requests complete within 250 ms latency.
- **Usability**: API is straightforward; clients need only two HTTP calls (upload and receive response).

