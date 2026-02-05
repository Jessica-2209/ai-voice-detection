from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from utils import extract_features_from_base64

API_KEY = "my-hackathon-key"

app = FastAPI(title="AI Voice Detection API")


# ✅ GUVI request format
class AudioRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


@app.post("/detect-voice")
def detect_voice(request: AudioRequest, x_api_key: str = Header(...)):

    # 🔐 Authentication
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # ✅ Format check
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 supported")

    try:
        features = extract_features_from_base64(request.audioBase64)

        # Extract key features safely
        flatness = float(features.get("spectral_flatness", 0))
        mfcc_std = float(features.get("mfcc_std", 1))
        zero_cross = float(features.get("zero_crossing", 0))

        # ---- Stable classifier logic ----
        # AI voices = smoother + less variation
        ai_score = (
            (flatness * 0.5) +
            (1 / (mfcc_std + 1) * 0.3) +
            ((0.1 - zero_cross) * 2 * 0.2)
        )

        # Clamp score
        ai_score = max(0.0, min(ai_score, 1.0))

        label = "AI_GENERATED" if ai_score > 0.55 else "HUMAN"

        confidence = round(0.5 + abs(ai_score - 0.5), 3)

        return {
            "classification": label,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )
