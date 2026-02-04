from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from utils import extract_features_from_base64

API_KEY = "my-hackathon-key"

app = FastAPI(title="AI Voice Detection API")

class AudioRequest(BaseModel):
    audio_base64: str

@app.post("/detect-voice")
def detect_voice(data: AudioRequest, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        features = extract_features_from_base64(data.audio_base64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    score = 0
    if features["spectral_flatness"] > 0.25:
        score += 1
    if features["mfcc_std"] < 20:
        score += 1
    if features["zero_crossing"] < 0.05:
        score += 1

    if score >= 2:
        classification = "AI_GENERATED"
        confidence = min(0.6 + 0.15 * score, 0.95)
    else:
        classification = "HUMAN"
        confidence = min(0.6 + 0.15 * (3 - score), 0.95)

    return {
        "classification": classification,
        "confidence": round(confidence, 2)
    }
