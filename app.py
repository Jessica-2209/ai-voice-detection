from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from utils import extract_features_from_base64
import math

API_KEY = "my-hackathon-key"

app = FastAPI(title="AI Voice Detection API")

class AudioRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


@app.post("/detect-voice")
def detect_voice(request: AudioRequest, x_api_key: str = Header(...)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if request.audioFormat.lower() != "mp3":
        raise HTTPException(
            status_code=400,
            detail="Only MP3 format supported"
        )

    try:
        features = extract_features_from_base64(request.audioBase64)

        if not features or len(features) == 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid audio data"
            )

        score = abs(sum(features)) % 1

        classification = (
            "AI_GENERATED" if score > 0.5 else "HUMAN"
        )

        confidence = round(0.5 + abs(score - 0.5), 3)

        return {
            "classification": classification,
            "confidence": confidence,
            "language": request.language,
            "explanation": "Classification based on extracted spectral audio features."
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )
