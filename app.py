from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from utils import extract_features_from_base64

API_KEY = "my-hackathon-key"

app = FastAPI(title="AI Voice Detection API")


class AudioRequest(BaseModel):
    audio_base64: str = Field(alias="audioBase64")

    class Config:
        populate_by_name = True


@app.post("/detect-voice")
async def detect_voice(
    request: AudioRequest,
    x_api_key: str = Header(...)
):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        features = extract_features_from_base64(request.audio_base64)

        # Convert dict values to numbers safely
        numeric_values = []

        for value in features.values():
            if isinstance(value, (int, float)):
                numeric_values.append(value)
            elif isinstance(value, list):
                numeric_values.extend(
                    [v for v in value if isinstance(v, (int, float))]
                )

        if not numeric_values:
            score = 0.5
        else:
            score = abs(sum(numeric_values)) % 1

        label = "AI_GENERATED" if score > 0.5 else "HUMAN"

        return {
            "classification": label,
            "confidence": round(score, 3)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )
