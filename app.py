from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import pickle
from utils import extract_features_from_base64
API_KEY = "my-hackathon-key"
app = FastAPI(title="AI Voice Detection API")
# Temporary dummy model (will replace)
class DummyModel:
    def predict(self, X):
        return [1]  # AI_GENERATED
    def predict_proba(self, X):
        return [[0.2, 0.8]]
model = DummyModel()
class AudioRequest(BaseModel):
    audio_base64: str
@app.post("/detect-voice")
def detect_voice(
    data: AudioRequest,
    x_api_key: str = Header(...)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    features = extract_features_from_base64(data.audio_base64)
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    label = "AI_GENERATED" if pred == 1 else "HUMAN"
    return {
        "classification": label,
        "confidence": round(max(prob), 2)
    }
