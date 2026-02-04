import base64
import io
import numpy as np
import librosa

def extract_features_from_base64(base64_audio):
    try:
        audio_bytes = base64.b64decode(base64_audio)
        audio_buffer = io.BytesIO(audio_bytes)

        y, sr = librosa.load(audio_buffer, sr=None)

        if len(y) < sr:  # less than 1 second
            raise ValueError("Audio too short")

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(y)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)

        return {
            "mfcc_std": float(np.std(mfccs)),
            "spectral_centroid": float(np.mean(spectral_centroid)),
            "zero_crossing": float(np.mean(zero_crossing)),
            "spectral_flatness": float(np.mean(spectral_flatness))
        }

    except Exception:
        raise ValueError("Invalid or corrupted audio input")
