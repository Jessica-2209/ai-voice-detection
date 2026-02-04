import librosa
import numpy as np
import base64
import io
import soundfile as sf
def extract_features_from_base64(base64_audio):
    audio_bytes = base64.b64decode(base64_audio)
    audio_buffer = io.ByterIO(audio_bytes)
    y, sr = sf.read(audio_buffer)
    if y.ndim > 1:
        y= np.mean(y, axis=1)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    zero_crossing = librosa.feature.zero_crossing_rate(y)
    features = np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(spectral_centroid),
        np.mean(zero_crossing)
    ])
    return features.reshape(1,-1)