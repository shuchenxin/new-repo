"""
Feature extraction helpers for ML client.
"""

# pylint: disable=invalid-name,import-error

import numpy as np
import librosa


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract normalized MFCC-based fingerprint for comparison."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=26)
    vec = np.mean(mfcc, axis=1)

    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.zeros_like(vec)

    return vec / norm


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def extract_features_audio(y, sr):
  
    rms = librosa.feature.rms(y=y).mean()                           
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean() 
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()  
    zcr = librosa.feature.zero_crossing_rate(y=y).mean()            
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)                  
    flatness = librosa.feature.spectral_flatness(y=y).mean()        

    
    acousticness = float(1.0 - flatness)        
    danceability = float(1.0 - zcr)            
    energy = float(rms)
    instrumentalness = float(bandwidth)         
    liveness = float(centroid)                  
    speechiness = float(zcr)                   
    tempo_feat = float(tempo)
    valence = float(
        centroid / (centroid + bandwidth + 1e-6)
    )  

    vec = np.array(
        [
            acousticness,
            danceability,
            energy,
            instrumentalness,
            liveness,
            speechiness,
            tempo_feat,
            valence,
        ],
        dtype=float,
    )

    return vec

