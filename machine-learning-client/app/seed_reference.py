"""
Seed reference fingerprints for genre classification.
"""

# pylint: disable=invalid-name,import-error

import time
import numpy as np
import librosa
from gridfs import GridFS
import pymongo


def seed_reference_tracks(db):
    """Generate and store reference fingerprints."""
    references = [
        ("vocal", "samples/vocal_example.wav"),
        ("electronic", "samples/electronic_example.wav"),
    ]

    docs = []

    for genre, path in references:
        audio, sample_rate = librosa.load(path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=26)
        fp = np.mean(mfcc, axis=1).astype(float)

        norm = np.linalg.norm(fp)
        if norm > 0:
            fp = fp / norm

        docs.append({"genre": genre, "fp": fp.tolist(), "created_at": time.time()})

    if docs:
        db.reference_tracks.insert_many(docs)


def main():
    """Manual entry point."""
    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client["ml_audio"]
    _ = GridFS(db)  # kept for symmetry
    seed_reference_tracks(db)


if __name__ == "__main__":
    main()
