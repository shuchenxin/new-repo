# test_local_model.py

import sys
from pathlib import Path

import numpy as np
import joblib
import soundfile as sf

# -----------------------------------------------------------
# machine-learning-client/local_test
# -----------------------------------------------------------
BASE = Path(__file__).resolve().parent  # local_test/
PROJECT_ROOT = BASE.parent  # machine-learning-client/


sys.path.append(str(PROJECT_ROOT))


from app.features import extract_features_audio  # noqa: E402


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def main() -> None:

    audio_path = BASE / "test_audio_rock.wav"

    if not audio_path.exists():
        raise FileNotFoundError(f"CAN NOT FIND AUDIO: {audio_path}")

    print(f"USE TEST AUDIO: {audio_path}")

    scaler_path = PROJECT_ROOT / "data" / "fma_metadata" / "scaler_rock_hiphop.joblib"
    model_path = PROJECT_ROOT / "data" / "fma_metadata" / "model_rock_hiphop.joblib"

    if not scaler_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "CAN NOT FIND scaler OR model，PLEASE CHECK PATH：\n"
            f"scaler: {scaler_path}\n"
            f"model : {model_path}"
        )

    scaler = joblib.load(scaler_path)
    # 打印特征的mean std
    print("SCALER mean :", scaler.mean_)
    print("SCALER std  :", scaler.scale_)
    model = joblib.load(model_path)

    print("MODEL COEFFICIENTS:", model.coef_)

    print("LOAD model SUCESSFULLY")

    audio, sr = load_audio_mono(audio_path)

    feat = extract_features_audio(audio, sr)  # shape: (8,)
    print("FEATURE VECTORS:", feat)
    feat = feat.reshape(1, -1)  # shape: (1, 8)

    print("ORIGINAL FEATURE VECTORS:", feat)

    print("INTERCEPT:", getattr(model, "intercept_", None))


    print("CLASSES:", getattr(model, "classes_", None))

    z = (feat - scaler.mean_) / scaler.scale_
    print("Z-SCORES:", z)

    feat_scaled = scaler.transform(feat)
    print("SCALED FEATURE:", feat_scaled)

    if hasattr(model, "decision_function"):
        print("DECISION:", model.decision_function(feat_scaled))

    pred = model.predict(feat_scaled)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat_scaled)[0]

    print("\n====== PREDICTED RESULT ======")
    print("PERDICTED CATEGORY:", pred)
    if proba is not None:
        print("PERDICT DISTRIBUTION:")
        print("classes_:", model.classes_)
        print("proba  :", proba)


if __name__ == "__main__":
    main()
