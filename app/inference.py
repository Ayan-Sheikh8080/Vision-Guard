"""Task 4.1 / 8: remote API if API_URL is set, otherwise local Keras model (loaded ONCE)."""
import io
import numpy as np
import requests
from PIL import Image
from app import config
_model = None

def _local_model():
    global _model
    if _model is None:
        import tensorflow as tf
        _model = tf.keras.models.load_model(config.MODEL_PATH)
        print("model loaded once from", config.MODEL_PATH, flush=True)
    return _model

def predict(data: bytes, filename: str):
    """returns (label, confidence, model_version)"""
    if config.API_URL:
        r = requests.post(config.API_URL + "/predict", files={"file": (filename, data)},
                          headers={"X-API-Key": config.API_KEY}, timeout=30)
        r.raise_for_status()
        d = r.json()
        return d["label"], d["confidence"], d.get("model_version", config.MODEL_VERSION)
    m = _local_model()
    size = m.input_shape[1]
    x = np.array(Image.open(io.BytesIO(data)).convert("RGB").resize((size, size)), dtype="float32")[None]
    p = float(m.predict(x, verbose=0)[0][0])   # same preprocessing as training (preprocessing lives inside the model)
    return ("AI-generated", p, config.MODEL_VERSION) if p >= config.THRESHOLD else ("Real", 1 - p, config.MODEL_VERSION)
