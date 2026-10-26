"""Task 7: model-serving API (runs in the cloud). POST /predict with header X-API-Key and form field 'file'."""
import io, os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
MODEL = tf.keras.models.load_model(os.environ.get("MODEL_PATH", "champion.keras"))   # loaded once
SIZE = MODEL.input_shape[1]
THR = float(os.environ.get("THRESHOLD", 0.5))
KEY = os.environ["API_KEY"]
VER = os.environ.get("MODEL_VERSION", "champion")

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():
    if request.headers.get("X-API-Key") != KEY:
        return jsonify(error="unauthorized"), 401
    f = request.files.get("file")
    if not f:
        return jsonify(error="no file"), 400
    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB").resize((SIZE, SIZE))
    except Exception:
        return jsonify(error="bad image"), 400
    p = float(MODEL.predict(np.array(img, dtype="float32")[None], verbose=0)[0][0])
    label, conf = ("AI-generated", p) if p >= THR else ("Real", 1 - p)
    return jsonify(label=label, confidence=conf, p_ai=p, model_version=VER)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
