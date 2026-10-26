import io
from flask import Flask, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename
from app import config, db, inference

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_MB * 1024 * 1024
ALLOWED = {"png", "jpg", "jpeg", "webp"}
db.init_db()

def err(msg, code):
    return render_template("error.html", msg=msg, code=code), code

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("file")
    if not f or not f.filename:
        return err("No file selected.", 400)
    name = secure_filename(f.filename)
    if "." not in name or name.rsplit(".", 1)[1].lower() not in ALLOWED:
        return err("Unsupported file type. Use PNG, JPG or WEBP.", 400)
    data = f.read()                                  # kept in memory, never written to disk
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        return err("This file looks like an image but cannot be decoded.", 400)
    try:
        label, conf, ver = inference.predict(data, name)
    except Exception as e:
        app.logger.error("inference failed: %s", e)
        return err("The model service is unavailable. Try again later.", 502)
    db.add(name, label, conf, config.MODEL_NAME, ver, config.THRESHOLD)
    return render_template("result.html", label=label, conf=round(conf * 100, 1), name=name, ver=ver)

@app.route("/history")
def history():
    return render_template("history.html", rows=db.recent())

@app.route("/health")
def health():
    return {"status": "ok", "mode": "remote-api" if config.API_URL else "local-model"}

@app.errorhandler(413)
def too_big(_):
    return err(f"File too large (max {config.MAX_MB} MB).", 413)

@app.errorhandler(404)
def nf(_):
    return err("Page not found.", 404)

@app.errorhandler(500)
def ise(_):
    return err("Internal server error.", 500)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)   # debug off: debugger allows remote code execution
