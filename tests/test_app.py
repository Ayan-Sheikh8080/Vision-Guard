import io, os
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
from PIL import Image
from app.app import app

def test_health():
    assert app.test_client().get("/health").json["status"] == "ok"

def test_rejects_bad_type():
    r = app.test_client().post("/predict", data={"file": (io.BytesIO(b"x"), "a.txt")}, content_type="multipart/form-data")
    assert r.status_code == 400

def test_rejects_fake_image():
    r = app.test_client().post("/predict", data={"file": (io.BytesIO(b"notanimage"), "a.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 400
