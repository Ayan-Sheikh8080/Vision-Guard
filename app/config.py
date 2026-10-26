import os
MODEL_PATH = os.environ.get("MODEL_PATH", "models/champion.keras")
THRESHOLD = float(os.environ.get("THRESHOLD", 0.5))
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///history.db")
API_URL = os.environ.get("API_URL", "").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "VisionGuard")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "champion")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev")
MAX_MB = 5
