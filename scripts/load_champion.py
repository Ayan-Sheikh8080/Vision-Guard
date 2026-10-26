"""Task 3.6: load by alias, classify one image, and save champion to models/champion.keras.
Usage: python scripts/load_champion.py path/to/image.jpg"""
import sys, glob, shutil, numpy as np, mlflow, tensorflow as tf
from dotenv import load_dotenv
from PIL import Image
load_dotenv()
d = mlflow.artifacts.download_artifacts(artifact_uri="models:/VisionGuard@champion")
f = glob.glob(f"{d}/**/*.keras", recursive=True)[0]
shutil.copy(f, "models/champion.keras")          # server uses this local copy, never DagsHub at request time
model = tf.keras.models.load_model("models/champion.keras")
size = model.input_shape[1]
x = np.array(Image.open(sys.argv[1]).convert("RGB").resize((size, size)), dtype="float32")[None]
p = float(model.predict(x, verbose=0)[0][0])
print("label:", "AI-generated" if p >= 0.5 else "Real", "| p(AI) =", round(p, 4))
