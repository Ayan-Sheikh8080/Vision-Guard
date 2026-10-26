"""Task 3.1/3.2 toy. Local: python scripts/mlflow_toy.py   DagsHub: python scripts/mlflow_toy.py --dagshub"""
import sys, mlflow, tensorflow as tf, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
if "--dagshub" in sys.argv:
    load_dotenv()                      # MLFLOW_TRACKING_URI/USERNAME/PASSWORD come from .env
else:
    mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("toy")
(x, y), (xt, yt) = tf.keras.datasets.mnist.load_data(); x, xt = x / 255.0, xt / 255.0
for lr in (1e-2, 1e-3):
    with mlflow.start_run(run_name=f"lr_{lr}"):
        mlflow.log_param("lr", lr); mlflow.set_tag("purpose", "toy")
        m = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dense(10, activation="softmax")])
        m.compile(tf.keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
        h = m.fit(x, y, epochs=5, validation_data=(xt, yt), verbose=2)
        for i, v in enumerate(h.history["val_accuracy"]):
            mlflow.log_metric("val_acc", v, step=i)
        fig = plt.figure(); plt.plot(h.history["val_accuracy"]); mlflow.log_figure(fig, "curve.png")
        m.save("toy.keras"); mlflow.log_artifact("toy.keras", "model")
