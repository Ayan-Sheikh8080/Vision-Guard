"""Task 2.7 apply: Keras Tuner on VisionGuard (>= 8 trials). Usage: python -m src.tune"""
import keras_tuner as kt, tensorflow as tf
from src.data import make_ds
from src.models import build_tl, unfreeze_last, compile_model
tr = make_ds("train", 160, True).take(60); va = make_ds("val", 160).take(20)   # subset for speed
def build(hp):
    model, base = build_tl("mobilenet", 160, hp.Float("dropout", 0.1, 0.5, step=0.1), 1e-3)
    n = hp.Choice("unfreeze", [0, 10, 30])
    lr = hp.Float("lr", 1e-5, 1e-3, sampling="log")
    if n:
        return unfreeze_last(model, base, n, lr)
    return compile_model(model, lr)
t = kt.RandomSearch(build, objective="val_loss", max_trials=8, directory="tuner", project_name="visionguard", overwrite=True)
t.search(tr, validation_data=va, epochs=3)
t.results_summary()
print(t.get_best_hyperparameters(1)[0].values)
