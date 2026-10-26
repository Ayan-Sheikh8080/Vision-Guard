"""Tasks 2.3/2.4/3.3-3.5. Examples:
  python -m src.train --model cnn --run-name cnn_baseline
  python -m src.train --model mobilenet --run-name mnv2_ft
  python -m src.train --model mobilenet --autolog --run-name mnv2_autolog
  python -m src.train --model mobilenet --no-aug --run-name mnv2_noaug
  python -m src.train --model mobilenet --img 160 --run-name mnv2_160
  python -m src.train --model efficientnet --run-name effb0
"""
import argparse, os, random, subprocess, numpy as np, tensorflow as tf, mlflow, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from src.data import make_ds
from src.models import build_cnn, build_tl, unfreeze_last, trainable_names
from src.evaluate import evaluate

load_dotenv()
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mobilenet", choices=["cnn", "mobilenet", "efficientnet", "resnet50"])
ap.add_argument("--run-name", required=True)
ap.add_argument("--img", type=int, default=224)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--ft-lr", type=float, default=1e-5)
ap.add_argument("--unfreeze", type=int, default=20)
ap.add_argument("--dropout", type=float, default=0.3)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--no-aug", action="store_true")
ap.add_argument("--autolog", action="store_true")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed); tf.random.set_seed(a.seed)
out = f"results/{a.run_name}"; os.makedirs(out, exist_ok=True); os.makedirs("models", exist_ok=True)

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return "n/a"

class MLflowCB(tf.keras.callbacks.Callback):          # manual logging
    def __init__(self, offset=0): super().__init__(); self.offset = offset
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=self.offset + epoch)

def cbs(offset=0):
    c = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
         tf.keras.callbacks.ModelCheckpoint(f"models/{a.run_name}_ckpt.keras", monitor="val_loss", save_best_only=True),
         tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)]
    if not a.autolog:
        c.append(MLflowCB(offset))
    return c

tr = make_ds("train", a.img, True, not a.no_aug, a.bs)
va = make_ds("val", a.img, False, bs=a.bs)
te = make_ds("test", a.img, False, bs=a.bs)

mlflow.set_experiment("VisionGuard")
if a.autolog:
    mlflow.tensorflow.autolog(log_models=False)
with mlflow.start_run(run_name=a.run_name) as run:
    mlflow.set_tags(dict(git_commit=sh("git rev-parse HEAD"), dvc_data_version=sh("grep -m1 md5 data/raw.dvc"),
                         seed=a.seed, logging="autolog" if a.autolog else "manual"))
    mlflow.log_params(dict(model=a.model, img=a.img, lr=a.lr, ft_lr=a.ft_lr, unfreeze=a.unfreeze,
                           dropout=a.dropout, epochs=a.epochs, bs=a.bs, augmentation=not a.no_aug))
    hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    def add(h):
        for k in hist: hist[k] += h.history[k]
    if a.model == "cnn":
        model = build_cnn(a.img, a.dropout, a.lr)
        model.summary(); add(model.fit(tr, validation_data=va, epochs=a.epochs, callbacks=cbs()))
    else:
        model, base = build_tl(a.model, a.img, a.dropout, a.lr)
        s1 = trainable_names(model)
        print("Stage 1 trainable:", s1, "| head only")
        add(model.fit(tr, validation_data=va, epochs=a.epochs, callbacks=cbs()))
        n1 = len(hist["loss"])
        model = unfreeze_last(model, base, a.unfreeze, a.ft_lr)
        s2 = trainable_names(model, base)
        print("Stage 2 trainable backbone layers:", s2)
        open(f"{out}/trainable_layers.txt", "w").write(f"stage1: head only\nstage2:\n" + "\n".join(s2))
        add(model.fit(tr, validation_data=va, epochs=a.epochs, callbacks=cbs(n1)))
    best = int(np.argmin(hist["val_loss"])) + 1
    mlflow.log_metric("best_epoch", best)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, k in enumerate(["loss", "accuracy"]):
        ax[i].plot(hist[k], label="train"); ax[i].plot(hist["val_" + k], label="val")
        ax[i].axvline(best - 1, color="r", ls="--", label=f"best epoch {best}"); ax[i].set_title(k); ax[i].legend()
    plt.savefig(f"{out}/curves.png", dpi=120); plt.close()
    m = evaluate(model, va, te, a.img, out)
    mlflow.log_metrics({f"test_{k}": v for k, v in m.items() if k not in ("threshold",)})
    mlflow.log_metric("threshold", m["threshold"])
    model.save(f"models/{a.run_name}.keras")
    os.system("pip freeze > results/pip_freeze.txt")
    for f in ["confusion_matrix.png", "roc.png", "curves.png", "test_preds.csv", "metrics.json"]:
        mlflow.log_artifact(f"{out}/{f}")
    for s in ["train", "val", "test"]:
        mlflow.log_artifact(f"splits/{s}.txt", "splits")
    mlflow.log_artifact("results/pip_freeze.txt")
    mlflow.log_artifact(f"models/{a.run_name}.keras", "model")
    print("RUN_ID", run.info.run_id, "best epoch", best, m)
