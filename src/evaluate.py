"""Task 2.5: evaluation. Threshold chosen on VALIDATION only; test used once."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay)
from src.data import read_split

def probs(model, ds):
    return model.predict(ds, verbose=0).ravel()

def best_threshold(y, p):
    ts = np.linspace(0.05, 0.95, 91)
    return float(ts[int(np.argmax([f1_score(y, p >= t) for t in ts]))])

def evaluate(model, val_ds, test_ds, img, out):
    yv = np.array(read_split("val")[1]); yt = np.array(read_split("test")[1])
    pv, pt = probs(model, val_ds), probs(model, test_ds)
    thr = best_threshold(yv, pv)
    pred = (pt >= thr).astype(int)
    m = dict(threshold=thr, accuracy=accuracy_score(yt, pred), precision=precision_score(yt, pred),
             recall=recall_score(yt, pred), f1=f1_score(yt, pred), roc_auc=roc_auc_score(yt, pt),
             val_f1=f1_score(yv, pv >= thr), val_auc=roc_auc_score(yv, pv))
    ConfusionMatrixDisplay(confusion_matrix(yt, pred), display_labels=["Real", "AI"]).plot()
    plt.savefig(f"{out}/confusion_matrix.png", dpi=120); plt.close()
    fpr, tpr, _ = roc_curve(yt, pt)
    plt.plot(fpr, tpr, label=f"AUC={m['roc_auc']:.3f}"); plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
    plt.savefig(f"{out}/roc.png", dpi=120); plt.close()
    paths = read_split("test")[0]
    pd.DataFrame(dict(path=paths, label=yt, prob=pt, pred=pred)).to_csv(f"{out}/test_preds.csv", index=False)
    json.dump(m, open(f"{out}/metrics.json", "w"), indent=2)
    return m
