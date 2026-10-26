"""Task 2.6: 10 misclassified test images. Usage: python -m src.errors <run_name>"""
import sys, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
run = sys.argv[1]
d = pd.read_csv(f"results/{run}/test_preds.csv")
w = d[d.label != d.pred].head(10)
fig, ax = plt.subplots(2, 5, figsize=(20, 8))
for a, (_, r) in zip(ax.ravel(), w.iterrows()):
    a.imshow(Image.open(r.path).convert("RGB")); a.axis("off")
    a.set_title(f"true={'AI' if r.label else 'Real'} pred={'AI' if r.pred else 'Real'}\np(AI)={r.prob:.2f}", fontsize=9)
plt.tight_layout(); plt.savefig(f"results/{run}/errors.png", dpi=120); print(w[["path", "label", "prob"]])
