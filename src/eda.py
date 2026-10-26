"""Task 1.3: EDA figures. Usage: python -m src.eda"""
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from PIL import Image
os_df = pd.read_csv("audit.csv")
df = os_df[~os_df.corrupt]
classes = sorted(df.cls.unique())
fig, ax = plt.subplots(len(classes), 8, figsize=(20, 5))
for i, c in enumerate(classes):
    for j, p in enumerate(df[df.cls == c].sample(8, random_state=1).path):
        ax[i][j].imshow(Image.open(p).convert("RGB")); ax[i][j].axis("off")
    ax[i][0].set_title(c, loc="left")
plt.tight_layout(); plt.savefig("results/eda_grid.png", dpi=120)
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for a, col in zip(ax, ["width", "height", "size"]):
    sns.histplot(df, x=col, hue="cls", bins=40, ax=a)
plt.tight_layout(); plt.savefig("results/eda_hist.png", dpi=120)
print("saved results/eda_grid.png, results/eda_hist.png")
