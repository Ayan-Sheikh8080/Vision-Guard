"""Task 1.2: dataset audit. Usage: python -m src.audit"""
import os, hashlib, pandas as pd
from PIL import Image
ROOT = "data/raw"
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
rows = []
for cls in sorted(os.listdir(ROOT)):
    top = os.path.join(ROOT, cls)
    if not os.path.isdir(top):
        continue
    for dirpath, _, files in os.walk(top):          # walks nested folders too
        for f in files:
            if os.path.splitext(f)[1].lower() not in EXT:
                continue
            p = os.path.join(dirpath, f)
            row = dict(path=p.replace("\\", "/"), cls=cls, size=os.path.getsize(p), corrupt=False)
            with open(p, "rb") as fh:
                row["sha256"] = hashlib.sha256(fh.read()).hexdigest()
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
            except Exception:
                row["corrupt"] = True
            rows.append(row)
df = pd.DataFrame(rows)
df["duplicate"] = df.duplicated("sha256", keep="first")
df.to_csv("audit.csv", index=False)
print(df.groupby("cls").size())
print(df.groupby("cls")[["width", "height", "size"]].agg(["min", "median", "max"]))
print("Corrupt:", int(df.corrupt.sum()), "| Duplicates:", int(df.duplicate.sum()))