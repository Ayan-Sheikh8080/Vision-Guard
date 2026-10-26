"""Clean + stratified 70/15/15 split. Usage: python -m src.split"""
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("audit.csv")
df = df[~df.corrupt & ~df.duplicate].copy()      # duplicates dropped => no leakage across splits
df["label"] = df.cls.str.lower().str.contains("ai").astype(int)   # 1 = AI-generated
print(df.groupby(["cls", "label"]).size())
tr, tmp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=42)
va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=42)
for n, d in [("train", tr), ("val", va), ("test", te)]:
    d[["path", "label"]].to_csv(f"splits/{n}.txt", index=False, header=False)
    print(n, len(d))
