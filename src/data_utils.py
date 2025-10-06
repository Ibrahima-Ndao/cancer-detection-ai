# src/data_utils.py
import os, csv, random
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import train_test_split

EXTS = (".tif", ".png", ".jpg", ".jpeg")

def load_labels(csv_path: str) -> List[Tuple[str,int]]:
    items = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            items.append((row["id"], int(row["label"])))
    return items

def verify_images_exist(image_dir: str, items: List[Tuple[str,int]]) -> int:
    ok = 0
    for _id, _ in items:
        if any(os.path.exists(os.path.join(image_dir, _id + ext)) for ext in EXTS):
            ok += 1
    return ok

def stratified_split(items: List[Tuple[str,int]], val_ratio=0.2, seed=1337):
    X = [i for i,_ in items]; y = [l for _,l in items]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_ratio, random_state=seed, stratify=y)
    return list(zip(X_tr, y_tr)), list(zip(X_val, y_val))

def save_split_csv(split, out_csv: str):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id","label"])
        for _id, y in split: w.writerow([_id, y])
