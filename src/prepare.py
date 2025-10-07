# src/prepare.py
from src.data_utils import load_labels, verify_images_exist, stratified_split, save_split_csv

labels = load_labels("data/train_labels.csv")
print("Lignes labels :", len(labels))
print("Images présentes :", verify_images_exist("data/train", labels))
train, val = stratified_split(labels, val_ratio=0.2)
save_split_csv(train, "data/splits/train.csv")
save_split_csv(val,   "data/splits/val.csv")
print("OK: data/splits/train.csv & val.csv créés")
