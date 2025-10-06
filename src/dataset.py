# src/dataset.py
import os, csv
from typing import Tuple
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

EXTS = (".tif",".png",".jpg",".jpeg")

def _tfms(img_size=96, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

class PCamCSV(Dataset):
    def __init__(self, csv_file: str, img_root: str, transform=None):
        self.items = []
        self.img_root = img_root
        self.transform = transform
        with open(csv_file, newline="") as f:
            r = csv.DictReader(f)
            for row in r: self.items.append((row["id"], int(row["label"])))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        _id, y = self.items[idx]
        for ext in EXTS:
            p = os.path.join(self.img_root, _id + ext)
            if os.path.exists(p):
                img = Image.open(p).convert("RGB"); break
        else:
            raise FileNotFoundError(f"Image not found: {_id}")
        img = self.transform(img) if self.transform else img
        return img, y

def get_loaders(batch_size=128, num_workers=2, img_size=96) -> Tuple[DataLoader,DataLoader]:
    train_ds = PCamCSV("data/splits/train.csv", "data/train", _tfms(img_size, True))
    val_ds   = PCamCSV("data/splits/val.csv",   "data/train", _tfms(img_size, False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
