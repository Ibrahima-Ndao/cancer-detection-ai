# src/train.py
import argparse, torch, torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from src.dataset import get_loaders
from src.models import build_model

def train_one_epoch(model, loader, device, optim, criterion):
    model.train(); total=0.0
    for xb,yb in tqdm(loader, desc="Train", leave=False):
        xb, yb = xb.to(device), yb.float().unsqueeze(1).to(device)
        optim.zero_grad(); logits = model(xb); loss = criterion(logits, yb)
        loss.backward(); optim.step(); total += loss.item()*xb.size(0)
    return total/len(loader.dataset)

@torch.no_grad()
def eval_auc(model, loader, device):
    model.eval(); ys, ps = [], []
    for xb,yb in tqdm(loader, desc="Val", leave=False):
        xb = xb.to(device)
        prob = torch.sigmoid(model(xb)).squeeze(1).cpu().numpy().tolist()
        ys.extend(yb.numpy().tolist()); ps.extend(prob)
    try: return float(roc_auc_score(ys, ps))
    except: return 0.5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tr, va = get_loaders(batch_size=args.batch_size, img_size=args.img_size)
    model = build_model(args.model, num_classes=1, pretrained=False).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit  = nn.BCEWithLogitsLoss()

    best_auc, best_path = 0.0, f"best_{args.model}.pt"
    for e in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, tr, device, optim, crit)
        val_auc = eval_auc(model, va, device)
        print(f"Epoch {e}/{args.epochs}  loss={tr_loss:.4f}  val_auc={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc; torch.save(model.state_dict(), best_path)
            print(f"  → new best, saved {best_path}")
if __name__=="__main__": main()
