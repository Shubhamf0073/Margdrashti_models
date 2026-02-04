import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from model_smallcnn import SmallCNN
from data_utils import pil_to_tensor

class TileDataset(Dataset):
    def __init__(self, csv_path, data_root, class_names):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["label_name"].notna()].copy()
        self.data_root = Path(data_root)
        self.class_names = class_names
        self.label_to_idx = {n:i for i,n in enumerate(class_names)}
        self.df["y"] = self.df["label_name"].map(self.label_to_idx).astype(int)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = self.data_root / r["tile_path"]
        img = Image.open(p).convert("RGB")
        x = pil_to_tensor(img)
        y = int(r["y"])
        return x, y

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    class_names = ckpt["class_names"]

    ds = TileDataset(args.test_csv, args.data_root, class_names)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = SmallCNN(num_classes=len(class_names))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    ys, ps = [], []
    for x, y in dl:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        ys.append(y.numpy()); ps.append(pred)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

if __name__ == "__main__":
    main()
