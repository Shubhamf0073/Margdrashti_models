import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_smallcnn import SmallCNN
from data_utils import pil_to_tensor

class TileDataset(Dataset):
    def __init__(self, labels_csv, data_root):
        self.df = pd.read_csv(labels_csv)
        self.data_root = Path(data_root)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = self.data_root / r["tile_path"]
        img = Image.open(p).convert("RGB")
        x = pil_to_tensor(img)
        return x, i

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    class_names = ckpt["class_names"]

    ds = TileDataset(args.labels_csv, args.data_root)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = SmallCNN(num_classes=len(class_names))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    probs = np.zeros((len(ds), len(class_names)), dtype=np.float32)

    for x, idx in tqdm(dl, desc="predict"):
        x = x.to(device)
        p = F.softmax(model(x), dim=1).cpu().numpy()
        probs[idx.numpy(), :] = p

    df = pd.read_csv(args.labels_csv)
    for j, name in enumerate(class_names):
        df[f"p_{name}"] = probs[:, j]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)

if __name__ == "__main__":
    main()
