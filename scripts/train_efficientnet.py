import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import time

from model_efficientnet import RoadAuditEfficientNet, count_parameters
from data_utils import augment_pil, pil_to_tensor

LABELS_2 = ["clean", "unclean"]
LABELS_3 = ["clean", "unclean", "ignore"]


class TileDataset(Dataset):
    def __init__(self, csv_path, data_root, drop_ignore=False, train=False):
        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.drop_ignore = drop_ignore
        self.train = train

        self.df = self.df[self.df["label_name"].notna()].copy()
        if drop_ignore:
            self.df = self.df[self.df["label_name"] != "ignore"].copy()

        if not drop_ignore:
            self.class_names = LABELS_3
        else:
            self.class_names = LABELS_2

        self.label_to_idx = {n: i for i, n in enumerate(self.class_names)}

        self.df["y"] = self.df["label_name"].map(self.label_to_idx)
        self.df = self.df[self.df["y"].notna()].copy()
        self.df["y"] = self.df["y"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = self.data_root / r["tile_path"]
        img = Image.open(p).convert("RGB")
        if self.train:
            img = augment_pil(img)
        x = pil_to_tensor(img)
        y = int(r["y"])
        return x, y


def compute_class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    w = 1.0 / counts
    w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names, verbose=True):
    model.eval()
    ys, ps = [], []
    running_loss = 0.0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        
        pred = logits.argmax(1).cpu().numpy()
        ys.append(y.cpu().numpy())
        ps.append(pred)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    
    loss = running_loss / len(loader.dataset)
    acc = (y_true == y_pred).mean()
    
    if verbose:
        labels = list(range(len(class_names)))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred, labels=labels))
        print("\nClassification Report:")
        print(classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=class_names,
            digits=4,
            zero_division=0
        ))
    
    return loss, acc, y_true, y_pred


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    return running_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--train_csv", required=True, help="Training CSV with labels")
    ap.add_argument("--val_csv", required=True, help="Validation CSV with labels")
    ap.add_argument("--data_root", required=True, help="Root directory containing tiles")
    ap.add_argument("--out_dir", required=True, help="Output directory for checkpoints")
    
    ap.add_argument("--stage1_epochs", type=int, default=10, help="Epochs for stage 1 (frozen backbone)")
    ap.add_argument("--stage2_epochs", type=int, default=10, help="Epochs for stage 2 (full fine-tuning)")
    ap.add_argument("--skip_stage2", action="store_true", help="Skip stage 2 fine-tuning")
    ap.add_argument("--batch", type=int, default=64, help="Batch size")
    ap.add_argument("--lr_stage1", type=float, default=1e-3, help="Learning rate for stage 1")
    ap.add_argument("--lr_stage2", type=float, default=3e-5, help="Learning rate for stage 2")
    ap.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    
    ap.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    ap.add_argument("--drop_ignore", action="store_true", help="Drop 'ignore' class")
    
    ap.add_argument("--num_workers", type=int, default=4, help="Data loading workers")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("LOADING DATASETS")
    print("="*60)
    train_ds = TileDataset(args.train_csv, args.data_root, drop_ignore=args.drop_ignore, train=True)
    val_ds = TileDataset(args.val_csv, args.data_root, drop_ignore=args.drop_ignore, train=False)

    n_classes = len(train_ds.class_names)
    
    print(f"\nDataset info:")
    print(f"  Training:   {len(train_ds):,} samples")
    print(f"  Validation: {len(val_ds):,} samples")
    print(f"  Classes:    {train_ds.class_names}")
    
    train_counts = train_ds.df["y"].value_counts().sort_index()
    print(f"\nClass distribution (training):")
    for idx, name in enumerate(train_ds.class_names):
        count = train_counts.get(idx, 0)
        pct = 100 * count / len(train_ds)
        print(f"  {name:10s}: {count:5d} ({pct:5.1f}%)")

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"INITIALIZING MODEL")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    model = RoadAuditEfficientNet(num_classes=n_classes, pretrained=True, dropout=args.dropout)
    model = model.to(device)
    
    print(f"Architecture: EfficientNet-B0")
    print(f"Parameters:   {count_parameters(model):,}")

    y_train = train_ds.df["y"].values
    class_weights = compute_class_weights(y_train, n_classes).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'stage': []
    }

    best_val_loss = float('inf')
    best_val_acc = 0.0

    print(f"\n{'='*60}")
    print(f"STAGE 1: TRAINING WITH FROZEN BACKBONE")
    print(f"{'='*60}")
    print(f"Epochs: {args.stage1_epochs}")
    print(f"Learning rate: {args.lr_stage1}")
    
    model.freeze_backbone()
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_stage1,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.stage1_epochs,
        eta_min=args.lr_stage1 * 0.01
    )

    start_time = time.time()
    
    for epoch in range(1, args.stage1_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, args.stage1_epochs
        )
        
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device, train_ds.class_names,
            verbose=(epoch % 5 == 0 or epoch == args.stage1_epochs)
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['stage'].append(1)
        
        print(f"\nEpoch {epoch}/{args.stage1_epochs}")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")
        print(f"  LR:    {current_lr:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'stage': 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'class_names': train_ds.class_names,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'args': vars(args)
            }
            torch.save(checkpoint, out / "best_stage1.pt")
            print(f"  ★ Saved best_stage1.pt (val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f})")

    stage1_time = time.time() - start_time
    print(f"\nStage 1 completed in {stage1_time/60:.1f} minutes")
    print(f"Best val_loss: {best_val_loss:.4f}, val_acc: {best_val_acc:.4f}")

    if not args.skip_stage2:
        print(f"\n{'='*60}")
        print(f"STAGE 2: FULL FINE-TUNING")
        print(f"{'='*60}")
        print(f"Epochs: {args.stage2_epochs}")
        print(f"Learning rate: {args.lr_stage2}")
        
        checkpoint = torch.load(out / "best_stage1.pt")
        model.load_state_dict(checkpoint['model'])
        
        model.unfreeze_backbone()
        print(f"Trainable parameters: {count_parameters(model):,}")
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr_stage2,
            weight_decay=args.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.stage2_epochs,
            eta_min=args.lr_stage2 * 0.01
        )

        start_time = time.time()
        
        for epoch in range(1, args.stage2_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                epoch, args.stage2_epochs
            )
            
            val_loss, val_acc, _, _ = evaluate(
                model, val_loader, criterion, device, train_ds.class_names,
                verbose=(epoch % 5 == 0 or epoch == args.stage2_epochs)
            )
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['stage'].append(2)
            
            print(f"\nEpoch {epoch}/{args.stage2_epochs} (Stage 2)")
            print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
            print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")
            print(f"  LR:    {current_lr:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'stage': 2,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'class_names': train_ds.class_names,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'args': vars(args)
                }
                torch.save(checkpoint, out / "best.pt")
                print(f"  ★ Saved best.pt (val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f})")
        
        stage2_time = time.time() - start_time
        print(f"\nStage 2 completed in {stage2_time/60:.1f} minutes")

    checkpoint = {
        'model': model.state_dict(),
        'class_names': train_ds.class_names,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'args': vars(args),
        'history': history
    }
    torch.save(checkpoint, out / "last.pt")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation acc:  {best_val_acc:.4f}")
    print(f"\nSaved models:")
    print(f"  {out}/best_stage1.pt  (stage 1 best)")
    if not args.skip_stage2:
        print(f"  {out}/best.pt         (overall best)")
    print(f"  {out}/last.pt         (final epoch)")
    print(f"\nNext steps:")
    print(f"  1. Evaluate: python eval_efficientnet.py --ckpt {out}/best.pt ...")
    print(f"  2. Predict:  python predict_efficientnet.py --ckpt {out}/best.pt ...")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
