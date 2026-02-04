import argparse
import pandas as pd
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="/Users/shubhamfufal/kaushal/data/road_audit_labeled_6792_clean/labels.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    df = df[df["label_name"].notna()].copy()

    vids = df["video_id"].unique().tolist()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(vids)

    n = len(vids)
    n_test = max(1, int(round(n * args.test_frac)))
    n_val  = max(1, int(round(n * args.val_frac)))
    test_vids = set(vids[:n_test])
    val_vids  = set(vids[n_test:n_test+n_val])
    train_vids = set(vids[n_test+n_val:])

    def sub(vset):
        return df[df["video_id"].isin(vset)].copy()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sub(train_vids).to_csv(out/"train.csv", index=False)
    sub(val_vids).to_csv(out/"val.csv", index=False)
    sub(test_vids).to_csv(out/"test.csv", index=False)

    print("Videos:", n, "train/val/test:", len(train_vids), len(val_vids), len(test_vids))
    print("Wrote:", out/"train.csv", out/"val.csv", out/"test.csv")

if __name__ == "__main__":
    main()
