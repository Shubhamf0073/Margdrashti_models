import argparse, os, shutil
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="prep_out/labels.csv")
    ap.add_argument("--out_dir", default="prep_out_labelsubset")
    ap.add_argument("--per_video", type=int, default=200, help="tiles per video")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_csv(args.labels)

    df = df[df["tile_path"].apply(lambda p: Path(p).exists())].copy()

    sub = (
        df.groupby("video_id", group_keys=False)
          .apply(lambda g: g.sample(n=min(args.per_video, len(g)), random_state=args.seed))
          .reset_index(drop=True)
    )

    out_dir = Path(args.out_dir)
    tiles_out = out_dir / "tiles"
    tiles_out.mkdir(parents=True, exist_ok=True)

    new_paths = []
    for p in sub["tile_path"].tolist():
        src = Path(p)
        dst = tiles_out / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        new_paths.append(str(dst))

    sub["tile_path"] = new_paths
    sub["label_id"] = ""
    sub["label_name"] = ""
    (out_dir / "labels_subset.csv").write_text(sub.to_csv(index=False))

    print("Wrote:", out_dir / "labels_subset.csv")
    print("Tiles:", tiles_out)
    print("Total subset tiles:", len(sub))
    print("Videos covered:", sub["video_id"].nunique())

if __name__ == "__main__":
    main()
