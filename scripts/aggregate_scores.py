import argparse
import pandas as pd
import numpy as np

def topk_mean(x: np.ndarray, frac: float) -> float:
    if x.size == 0:
        return float("nan")
    k = max(1, int(round(x.size * frac)))
    xs = np.sort(x)[-k:]
    return float(xs.mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_csv", required=True, help="Output of predict_tiles.py")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--topk_frac", type=float, default=0.15, help="fraction of tiles used for score (robust peakiness)")
    ap.add_argument("--ignore_threshold", type=float, default=0.5, help="if p_ignore > this, tile is excluded from scoring")
    args = ap.parse_args()

    df = pd.read_csv(args.preds_csv)

    p_unclean_cols = [c for c in df.columns if c == "p_unclean"]
    if not p_unclean_cols:
        raise SystemExit("Missing p_unclean column. Train/predict with class name 'unclean'.")

    p_ignore = df["p_ignore"].values if "p_ignore" in df.columns else None
    use = np.ones(len(df), dtype=bool)
    if p_ignore is not None:
        use = p_ignore <= args.ignore_threshold

    df_use = df[use].copy()

    rows = []
    for vid, g in df_use.groupby("video_id"):
        pu = g["p_unclean"].values.astype(np.float32)
        score = topk_mean(pu, args.topk_frac)
        frac_dirty = float((pu > 0.5).mean()) if pu.size else float("nan")
        rows.append({
            "video_id": vid,
            "n_tiles_used": int(pu.size),
            "score_topk_mean": score,
            "dirty_frac_p>0.5": frac_dirty,
        })

    out = pd.DataFrame(rows).sort_values("score_topk_mean", ascending=False)
    out.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)

if __name__ == "__main__":
    main()
